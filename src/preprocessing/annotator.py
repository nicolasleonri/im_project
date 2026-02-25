from utils_preprocessing import *
import argparse
import json
import re
import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

"""
annotator.py

Pipeline for LLM-assisted annotation of argumentative components in sentences
extracted from the CGEC13-20 informal economy subset.

Workflow:
    1. Read preprocessed CSV (output of paragrapher.py)
    2. Load annotation guidelines from markdown file
    3. LLM annotates each sentence in paragraph + article context
    4. Output CSV with annotation columns

Input columns (from paragrapher.py):
    article_id, newspaper, date, year, headline, content,
    paragraph_i, paragraph_text, sentence_j, sentence_text

Output columns (added):
    label_binary     -- argumentative | non-argumentative
    label_component  -- claim | premise | none
    confidence_binary    -- float [0-1]
    confidence_component -- float [0-1]
    reasoning        -- brief justification in Spanish

Usage:
    python annotator.py \\
        --input data/preprocessing/test_set.csv \\
        --output data/preprocessing/test_set_annotated.csv \\
        --guidelines docs/annotation_guidelines_v1.md \\
        --model mistralai/Mistral-Small-3.2-24B-Instruct-2506
"""

def parse_annotation_response(
    response_text: str,
    article_id: str,
    sentence_j: int,
) -> dict:
    """
    Parse LLM JSON response into annotation fields.
    Returns a dict with label_binary, label_component,
    confidence_binary, confidence_component, reasoning.
    Falls back to safe defaults if parsing fails.
    """
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())

            label_binary = str(data.get("label_binary", "")).strip().lower()
            label_component = str(data.get("label_component", "")).strip().lower()
            confidence_binary = float(data.get("confidence_binary", 0.0))
            confidence_component = float(data.get("confidence_component", 0.0))
            reasoning = str(data.get("reasoning", "")).strip()

            # Validate values
            if label_binary not in VALID_BINARY:
                raise ValueError(f"Invalid label_binary: {label_binary}")
            if label_component not in VALID_COMPONENT:
                raise ValueError(f"Invalid label_component: {label_component}")

            # Enforce consistency: non-argumentative → none
            if label_binary == "non-argumentative" and label_component != "none":
                log.warning(
                    f"[{article_id}|s{sentence_j}] Inconsistent labels: "
                    f"non-argumentative but label_component={label_component}. "
                    f"Correcting to none."
                )
                label_component = "none"

            return {
                "label_binary": label_binary,
                "label_component": label_component,
                "confidence_binary": round(confidence_binary, 4),
                "confidence_component": round(confidence_component, 4),
                "reasoning": reasoning,
                "annotation_error": False,
            }

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.warning(f"[{article_id}|s{sentence_j}] Parsing error: {e}")

    log.warning(
        f"[{article_id}|s{sentence_j}] Annotation failed. Recording as NC."
    )
    return {
        "label_binary": "NC",
        "label_component": "NC",
        "confidence_binary": None,
        "confidence_component": None,
        "reasoning": f"Parsing failed. Raw response: {response_text[:200]}",
        "annotation_error": True,
    }


def annotate_sentences(
    df: pd.DataFrame,
    guidelines: str,
    model_name: str,
    tokenizer: str = None,
    tokenizer_mode: str = "auto",
    tensor_parallel_size: int = 1,
    batch_size: int = 32,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_model_len: int = 32768,
) -> pd.DataFrame:
    """
    Annotate all sentences in df using vLLM.
    Returns df with annotation columns added.
    """
    log.info(f"Loading vLLM model: {model_name}")
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer if tokenizer else model_name,
        tokenizer_mode=tokenizer_mode,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 0},
        # enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=42,
    )

    log.info(f"Building annotation prompts for {len(df)} sentences...")
    prompts = build_annotation_prompts(df, guidelines, model_name)

    annotation_results = []
    log.info(f"Running annotation inference (batch_size={batch_size})...")

    for start in tqdm(range(0, len(prompts), batch_size), desc="Annotation batches"):
        batch_prompts = prompts[start: start + batch_size]
        batch_rows = df.iloc[start: start + batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        for output, (_, row) in zip(outputs, batch_rows.iterrows()):
            response_text = output.outputs[0].text
            result = parse_annotation_response(
                response_text,
                article_id=row["article_id"],
                sentence_j=int(row["sentence_j"]),
            )
            annotation_results.append(result)

    # Merge annotations back into df
    annotation_df = pd.DataFrame(annotation_results)
    result_df = pd.concat(
        [df.reset_index(drop=True), annotation_df.reset_index(drop=True)],
        axis=1,
    )

    # Report annotation statistics
    n_errors = annotation_df["annotation_error"].sum()
    n_total = len(annotation_df)
    log.info(f"Annotation complete: {n_total} sentences, {n_errors} errors ({100*n_errors/n_total:.1f}%).")
    log.info(f"\nLabel distribution (binary):\n{result_df['label_binary'].value_counts().to_string()}")
    log.info(f"\nLabel distribution (component):\n{result_df['label_component'].value_counts().to_string()}")

    return result_df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-assisted annotation of argumentative components."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to preprocessed CSV (output of paragrapher.py)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for annotated output CSV."
    )
    parser.add_argument(
        "--guidelines", required=True,
        help="Path to annotation guidelines markdown file."
    )
    parser.add_argument(
        "--model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        help="vLLM model name for annotation."
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="Tokenizer name or path (defaults to model)."
    )
    parser.add_argument(
        "--tokenizer_mode", default="auto",
        help="Tokenizer mode for vLLM (default: auto)."
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for vLLM inference (default: 32)."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="Max tokens for annotation response (default: 512)."
    )
    parser.add_argument(
        "--max_model_len", type=int, default=32768,
        help="Max model context length (default: 32768)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load input
    log.info(f"Loading input from: {args.input}")
    df = pd.read_csv(
        args.input,
        sep=";",
        decimal=".",
        na_values="NA",
        quotechar='"',
        encoding="utf-8",
    )
    log.info(f"Loaded {len(df)} sentence records.")

    # Load guidelines
    guidelines = load_guidelines(args.guidelines)

    # Annotate
    result_df = annotate_sentences(
        df=df,
        guidelines=guidelines,
        model_name=args.model,
        tokenizer=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
    )

    clear_gpu_memory()
    log.info("GPU memory cleared after LLM inference and sentence segmentation.")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(
        args.output,
        index=False,
        header=True,
        encoding="utf-8",
        na_rep="NA",
        sep=";",
        quotechar='"',
        date_format="%Y-%m-%d",
        quoting=csv.QUOTE_ALL,
        decimal=".",
        errors="strict",
    )

    clear_gpu_memory()
    log.info(f"Annotated output saved to: {args.output}")
    log.info(f"Final shape: {result_df.shape[0]} rows x {result_df.shape[1]} columns.")


if __name__ == "__main__":
    main()