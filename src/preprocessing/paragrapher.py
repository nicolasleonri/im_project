from utils_preprocessing import *
import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import spacy_udpipe
from tqdm import tqdm
from vllm import LLM, SamplingParams

"""
paragrapher.py

Pipeline for extracting and preprocessing a stratified test set from the CGEC13-20 informal economy subset.

Workflow:
    1. Read and preprocess input CSV
    2. LLM (Mistral via vLLM) extracts paragraphs from article content
    3. Programmatically fill output CSV with paragraphs
    4. Programmatically fill output CSV with sentences (spacy_udpipe)
    5. Stratified sampling of target N paragraphs

Output columns:
    article_id, newspaper, date, title, content,
    paragraph_i, paragraph_text,
    sentence_j, sentence_text

Usage:
    python paragrapher.py \
        --input data/informal_economy.csv \
        --output data/test_set.csv \
        --n_samples 600 \
        --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
        --min_stratum_size 10
"""

def parse_llm_response(response_text: str, article_id: str) -> list[str]:
    """
    Parse LLM JSON response into a list of paragraph strings.
    Falls back to newline splitting if JSON parsing fails.
    """
    # Try to extract JSON block from response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            paragraphs = data.get("paragraphs", [])
            if isinstance(paragraphs, list) and all(isinstance(p, str) for p in paragraphs):
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
                if paragraphs:
                    return paragraphs
        except json.JSONDecodeError:
            pass

    log.warning(f"[{article_id}] JSON parsing failed. Falling back to newline split.")
    # Fallback: split by double newline or single newline
    fallback = [p.strip() for p in re.split(r'\n{2,}|\n', response_text) if p.strip()]
    return fallback if fallback else [response_text.strip()]

def extract_paragraphs_llm(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> list[dict]:
    """
    Use vLLM to extract paragraphs from each article.

    Returns a list of dicts:
        {article_id, newspaper, date, year, title, content,
         paragraph_i, paragraph_text}
    """
    log.info(f"Loading vLLM model: {model_name}")
    llm = LLM(model=model_name)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    prompts = build_prompts(df)
    article_ids = df["article_id"].tolist()
    records = []

    log.info(f"Running LLM inference on {len(prompts)} articles (batch_size={batch_size})")
    for start in tqdm(range(0, len(prompts), batch_size), desc="LLM batches"):
        batch_prompts = prompts[start : start + batch_size]
        batch_ids = article_ids[start : start + batch_size]
        batch_rows = df.iloc[start : start + batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        for i, (output, article_id, (_, row)) in enumerate(
            zip(outputs, batch_ids, batch_rows.iterrows())
        ):
            response_text = output.outputs[0].text
            paragraphs = parse_llm_response(response_text, article_id)

            for p_idx, para_text in enumerate(paragraphs):
                records.append({
                    "article_id":      article_id,
                    "newspaper":       row["newspaper"],
                    "date":            row["date"],
                    "year":            row["year"],
                    "title":           row["title"],
                    "content":         row["content"],
                    "paragraph_i":     p_idx,
                    "paragraph_text":  para_text,
                })

    log.info(f"Extracted {len(records)} paragraphs from {len(df)} articles.")
    return records


def segment_sentences(records: list[dict], nlp) -> list[dict]:
    """
    For each paragraph record, segment into sentences using spacy_udpipe.

    Returns a list of dicts with an additional sentence_j and sentence_text.
    """
    log.info("Segmenting paragraphs into sentences with spacy_udpipe...")
    sentence_records = []

    paragraph_texts = [r["paragraph_text"] for r in records]

    # Process in batches for efficiency
    for record, doc in tqdm(
        zip(records, nlp.pipe(paragraph_texts, batch_size=64)),
        total=len(records),
        desc="Sentence segmentation",
    ):
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        for s_idx, sent_text in enumerate(sentences):
            sentence_records.append({
                **record,
                "sentence_j":    s_idx,
                "sentence_text": sent_text,
            })

    log.info(f"Produced {len(sentence_records)} sentence records.")
    return sentence_records

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and preprocess stratified test set from CGEC13-20 subset."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV (must contain: title, content, newspaper, date)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for output CSV."
    )
    parser.add_argument(
        "--n_samples", type=int, default=600,
        help="Target number of paragraphs to sample (default: 600)."
    )
    parser.add_argument(
        "--model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        help="vLLM model name for paragraph extraction."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for vLLM inference (default: 32)."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Max tokens for LLM response (default: 4096)."
    )
    parser.add_argument(
        "--min_stratum_size", type=int, default=10,
        help="Minimum paragraphs per stratum before falling back (default: 10)."
    )
    parser.add_argument(
        "--no_sample", action="store_true",
        help="If set, output all extracted paragraphs without sampling."
    )
    parser.add_argument(
        "--udpipe_lang", default="es",
        help="Language code for spacy_udpipe (default: es)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Step 1: Load input
    df = load_input(args.input)

    # Steps 2 & 3: LLM paragraph extraction
    paragraph_records = extract_paragraphs_llm(
        df,
        model_name=args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )
    para_df = pd.DataFrame(paragraph_records)

    # Steps 4 & 5: Sentence segmentation
    nlp = load_udpipe_model(args.udpipe_lang)
    sentence_records = segment_sentences(para_df, nlp)
    sent_df = pd.DataFrame(sentence_records)

    # Step 6: Stratified sampling (optional)
    if args.no_sample:
        log.info("Skipping sampling. Outputting all extracted data.")
        output_df = sent_df
    else:
        output_df = stratified_sample(
            sent_df,
            n_samples=args.n_samples,
            min_stratum_size=args.min_stratum_size,
        )

    # Reorder columns for output
    output_cols = [
        "article_id", "newspaper", "date", "year",
        "title", "content",
        "paragraph_i", "paragraph_text",
        "sentence_j", "sentence_text",
    ]
    output_df = output_df[output_cols]

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False, encoding="utf-8")
    log.info(f"Output saved to: {args.output}")
    log.info(f"Final shape: {output_df.shape[0]} rows x {output_df.shape[1]} columns.")


if __name__ == "__main__":
    main()