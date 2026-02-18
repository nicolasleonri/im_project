"""
extract_test_set.py

Pipeline for extracting and preprocessing a stratified test set
from the CGEC13-20 informal economy subset.

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
    python extract_test_set.py \
        --input data/informal_economy.csv \
        --output data/test_set.csv \
        --n_samples 600 \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --min_stratum_size 10
"""

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"title", "content", "newspaper", "date"}

PARAGRAPH_PROMPT = """You are a text segmentation assistant. Your task is to identify and extract the paragraphs from a Spanish journalistic article.

Rules:
- Extract the paragraphs exactly as they appear in the text, preserving the original wording.
- A paragraph is a thematically coherent block of text. Short transitional sentences may form their own paragraph.
- Ignore headers, captions, or metadata that are not part of the article body.
- Return ONLY a valid JSON object with a single key "paragraphs" containing a list of strings.
- Do not add explanations, comments or any text outside the JSON.

Example output format:
{{"paragraphs": ["First paragraph text.", "Second paragraph text.", "Third paragraph text."]}}

Article:
{content}"""


# ---------------------------------------------------------------------------
# Step 1: Read and preprocess input
# ---------------------------------------------------------------------------

def load_input(path: str) -> pd.DataFrame:
    """Load input CSV and validate required columns."""
    log.info(f"Loading input from: {path}")
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    original_len = len(df)
    df = df.dropna(subset=["content", "title"])
    df = df[df["content"].str.strip().str.len() > 0]
    log.info(f"Loaded {original_len} articles, {len(df)} after dropping empty content.")

    # Normalize date to year for stratification
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Assign stable article_id
    df = df.reset_index(drop=True)
    df["article_id"] = df.index.map(lambda i: f"ART_{i:06d}")

    return df


# ---------------------------------------------------------------------------
# Step 2 & 3: LLM paragraph extraction
# ---------------------------------------------------------------------------

def build_prompts(df: pd.DataFrame) -> list[str]:
    """Build one prompt per article."""
    return [
        PARAGRAPH_PROMPT.format(content=row["content"])
        for _, row in df.iterrows()
    ]


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
    batch_size: int = 32,
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


# ---------------------------------------------------------------------------
# Step 4 & 5: Sentence segmentation with spacy_udpipe
# ---------------------------------------------------------------------------

def load_udpipe_model(lang: str = "es") -> object:
    """Download and load the spacy_udpipe model for Spanish."""
    log.info(f"Loading spacy_udpipe model for language: {lang}")
    try:
        nlp = spacy_udpipe.load(lang)
    except Exception:
        log.info("Model not found locally. Downloading...")
        spacy_udpipe.download(lang)
        nlp = spacy_udpipe.load(lang)
    return nlp


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


# ---------------------------------------------------------------------------
# Step 6: Stratified sampling
# ---------------------------------------------------------------------------

def report_distribution(df: pd.DataFrame, strat_cols: list[str]) -> None:
    """Log stratum sizes for a given stratification."""
    dist = df.groupby(strat_cols)["paragraph_i"].nunique().reset_index()
    dist.columns = strat_cols + ["n_paragraphs"]
    log.info(f"\nParagraph distribution by {strat_cols}:\n{dist.to_string(index=False)}")


def stratified_sample(
    df: pd.DataFrame,
    n_samples: int,
    min_stratum_size: int = 10,
) -> pd.DataFrame:
    """
    Sample n_samples paragraphs with stratification.

    Strategy:
        1. Try stratification by newspaper + year.
        2. If any stratum has fewer than min_stratum_size paragraphs,
           fall back to stratification by newspaper only.
        3. Allocate samples proportionally to stratum size.
        4. Report final distribution.

    Sampling is at the paragraph level (one row per paragraph,
    all sentences of sampled paragraphs are included).
    """
    # Work at paragraph level (unique paragraph per article)
    para_df = (
        df.groupby(["article_id", "paragraph_i"])
        .first()
        .reset_index()[["article_id", "newspaper", "year", "paragraph_i", "paragraph_text"]]
    )

    # --- Attempt 1: stratify by newspaper + year ---
    strat_cols = ["newspaper", "year"]
    stratum_sizes = para_df.groupby(strat_cols).size()
    small_strata = (stratum_sizes < min_stratum_size).sum()

    if small_strata > 0:
        log.warning(
            f"{small_strata} strata have fewer than {min_stratum_size} paragraphs "
            f"under newspaper+year stratification. Falling back to newspaper only."
        )
        strat_cols = ["newspaper"]
        stratum_sizes = para_df.groupby(strat_cols).size()

    report_distribution(df, strat_cols)

    # --- Proportional allocation ---
    total_paragraphs = stratum_sizes.sum()
    allocations = (stratum_sizes / total_paragraphs * n_samples).round().astype(int)

    # Adjust rounding errors to hit exactly n_samples
    diff = n_samples - allocations.sum()
    if diff != 0:
        # Add/subtract from largest stratum
        largest = allocations.idxmax()
        allocations[largest] += diff

    log.info(f"\nSample allocation per stratum:\n{allocations.to_string()}")

    # --- Sample ---
    sampled_parts = []
    for stratum_key, n_alloc in allocations.items():
        if isinstance(stratum_key, str):
            stratum_key = (stratum_key,)
        mask = pd.Series([True] * len(para_df))
        for col, val in zip(strat_cols, stratum_key):
            mask &= para_df[col] == val
        stratum_df = para_df[mask]

        if len(stratum_df) < n_alloc:
            log.warning(
                f"Stratum {stratum_key} has only {len(stratum_df)} paragraphs, "
                f"requested {n_alloc}. Sampling with replacement."
            )
            sampled = stratum_df.sample(n=n_alloc, replace=True, random_state=42)
        else:
            sampled = stratum_df.sample(n=n_alloc, replace=False, random_state=42)

        sampled_parts.append(sampled)

    sampled_paras = pd.concat(sampled_parts, ignore_index=True)
    sampled_keys = set(
        zip(sampled_paras["article_id"], sampled_paras["paragraph_i"])
    )

    # Retrieve all sentence rows for sampled paragraphs
    mask = df.apply(
        lambda r: (r["article_id"], r["paragraph_i"]) in sampled_keys, axis=1
    )
    result = df[mask].copy()

    log.info(
        f"\nFinal sample: {len(sampled_paras)} paragraphs, "
        f"{len(result)} sentence rows."
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        "--model", default="mistralai/Mistral-7B-Instruct-v0.3",
        help="vLLM model name for paragraph extraction."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for vLLM inference (default: 32)."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048,
        help="Max tokens for LLM response (default: 2048)."
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
    sentence_records = segment_sentences(paragraph_records, nlp)
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