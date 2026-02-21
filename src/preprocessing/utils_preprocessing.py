import pandas as pd
import logging
import json
import re
import sys
from pathlib import Path
from vllm import LLM, SamplingParams
import spacy_udpipe
from transformers import AutoTokenizer

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Constants
REQUIRED_COLUMNS = {"headline", "content", "newspaper", "date"}

KEEP_COLUMNS = ["headline", "content", "newspaper", "date"]

PROMPT = """Your task is to identify and extract the paragraphs from the following Peruvian Spanish journalistic article.

Rules:
- Extract the paragraphs exactly as they appear in the text, preserving the original wording.
- A paragraph is a thematically coherent block of text. Short transitional sentences may form their own paragraph.
- Ignore headers, captions, or metadata that are not part of the article body.
- Return ONLY a valid JSON object with a single key "paragraphs" containing a list of strings.
- Do not add explanations, comments or any text outside the JSON.

Example output format:
{{"paragraphs": ["First paragraph text.", "Second paragraph text."]}}

Article:
{content}"""

# Functions
def load_input(path: str) -> pd.DataFrame:
    """Load input CSV and validate required columns."""
    log.info(f"Loading input from: {path}")
    df = pd.read_csv(path,
                    sep=";", 
                    decimal=".",
                    na_values="NA", 
                    quotechar='"', 
                    encoding="utf-8")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df[KEEP_COLUMNS].copy() # Discard unnecessary columns early to reduce memory fragmentation

    original_len = len(df)
    df = df.dropna(subset=["content", "headline"])
    df = df[df["content"].str.strip().str.len() > 0]
    log.info(f"Loaded {original_len} articles, {len(df)} after dropping empty content.")

    # Normalize date to year for stratification
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["year"] = df["date"].dt.year

    # Assign stable article_id
    df = df.reset_index(drop=True)
    df["article_id"] = df.index.map(lambda i: f"ART_{i:06d}")

    return df

def build_prompts(df: pd.DataFrame, model_name: str) -> list[str]:    
    if "mistral" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            fix_mistral_regex=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

    prompts = []
    for _, row in df.iterrows():
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a text segmentation assistant specialized in "
                    "Peruvian Spanish journalistic articles. You always respond "
                    "with valid JSON only, no additional text."
                )
            },
            {
                "role": "user",
                "content": PROMPT.format(content=row["content"])
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    
    return prompts

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