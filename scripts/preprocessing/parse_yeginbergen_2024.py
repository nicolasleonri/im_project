"""
parse_bio_json.py — BIO-tagged JSONL Argumentative Corpus Parser

Converts BIO-tagged JSONL files into the project's unified schema.
Accepts a single .json/.jsonl file or a folder, producing a single
merged output CSV.

Input structure (one JSON object per line):
    {
        "text": ["token1", "token2", ...],
        "tags": ["B-Claim", "I-Claim", "O", "B-Premise", ...]
    }

BIO tag mapping:
    B-Claim / I-Claim     → argumentative + claim
    B-Premise / I-Premise → argumentative + premise
    O                     → non-argumentative + none

Each contiguous span of the same label becomes one row.

Output schema:
    sentence_text | label_binary | label_component

Usage:
    # Single file
    python parse_bio_json.py \\
        --input  data/external/corpus.jsonl \\
        --output data/adapted/bio_adapted.csv

    # Folder of .jsonl files
    python parse_bio_json.py \\
        --input  data/external/bio/ \\
        --output data/adapted/bio_adapted.csv

Requirements:
    pip install pandas
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATASET_NAME = "bio_json_corpus"

TAG_MAP = {
    "Claim":   ("argumentative",     "claim"),
    "Premise": ("argumentative",     "premise"),
    "O":       ("non-argumentative", "none"),
}

CSV_OPTS = dict(
    index=False,
    header=True,
    encoding="utf-8",
    na_rep="NA",
    sep=";",
    quotechar='"',
    quoting=csv.QUOTE_ALL,
    decimal=".",
    errors="strict",
)


# ------------------------------------------------------------------
# BIO span extraction
# ------------------------------------------------------------------

def extract_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """
    Collapse BIO-tagged tokens into contiguous spans, each becoming one row.
    Consecutive O tags are merged into a single non-argumentative span.
    """
    if len(tokens) != len(tags):
        raise ValueError(f"Token/tag length mismatch: {len(tokens)} vs {len(tags)}")

    records = []
    current_label = None
    current_tokens = []

    def flush(label, toks):
        if not toks:
            return
        text = " ".join(toks).strip()
        if not text:
            return
        # Normalize label: strip B-/I- prefix, or keep "O"
        base = label[2:] if label.startswith(("B-", "I-")) else label
        if base not in TAG_MAP:
            log.warning(f"  Unknown tag base '{base}' — skipping span.")
            return
        label_binary, label_component = TAG_MAP[base]
        records.append({
            "sentence_text":   text,
            "label_binary":    label_binary,
            "label_component": label_component,
        })

    for token, tag in zip(tokens, tags):
        base = tag[2:] if tag.startswith(("B-", "I-")) else tag  # "Claim", "Premise", "O"
        is_begin = tag.startswith("B-") or tag == "O"

        if current_label is None:
            current_label = base
            current_tokens = [token]
        elif base == current_label and not is_begin:
            # Continuation of same span (I- tag)
            current_tokens.append(token)
        elif base == "O" and current_label == "O":
            # Merge consecutive O tokens into one span
            current_tokens.append(token)
        else:
            # New span starts — flush previous
            flush(current_label, current_tokens)
            current_label = base
            current_tokens = [token]

    flush(current_label, current_tokens)
    return records


# ------------------------------------------------------------------
# Core parsing
# ------------------------------------------------------------------

def parse_jsonl(path: Path) -> list[dict]:
    """Parse a single JSON file into a list of segment dicts."""
    records = []
    n_sentences = 0
    n_errors = 0

    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)  # Un solo JSON, no JSONL
        except json.JSONDecodeError as e:
            log.warning(f"  [{path.name}] JSON parse error: {e} — skipping.")
            return []

    texts = obj.get("text", [])
    tags  = obj.get("tags", [])

    if len(texts) != len(tags):
        log.warning(f"  [{path.name}] 'text' and 'tags' have different lengths — skipping.")
        return []

    for sent_idx, (tokens, sent_tags) in enumerate(zip(texts, tags), start=1):
        if not tokens or not sent_tags:
            log.warning(f"  [{path.name}] Sentence {sent_idx}: empty tokens or tags — skipping.")
            n_errors += 1
            continue

        try:
            spans = extract_spans(tokens, sent_tags)
        except ValueError as e:
            log.warning(f"  [{path.name}] Sentence {sent_idx}: {e} — skipping.")
            n_errors += 1
            continue

        n_sentences += 1
        records.extend(spans)

    log.info(
        f"  {path.name}: {n_sentences} sentences, {n_errors} errors → "
        f"{len(records)} segments"
    )
    return records


# ------------------------------------------------------------------
# Input resolution
# ------------------------------------------------------------------

def resolve_input_files(input_path: str, ext: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob(f"*{ext}"))
        if not files:
            raise FileNotFoundError(
                f"No '{ext}' files found in directory: {p}\n"
                f"Use --ext to specify a different extension."
            )
        return files
    raise FileNotFoundError(f"Input path does not exist: {p}")


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    valid_binary    = {"argumentative", "non-argumentative"}
    valid_component = {"claim", "premise", "none"}
    bad_b = ~df["label_binary"].isin(valid_binary)
    bad_c = ~df["label_component"].isin(valid_component)
    if bad_b.any():
        raise ValueError(f"Invalid label_binary values: {df.loc[bad_b, 'label_binary'].unique()}")
    if bad_c.any():
        raise ValueError(f"Invalid label_component values: {df.loc[bad_c, 'label_component'].unique()}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse BIO-tagged JSONL file(s) into project schema CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single .jsonl file OR a folder containing multiple files."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the merged output CSV."
    )
    parser.add_argument(
        "--ext", default=".json",
        help="File extension to glob when --input is a folder (default: .json)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    files = resolve_input_files(args.input, args.ext)
    log.info(f"Found {len(files)} file(s) to process:")
    for f in files:
        log.info(f"  {f}")

    all_records = []
    for f in files:
        all_records.extend(parse_jsonl(f))

    if not all_records:
        log.error("No segments extracted. Check file format and structure.")
        sys.exit(1)

    df = pd.DataFrame(all_records).reset_index(drop=True)
    validate(df)

    log.info(
        f"\nAdapted dataset summary ({DATASET_NAME}):\n"
        f"  Files processed  : {len(files)}\n"
        f"  Total segments   : {len(df)}\n\n"
        f"  label_binary:\n{df['label_binary'].value_counts().to_string()}\n\n"
        f"  label_component:\n{df['label_component'].value_counts().to_string()}"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, **CSV_OPTS)
    log.info(f"\nSaved → {args.output}  ({len(df)} rows × {len(df.columns)} columns)")


if __name__ == "__main__":
    main()