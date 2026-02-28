"""
parse_asohmo.py — ASOHMO CoNLL Parser

Converts ASOHMO token-level CoNLL annotations into the project's unified schema.
Accepts either a single .conll file or a folder of .conll files, producing
a single merged output CSV.

ASOHMO annotation structure (tab-separated, 9 columns):
    Col 0: token
    Col 1: O (unused)
    Col 2: Justification tag   (Premise2Justification | O)
    Col 3: Conclusion tag      (Premise1Conclusion | O)
    Col 4: Collective tag      (Collective | O)         → discarded
    Col 5: Property tag        (Property | O)           → discarded
    Col 6: Pivot tag           (pivot | O)              → discarded
    Col 7: Proposition type    (fact | policy | value)  → discarded
    Col 8: O (unused)

Mapping:
    Premise2Justification → argumentative + premise
    Premise1Conclusion    → argumentative + claim
    Counter narratives    → discarded (not part of original argument)
    O segments            → discarded (non-annotated tokens between segments)

Segmentation logic:
    - Tokens with the same active tag (col 2 or col 3) are grouped into a segment
    - A blank line in the CoNLL file signals a new tweet/document
    - Segments are reconstructed as full sentences by joining tokens with spaces

Output schema:
    sentence_text  | label_binary | label_component | source_dataset | source_file

Usage:
    # Single file
    python parse_asohmo.py \\
        --input  data/external/asohmo.conll \\
        --output data/adapted/asohmo_adapted.csv

    # Folder of .conll files
    python parse_asohmo.py \\
        --input  data/external/asohmo/ \\
        --output data/adapted/asohmo_adapted.csv

    # Folder with a different extension
    python parse_asohmo.py \\
        --input  data/external/asohmo/ \\
        --output data/adapted/asohmo_adapted.csv \\
        --ext .txt

Requirements:
    pip install pandas
"""

import argparse
import csv
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

DATASET_NAME = "asohmo_2023"

# Column indices
COL_TOKEN         = 0
COL_JUSTIFICATION = 2
COL_CONCLUSION    = 3

# Tag to label mapping
TAG_MAP = {
    "premise2justification": ("argumentative", "premise"),
    "premise1conclusion":    ("argumentative", "claim"),
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
# Core parsing
# ------------------------------------------------------------------

def get_active_tag(cols: list[str]) -> str | None:
    """
    Return the active argumentative tag for a token row.
    Priority: justification > conclusion (they should be mutually exclusive).
    Returns None if the token is outside any annotated segment.
    """
    justif = cols[COL_JUSTIFICATION].strip().lower()
    conclu = cols[COL_CONCLUSION].strip().lower()
    if justif != "o" and justif:
        return justif
    if conclu != "o" and conclu:
        return conclu
    return None


def parse_conll(path: Path) -> list[dict]:
    """
    Parse a single CoNLL file into a list of segment dicts.
    Each segment is a contiguous block of tokens sharing the same tag.
    Blank lines signal document boundaries (reset current segment).
    """
    records = []
    current_tag    = None
    current_tokens: list[str] = []
    n_docs  = 0
    n_lines = 0

    def flush_segment():
        if current_tokens and current_tag and current_tag in TAG_MAP:
            label_binary, label_component = TAG_MAP[current_tag]
            sentence_text = " ".join(current_tokens).strip()
            if sentence_text:
                records.append({
                    "sentence_text":   sentence_text,
                    "label_binary":    label_binary,
                    "label_component": label_component,
                })

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            line = line.rstrip("\n")

            # Blank line = document boundary
            if not line.strip():
                flush_segment()
                current_tag    = None
                current_tokens = []
                n_docs += 1
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue  # skip malformed lines, headers, comments

            token = cols[COL_TOKEN].strip()
            tag   = get_active_tag(cols)

            if tag != current_tag:
                flush_segment()
                current_tag    = tag
                current_tokens = [token] if token else []
            else:
                if token:
                    current_tokens.append(token)

    flush_segment()  # flush final segment (file may not end with blank line)

    log.info(
        f"  {path.name}: {n_lines} lines, {n_docs} doc boundaries → "
        f"{len(records)} segments"
    )
    return records


# ------------------------------------------------------------------
# Input resolution: file or folder
# ------------------------------------------------------------------

def resolve_input_files(input_path: str, ext: str) -> list[Path]:
    """
    Return a sorted list of CoNLL files to process.
    If input_path is a file, return [input_path].
    If it is a directory, return all files matching ext inside it.
    """
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
    valid_component = {"claim", "premise", "none", "unknown"}
    bad_b = ~df["label_binary"].isin(valid_binary)
    bad_c = ~df["label_component"].isin(valid_component)
    if bad_b.any():
        raise ValueError(
            f"Invalid label_binary values: {df.loc[bad_b, 'label_binary'].unique()}"
        )
    if bad_c.any():
        raise ValueError(
            f"Invalid label_component values: {df.loc[bad_c, 'label_component'].unique()}"
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse ASOHMO CoNLL file(s) into project schema CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single .conll file OR a folder containing multiple .conll files."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the merged output CSV."
    )
    parser.add_argument(
        "--ext", default=".conll",
        help="File extension to glob when --input is a folder (default: .conll)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve input files
    files = resolve_input_files(args.input, args.ext)
    log.info(f"Found {len(files)} file(s) to process:")
    for f in files:
        log.info(f"  {f}")

    # 2. Parse all files
    all_records = []
    for f in files:
        all_records.extend(parse_conll(f))

    if not all_records:
        log.error("No segments extracted from any file. Check column indices and format.")
        sys.exit(1)

    # 3. Build DataFrame
    df = pd.DataFrame(all_records).reset_index(drop=True)

    # 4. Validate
    validate(df)

    # 5. Report
    log.info(
        f"\nAdapted dataset summary ({DATASET_NAME}):\n"
        f"  Files processed  : {len(files)}\n"
        f"  Total segments   : {len(df)}\n\n"
        f"  label_binary:\n{df['label_binary'].value_counts().to_string()}\n\n"
        f"  label_component:\n{df['label_component'].value_counts().to_string()}\n\n"
    )

    # 6. Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, **CSV_OPTS)
    log.info(f"\nSaved → {args.output}  ({len(df)} rows × {len(df.columns)} columns)")


if __name__ == "__main__":
    main()