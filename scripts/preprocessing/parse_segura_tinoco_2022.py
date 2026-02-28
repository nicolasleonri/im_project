"""
parse_comp_csv.py — Argumentative Component CSV Parser

Converts _comp.csv annotated files into the project's unified schema.
Accepts a single _comp.csv file or a folder of _comp.csv files, producing
a single merged output CSV.

Input structure:
    ac_id | ac_text | ac_type | annotator | timestamp

    ac_type values:
        major claim → argumentative + claim
        claim       → argumentative + claim
        premise     → argumentative + premise

Discarded: ac_id, annotator, timestamp.

Output schema:
    sentence_text | label_binary | label_component

Usage:
    # Single file
    python parse_comp_csv.py \\
        --input  data/external/file_comp.csv \\
        --output data/adapted/comp_adapted.csv

    # Folder of _comp.csv files
    python parse_comp_csv.py \\
        --input  data/external/comp/ \\
        --output data/adapted/comp_adapted.csv

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

DATASET_NAME = "comp_csv"

AC_TYPE_MAP = {
    "major claim": ("argumentative", "claim"),
    "claim":       ("argumentative", "claim"),
    "premise":     ("argumentative", "premise"),
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

def parse_csv(path: Path) -> list[dict]:
    """Parse a single _comp.csv file into a list of segment dicts."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        log.warning(f"  [{path.name}] Failed to read CSV: {e} — skipping.")
        return []

    records = []
    n_errors = 0

    for _, row in df.iterrows():
        ac_type = str(row.get("ac_type", "")).strip().lower()
        text    = str(row.get("ac_text", "")).strip()

        if not text or text == "nan":
            log.warning(f"  [{path.name}] Empty ac_text — skipping.")
            n_errors += 1
            continue

        if ac_type not in AC_TYPE_MAP:
            log.warning(f"  [{path.name}] Unknown ac_type '{ac_type}' — skipping.")
            n_errors += 1
            continue

        label_binary, label_component = AC_TYPE_MAP[ac_type]
        records.append({
            "sentence_text":   text,
            "label_binary":    label_binary,
            "label_component": label_component,
        })

    log.info(
        f"  {path.name}: {len(df)} entries, {n_errors} errors → "
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
        description="Parse _comp.csv file(s) into project schema CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single _comp.csv file OR a folder containing multiple _comp.csv files."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the merged output CSV."
    )
    parser.add_argument(
        "--ext", default="_comp.csv",
        help="File extension to glob when --input is a folder (default: _comp.csv)."
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
        all_records.extend(parse_csv(f))

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