"""
parse_nlas.py — NLAS-multi Spanish Subset Parser

Converts NLAS-multi JSON annotations into the project's unified schema.
Accepts either a single .json file or a folder of .json files, producing
a single merged output CSV.

NLAS-multi structure:
    {
        "esp": {
            "0": {
                "topic": "...",
                "stance": "...",
                "argumentation scheme": "...",
                "argument": "{
                    \"major premise\": \"...\",
                    \"minor premise\": \"...\",
                    \"conclusion\": \"...\"
                }",
                "label": "yes"
            }, ...
        }
    }

Mapping (each entry generates 3 rows):
    major premise → argumentative + premise
    minor premise → argumentative + premise
    conclusion    → argumentative + claim

Discarded: topic, stance, argumentation scheme, label.

Output schema:
    sentence_text | label_binary | label_component | source_dataset

Usage:
    # Single file
    python parse_nlas.py \\
        --input  data/external/nlas.json \\
        --output data/adapted/nlas_adapted.csv

    # Folder of .json files
    python parse_nlas.py \\
        --input  data/external/nlas/ \\
        --output data/adapted/nlas_adapted.csv

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

DATASET_NAME = "nlas_multi_2024"

COMPONENT_MAP = {
    "major premise": ("argumentative", "premise"),
    "minor premise": ("argumentative", "premise"),
    "conclusion":    ("argumentative", "claim"),
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

def parse_json(path: Path) -> list[dict]:
    """Parse a single NLAS JSON file into a list of segment dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    n_entries = 0
    n_errors  = 0

    # Top-level key may be "esp" or other language codes — process all
    for lang_key, entries in data.items():
        for idx, entry in entries.items():
            n_entries += 1
            raw_argument = entry.get("argument", "")

            # The argument field is a JSON-encoded string — parse it
            try:
                argument = json.loads(raw_argument)
            except (json.JSONDecodeError, TypeError):
                log.warning(f"  [{path.name}] Entry {idx}: failed to parse argument JSON — skipping.")
                n_errors += 1
                continue

            for component_key, (label_binary, label_component) in COMPONENT_MAP.items():
                text = argument.get(component_key, "").strip()
                if not text:
                    log.warning(
                        f"  [{path.name}] Entry {idx}: missing '{component_key}' — skipping component."
                    )
                    continue
                records.append({
                    "sentence_text":   text,
                    "label_binary":    label_binary,
                    "label_component": label_component,
                    # "source_dataset":  DATASET_NAME,
                })

    log.info(
        f"  {path.name}: {n_entries} entries, {n_errors} parse errors → "
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
        description="Parse NLAS-multi JSON file(s) into project schema CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single .json file OR a folder containing multiple .json files."
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
        all_records.extend(parse_json(f))

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