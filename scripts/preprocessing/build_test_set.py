"""
build_test_set.py — Assemble Final Annotated Test Set

Merges the three CSVs produced by iaa.py (after manual annotation of the
PENDING rows) into a single clean test_set_final.csv.

Input files (from iaa.py output dir):
    agreed.csv             Auto-labeled sentences (no manual work needed)
    review_sentences.csv   Manually labeled disagreements
    errors.csv             Manually labeled NC sentences

Validation performed before writing:
    - No PENDING values remain in label_binary_final / label_component_final
    - label_binary_final values are within the valid set
    - label_component_final values are within the valid set
    - Consistency: non-argumentative → component must be none
    - No duplicate sentence IDs across the three files

Output:
    test_set_final.csv     One row per sentence with columns:
                               article_id, newspaper, date, year,
                               headline, content,
                               paragraph_i, paragraph_text,
                               sentence_j, sentence_text,
                               label_binary_final, label_component_final,
                               label_source        ("auto" | "manual")

Usage:
    python build_test_set.py \\
        --iaa_dir  data/iaa/ \\
        --output   data/test_set_final.csv

    # Use a different output dir for the three input files:
    python build_test_set.py \\
        --agreed   data/iaa/agreed.csv \\
        --review   data/iaa/review_sentences.csv \\
        --errors   data/iaa/errors.csv \\
        --output   data/test_set_final.csv

    # Skip errors.csv if you have none (or haven't resolved them yet):
    python build_test_set.py \\
        --iaa_dir data/iaa/ --skip_errors
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

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
VALID_BINARY    = {"argumentative", "non-argumentative"}
VALID_COMPONENT = {"claim", "premise", "none"}

FINAL_LABEL_COLS = ["label_binary_final", "label_component_final"]

OUTPUT_COLS = [
    "article_id", "newspaper", "date", "year",
    "headline", "content",
    "paragraph_i", "paragraph_text",
    "sentence_j", "sentence_text",
    "label_binary_final", "label_component_final",
    "label_source",
]

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

ID_COLS = ["article_id", "sentence_j"]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load one of the three input CSVs with consistent dtypes."""
    log.info(f"Loading {label}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(
        path,
        sep=";",
        decimal=".",
        na_values="NA",
        quotechar='"',
        encoding="utf-8",
        dtype=str,          # read everything as str to avoid silent coercions
    )
    log.info(f"  → {len(df)} rows")
    return df


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate(df: pd.DataFrame) -> list[str]:
    """
    Run all validation checks on the merged DataFrame.
    Returns a list of error messages (empty = all good).
    """
    errors = []

    # 1. No PENDING values
    for col in FINAL_LABEL_COLS:
        pending = df[col].str.upper().eq("PENDING")
        if pending.any():
            n = pending.sum()
            examples = df.loc[pending, ID_COLS].head(3).to_dict("records")
            errors.append(
                f"[{col}] {n} row(s) still have PENDING values. Examples: {examples}"
            )

    # 2. Valid label_binary_final values
    bad_binary = ~df["label_binary_final"].isin(VALID_BINARY)
    if bad_binary.any():
        bad_vals = df.loc[bad_binary, "label_binary_final"].unique().tolist()
        errors.append(
            f"[label_binary_final] Invalid values found: {bad_vals}. "
            f"Allowed: {sorted(VALID_BINARY)}"
        )

    # 3. Valid label_component_final values
    bad_comp = ~df["label_component_final"].isin(VALID_COMPONENT)
    if bad_comp.any():
        bad_vals = df.loc[bad_comp, "label_component_final"].unique().tolist()
        errors.append(
            f"[label_component_final] Invalid values found: {bad_vals}. "
            f"Allowed: {sorted(VALID_COMPONENT)}"
        )

    # 4. Consistency: non-argumentative → label_component_final must be none
    inconsistent = (
        df["label_binary_final"].eq("non-argumentative") &
        ~df["label_component_final"].eq("none")
    )
    if inconsistent.any():
        n = inconsistent.sum()
        examples = df.loc[inconsistent, ID_COLS + FINAL_LABEL_COLS].head(3).to_dict("records")
        errors.append(
            f"[consistency] {n} row(s) have label_binary_final='non-argumentative' "
            f"but label_component_final != 'none'. Examples: {examples}"
        )

    # 5. No duplicate sentence IDs
    dupes = df.duplicated(subset=ID_COLS, keep=False)
    if dupes.any():
        n = dupes.sum()
        examples = df.loc[dupes, ID_COLS].head(3).to_dict("records")
        errors.append(
            f"[duplicates] {n} rows share the same {ID_COLS}. "
            f"Check for overlapping sentences across input files. Examples: {examples}"
        )

    return errors


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge iaa.py outputs into a clean final test set CSV."
    )
    # Convenience shorthand: point to the whole iaa dir
    parser.add_argument(
        "--iaa_dir", default=None,
        help="Directory containing agreed.csv, review_sentences.csv, errors.csv. "
             "Overridden by --agreed / --review / --errors if provided."
    )
    # Or specify each file individually
    parser.add_argument("--agreed", default=None, help="Path to agreed.csv")
    parser.add_argument("--review", default=None, help="Path to review_sentences.csv")
    parser.add_argument("--errors", default=None, help="Path to errors.csv")
    parser.add_argument(
        "--skip_errors", action="store_true",
        help="Skip errors.csv (use if no NC sentences exist or not yet resolved)."
    )
    parser.add_argument(
        "--output", default="data/test_set_final.csv",
        help="Output path for the final test set CSV (default: data/test_set_final.csv)."
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    """Resolve file paths from --iaa_dir or explicit flags."""
    if args.iaa_dir:
        base = Path(args.iaa_dir)
        agreed_path = Path(args.agreed)  if args.agreed else base / "agreed.csv"
        review_path = Path(args.review)  if args.review else base / "review_sentences.csv"
        errors_path = Path(args.errors)  if args.errors else base / "errors.csv"
    else:
        if not args.agreed or not args.review:
            raise ValueError("Provide --iaa_dir OR both --agreed and --review.")
        agreed_path = Path(args.agreed)
        review_path = Path(args.review)
        errors_path = Path(args.errors) if args.errors else None

    if args.skip_errors:
        errors_path = None

    return agreed_path, review_path, errors_path


def main():
    args = parse_args()
    agreed_path, review_path, errors_path = resolve_paths(args)

    # ── Load ─────────────────────────────────────────────────────────
    parts = [
        load_csv(agreed_path, "agreed"),
        load_csv(review_path, "review"),
    ]
    if errors_path is not None:
        parts.append(load_csv(errors_path, "errors"))
    else:
        log.info("Skipping errors.csv.")

    merged = pd.concat(parts, ignore_index=True)
    log.info(f"Total rows before validation: {len(merged)}")

    # ── Normalise label case ─────────────────────────────────────────
    for col in FINAL_LABEL_COLS:
        if col in merged.columns:
            merged[col] = merged[col].str.strip().str.lower()

    # ── Validate ─────────────────────────────────────────────────────
    validation_errors = validate(merged)
    if validation_errors:
        log.error("Validation failed — fix the issues below before building the test set:\n")
        for i, err in enumerate(validation_errors, 1):
            log.error(f"  {i}. {err}")
        sys.exit(1)

    log.info("Validation passed ✓")

    # ── Select & reorder output columns ──────────────────────────────
    available = [c for c in OUTPUT_COLS if c in merged.columns]
    missing_out = [c for c in OUTPUT_COLS if c not in merged.columns]
    if missing_out:
        log.warning(f"Output columns not found in data (skipped): {missing_out}")

    output_df = merged[available].copy()

    # ── Sort by article and sentence order ───────────────────────────
    sort_cols = [c for c in ["article_id", "paragraph_i", "sentence_j"] if c in output_df.columns]
    output_df = output_df.sort_values(sort_cols).reset_index(drop=True)

    # ── Summary stats ─────────────────────────────────────────────────
    log.info(
        f"\nFinal test set summary:\n"
        f"  Total sentences : {len(output_df)}\n"
        f"  label_source breakdown:\n"
        f"{output_df['label_source'].value_counts().to_string()}\n\n"
        f"  label_binary_final distribution:\n"
        f"{output_df['label_binary_final'].value_counts().to_string()}\n\n"
        f"  label_component_final distribution:\n"
        f"{output_df['label_component_final'].value_counts().to_string()}"
    )

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, **CSV_OPTS)
    log.info(f"\nTest set saved → {out_path}  ({len(output_df)} rows × {len(available)} columns)")


if __name__ == "__main__":
    main()