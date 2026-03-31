"""
combine_results.py
──────────────────
Walks subdirectories of a results folder, finds results JSON files,
extracts metadata from the directory name, and writes a single combined CSV.

Each output row contains:
  - metadata parsed from the folder name (model, is_cgec, dataset, dataset_year, task)
  - flattened best_hyperparameters (e.g. learning_rate, num_train_epochs)
  - flattened test_set_metrics (e.g. accuracy, f1_macro)

Folder name convention:
  {model}_{dataset}_{dataset_year}_{task}

  model        — up to the first underscore; hyphens are its internal separator
                 (e.g. beto-base, beto-cgec)
  dataset      — everything between model and dataset_year; may contain
                 underscores (e.g. furman, kovatchev_taule)
  dataset_year — last integer-looking segment before task (e.g. 2022, 2023)
  task         — final segment (e.g. binary, component)

Derived column:
  is_cgec      — 1 if "cgec" appears in model name, else 0

Examples:
  beto-base_furman_2023_component
    → model=beto-base, dataset=furman, dataset_year=2023, task=component, is_cgec=0

  beto-base_kovatchev_taule_2022_component
    → model=beto-base, dataset=kovatchev_taule, dataset_year=2022, task=component, is_cgec=0

  beto-cgec_asohmo_2021_binary
    → model=beto-cgec, dataset=asohmo, dataset_year=2021, task=binary, is_cgec=1

Usage
─────
  # Combine all results.json files under results/
  python combine_results.py --input_dir results/ --output combined.csv

  # Use a different JSON filename
  python combine_results.py --input_dir results/ --output combined.csv --json_filename metrics.json

  # Combine all errors.csv files under results/
  python combine_results.py --errors_dir results/ --output combined_errors.csv

  # Sample an existing combined CSV (10 % stratified by model × dataset × task)
  python combine_results.py --sample combined.csv --output sampled.csv

Options
───────
  --input_dir     Root results directory (combine mode, required unless --sample or --errors_dir)
  --errors_dir    Root results directory (errors mode); walks subdirs for errors.csv files
  --output        Path for the output CSV (required)
  --sample        Path to an existing combined CSV to sample instead of combining
  --frac          Sampling fraction, default 0.10 (10 %)
  --seed          Random seed for reproducibility (default: 42)
  --json_filename JSON file to collect from each subdirectory (default: results.json)
  --errors_filename
                  CSV file to collect from each subdirectory in errors mode (default: errors.csv)
  --sep           Delimiter used in output file (default: ;)
  --skip_errors   Warn and skip unparseable folders instead of aborting

Modes are mutually exclusive: --input_dir, --errors_dir, and --sample cannot be combined.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


# ── folder name parser ────────────────────────────────────────────────────────

def parse_folder_name(folder_name: str) -> dict:
    """
    Parse a folder name into metadata fields.

    Strategy:
      1. Split on '_' to get all segments.
      2. model        = segments[0]           (hyphens are its internal separator)
      3. task         = segments[-1]          (last segment)
      4. dataset_year = segments[-2]          (must be a 4-digit integer)
      5. dataset      = '_'.join(segments[1:-2])  (everything in between)
    """
    parts = folder_name.split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Folder '{folder_name}' has too few underscore-separated parts "
            f"(expected at least 4, got {len(parts)}). "
            f"Required pattern: {{model}}_{{dataset}}_{{dataset_year}}_{{task}}"
        )

    model    = parts[0]
    task     = parts[-1]
    year_str = parts[-2]
    dataset  = "_".join(parts[1:-2])

    if not re.fullmatch(r"\d{4}", year_str):
        raise ValueError(
            f"Folder '{folder_name}': expected a 4-digit year as the "
            f"second-to-last segment, got '{year_str}'."
        )
    dataset_year = int(year_str)

    is_cgec = 1 if "cgec" in model.lower() else 0

    return {
        "model":        model,
        "is_cgec":      is_cgec,
        "dataset":      dataset,
        "dataset_year": dataset_year,
        "task":         task,
    }


# ── JSON flattener ────────────────────────────────────────────────────────────

def flatten_json(data: dict) -> dict:
    """
    Flatten a results JSON into a single-row dict.

    Sections used:
      best_hyperparameters  → flat keys as-is (e.g. learning_rate)
      test_set_metrics      → flat keys as-is (e.g. accuracy, f1_macro)

    All other top-level keys are ignored.
    """
    row: dict = {}
    for key, val in data.get("best_hyperparameters", {}).items():
        row[key] = val
    for key, val in data.get("test_set_metrics", {}).items():
        row[key] = val
    return row


# ── stratified sampler ────────────────────────────────────────────────────────

STRATA_COLS = ["model", "dataset", "task"]

def stratified_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """
    Sample `frac` of rows from df, stratified by model × dataset × task.

    For strata too small to yield even 1 row at the requested fraction,
    1 row is taken so every stratum is represented.
    """
    missing = [c for c in STRATA_COLS if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: Cannot stratify — column(s) {missing} not found in the CSV.\n"
            f"Expected columns: {STRATA_COLS}"
        )

    pieces = []
    for key, group in df.groupby(STRATA_COLS, sort=False):
        n = max(1, round(len(group) * frac))
        pieces.append(group.sample(n=n, random_state=seed))

    sampled = pd.concat(pieces).sample(frac=1, random_state=seed)
    return sampled.reset_index(drop=True)


# ── errors combiner ───────────────────────────────────────────────────────────

def combine_errors(input_dir: Path, output: Path, errors_filename: str,
                   sep: str, skip_errors: bool) -> None:
    """
    Walk subdirectories of `input_dir`, find `errors_filename` files,
    prepend parsed folder metadata to every row, and write a combined CSV
    to `output`.
    """
    csv_paths = sorted(input_dir.glob(f"*/{errors_filename}"))
    if not csv_paths:
        sys.exit(
            f"ERROR: No files named '{errors_filename}' found under '{input_dir}'."
        )

    frames:  list[pd.DataFrame] = []
    skipped: int = 0

    for csv_path in csv_paths:
        folder_name = csv_path.parent.name

        try:
            meta = parse_folder_name(folder_name)
        except ValueError as exc:
            if skip_errors:
                print(f"WARNING: {exc}")
                skipped += 1
                continue
            sys.exit(f"ERROR: {exc}\n(Use --skip_errors to skip bad folder names.)")

        try:
            df = pd.read_csv(csv_path, sep=sep, encoding="utf-8")
        except Exception as exc:
            if skip_errors:
                print(f"WARNING: Could not read '{csv_path}': {exc}")
                skipped += 1
                continue
            sys.exit(f"ERROR: Could not read '{csv_path}': {exc}")

        # Prepend metadata columns to the left of every row.
        for col, val in reversed(meta.items()):
            df.insert(0, col, val)

        frames.append(df)
        print(
            f"  ✓  {folder_name}  ({len(df):,} rows)\n"
            f"     model={meta['model']}, is_cgec={meta['is_cgec']}, "
            f"dataset={meta['dataset']}, year={meta['dataset_year']}, "
            f"task={meta['task']}"
        )

    if not frames:
        sys.exit("ERROR: No data collected — nothing to write.")

    combined = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, sep=sep, index=False, encoding="utf-8")

    print(
        f"\nDone. Combined {len(frames)} file(s)"
        + (f", skipped {skipped}" if skipped else "")
        + f"  →  {output}  ({len(combined):,} rows total)"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-experiment results JSONs under a results directory, "
            "prepending metadata parsed from each folder name. "
            "Alternatively, combine errors CSVs with --errors_dir, or "
            "stratified-sample an existing combined CSV with --sample."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Root results directory whose subdirectories hold results JSONs (combine mode).",
    )
    parser.add_argument(
        "--errors_dir",
        type=Path,
        default=None,
        help=(
            "Root results directory whose subdirectories hold errors CSVs (errors mode). "
            "Mutually exclusive with --input_dir and --sample."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the output CSV.",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        metavar="COMBINED_CSV",
        help="Sample mode: path to an existing combined CSV. "
             "Stratifies by model × dataset × task and writes --frac of each stratum.",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.10,
        help="Fraction of rows to sample per stratum (default: 0.10 = 10%%). "
             "Only used with --sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). Only used with --sample.",
    )
    parser.add_argument(
        "--json_filename",
        default="results.json",
        help="JSON file to collect from each subdirectory (default: results.json).",
    )
    parser.add_argument(
        "--errors_filename",
        default="errors.csv",
        help="CSV file to collect from each subdirectory in errors mode (default: errors.csv).",
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="Delimiter used in the output CSV (default: ';').",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Warn and skip folders that cannot be parsed instead of aborting.",
    )
    args = parser.parse_args()

    # ── mutual exclusivity check ──────────────────────────────────────────────
    modes = [m for m in ("input_dir", "errors_dir", "sample") if getattr(args, m) is not None]
    if len(modes) > 1:
        sys.exit(
            f"ERROR: --input_dir, --errors_dir, and --sample are mutually exclusive "
            f"(got: {', '.join('--' + m for m in modes)})."
        )

    # ── errors mode ───────────────────────────────────────────────────────────
    if args.errors_dir is not None:
        errors_dir: Path = args.errors_dir.resolve()
        if not errors_dir.is_dir():
            sys.exit(f"ERROR: '{errors_dir}' is not a directory.")
        combine_errors(
            input_dir=errors_dir,
            output=args.output,
            errors_filename=args.errors_filename,
            sep=args.sep,
            skip_errors=args.skip_errors,
        )
        return

    # ── sample mode ───────────────────────────────────────────────────────────
    if args.sample is not None:
        sample_path: Path = args.sample.resolve()
        if not sample_path.is_file():
            sys.exit(f"ERROR: '{sample_path}' does not exist or is not a file.")
        if not (0 < args.frac <= 1):
            sys.exit("ERROR: --frac must be between 0 (exclusive) and 1 (inclusive).")

        print(f"Sample mode: reading '{sample_path}' …")
        try:
            df = pd.read_csv(sample_path, sep=args.sep, encoding="utf-8")
        except Exception as exc:
            sys.exit(f"ERROR: Could not read '{sample_path}': {exc}")

        print(
            f"  {len(df):,} rows — stratifying by {STRATA_COLS} "
            f"at {args.frac:.0%} (seed={args.seed}) …"
        )
        sampled = stratified_sample(df, frac=args.frac, seed=args.seed)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_csv(args.output, sep=args.sep, index=False, encoding="utf-8")

        print(f"\n  {'stratum':<45} {'total':>7}  {'sampled':>7}")
        print(f"  {'-'*45} {'-------':>7}  {'-------':>7}")
        for key, grp in df.groupby(STRATA_COLS, sort=True):
            label = "  ×  ".join(str(k) for k in key)
            n_sampled = len(sampled[
                (sampled[STRATA_COLS[0]] == key[0]) &
                (sampled[STRATA_COLS[1]] == key[1]) &
                (sampled[STRATA_COLS[2]] == key[2])
            ])
            print(f"  {label:<45} {len(grp):>7,}  {n_sampled:>7,}")

        print(
            f"\nDone. {len(df):,} → {len(sampled):,} rows "
            f"({len(sampled)/len(df):.1%})  →  {args.output}"
        )
        return

    # ── combine mode ──────────────────────────────────────────────────────────
    if args.input_dir is None:
        sys.exit(
            "ERROR: Provide one of --input_dir (combine mode), "
            "--errors_dir (errors mode), or --sample (sample mode)."
        )

    input_dir: Path = args.input_dir.resolve()
    if not input_dir.is_dir():
        sys.exit(f"ERROR: '{input_dir}' is not a directory.")

    json_paths = sorted(input_dir.glob(f"*/{args.json_filename}"))
    if not json_paths:
        sys.exit(f"ERROR: No files named '{args.json_filename}' found under '{input_dir}'.")

    frames:  list[pd.DataFrame] = []
    skipped: int = 0

    for json_path in json_paths:
        folder_name = json_path.parent.name

        try:
            meta = parse_folder_name(folder_name)
        except ValueError as exc:
            if args.skip_errors:
                print(f"WARNING: {exc}")
                skipped += 1
                continue
            sys.exit(f"ERROR: {exc}\n(Use --skip_errors to skip bad folder names.)")

        try:
            with json_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            row = {**meta, **flatten_json(data)}
            df  = pd.DataFrame([row])
        except Exception as exc:
            if args.skip_errors:
                print(f"WARNING: Could not read '{json_path}': {exc}")
                skipped += 1
                continue
            sys.exit(f"ERROR: Could not read '{json_path}': {exc}")

        frames.append(df)
        print(
            f"  ✓  {folder_name}\n"
            f"     model={meta['model']}, is_cgec={meta['is_cgec']}, "
            f"dataset={meta['dataset']}, year={meta['dataset_year']}, "
            f"task={meta['task']}"
        )

    if not frames:
        sys.exit("ERROR: No data collected — nothing to write.")

    combined = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, sep=args.sep, index=False, encoding="utf-8")

    print(
        f"\nDone. Combined {len(frames)} file(s)"
        + (f", skipped {skipped}" if skipped else "")
        + f"  →  {args.output}  ({len(combined):,} rows total)"
    )


if __name__ == "__main__":
    main()