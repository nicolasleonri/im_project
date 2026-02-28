"""
iaa.py — Inter-Annotator Agreement for LLM Argument Mining Annotations

Computes agreement metrics across 2+ annotated CSVs (output of annotator.py)
and splits sentences into three output CSVs for downstream test set construction.

Metrics computed:
    - Pairwise Cohen's Kappa     (scikit-learn, for every model pair)
    - Fleiss' Kappa              (statsmodels, all raters at once)
    - Krippendorff's Alpha       (krippendorff, nominal)

Output files:
    iaa_report.txt        Human-readable agreement report
    iaa_pairwise.csv      Pairwise Cohen's Kappa table
    agreed.csv            All models agree on both labels, no NC present
                          → label_binary_final / label_component_final filled automatically
                          → label_source = "auto"
    review_sentences.csv  Any label disagreement, no NC involved
                          → sorted most-contested first; fill final labels manually
                          → label_source = "manual"
    errors.csv            Any model returned NC on this sentence
                          → needs manual inspection regardless of agreement
                          → label_source = "manual"

Next step: after manually filling label_binary_final / label_component_final
in review_sentences.csv and errors.csv, run build_test_set.py to produce
the clean test_set_final.csv.

Usage:
    python3 iaa.py \\
        --inputs data/annotated_modelA.csv data/annotated_modelB.csv data/annotated_modelC.csv \\
        --output_dir data/iaa/

Requirements:
    pip install pandas scikit-learn statsmodels krippendorff
"""

import argparse
import csv
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import krippendorff

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
LABEL_COLS = ["label_binary", "label_component"]

ORDINAL_MAP = {
    "label_binary":    {"non-argumentative": 0, "argumentative": 1, "nc": -1},
    "label_component": {"none": 0, "premise": 1, "claim": 2, "nc": -1},
}

META_COLS = [
    "article_id", "newspaper", "date", "year",
    "headline", "content",
    "paragraph_i", "paragraph_text",
    "sentence_j", "sentence_text",
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


# ------------------------------------------------------------------
# Loading & merging
# ------------------------------------------------------------------

def load_and_validate(paths: list[str], id_cols: list[str]) -> dict[str, pd.DataFrame]:
    """Load each CSV, validate required columns, return {model_name: df}."""
    dfs = {}
    required = set(id_cols + LABEL_COLS)
    for path in paths:
        name = Path(path).stem
        log.info(f"Loading: {path}  →  model '{name}'")
        df = pd.read_csv(
            path,
            sep=";",
            decimal=".",
            na_values="NA",
            quotechar='"',
            encoding="utf-8",
        )
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"File '{path}' is missing columns: {missing}")
        for col in LABEL_COLS:
            df[col] = df[col].fillna("NC").astype(str).str.strip().str.lower()
        dfs[name] = df
    return dfs


def merge_annotations(dfs: dict[str, pd.DataFrame], id_cols: list[str]) -> pd.DataFrame:
    """
    Inner-join all model DataFrames on id_cols.
    Label columns are suffixed with the model name: label_binary_<model>, etc.
    Metadata columns (META_COLS) are taken from the first model's file only.
    """
    merged = None
    for name, df in dfs.items():
        rename = {col: f"{col}_{name}" for col in LABEL_COLS}
        available_meta = [c for c in META_COLS if c in df.columns]
        keep_cols = list(set(id_cols) | set(available_meta) | set(LABEL_COLS))
        tmp = df[[c for c in keep_cols if c in df.columns]].rename(columns=rename)

        if merged is None:
            merged = tmp
        else:
            label_cols_new = list(rename.values())
            merged = merged.merge(
                tmp[id_cols + label_cols_new],
                on=id_cols,
                how="inner",
            )

    log.info(f"Merged dataset: {len(merged)} sentences common to all models.")
    return merged


# ------------------------------------------------------------------
# Fleiss' Kappa
# ------------------------------------------------------------------

def fleiss_kappa_from_labels(label_matrix: np.ndarray, categories: list) -> float:
    """Build a count matrix and compute Fleiss' kappa via statsmodels."""
    n_subjects, n_raters = label_matrix.shape
    n_cat = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((n_subjects, n_cat), dtype=int)
    for i in range(n_subjects):
        for j in range(n_raters):
            val = label_matrix[i, j]
            if val in cat_to_idx:
                counts[i, cat_to_idx[val]] += 1
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa
        return float(fleiss_kappa(counts, method="fleiss"))
    except Exception as e:
        log.warning(f"Fleiss kappa computation failed: {e}")
        return float("nan")


# ------------------------------------------------------------------
# Agreement metrics
# ------------------------------------------------------------------

def compute_pairwise_kappa(
    merged: pd.DataFrame,
    model_names: list[str],
    label_col: str,
) -> pd.DataFrame:
    """Pairwise Cohen's Kappa for every model pair on one label column."""
    rows = []
    for m1, m2 in itertools.combinations(model_names, 2):
        y1 = merged[f"{label_col}_{m1}"].tolist()
        y2 = merged[f"{label_col}_{m2}"].tolist()
        try:
            kappa = cohen_kappa_score(y1, y2)
        except Exception as e:
            log.warning(f"Cohen's kappa failed for {m1} vs {m2} on {label_col}: {e}")
            kappa = float("nan")
        rows.append({
            "model_1": m1, "model_2": m2,
            "label": label_col,
            "cohen_kappa": round(kappa, 4),
        })
    return pd.DataFrame(rows)


def compute_fleiss(
    merged: pd.DataFrame,
    model_names: list[str],
    label_col: str,
) -> float:
    """Fleiss' kappa across all models for one label column."""
    cols = [f"{label_col}_{m}" for m in model_names]
    matrix = merged[cols].values
    all_vals = sorted(set(v for row in matrix for v in row))
    return fleiss_kappa_from_labels(matrix, all_vals)


def compute_krippendorff(
    merged: pd.DataFrame,
    model_names: list[str],
    label_col: str,
) -> float:
    """Krippendorff's Alpha (nominal) for one label column."""
    cols = [f"{label_col}_{m}" for m in model_names]
    mapping = ORDINAL_MAP[label_col]
    data = []
    for col in cols:
        encoded = merged[col].map(mapping).replace(-1, np.nan).tolist()
        data.append(encoded)
    try:
        alpha = krippendorff.alpha(
            reliability_data=data,
            level_of_measurement="nominal",
        )
        return float(alpha)
    except Exception as e:
        log.warning(f"Krippendorff alpha failed for {label_col}: {e}")
        return float("nan")


# ------------------------------------------------------------------
# 3-way split
# ------------------------------------------------------------------

def split_sentences(
    merged: pd.DataFrame,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split merged DataFrame into three non-overlapping groups:

        errors  — any model returned NC (parsing failure); isolated first
        agreed  — all models agree on BOTH labels, no NC present
        review  — any label disagreement, no NC present

    Returns (agreed_df, review_df, errors_df).

    agreed_df already has label_binary_final / label_component_final populated.
    review_df and errors_df have those columns set to 'PENDING' for manual filling.
    """
    # ── 1. Isolate NC rows ────────────────────────────────────────────
    nc_mask = pd.Series(False, index=merged.index)
    for label_col in LABEL_COLS:
        for m in model_names:
            nc_mask |= merged[f"{label_col}_{m}"] == "nc"
    errors_df = merged[nc_mask].copy()
    clean = merged[~nc_mask].copy()

    # ── 2. Agreed vs review among NC-free rows ───────────────────────
    disagree_mask = pd.Series(False, index=clean.index)
    for label_col in LABEL_COLS:
        cols = [f"{label_col}_{m}" for m in model_names]
        disagree_mask |= clean[cols].nunique(axis=1) > 1

    review_df = clean[disagree_mask].copy()
    agreed_df = clean[~disagree_mask].copy()

    # ── 3. Enrich review_df ──────────────────────────────────────────
    for label_col in LABEL_COLS:
        cols = [f"{label_col}_{m}" for m in model_names]
        review_df[f"{label_col}_majority"] = review_df[cols].mode(axis=1).iloc[:, 0]
        majority = review_df[f"{label_col}_majority"]
        review_df[f"{label_col}_n_agree"] = review_df[cols].eq(majority, axis=0).sum(axis=1)

    # Sort review: most contested (lowest total agreement) first
    review_df["_total_agree"] = (
        review_df["label_binary_n_agree"] + review_df["label_component_n_agree"]
    )
    review_df = (
        review_df.sort_values("_total_agree")
        .drop(columns=["_total_agree"])
        .reset_index(drop=True)
    )

    # ── 4. Final label columns ───────────────────────────────────────
    # agreed: fill automatically from unanimous label
    first = model_names[0]
    agreed_df["label_binary_final"]    = agreed_df[f"label_binary_{first}"]
    agreed_df["label_component_final"] = agreed_df[f"label_component_{first}"]
    agreed_df["label_source"]          = "auto"

    # review & errors: placeholder — fill manually before build_test_set.py
    for df in [review_df, errors_df]:
        df["label_binary_final"]    = "PENDING"
        df["label_component_final"] = "PENDING"
        df["label_source"]          = "manual"

    log.info(
        f"\nSplit results:\n"
        f"  agreed  : {len(agreed_df):>6} sentences  (label_source='auto', ready for test set)\n"
        f"  review  : {len(review_df):>6} sentences  (disagreement — fill PENDING labels)\n"
        f"  errors  : {len(errors_df):>6} sentences  (NC present  — fill PENDING labels)"
    )
    return agreed_df, review_df, errors_df


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def interpret_kappa(k: float) -> str:
    if np.isnan(k): return "N/A"
    if k < 0:       return "Poor (< 0)"
    if k < 0.20:    return "Slight"
    if k < 0.40:    return "Fair"
    if k < 0.60:    return "Moderate"
    if k < 0.80:    return "Substantial"
    return "Almost Perfect"


def build_report(
    model_names: list[str],
    n_total: int,
    n_agreed: int,
    n_review: int,
    n_errors: int,
    pairwise_df: pd.DataFrame,
    fleiss_scores: dict,
    kripp_scores: dict,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  INTER-ANNOTATOR AGREEMENT REPORT")
    lines.append("=" * 70)
    lines.append(f"\nModels compared  : {', '.join(model_names)}")
    lines.append(f"Total sentences  : {n_total}")
    lines.append(f"  → agreed       : {n_agreed:>6}  ({100*n_agreed/n_total:.1f}%)  auto-labeled")
    lines.append(f"  → review       : {n_review:>6}  ({100*n_review/n_total:.1f}%)  needs manual labels")
    lines.append(f"  → errors (NC)  : {n_errors:>6}  ({100*n_errors/n_total:.1f}%)  needs manual labels\n")

    if n_errors > 0:
        lines.append(
            f"  ⚠  {n_errors} sentences contain NC labels. They are included in the metrics\n"
            f"     below as a separate 'nc' category, which may deflate agreement scores.\n"
        )

    for label_col in LABEL_COLS:
        lines.append("-" * 70)
        lines.append(f"  Label: {label_col}")
        lines.append("-" * 70)
        fk = fleiss_scores[label_col]
        ka = kripp_scores[label_col]
        lines.append(f"  Fleiss' Kappa        : {fk:.4f}  [{interpret_kappa(fk)}]")
        lines.append(f"  Krippendorff's Alpha : {ka:.4f}  [{interpret_kappa(ka)}]")
        lines.append(f"\n  Pairwise Cohen's Kappa:")
        sub = pairwise_df[pairwise_df["label"] == label_col]
        for _, row in sub.iterrows():
            k = row["cohen_kappa"]
            lines.append(
                f"    {row['model_1']:30s} vs {row['model_2']:30s}"
                f"  κ = {k:.4f}  [{interpret_kappa(k)}]"
            )
        lines.append("")

    lines.append("=" * 70)
    lines.append("  Kappa interpretation (Landis & Koch 1977):")
    lines.append("    < 0.00  Poor  |  0.00–0.20  Slight  |  0.21–0.40  Fair")
    lines.append("    0.41–0.60  Moderate  |  0.61–0.80  Substantial  |  0.81–1.00  Almost Perfect")
    lines.append("=" * 70)
    lines.append("\n  Next steps:")
    lines.append("    1. Open review_sentences.csv — fill label_binary_final and")
    lines.append("       label_component_final for every PENDING row.")
    lines.append("    2. Open errors.csv — do the same.")
    lines.append("    3. Run build_test_set.py to merge all three CSVs into")
    lines.append("       test_set_final.csv.")
    lines.append("=" * 70)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, **CSV_OPTS)
    log.info(f"Saved {len(df):>6} rows → {path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute IAA and split sentences into agreed / review / errors."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="2+ annotated CSV files (output of annotator.py)."
    )
    parser.add_argument(
        "--output_dir", default="data/iaa/",
        help="Directory for output files (default: data/iaa/)."
    )
    parser.add_argument(
        "--id_cols", nargs="+", default=["article_id", "sentence_j"],
        help="Columns that uniquely identify a sentence (default: article_id sentence_j)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.inputs) < 2:
        raise ValueError("At least 2 input files are required.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & merge
    dfs = load_and_validate(args.inputs, args.id_cols)
    model_names = list(dfs.keys())
    merged = merge_annotations(dfs, args.id_cols)

    # 2. Metrics (computed on full merged set, including NC rows)
    pairwise_parts, fleiss_scores, kripp_scores = [], {}, {}
    for label_col in LABEL_COLS:
        log.info(f"Computing metrics for: {label_col}")
        pairwise_parts.append(compute_pairwise_kappa(merged, model_names, label_col))
        fleiss_scores[label_col] = compute_fleiss(merged, model_names, label_col)
        kripp_scores[label_col]  = compute_krippendorff(merged, model_names, label_col)
    pairwise_df = pd.concat(pairwise_parts, ignore_index=True)

    # 3. 3-way split
    agreed_df, review_df, errors_df = split_sentences(merged, model_names)

    # 4. Report
    report_str = build_report(
        model_names=model_names,
        n_total=len(merged),
        n_agreed=len(agreed_df),
        n_review=len(review_df),
        n_errors=len(errors_df),
        pairwise_df=pairwise_df,
        fleiss_scores=fleiss_scores,
        kripp_scores=kripp_scores,
    )
    print("\n" + report_str)
    (out_dir / "iaa_report.txt").write_text(report_str, encoding="utf-8")
    log.info(f"Report saved → {out_dir / 'iaa_report.txt'}")

    # 5. Save all outputs
    save_csv(pairwise_df, out_dir / "iaa_pairwise.csv")
    save_csv(agreed_df,   out_dir / "agreed.csv")
    save_csv(review_df,   out_dir / "review_sentences.csv")
    save_csv(errors_df,   out_dir / "errors.csv")

    log.info(
        "\nDone. Fill PENDING labels in review_sentences.csv and errors.csv, "
        "then run build_test_set.py."
    )


if __name__ == "__main__":
    main()