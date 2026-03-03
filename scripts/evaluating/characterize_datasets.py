"""
characterize_datasets.py
------------------------
Computes quantitative characterization attributes for all source datasets
in a given folder. Produces a single semicolon-separated CSV with one row
per dataset, containing all attributes from tab:characterization.

Automatically computed:
    Size & Composition, Class Distribution, Computed Transfer Variables
    (vocabulary overlap, TF-IDF similarity, perplexity x6, size ratio)

Manually filled (left empty in output CSV):
    Identification, Linguistic, Domain, Annotation Scheme,
    Quality Metrics, Dialectal Distance

Missing value conventions:
    NA  - not applicable by design
    NR  - information exists but not documented in original source

Usage:
    # Minimal (no GPU)
    python characterize_datasets.py \
        --datasets_dir  data/preprocessing/datasets/ \
        --target_corpus data/mlm/corpus_clean.txt \
        --output        results/characterization.csv

    # Full (after all pre-trainings complete)
    python characterize_datasets.py \
        --datasets_dir       data/preprocessing/datasets/ \
        --target_corpus      data/mlm/corpus_clean.txt \
        --target_test_set    data/test_set_final.csv \
        --beto_base          dccuchile/bert-base-spanish-wwm-uncased \
        --beto_cgec          /scratch/nicolasal97/im_project/beto_cgec \
        --roberta_base       PlanTL-GOB-ES/roberta-base-bne \
        --roberta_cgec       /scratch/nicolasal97/im_project/roberta_bne_cgec \
        --xlmr_base          xlm-roberta-base \
        --xlmr_cgec          /scratch/nicolasal97/im_project/xlm_roberta_cgec \
        --output             results/characterization.csv
"""

import os
import re
import math
import logging
import argparse
import warnings
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────

MANUAL_COLS = [
    "year", "doi_url",
    "variety", "register", "text_genre",
    "primary_domain", "topical_scope", "domain_specificity",
    "theoretical_framework", "granularity_level", "component_types",
    "task_framing", "annotation_scheme_compatibility",
    "granularity_match_with_target",
    "iaa_metric", "iaa_value",
    "dialectal_distance",
]

COL_ORDER = (
    ["dataset"]
    + ["year", "doi_url"]
    + ["variety", "register", "text_genre"]
    + ["primary_domain", "topical_scope", "domain_specificity"]
    + ["total_instances", "total_tokens", "unique_tokens",
       "avg_instance_length", "median_instance_length"]
    + ["theoretical_framework", "granularity_level", "component_types",
       "task_framing", "annotation_scheme_compatibility",
       "granularity_match_with_target"]
    + ["iaa_metric", "iaa_value"]
    + ["binary_argumentative_pct", "binary_nonargumentative_pct",
       "binary_balance_ratio", "component_premise_pct", "component_claim_pct"]
    + ["vocabulary_overlap", "domain_similarity_tfidf",
       "linguistic_distance_beto",    "domain_distance_beto",
       "linguistic_distance_roberta", "domain_distance_roberta",
       "linguistic_distance_xlmr",    "domain_distance_xlmr",
       "size_ratio_log", "dialectal_distance"]
)


# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def compute_size_metrics(df):
    lengths = df["sentence_text"].apply(lambda x: len(tokenize(str(x))))
    all_tokens = []
    for text in df["sentence_text"]:
        all_tokens.extend(tokenize(str(text)))
    return {
        "total_instances":        len(df),
        "total_tokens":           len(all_tokens),
        "unique_tokens":          len(set(all_tokens)),
        "avg_instance_length":    round(float(lengths.mean()), 2),
        "median_instance_length": round(float(lengths.median()), 2),
    }


def compute_class_distribution(df):
    result = {}
    if "label_binary" in df.columns and df["label_binary"].notna().any():
        counts = df["label_binary"].str.strip().str.lower().value_counts()
        total  = len(df)
        maj, minn = counts.max(), counts.min()
        result.update({
            "binary_argumentative_pct":    round(counts.get("argumentative",     0) / total * 100, 2),
            "binary_nonargumentative_pct": round(counts.get("non-argumentative", 0) / total * 100, 2),
            "binary_balance_ratio":        round(maj / minn, 3) if minn > 0 else "NA",
        })
    else:
        result.update({
            "binary_argumentative_pct": "NA",
            "binary_nonargumentative_pct": "NA",
            "binary_balance_ratio": "NA",
        })

    if "label_component" in df.columns and df["label_component"].notna().any():
        comp     = df["label_component"].str.strip().str.lower()
        comp_arg = comp[comp != "none"]
        n        = len(comp_arg)
        result.update({
            "component_premise_pct": round(comp_arg.eq("premise").sum() / n * 100, 2) if n > 0 else "NA",
            "component_claim_pct":   round(comp_arg.eq("claim").sum()   / n * 100, 2) if n > 0 else "NA",
        })
    else:
        result.update({"component_premise_pct": "NA", "component_claim_pct": "NA"})
    return result


def compute_vocabulary_overlap(source_tokens, target_tokens):
    intersection = len(source_tokens & target_tokens)
    union        = len(source_tokens | target_tokens)
    return round(intersection / union, 4) if union > 0 else 0.0


def compute_domain_similarity(source_text, target_text, max_features=1000):
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
        tfidf = vectorizer.fit_transform([source_text, target_text])
        return round(float(cosine_similarity(tfidf[0], tfidf[1])[0][0]), 4)
    except Exception as e:
        log.warning(f"  TF-IDF failed: {e}")
        return None


def compute_perplexity(texts, model_path, batch_size=32, max_length=512):
    """
    Per-token perplexity of a masked LM on source texts.
    Lower = source is more predictable to the model = more similar.

    Base model  -> linguistic distance (similarity to general Spanish/multilingual)
    CGEC model  -> domain distance (similarity to Peruvian journalism domain)

    NOTE: outputs.loss is already mean cross-entropy over non-padded tokens
    in the batch. We accumulate batch-level means weighted by batch size
    (number of sequences), then exponentiate the average.
    This avoids the overflow bug caused by multiplying loss by token count.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        log.info(f"    Loading: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModelForMaskedLM.from_pretrained(model_path)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        total_loss   = 0.0
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch  = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)

            # Build labels: -100 on padding positions so CrossEntropyLoss
            # ignores them. This is critical for XLM-R which uses SentencePiece
            # and pads differently — without this, padded positions inflate loss
            # to astronomically large values.
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100

            with torch.no_grad():
                outputs  = model(**inputs, labels=labels)
                # outputs.loss = mean CE over non-ignored (-100) tokens only
                n_tokens  = (labels != -100).sum().item()
                total_loss   += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

        avg_loss   = total_loss / total_tokens if total_tokens > 0 else None
        perplexity = math.exp(avg_loss) if avg_loss is not None else None

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return round(perplexity, 4) if perplexity else None

    except Exception as e:
        log.warning(f"  Perplexity failed for {model_path}: {e}")
        return None


def load_target_corpus(corpus_path, max_lines=100_000):
    log.info(f"Loading target corpus: {corpus_path}")
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= max_lines:
                break
    log.info(f"  Loaded {len(lines):,} lines")
    corpus_text = " ".join(lines)
    return corpus_text, set(tokenize(corpus_text))


def count_target_tokens(test_set_path):
    df    = pd.read_csv(test_set_path, sep=";", encoding="utf-8")
    total = sum(len(tokenize(str(t))) for t in df["sentence_text"])
    log.info(f"  Target test set tokens: {total:,}")
    return total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    csv_files = sorted(Path(args.datasets_dir).glob("*.csv"))
    if not csv_files:
        log.error(f"No CSV files found in {args.datasets_dir}")
        return
    log.info(f"Found {len(csv_files)} datasets")

    # Load target corpus
    target_text, target_tokens, target_n_tokens = None, None, None
    if args.target_corpus and os.path.exists(args.target_corpus):
        target_text, target_tokens = load_target_corpus(args.target_corpus)
    if args.target_test_set and os.path.exists(args.target_test_set):
        target_n_tokens = count_target_tokens(args.target_test_set)

    # Perplexity model pairs: (output_column, model_path)
    # BUG FIX: domain_distance_* must use CGEC-adapted models, not base models
    perplexity_models = [
        ("linguistic_distance_beto",     args.beto_base),    # base -> linguistic
        ("domain_distance_beto",         args.beto_cgec),    # CGEC -> domain
        ("linguistic_distance_roberta",  args.roberta_base), # base -> linguistic
        ("domain_distance_roberta",      args.roberta_cgec), # CGEC -> domain
        ("linguistic_distance_xlmr",     args.xlmr_base),    # base -> linguistic
        ("domain_distance_xlmr",         args.xlmr_cgec),    # CGEC -> domain
    ]

    rows = []
    for csv_path in csv_files:
        name = csv_path.stem
        log.info(f"\n{'='*60}\nProcessing: {name}")

        try:
            df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        except Exception as e:
            log.error(f"  Load failed: {e}")
            continue

        if "sentence_text" not in df.columns:
            log.error("  Missing 'sentence_text' — skipping")
            continue

        df  = df.dropna(subset=["sentence_text"])
        row = {"dataset": name}

        for col in MANUAL_COLS:
            row[col] = ""

        log.info("  Size metrics...")
        row.update(compute_size_metrics(df))

        log.info("  Class distribution...")
        row.update(compute_class_distribution(df))

        source_text       = " ".join(df["sentence_text"].astype(str).tolist())
        source_tokens_set = set(tokenize(source_text))
        texts_list        = df["sentence_text"].astype(str).tolist()

        if target_tokens is not None:
            log.info("  Vocabulary overlap (Jaccard)...")
            row["vocabulary_overlap"] = compute_vocabulary_overlap(
                source_tokens_set, target_tokens
            )
        else:
            row["vocabulary_overlap"] = None

        if target_text is not None:
            log.info("  Domain similarity (TF-IDF)...")
            row["domain_similarity_tfidf"] = compute_domain_similarity(
                source_text, target_text, max_features=args.tfidf_features
            )
        else:
            row["domain_similarity_tfidf"] = None

        for col_name, model_path in perplexity_models:
            if model_path:
                log.info(f"  {col_name}...")
                row[col_name] = compute_perplexity(
                    texts_list, model_path,
                    batch_size=args.perplexity_batch_size
                )
            else:
                row[col_name] = None

        if target_n_tokens and row["total_tokens"] > 0:
            row["size_ratio_log"] = round(
                math.log10(row["total_tokens"] / target_n_tokens), 4
            )
        else:
            row["size_ratio_log"] = None

        rows.append(row)
        log.info(f"  Done: {name}")

    out_df = pd.DataFrame(rows)
    for col in COL_ORDER:
        if col not in out_df.columns:
            out_df[col] = None
    out_df = out_df[COL_ORDER]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_csv(args.output, sep=";", index=False, encoding="utf-8")

    log.info(f"\nDone. {len(rows)} datasets saved to {args.output}")
    summary_cols = ["dataset", "total_instances", "binary_balance_ratio",
                    "vocabulary_overlap", "domain_similarity_tfidf",
                    "linguistic_distance_beto", "domain_distance_beto"]
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    print(out_df[[c for c in summary_cols if c in out_df.columns]].to_string(index=False))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute quantitative characterization for all source datasets"
    )
    parser.add_argument("--datasets_dir",          required=True)
    parser.add_argument("--output",                required=True)
    parser.add_argument("--target_corpus",         default=None)
    parser.add_argument("--target_test_set",       default=None)
    parser.add_argument("--beto_base",             default=None)
    parser.add_argument("--beto_cgec",             default=None)
    parser.add_argument("--roberta_base",          default=None)
    parser.add_argument("--roberta_cgec",          default=None)
    parser.add_argument("--xlmr_base",             default=None)
    parser.add_argument("--xlmr_cgec",             default=None)
    parser.add_argument("--tfidf_features",        type=int, default=1000)
    parser.add_argument("--perplexity_batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)