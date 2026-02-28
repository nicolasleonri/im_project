"""
finetune_beto.py
----------------
Fine-tuning of BETO (base or domain-adapted) for argumentation component detection
on a source dataset, with Optuna hyperparameter search and evaluation on the
Peruvian Spanish test set.

Pipeline:
    1. Split source dataset 70/30 (train/val)
    2. Optuna search over train/val to find best hyperparameters
    3. Final fine-tuning with best hyperparameters
    4. Evaluation on unseen Peruvian Spanish test set

Input format (CSV with semicolon separator):
    sentence_text;label_binary;label_component

Two tasks:
    - binary   : argumentative | non-argumentative
    - component: claim | premise | none

Usage:
    python finetune_beto.py \
        --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
        --source_dataset data/datasets/guzman.csv \
        --test_dataset data/test_set/test_set_final.csv \
        --task binary \
        --output_dir results/beto-base_guzman_binary \
        --n_trials 20 \
        --bf16

    # With domain-adapted BETO:
    python finetune_beto.py \
        --model_name_or_path models/beto-cgec \
        --source_dataset data/datasets/asohmo.csv \
        --test_dataset data/test_set/test_set_final.csv \
        --task component \
        --output_dir results/beto-cgec_asohmo_component \
        --n_trials 20 \
        --bf16
"""

import os
import json
import logging
import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

SEED = 42


# ─────────────────────────────────────────────
# Label mappings
# ─────────────────────────────────────────────

LABEL_MAPS = {
    "binary": {
        "non-argumentative": 0,
        "argumentative":     1,
    },
    "component": {
        "none":    0,
        "claim":   1,
        "premise": 2,
    }
}


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Loads a semicolon-separated CSV with sentence_text, label_binary, label_component."""
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    required = {"sentence_text", "label_binary", "label_component"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df.dropna(subset=["sentence_text"])
    return df


def load_test_set(path: str) -> pd.DataFrame:
    """
    Loads the annotated Peruvian Spanish test set.
    Expects columns: sentence_text, paragraph_text, label_binary_final, label_component_final
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df = df.rename(columns={
        "label_binary_final":    "label_binary",
        "label_component_final": "label_component",
    })
    df = df.dropna(subset=["sentence_text", "label_binary", "label_component"])
    return df


# ─────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────

def tokenize_dataset(df: pd.DataFrame, tokenizer, task: str, label_map: dict,
                     max_length: int = 512, has_paragraph: bool = False) -> Dataset:
    """
    Tokenizes the dataset for sequence classification.

    If paragraph_text is available (test set), uses sentence + paragraph as input pair:
        [CLS] sentence [SEP] paragraph [SEP]
    Otherwise uses sentence only:
        [CLS] sentence [SEP]
    """
    label_col = "label_binary" if task == "binary" else "label_component"

    df = df.copy()
    df["label"] = df[label_col].str.strip().str.lower().map(label_map)

    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        log.warning(f"Dropping {unmapped} rows with unrecognized labels in '{label_col}'")
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    sentences  = df["sentence_text"].tolist()
    labels     = df["label"].tolist()
    paragraphs = df["paragraph_text"].tolist() \
        if has_paragraph and "paragraph_text" in df.columns else None

    all_encodings = {"label": labels}

    if paragraphs:
        enc = tokenizer(
            sentences, paragraphs,
            truncation=True, max_length=max_length, padding=False,
        )
    else:
        enc = tokenizer(
            sentences,
            truncation=True, max_length=max_length, padding=False,
        )

    all_encodings.update(enc)

    # token_type_ids not always present
    if "token_type_ids" not in enc:
        all_encodings.pop("token_type_ids", None)

    return Dataset.from_dict(all_encodings)


# ─────────────────────────────────────────────
# Weighted loss Trainer
# ─────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that applies class weights to CrossEntropyLoss
    to handle class imbalance in source datasets.
    """
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
            loss_fn = CrossEntropyLoss(weight=weights)
        else:
            loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def build_compute_metrics(label_map: dict):
    """Returns a compute_metrics function for the given label map."""
    id2label = {v: k for k, v in label_map.items()}

    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {
            "accuracy":        accuracy,
            "precision_macro": precision,
            "recall_macro":    recall,
            "f1_macro":        f1,
        }

    return compute_metrics


# ─────────────────────────────────────────────
# Model init factory (required by Optuna search)
# ─────────────────────────────────────────────

def make_model_init(model_name_or_path, num_labels, id2label, label_map):
    """
    Returns a model_init function for Trainer.hyperparameter_search.
    Optuna calls this function to reinitialize the model for each trial.
    """
    def model_init(trial=None):
        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label_map,
            ignore_mismatched_sizes=True,
        )
    return model_init


# ─────────────────────────────────────────────
# Optuna hyperparameter search space
# ─────────────────────────────────────────────

def optuna_hp_space(trial):
    """
    Defines the hyperparameter search space for Optuna.
    Covers the most impactful hyperparameters for BERT fine-tuning.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32]
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.15),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    set_seed(SEED)

    label_map  = LABEL_MAPS[args.task]
    id2label   = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)
    label_col  = "label_binary" if args.task == "binary" else "label_component"

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    log.info(f"Loading source dataset : {args.source_dataset}")
    source_df = load_csv(args.source_dataset)
    log.info(f"  Source samples: {len(source_df):,}")

    log.info(f"Loading test set       : {args.test_dataset}")
    test_df = load_test_set(args.test_dataset)
    log.info(f"  Test samples  : {len(test_df):,}")

    # ── Train/validation split (70/30) ────────────────────────────────────
    train_df, val_df = train_test_split(
        source_df,
        test_size=0.3,
        random_state=SEED,
        stratify=source_df[label_col],
    )
    log.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    # ── Compute class weights from training set ───────────────────────────
    train_labels = (
        train_df[label_col].str.strip().str.lower().map(label_map).dropna().astype(int)
    )
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=train_labels,
    )
    log.info(f"  Class weights: { {id2label[i]: round(w, 3) for i, w in enumerate(class_weights)} }")

    # ── Load tokenizer ────────────────────────────────────────────────────
    log.info(f"Loading tokenizer from : {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # ── Tokenize datasets ─────────────────────────────────────────────────
    log.info("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_df, tokenizer, args.task, label_map)
    val_dataset   = tokenize_dataset(val_df,   tokenizer, args.task, label_map)
    test_dataset  = tokenize_dataset(
        test_df, tokenizer, args.task, label_map, has_paragraph=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Base training arguments (overridden by Optuna per trial) ──────────
    base_training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "optuna_trials"),
        eval_strategy="epoch",
        save_strategy="no",
        fp16=args.fp16,
        bf16=args.bf16,
        seed=SEED,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2,
        # Defaults — will be overridden by Optuna
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
    )

    # ── Optuna hyperparameter search ──────────────────────────────────────
    log.info(f"Starting Optuna search ({args.n_trials} trials)...")

    search_trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model_init=make_model_init(args.model_name_or_path, num_labels, id2label, label_map),
        args=base_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(label_map),
    )

    best_run = search_trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=args.n_trials,
        direction="maximize",
        compute_objective=lambda metrics: metrics["eval_f1_macro"],
    )

    log.info(f"Best hyperparameters found:")
    for k, v in best_run.hyperparameters.items():
        log.info(f"  {k}: {v}")

    # Save best hyperparameters
    hp_path = os.path.join(args.output_dir, "best_hyperparameters.json")
    with open(hp_path, "w") as f:
        json.dump(best_run.hyperparameters, f, indent=2)

    # ── Final fine-tuning with best hyperparameters ───────────────────────
    log.info("Starting final fine-tuning with best hyperparameters...")

    final_training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "final_model"),
        num_train_epochs=best_run.hyperparameters["num_train_epochs"],
        per_device_train_batch_size=best_run.hyperparameters["per_device_train_batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=best_run.hyperparameters["learning_rate"],
        warmup_ratio=best_run.hyperparameters["warmup_ratio"],
        weight_decay=best_run.hyperparameters["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=SEED,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=2,
    )

    final_trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label_map,
            ignore_mismatched_sizes=True,
        ),
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(label_map),
    )

    final_trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────────────
    log.info("Evaluating on Peruvian Spanish test set...")
    test_results = final_trainer.predict(test_dataset)
    preds  = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids

    report = classification_report(
        labels, preds,
        target_names=[id2label[i] for i in sorted(id2label)],
        zero_division=0,
        output_dict=True,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "experiment": {
            "model":          args.model_name_or_path,
            "source_dataset": args.source_dataset,
            "task":           args.task,
            "n_trials":       args.n_trials,
        },
        "best_hyperparameters": best_run.hyperparameters,
        "test_set_metrics": {
            "accuracy":        round(accuracy, 4),
            "precision_macro": round(precision, 4),
            "recall_macro":    round(recall, 4),
            "f1_macro":        round(f1, 4),
        },
        "classification_report": report,
    }

    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save predictions for error analysis (RQ2)
    test_df_out = test_df.copy()
    test_df_out["prediction"] = [id2label[p] for p in preds]
    test_df_out["correct"]    = (test_df_out["prediction"] == test_df_out[label_col])
    preds_path = os.path.join(args.output_dir, "test_predictions.csv")
    test_df_out.to_csv(preds_path, sep=";", index=False, encoding="utf-8")

    # Cleanup checkpoints and optuna trials
    for folder in ["final_model", "optuna_trials"]:
        folder_path = Path(args.output_dir) / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
            log.info(f"Deleted temporary folder: {folder_path}")

    log.info(f"Results saved to      : {results_path}")
    log.info(f"Predictions saved to  : {preds_path}")
    log.info(f"Best hyperparameters  : {hp_path}")
    log.info(f"Test F1 (macro)       : {f1:.4f}")
    log.info(f"Test Accuracy         : {accuracy:.4f}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune BETO for argumentation component detection with Optuna HP search"
    )
    parser.add_argument("--model_name_or_path", required=True,
                        help="BETO base or domain-adapted model path")
    parser.add_argument("--source_dataset",     required=True,
                        help="Path to source dataset CSV (semicolon-separated)")
    parser.add_argument("--test_dataset",       required=True,
                        help="Path to Peruvian Spanish test set CSV")
    parser.add_argument("--task",               required=True,
                        choices=["binary", "component"],
                        help="Classification task")
    parser.add_argument("--output_dir",         required=True,
                        help="Directory to save results")
    parser.add_argument("--n_trials",           type=int, default=20,
                        help="Number of Optuna trials (default: 20)")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use fp16 mixed precision (A5000)")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bf16 mixed precision (H100, recommended)")

    args = parser.parse_args()
    main(args)