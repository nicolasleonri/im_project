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
    confusion_matrix
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

import matplotlib.pyplot as plt
import seaborn as sns

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
# Data saving
# ─────────────────────────────────────────────

def save_annotation_results(test_df, predictions, labels, id2label, label_col, output_dir):
    """
    Saves comprehensive annotation results including:
    - Full predictions with metadata
    - Confidence scores
    - Error analysis breakdown
    - Per-class performance metrics
    """
    
    # Create output dataframe with all annotations
    results_df = test_df.copy()
    
    # Add predictions and confidence scores
    results_df["predicted_label"] = [id2label[p] for p in predictions]
    results_df["true_label"] = test_df[label_col].values
    results_df["correct"] = (results_df["predicted_label"] == results_df["true_label"])
    
    # Add confidence scores if available (probabilities)
    if hasattr(test_results, 'predictions'):
        probs = torch.nn.functional.softmax(
            torch.tensor(test_results.predictions), dim=-1
        ).numpy()
        results_df["confidence"] = np.max(probs, axis=-1)
        
        # Add per-class probabilities
        for idx, label_name in id2label.items():
            results_df[f"prob_{label_name}"] = probs[:, idx]
    
    # Save full annotations
    full_path = os.path.join(output_dir, "full_annotations.csv")
    results_df.to_csv(full_path, sep=";", index=False, encoding="utf-8")
    log.info(f"Full annotations saved to: {full_path}")
    
    # Save error cases for analysis
    error_df = results_df[~results_df["correct"]].copy()
    if len(error_df) > 0:
        error_path = os.path.join(output_dir, "error_analysis.csv")
        error_df.to_csv(error_path, sep=";", index=False, encoding="utf-8")
        log.info(f"Error cases saved to: {error_path}")
    
    # Save correct cases
    correct_df = results_df[results_df["correct"]].copy()
    correct_path = os.path.join(output_dir, "correct_predictions.csv")
    correct_df.to_csv(correct_path, sep=";", index=False, encoding="utf-8")
    
    # Generate per-class statistics
    class_stats = {}
    for label_name in id2label.values():
        class_mask = results_df["true_label"] == label_name
        class_total = class_mask.sum()
        class_correct = (results_df["true_label"] == results_df["predicted_label"])[class_mask].sum()
        
        class_stats[label_name] = {
            "total": int(class_total),
            "correct": int(class_correct),
            "accuracy": float(class_correct / class_total) if class_total > 0 else 0.0,
            "error_rate": float(1 - (class_correct / class_total)) if class_total > 0 else 0.0
        }
    
    # Save class statistics
    stats_path = os.path.join(output_dir, "class_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(class_stats, f, ensure_ascii=False, indent=2)
    
    return results_df, class_stats

def save_detailed_predictions_with_metadata(test_dataset, predictions, labels, 
                                           test_df, tokenizer, output_dir):
    """
    Saves detailed predictions including token-level information and metadata.
    """
    detailed_results = []
    
    for idx, (pred, true_label, row) in enumerate(zip(predictions, labels, test_df.itertuples())):
        # Get input text
        sentence = row.sentence_text
        paragraph = getattr(row, 'paragraph_text', '')
        
        # Calculate token-level information if needed
        tokens = tokenizer.tokenize(sentence)
        
        detailed_results.append({
            "index": idx,
            "sentence": sentence,
            "paragraph": paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
            "true_label": true_label,
            "predicted_label": pred,
            "correct": true_label == pred,
            "sentence_length": len(sentence.split()),
            "token_count": len(tokens),
            "has_paragraph": bool(paragraph)
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(output_dir, "detailed_predictions.csv")
    detailed_df.to_csv(detailed_path, sep=";", index=False, encoding="utf-8")
    log.info(f"Detailed predictions saved to: {detailed_path}")
    
    return detailed_df

def save_confusion_matrix_visualization(labels, preds, id2label, output_dir):
    """
    Creates and saves confusion matrix visualization.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        class_names = [id2label[i] for i in sorted(id2label.keys())]
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Confusion matrix saved to: {cm_path}")
        
        # Also save raw confusion matrix as JSON
        cm_dict = {
            "labels": class_names,
            "matrix": cm.tolist()
        }
        cm_json_path = os.path.join(output_dir, "confusion_matrix.json")
        with open(cm_json_path, "w", encoding="utf-8") as f:
            json.dump(cm_dict, f, indent=2)
            
    except ImportError:
        log.warning("matplotlib/seaborn not installed. Skipping confusion matrix visualization.")


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
    present_classes = np.sort(train_labels.unique())
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=train_labels,
    )
    # Build full weight array — classes missing from train get weight 1.0
    class_weights = np.ones(num_labels)
    for cls, w in zip(present_classes, raw_weights):
        class_weights[cls] = w
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
        processing_class=tokenizer,
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
        save_total_limit=1, # Keep only the best checkpoint for testing
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
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(label_map),
    )

    final_trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────────────
    log.info("Evaluating on Peruvian Spanish test set...")
    test_results = final_trainer.predict(test_dataset)
    preds = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids

    # Get probabilities for confidence scores
    probs = torch.nn.functional.softmax(torch.tensor(test_results.predictions), dim=-1).numpy()

    # Generate classification report
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

    # ── Save comprehensive annotation results ────────────────────────────
    log.info("Saving annotation results...")

    # 1. Full annotations with metadata
    test_df_out = test_df.copy()
    test_df_out["predicted_label"] = [id2label[p] for p in preds]
    test_df_out["true_label"] = test_df[label_col].values
    test_df_out["correct"] = (test_df_out["predicted_label"] == test_df_out["true_label"])
    test_df_out["confidence"] = np.max(probs, axis=-1)

    # Add per-class probabilities
    for idx, label_name in id2label.items():
        test_df_out[f"prob_{label_name}"] = probs[:, idx]

    # Save full annotations
    full_annotations_path = os.path.join(args.output_dir, "full_annotations.csv")
    test_df_out.to_csv(full_annotations_path, sep=";", index=False, encoding="utf-8")
    log.info(f"Full annotations saved to: {full_annotations_path}")

    # 2. Save confusion matrix visualization
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {args.task} classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw confusion matrix
    cm_dict = {"labels": class_names, "matrix": cm.tolist()}
    cm_json_path = os.path.join(args.output_dir, "confusion_matrix.json")
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)

    # 3. Save error cases for analysis
    error_df = test_df_out[~test_df_out["correct"]].copy()
    if len(error_df) > 0:
        error_path = os.path.join(args.output_dir, "error_analysis.csv")
        error_df.to_csv(error_path, sep=";", index=False, encoding="utf-8")
        log.info(f"Error cases saved to: {error_path}")

    # 4. Save per-class statistics
    class_stats = {}
    for label_name in id2label.values():
        class_mask = test_df_out["true_label"] == label_name
        class_total = class_mask.sum()
        if class_total > 0:
            class_correct = (test_df_out["true_label"] == test_df_out["predicted_label"])[class_mask].sum()
            class_stats[label_name] = {
                "total": int(class_total),
                "correct": int(class_correct),
                "accuracy": float(class_correct / class_total),
                "avg_confidence": float(test_df_out.loc[class_mask, "confidence"].mean())
            }

    stats_path = os.path.join(args.output_dir, "class_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(class_stats, f, ensure_ascii=False, indent=2)

    # 5. Save detailed token-level analysis (optional, for deeper analysis)
    detailed_results = []
    for idx, row in test_df_out.iterrows():
        tokens = tokenizer.tokenize(row["sentence_text"])
        detailed_results.append({
            "index": idx,
            "sentence": row["sentence_text"][:100] + "..." if len(row["sentence_text"]) > 100 else row["sentence_text"],
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "correct": row["correct"],
            "confidence": row["confidence"],
            "sentence_length": len(row["sentence_text"].split()),
            "token_count": len(tokens),
            "paragraph_preview": row.get("paragraph_text", "")[:100] if pd.notna(row.get("paragraph_text", "")) else ""
        })

    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(args.output_dir, "detailed_predictions.csv")
    detailed_df.to_csv(detailed_path, sep=";", index=False, encoding="utf-8")
    log.info(f"Detailed predictions saved to: {detailed_path}")

    # ── Save main results JSON ────────────────────────────────────────────
    results = {
        "experiment": {
            "model": args.model_name_or_path,
            "source_dataset": args.source_dataset,
            "test_dataset": args.test_dataset,
            "task": args.task,
            "n_trials": args.n_trials,
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        "best_hyperparameters": best_run.hyperparameters,
        "test_set_metrics": {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        },
        "per_class_metrics": {
            label_name: {
                "precision": report[label_name]["precision"],
                "recall": report[label_name]["recall"],
                "f1-score": report[label_name]["f1-score"],
                "support": report[label_name]["support"]
            }
            for label_name in id2label.values()
        },
        "classification_report": report,
        "class_statistics": class_stats,
    }

    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Keep original predictions file for backward compatibility
    preds_path = os.path.join(args.output_dir, "test_predictions.csv")
    test_df_out[["sentence_text", "true_label", "predicted_label", "correct", "confidence"]].to_csv(
        preds_path, sep=";", index=False, encoding="utf-8"
    )

    # ── Cleanup: Remove model checkpoints and temporary files ────────────
    log.info("Cleaning up temporary files...")

    # Remove the entire final_model folder (including checkpoints)
    final_model_path = Path(args.output_dir) / "final_model"
    if final_model_path.exists():
        shutil.rmtree(final_model_path)
        log.info(f"Deleted model checkpoints: {final_model_path}")

    # Remove Optuna trial folders if they exist
    optuna_path = Path(args.output_dir) / "optuna_trials"
    if optuna_path.exists():
        shutil.rmtree(optuna_path)
        log.info(f"Deleted Optuna trial folders: {optuna_path}")

    # Also remove any other temporary directories
    for temp_dir in ["checkpoints", "runs", "tmp"]:
        temp_path = Path(args.output_dir) / temp_dir
        if temp_path.exists():
            shutil.rmtree(temp_path)
            log.info(f"Deleted temporary folder: {temp_path}")

    log.info("Cleanup complete - only results and annotations preserved")

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
    parser.add_argument("--save_confidence_scores", action="store_true", default=True,
                    help="Save confidence scores for predictions")
    parser.add_argument("--save_error_analysis", action="store_true", default=True,
                        help="Save detailed error analysis files")
    parser.add_argument("--save_visualizations", action="store_true", default=True,
                        help="Generate confusion matrix plots")

    args = parser.parse_args()
    main(args)