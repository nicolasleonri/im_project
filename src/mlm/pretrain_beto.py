"""
pretrain_beto.py
----------------
Domain-adaptive MLM pre-training of BETO on the CGEC13-20 corpus
(Peruvian Spanish journalism).

Continues training from the existing BETO checkpoint using Whole Word Masking (WWM),
consistent with the original BETO pre-training setup.

Usage:
    python pretrain_beto.py \
        --corpus_file data/mlm/corpus_clean.txt \
        --output_dir models/beto-cgec \
        --num_train_epochs 3 \
        --per_device_train_batch_size 64

For multi-GPU (e.g. 2x H100):
    torchrun --nproc_per_node=2 pretrain_beto.py \
        --corpus_file data/mlm/corpus_clean.txt \
        --output_dir models/beto-cgec
"""

import os
import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL_NAME      = "dccuchile/bert-base-spanish-wwm-uncased"
MAX_SEQ_LENGTH  = 512       # BERT max — use full context
MLM_PROBABILITY = 0.15      # Standard masking probability
SEED            = 42


# ─────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────

def tokenize_and_group(dataset, tokenizer, max_seq_length):
    """
    Tokenizes the corpus and groups tokens into fixed-length chunks of max_seq_length.
    This is the standard approach for MLM pre-training:
    - Tokenize all text
    - Concatenate into a single long sequence
    - Split into chunks of max_seq_length
    This maximizes GPU utilization and avoids padding waste.
    """
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=False,       # Do NOT truncate — we chunk manually below
            padding=False,
            return_special_tokens_mask=True,
        )

    def group_texts(examples):
        # Concatenate all tokenized sequences
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # Drop the last incomplete chunk
        total_length = (total_length // max_seq_length) * max_seq_length

        # Split into chunks
        result = {
            k: [v[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, v in concatenated.items()
        }
        return result

    log.info("Tokenizing corpus...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
        desc="Tokenizing",
    )

    log.info("Grouping into fixed-length chunks...")
    grouped = tokenized.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Grouping",
    )

    log.info(f"Total training chunks: {len(grouped):,}")
    return grouped


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    set_seed(SEED)

    # ── Load tokenizer and model ──────────────────────────────────────────
    log.info(f"Loading tokenizer and model from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    log.info(f"Model parameters: {model.num_parameters():,}")

    # ── Load corpus ───────────────────────────────────────────────────────
    log.info(f"Loading corpus from: {args.corpus_file}")
    raw_dataset = load_dataset(
        "text",
        data_files={"train": args.corpus_file},
        split="train",
    )
    # Filter out empty lines (document boundary separators)
    raw_dataset = raw_dataset.filter(lambda x: len(x["text"].strip()) > 0)
    log.info(f"Corpus lines loaded: {len(raw_dataset):,}")

    # ── Tokenize and chunk ────────────────────────────────────────────────
    # Tokenization is expensive — cache to disk so it only runs once.
    cache_path = os.path.join(os.path.dirname(args.corpus_file), "tokenized_cache")

    if os.path.exists(cache_path):
        log.info(f"Loading tokenized dataset from cache: {cache_path}")
        from datasets import load_from_disk
        train_dataset = load_from_disk(cache_path)
        log.info(f"Loaded {len(train_dataset):,} chunks from cache.")
    else:
        log.info("Tokenizing corpus (this may take a while)...")
        train_dataset = tokenize_and_group(raw_dataset, tokenizer, MAX_SEQ_LENGTH)
        log.info(f"Saving tokenized dataset to cache: {cache_path}")
        train_dataset.save_to_disk(cache_path)
        log.info("Cache saved — next run will skip tokenization.")

    # ── Data collator (Whole Word Masking) ────────────────────────────────
    # Consistent with original BETO pre-training
    data_collator = DataCollatorForLanguageModeling(
        processing_class=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
        pad_to_multiple_of=8,
        whole_word_mask=True,
    )

    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Training duration
        num_train_epochs=args.num_train_epochs,

        # Batch size — tuned for H100 (80GB) with fp16
        # For A5000 (24GB), reduce to 32 or use gradient_accumulation_steps=2
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimizer
        learning_rate=5e-5,             # Lower LR for continued pre-training
        weight_decay=0.01,
        warmup_steps=500,              # 6% warmup — standard for BERT continued pre-training
        lr_scheduler_type="linear",

        # Precision
        fp16=args.fp16,
        bf16=args.bf16,                 # Prefer bf16 on H100

        # Logging and saving
        # logging_dir removed (use TENSORBOARD_LOGGING_DIR env var)
        logging_steps=500,
        save_steps=5000,
        save_total_limit=3,             # Keep only last 3 checkpoints

        # Reproducibility
        seed=SEED,
        data_seed=SEED,

        # Performance
        dataloader_num_workers=4,
        # group_by_length removed in newer transformers versions
        report_to="none",               # Change to "wandb" if you use W&B
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    log.info("Starting domain-adaptive pre-training...")

    # Resume from checkpoint if available
    checkpoint = None
    if os.path.isdir(args.output_dir):
        checkpoints = [
            d for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoint = os.path.join(
                args.output_dir,
                sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            )
            log.info(f"Resuming from checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # ── Save final model ──────────────────────────────────────────────────
    log.info(f"Saving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── Log final metrics ─────────────────────────────────────────────────
    metrics = train_result.metrics
    perplexity = math.exp(metrics["train_loss"])
    metrics["perplexity"] = perplexity
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    log.info(f"Training complete.")
    log.info(f"  Final loss       : {metrics['train_loss']:.4f}")
    log.info(f"  Final perplexity : {perplexity:.2f}")
    log.info(f"  Model saved to   : {args.output_dir}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Domain-adaptive MLM pre-training of BETO on CGEC13-20"
    )
    parser.add_argument("--corpus_file",   required=True,
                        help="Path to corpus_clean.txt")
    parser.add_argument("--output_dir",    required=True,
                        help="Directory to save model checkpoints and final model")
    parser.add_argument("--num_train_epochs",            type=int,   default=3)
    parser.add_argument("--per_device_train_batch_size", type=int,   default=64,
                        help="64 for H100 80GB with bf16; use 32 for A5000")
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=1)
    parser.add_argument("--fp16",  action="store_true",  default=False,
                        help="Use fp16 mixed precision (for V100/A10/A5000)")
    parser.add_argument("--bf16",  action="store_true",  default=True,
                        help="Use bf16 mixed precision (recommended for H100)")

    args = parser.parse_args()
    main(args)