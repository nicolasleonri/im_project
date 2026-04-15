# Argument Mining Pipeline — "Cross-domain dataset reuse for Argumentation Mining: A mixed-methods study on Peruvian Spanish journalism"

This repository contains the pipeline used to build a test set for Argumentation Mining in Peruvian Spanish and to run cross-domain transfer experiments.

In short, it covers three main steps:
1. Constructing an annotated test set from journalistic data
2. Domain-adaptive pre-training of BETO
3. Fine-tuning and evaluation on multiple source datasets

---

## Overview

The pipeline starts from raw articles and ends with evaluated transfer models:

Raw CSV
  → paragraph extraction + sentence segmentation
  → LLM-based annotation (multiple models)
  → agreement analysis + manual review
  → final test set
  → domain-adaptive pre-training
  → fine-tuning + evaluation

---

## Repository Structure

- data/ – input data, intermediate files, and final test set  
- docs/ – annotation guidelines  
- models/ – domain-adapted BETO checkpoints  
- results/ – experiment outputs  
- mlm/ – cleaned corpus for pre-training  
- scripts (*.py) – pipeline steps  

---

## 1. Test Set Construction

The test set is built in several stages.

First, paragrapher.py extracts paragraphs from raw articles using an LLM and splits them into sentences. It also performs stratified sampling.

Then, annotator.py labels each sentence using different LLMs, based on the annotation guidelines.

The outputs are compared in iaa.py, which computes agreement and separates:
- sentences with agreement
- sentences requiring manual review
- annotation errors

After manual correction, build_test_set.py merges everything into the final test set.

---

## 2. Domain-Adaptive Pre-training

The full CGEC13-20 corpus is cleaned using preprocess_corpus.py to remove OCR noise.

pretrain_beto.py then performs masked language modeling (MLM) on this corpus to obtain a domain-adapted BETO model.

---

## 3. Fine-tuning and Evaluation

finetune_beto.py fine-tunes BETO (base or adapted) on each source dataset.

Each experiment includes:
- train/validation split
- hyperparameter search (Optuna)
- evaluation on the Peruvian Spanish test set

Two tasks are considered:
- binary classification (argumentative vs non-argumentative)
- component classification (claim, premise, none)

---

## Running the Pipeline

Typical usage:

# Build test set
python paragrapher.py ...
python annotator.py ...
python iaa.py ...
python build_test_set.py ...

# Pre-training
python preprocess_corpus.py ...
python pretrain_beto.py ...

# Fine-tuning
python finetune_beto.py ...

---

## Notes

- All CSV files use ; as delimiter
- Fine-tuning results are saved per experiment (metrics + predictions)
- The test set is strictly held out from training
