# Argument Mining Pipeline — CGEC13-20 Informal Economy Subset

A fully automated pipeline for:
1. Building an annotated test set of argumentative components in Peruvian Spanish journalistic articles about informal economy.
2. Domain-adaptive pre-training of BETO on the full CGEC13-20 corpus.
3. Fine-tuning and evaluating transfer experiments across multiple source datasets.

---

## Overview

```
Raw CSV
  └─▶ paragrapher.py          LLM paragraph extraction + sentence segmentation + stratified sampling
        └─▶ annotator.py      (×N models) LLM annotation of each sentence
              └─▶ iaa.py      Inter-annotator agreement metrics + 3-way split
                    └─▶ [manual review of PENDING rows]
                          └─▶ build_test_set.py      Final validated test set
                                    │
                                    ▼
                         preprocess_corpus.py     OCR corpus cleaning for MLM
                                    │
                                    ▼
                           pretrain_beto.py       Domain-adaptive MLM pre-training
                                    │
                                    ▼
                           finetune_beto.py       Fine-tuning + Optuna HP search + evaluation
```

---

## Repository Structure

```
.
├── data/
│   ├── informal_economy.csv              Raw input corpus (CGEC13-20 informal economy subset)
│   ├── preprocessing/
│   │   ├── paragraphs_<model>.csv        Intermediate: extracted paragraphs
│   │   ├── sentences_<model>.csv         Intermediate: segmented sentences
│   │   ├── datasets/                     Source datasets for fine-tuning (semicolon-separated CSVs)
│   │   └── tokenized_cache/              Cached tokenized MLM corpus (auto-generated)
│   ├── iaa/
│   │   ├── agreed.csv                    Auto-labeled unanimous sentences
│   │   ├── review_sentences.csv          Disagreements — fill PENDING labels manually
│   │   ├── errors.csv                    NC sentences  — fill PENDING labels manually
│   │   ├── iaa_report.txt                Human-readable agreement report
│   │   └── iaa_pairwise.csv              Pairwise Cohen's Kappa table
│   └── test_set_final.csv                ✅ Final annotated test set
├── docs/
│   └── annotation_guidelines_v1.md       Annotation guidelines (loaded at runtime)
├── models/
│   └── beto-cgec/                        Domain-adapted BETO checkpoint (output of pretrain_beto.py)
├── results/
│   └── {model}_{dataset}_{task}/         Per-experiment results
│       ├── test_results.json             Metrics + best hyperparameters
│       └── test_predictions.csv         Predictions for error analysis (RQ2)
├── mlm/
│   └── corpus_clean.txt                  Preprocessed OCR corpus for MLM
├── paragrapher.py
├── annotator.py
├── iaa.py
├── build_test_set.py
├── preprocess_corpus.py
├── pretrain_beto.py
├── finetune_beto.py
├── utils_preprocessing.py
└── README.md
```

---

## Part 1 — Test Set Construction

### `utils_preprocessing.py`
Shared utilities imported by all annotation scripts. Contains:
- Prompt templates (`PARAGRAPHER_PROMPT`, `ANNOTATION_PROMPT`, `DECISION_TREE`)
- `load_input` / `load_guidelines` — data loading and validation
- `build_paragrapher_prompts` / `build_annotation_prompts` — chat template formatting via HuggingFace tokenizers
- `load_spacy_model` — spaCy model loader with `spacy_udpipe` fallback
- `stratified_sample` — proportional sampling by newspaper × year (falls back to newspaper-only)
- `clear_gpu_memory` — vLLM / CUDA cleanup between runs

---

### `paragrapher.py`
Extracts paragraphs from raw article text via an LLM, segments them into sentences, and produces a stratified sample of target size.

**Workflow:**
1. Load and validate input CSV
2. LLM (vLLM) extracts paragraphs as structured JSON
3. Save intermediate paragraph CSV
4. Sentence segmentation with spaCy (`es_dep_news_trf`, falls back to `spacy_udpipe`)
5. Save intermediate sentence CSV
6. Stratified sampling of N paragraphs (by newspaper × year)

**Input CSV required columns:** `headline`, `content`, `newspaper`, `date`

**Output CSV columns:** `article_id`, `newspaper`, `date`, `year`, `headline`, `content`, `paragraph_i`, `paragraph_text`, `sentence_j`, `sentence_text`

```bash
python paragrapher.py \
  --input  data/informal_economy.csv \
  --output data/preprocessing/test_set.csv \
  --model  mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --n_samples 600
```

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to raw input CSV |
| `--output` | *(required)* | Path for output CSV |
| `--model` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | vLLM model for paragraph extraction |
| `--tokenizer` | same as model | HuggingFace tokenizer name |
| `--n_samples` | `600` | Target number of sampled paragraphs |
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `--min_stratum_size` | `10` | Min paragraphs per stratum before fallback |
| `--spacy_model` | `es_dep_news_trf` | spaCy model for sentence segmentation |
| `--udpipe_lang` | `es` | Fallback spacy_udpipe language code |
| `--no_sample` | `False` | Output all extracted data without sampling |
| `--skip_vllm` | `False` | Skip LLM step, load precomputed paragraphs |
| `--quantization` | `None` | vLLM quantization method |
| `--tokenizer_mode` | `auto` | vLLM tokenizer mode |

---

### `annotator.py`
Annotates each sentence with its argumentative role using an LLM, guided by annotation guidelines and a decision tree.

**Workflow:**
1. Load preprocessed CSV (output of `paragrapher.py`)
2. Load annotation guidelines from markdown file
3. LLM annotates each sentence in its paragraph + article context
4. Parse, validate, and enforce label consistency
5. Save annotated CSV

**Labels produced:**

| Column | Values |
|---|---|
| `label_binary` | `argumentative`, `non-argumentative` |
| `label_component` | `claim`, `premise`, `none` |
| `confidence_binary` | float [0.0 – 1.0] |
| `confidence_component` | float [0.0 – 1.0] |
| `reasoning` | brief justification in Spanish |
| `annotation_error` | `True` if parsing failed (label = `NC`) |

```bash
python annotator.py \
  --input      data/preprocessing/test_set.csv \
  --output     data/annotated_modelA.csv \
  --guidelines docs/annotation_guidelines_v1.md \
  --model      mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Preprocessed CSV from `paragrapher.py` |
| `--output` | *(required)* | Path for annotated output CSV |
| `--guidelines` | *(required)* | Annotation guidelines markdown file |
| `--model` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | vLLM model for annotation |
| `--tokenizer` | same as model | HuggingFace tokenizer name |
| `--tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `--batch_size` | `32` | vLLM inference batch size |
| `--max_tokens` | `512` | Max tokens in annotation response |
| `--max_model_len` | `32768` | Max model context length |
| `--temperature` | `0.0` | Sampling temperature |
| `--tokenizer_mode` | `auto` | vLLM tokenizer mode |

> Run this script once per model. The output CSVs are then compared by `iaa.py`.

---

### `iaa.py`
Computes inter-annotator agreement metrics across 2+ annotated CSVs and splits all sentences into three groups for downstream test set construction.

**Metrics computed** (for both `label_binary` and `label_component`):
- **Pairwise Cohen's κ** — every unique model pair
- **Fleiss' κ** — all raters simultaneously
- **Krippendorff's α** — nominal scale

**3-way split:**

| Output file | Contents | `label_source` |
|---|---|---|
| `agreed.csv` | All models agree on both labels, no NC | `auto` — final labels filled automatically |
| `review_sentences.csv` | Any label disagreement, no NC | `manual` — fill `PENDING` labels |
| `errors.csv` | Any model returned `NC` | `manual` — fill `PENDING` labels |

`review_sentences.csv` is sorted most-contested first and includes majority-vote helper columns (`label_binary_majority`, `label_component_majority`, `*_n_agree`) to speed up manual review.

```bash
python iaa.py \
  --inputs    data/annotated_modelA.csv data/annotated_modelB.csv \
  --output_dir data/iaa/
```

| Argument | Default | Description |
|---|---|---|
| `--inputs` | *(required)* | 2+ annotated CSVs from `annotator.py` |
| `--output_dir` | `data/iaa/` | Directory for all output files |
| `--id_cols` | `article_id sentence_j` | Columns that uniquely identify a sentence |

**Kappa interpretation (Landis & Koch 1977):**

| Range | Interpretation |
|---|---|
| < 0.00 | Poor |
| 0.00 – 0.20 | Slight |
| 0.21 – 0.40 | Fair |
| 0.41 – 0.60 | Moderate |
| 0.61 – 0.80 | Substantial |
| 0.81 – 1.00 | Almost Perfect |

---

### `build_test_set.py`
Merges the three CSVs from `iaa.py` (after manual annotation of `PENDING` rows) into a single validated `test_set_final.csv`.

**Validation checks before writing:**
- No `PENDING` values remain in `label_binary_final` / `label_component_final`
- All label values are within the valid sets
- Consistency: `non-argumentative` → `label_component_final` must be `none`
- No duplicate sentence IDs across the three input files

```bash
python build_test_set.py \
  --iaa_dir data/iaa/ \
  --output  data/test_set_final.csv
```

| Argument | Default | Description |
|---|---|---|
| `--iaa_dir` | `None` | Directory with `agreed.csv`, `review_sentences.csv`, `errors.csv` |
| `--agreed` | `None` | Override path to `agreed.csv` |
| `--review` | `None` | Override path to `review_sentences.csv` |
| `--errors` | `None` | Override path to `errors.csv` |
| `--output` | `data/test_set_final.csv` | Output path |
| `--skip_errors` | `False` | Skip `errors.csv` if none exist or unresolved |

---

## Part 2 — Domain-Adaptive Pre-training

### `preprocess_corpus.py`
Cleans the OCR-extracted CGEC13-20 corpus for MLM pre-training. Recursively processes all `.txt` files under the corpus root, applies noise filters, rejoins OCR-broken lines, and outputs a single clean corpus file.

**OCR noise filters applied:**
- Lines shorter than 25 characters (headers, page numbers)
- Non-alphanumeric character ratio > 30%
- Short token ratio > 50% (layout noise, column separators)
- Real Spanish word ratio < 35% (allows siglas, numbers, proper nouns)
- All-caps fragments (broken column headers)

```bash
python preprocess_corpus.py \
  --input_dir  /path/to/cgec1320 \
  --output_file data/mlm/corpus_clean.txt
```

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | *(required)* | Root folder with `.txt` files (searched recursively) |
| `--output_file` | `corpus_clean.txt` | Output file path |

**Output format:** One paragraph per line, blank lines as document boundaries — compatible with `datasets.load_dataset("text")`.

---

### `pretrain_beto.py`
Domain-adaptive MLM pre-training of BETO on the cleaned CGEC13-20 corpus using Whole Word Masking, consistent with the original BETO pre-training setup.

**Features:**
- Tokenized dataset cached to disk — skips re-tokenization on subsequent runs
- Automatic checkpoint resumption if job is interrupted
- Reports final perplexity as domain adaptation quality metric

```bash
# Single GPU (H100, recommended)
python pretrain_beto.py \
  --corpus_file data/mlm/corpus_clean.txt \
  --output_dir  models/beto-cgec \
  --bf16

# Multi-GPU
torchrun --nproc_per_node=2 pretrain_beto.py \
  --corpus_file data/mlm/corpus_clean.txt \
  --output_dir  models/beto-cgec \
  --bf16
```

| Argument | Default | Description |
|---|---|---|
| `--corpus_file` | *(required)* | Path to `corpus_clean.txt` |
| `--output_dir` | *(required)* | Directory for checkpoints and final model |
| `--num_train_epochs` | `3` | Number of training epochs |
| `--per_device_train_batch_size` | `64` | Batch size (64 for H100 bf16; 32 for A5000) |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation steps |
| `--fp16` | `False` | fp16 mixed precision (A5000/V100) |
| `--bf16` | `True` | bf16 mixed precision (H100, recommended) |

**HPC setup:**
```bash
# Download model on login node before submitting job
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
huggingface-cli download dccuchile/bert-base-spanish-wwm-uncased

# Set in SLURM job script
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

---

## Part 3 — Fine-tuning & Evaluation

### `finetune_beto.py`
Fine-tunes BETO (base or domain-adapted) on a source dataset with Optuna hyperparameter search, then evaluates on the Peruvian Spanish test set.

**Pipeline per experiment:**
1. Split source dataset 70/30 (train/val), stratified
2. Compute class weights for imbalance handling (weighted CrossEntropyLoss)
3. Optuna search over train/val to find best hyperparameters (N trials)
4. Final fine-tuning with best hyperparameters (`load_best_model_at_end=True`)
5. Evaluation on unseen test set
6. Save results and predictions; delete checkpoints

**Optuna search space:**

| Hyperparameter | Range |
|---|---|
| `learning_rate` | [1e-5, 5e-5] log-uniform |
| `num_train_epochs` | [3, 6] |
| `per_device_train_batch_size` | {16, 32} |
| `warmup_ratio` | [0.0, 0.15] |
| `weight_decay` | [0.0, 0.1] |

**Tasks:**

| Task | Labels | Metric |
|---|---|---|
| `binary` | `argumentative`, `non-argumentative` | F1 macro |
| `component` | `claim`, `premise`, `none` | F1 macro |

**Input format** (source datasets and test set, semicolon-separated CSV):
```
sentence_text;label_binary;label_component
```
Test set additionally requires `paragraph_text` for context-aware tokenization:
```
[CLS] sentence [SEP] paragraph [SEP]
```

```bash
# BETO base, binary task
python finetune_beto.py \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
  --source_dataset     data/preprocessing/datasets/guzman.csv \
  --test_dataset       data/test_set_final.csv \
  --task               binary \
  --output_dir         results/beto-base_guzman_binary \
  --n_trials           20 \
  --bf16

# Domain-adapted BETO, component task
python finetune_beto.py \
  --model_name_or_path models/beto-cgec \
  --source_dataset     data/preprocessing/datasets/asohmo.csv \
  --test_dataset       data/test_set_final.csv \
  --task               component \
  --output_dir         results/beto-cgec_asohmo_component \
  --n_trials           20 \
  --bf16
```

| Argument | Default | Description |
|---|---|---|
| `--model_name_or_path` | *(required)* | BETO base or domain-adapted model path |
| `--source_dataset` | *(required)* | Source dataset CSV |
| `--test_dataset` | *(required)* | Peruvian Spanish test set CSV |
| `--task` | *(required)* | `binary` or `component` |
| `--output_dir` | *(required)* | Directory to save results |
| `--n_trials` | `20` | Number of Optuna trials |
| `--fp16` | `False` | fp16 mixed precision |
| `--bf16` | `False` | bf16 mixed precision (H100, recommended) |

**Outputs per experiment:**

| File | Contents |
|---|---|
| `test_results.json` | Metrics, best hyperparameters, classification report |
| `test_predictions.csv` | Per-sentence predictions + `correct` flag for error analysis |
| `best_hyperparameters.json` | Best Optuna trial hyperparameters |

---

## Experiment Matrix

With 2 base models × 8 source datasets × 2 tasks = **32 experiments minimum**.

| Model | Datasets | Tasks |
|---|---|---|
| `dccuchile/bert-base-spanish-wwm-uncased` | Guzman, ASOHMO, INFERES, NLAS, CATyPI, Decide-Madrid, Yeginbergen, VivesDebate | `binary`, `component` |
| `models/beto-cgec` (domain-adapted) | idem | idem |

Results feed into:
- **RQ1** — Performance benchmark (F1 matrix across all transfer conditions)
- **RQ2** — Error analysis via `test_predictions.csv` (qualitative taxonomy)
- **RQ3** — Exploratory correlation of dataset characteristics with error patterns

---

## End-to-End Usage

```bash
# ── Part 1: Test set construction ────────────────────────────────────────

# 1. Extract paragraphs and sentences
python paragrapher.py \
  --input  data/informal_economy.csv \
  --output data/preprocessing/test_set.csv \
  --n_samples 600

# 2. Annotate with each model
python annotator.py \
  --input      data/preprocessing/test_set.csv \
  --output     data/annotated_mistral.csv \
  --guidelines docs/annotation_guidelines_v1.md \
  --model      mistralai/Mistral-Small-3.2-24B-Instruct-2506

python annotator.py \
  --input      data/preprocessing/test_set.csv \
  --output     data/annotated_deepseek.csv \
  --guidelines docs/annotation_guidelines_v1.md \
  --model      deepseek-ai/DeepSeek-R1-Distill-Llama-70B

# 3. Compute IAA and split
python iaa.py \
  --inputs     data/annotated_mistral.csv data/annotated_deepseek.csv \
  --output_dir data/iaa/

# 4. Manually fill PENDING rows in:
#      data/iaa/review_sentences.csv
#      data/iaa/errors.csv

# 5. Build final test set
python build_test_set.py \
  --iaa_dir data/iaa/ \
  --output  data/test_set_final.csv

# ── Part 2: Domain-adaptive pre-training ─────────────────────────────────

# 6. Preprocess OCR corpus
python preprocess_corpus.py \
  --input_dir  /path/to/cgec1320 \
  --output_file data/mlm/corpus_clean.txt

# 7. Pre-train BETO
python pretrain_beto.py \
  --corpus_file data/mlm/corpus_clean.txt \
  --output_dir  models/beto-cgec \
  --bf16

# ── Part 3: Fine-tuning experiments ──────────────────────────────────────

# 8. Run one experiment (repeat for all dataset × model × task combinations)
python finetune_beto.py \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
  --source_dataset     data/preprocessing/datasets/guzman.csv \
  --test_dataset       data/test_set_final.csv \
  --task               binary \
  --output_dir         results/beto-base_guzman_binary \
  --n_trials           20 \
  --bf16
```

---

## Dependencies

```bash
pip install pandas scikit-learn statsmodels krippendorff \
            vllm transformers spacy spacy-udpipe torch \
            datasets optuna accelerate
python -m spacy download es_dep_news_trf
```

| Package | Purpose |
|---|---|
| `vllm` | Fast LLM inference for paragraph extraction and annotation |
| `transformers` | BETO pre-training, fine-tuning, tokenization |
| `datasets` | Corpus loading and tokenized cache management |
| `optuna` | Hyperparameter search for fine-tuning |
| `accelerate` | Multi-GPU training support |
| `spacy` / `spacy-udpipe` | Sentence segmentation |
| `scikit-learn` | Class weights, Cohen's Kappa, metrics |
| `statsmodels` | Fleiss' Kappa |
| `krippendorff` | Krippendorff's Alpha |
| `pandas` | Data loading and CSV I/O |
| `torch` | GPU operations |

---

## Label Schema

### Binary label (`label_binary_final`)
| Value | Meaning |
|---|---|
| `argumentative` | Sentence participates in an argumentative move |
| `non-argumentative` | Sentence is descriptive, transitional, or contextual |

### Component label (`label_component_final`)
| Value | Meaning |
|---|---|
| `claim` | A contestable standpoint or evaluative position |
| `premise` | Evidence, data, causal explanation, or authority reference supporting a claim |
| `none` | No argumentative component role; always paired with `non-argumentative` |

**Constraint:** `non-argumentative` → `label_component_final` must be `none`.

---

## Notes

- All CSVs use `;` as delimiter, `"` as quotechar, UTF-8 encoding, and `NA` for missing values.
- GPU memory is explicitly cleared between vLLM runs via `clear_gpu_memory()`.
- Stratified sampling falls back from newspaper × year to newspaper-only if any stratum has fewer than `--min_stratum_size` paragraphs.
- `NC` labels (annotation parsing failures) are isolated into `errors.csv` and excluded from agreement scoring.
- Tokenized MLM corpus is cached to `data/preprocessing/tokenized_cache/` — delete this folder to force re-tokenization.
- Fine-tuning checkpoints are deleted after evaluation; only `test_results.json` and `test_predictions.csv` are kept.
- Set `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` in SLURM job scripts to prevent compute nodes from attempting internet access.