# Argument Mining Pipeline — CGEC13-20 Informal Economy Subset

A fully automated pipeline for building an annotated test set of argumentative components in Peruvian Spanish journalistic articles about informal economy. It chains LLM-based paragraph extraction, sentence segmentation, multi-model annotation, inter-annotator agreement analysis, and final test set assembly.

---

## Overview

```
Raw CSV
  └─▶ paragrapher.py       LLM paragraph extraction + sentence segmentation + stratified sampling
        └─▶ annotator.py   (×N models) LLM annotation of each sentence
              └─▶ iaa.py   Inter-annotator agreement metrics + 3-way split
                    └─▶ [manual review of PENDING rows]
                          └─▶ build_test_set.py   Final validated test set
```

---

## Repository Structure

```
.
├── data/
│   ├── informal_economy.csv          Raw input corpus
│   ├── preprocessing/
│   │   ├── paragraphs_<model>.csv    Intermediate: extracted paragraphs
│   │   └── sentences_<model>.csv     Intermediate: segmented sentences
│   ├── iaa/
│   │   ├── agreed.csv                Auto-labeled unanimous sentences
│   │   ├── review_sentences.csv      Disagreements — fill PENDING labels manually
│   │   ├── errors.csv                NC sentences  — fill PENDING labels manually
│   │   ├── iaa_report.txt            Human-readable agreement report
│   │   └── iaa_pairwise.csv          Pairwise Cohen's Kappa table
│   └── test_set_final.csv            ✅ Final annotated test set
├── docs/
│   └── annotation_guidelines_v1.md   Annotation guidelines (loaded at runtime)
├── paragrapher.py
├── annotator.py
├── iaa.py
├── build_test_set.py
├── utils_preprocessing.py
└── README.md
```

---

## Scripts

### `utils_preprocessing.py`
Shared utilities imported by all other scripts. Contains:
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
  --inputs    data/annotated_modelA.csv data/annotated_modelB.csv data/annotated_modelC.csv \
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

**Final output columns:** `article_id`, `newspaper`, `date`, `year`, `headline`, `content`, `paragraph_i`, `paragraph_text`, `sentence_j`, `sentence_text`, `label_binary_final`, `label_component_final`, `label_source`

---

## End-to-End Usage

```bash
# 1. Extract paragraphs and sentences
python paragrapher.py \
  --input  data/informal_economy.csv \
  --output data/preprocessing/test_set.csv \
  --model  mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --n_samples 600

# 2. Annotate with each model (repeat for each)
python annotator.py \
  --input      data/preprocessing/test_set.csv \
  --output     data/annotated_modelA.csv \
  --guidelines docs/annotation_guidelines_v1.md \
  --model      mistralai/Mistral-Small-3.2-24B-Instruct-2506

python annotator.py \
  --input      data/preprocessing/test_set.csv \
  --output     data/annotated_modelB.csv \
  --guidelines docs/annotation_guidelines_v1.md \
  --model      meta-llama/Llama-3.1-70B-Instruct

# 3. Compute IAA and split
python iaa.py \
  --inputs     data/annotated_modelA.csv data/annotated_modelB.csv \
  --output_dir data/iaa/

# 4. Manually fill PENDING rows in:
#      data/iaa/review_sentences.csv
#      data/iaa/errors.csv

# 5. Build final test set
python build_test_set.py \
  --iaa_dir data/iaa/ \
  --output  data/test_set_final.csv
```

---

## Dependencies

```bash
pip install pandas scikit-learn statsmodels krippendorff \
            vllm transformers spacy spacy-udpipe torch
python -m spacy download es_dep_news_trf
```

| Package | Purpose |
|---|---|
| `vllm` | Fast LLM inference for paragraph extraction and annotation |
| `transformers` | Chat template formatting via HuggingFace tokenizers |
| `spacy` / `spacy-udpipe` | Sentence segmentation (`es_dep_news_trf` preferred) |
| `scikit-learn` | Pairwise Cohen's Kappa |
| `statsmodels` | Fleiss' Kappa |
| `krippendorff` | Krippendorff's Alpha |
| `pandas` | Data loading, manipulation, and CSV I/O |
| `torch` | GPU memory management |

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

- All intermediate CSVs use `;` as delimiter, `"` as quotechar, UTF-8 encoding, and `NA` for missing values — consistent with the full pipeline.
- GPU memory is explicitly cleared between vLLM runs via `clear_gpu_memory()` in `utils_preprocessing.py`.
- Stratified sampling falls back from newspaper × year to newspaper-only if any stratum has fewer than `--min_stratum_size` paragraphs.
- `NC` labels (annotation parsing failures) are isolated into `errors.csv` by `iaa.py` and excluded from the agree/review split, preventing them from silently inflating agreement scores.