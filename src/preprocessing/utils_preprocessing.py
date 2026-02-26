import pandas as pd
import logging
import json
import re
import sys
from pathlib import Path
from vllm import LLM, SamplingParams
import spacy_udpipe
from transformers import AutoTokenizer
import gc
import torch
import spacy
import torch.distributed as dist

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Constants
PARAGRAPHER_PROMPT = """
You are a high-precision text structuring system being evaluated for strict rule compliance.

This task is part of a benchmark. Any deviation from the rules will be considered a failure.

The input is a raw, unformatted Peruvian Spanish journalistic article.
It may not contain explicit paragraph breaks.
Your task is to reconstruct the intended body paragraphs.

PARAGRAPH RECONSTRUCTION PRINCIPLES:
- A paragraph is a MAXIMAL coherent block of related sentences.
- Prefer grouping sentences together rather than splitting them.
- Only create a new paragraph when there is a clear topic shift, change of speaker, or structural transition.
- Under NO circumstances should each sentence be treated as its own paragraph unless the article clearly uses single-sentence paragraphs intentionally.
- When in doubt, group sentences together.

PARAGRAPH VALIDITY:
- A valid paragraph contains AT LEAST TWO complete sentences.
- Exception: a single-sentence paragraph is allowed ONLY if it is clearly a lead, a strong closing statement, or an intentional stylistic standalone sentence.

STRICT EXTRACTION RULES:
1. Preserve the text EXACTLY as written.
2. Do NOT rewrite, summarize, translate, normalize or correct anything.
3. Do NOT modify wording.
4. Do NOT reorder content.
5. Ignore bylines, captions, metadata and publication details.
6. The first sentence of the article body is NOT a header — include it in the first paragraph.
7. Preserve punctuation, accents, quotation marks and spacing exactly as they appear.

ANTI-FRAGMENTATION RULES:
- If multiple consecutive sentences discuss the same event, actor, or subtopic, they MUST remain in the same paragraph.
- Consecutive quoted speech attributions from the same speaker MUST stay in the same paragraph.

OUTPUT REQUIREMENTS (MANDATORY):
- Return ONLY a valid JSON object.
- The JSON must contain exactly one key: "paragraphs".
- The value must be a list of strings.
- Each string must be one reconstructed paragraph.
- Do NOT include explanations or any text outside the JSON.

If you separate sentences unnecessarily, the evaluation will fail.
When uncertain, group sentences together rather than splitting them.

Example:
{{"paragraphs": ["Primer párrafo completo con varias oraciones.", "Segundo párrafo completo."]}}

Article:
{content}
"""

# PARAGRAPHER_PROMPT = """Your task is to identify and extract the paragraphs from the following Peruvian Spanish journalistic article.

# Rules:
# - Extract the paragraphs exactly as they appear in the text, preserving the original wording.
# - A paragraph is a thematically coherent block of text that contains AT LEAST TWO sentences, unless the original text has a single-sentence paragraph of clear and explicit structural importance (e.g. a lead sentence or a closing statement).
# - Ignore headers, captions, or metadata that are not part of the article body.
# - Return ONLY a valid JSON object with a single key "paragraphs" containing a list of strings.
# - Do not add explanations, comments or any text outside the JSON.

# Example output format:
# {{"paragraphs": ["First paragraph text.", "Second paragraph text."]}}

# Article:
# {content}"""

VALID_BINARY = {"argumentative", "non-argumentative"}
VALID_COMPONENT = {"claim", "premise", "none"}
DECISION_TREE = """
ÁRBOL DE DECISIÓN PARA LA ANOTACIÓN (aplicar estrictamente en orden):

1. ¿La oración cumple una función argumentativa en su contexto local inmediato?
   → NO: label_binary = non-argumentative y label_component = none. STOP.
   → SÍ: continuar al paso 2.

2. ¿La oración principalmente APOYA otra oración del contexto?
   → SÍ: continuar al paso 3.
   → NO: continuar al paso 5.

3. ¿Puede responder naturalmente a la pregunta "¿Por qué?" respecto a otra oración?
   → SÍ: label_binary = argumentative y label_component = premise. STOP.
   → NO: continuar al paso 4.

4. ¿Aporta evidencia, datos, ejemplo, explicación causal o referencia a autoridad?
   → SÍ: label_binary = argumentative y label_component = premise. STOP.
   → NO: label_binary = non-argumentative y label_component = none. STOP.

5. ¿La oración expresa un punto de vista que podría ser razonablemente cuestionado?
   → NO: label_binary = non-argumentative y label_component = none. STOP.
   → SÍ: continuar al paso 6.

6. ¿Está formulada como conclusión (por ejemplo: "por lo tanto", "en consecuencia", "por ende")?
   → SÍ: label_binary = argumentative y label_component = claim. STOP.
   → NO: continuar al paso 7.

7. ¿Es discurso reportado que expresa una posición evaluativa o diagnóstica?
   → SÍ: label_binary = argumentative y label_component = claim. STOP.
   → NO: continuar al paso 8.

8. Si expresa un punto de vista defendible dentro del contexto local (sin requerir reconstrucción global):
   → SÍ: label_binary = argumentative y label_component = claim. STOP.
   → NO: label_binary = non-argumentative y label_component = none. STOP.

Nota de coherencia:
Si label_component es "claim" o "premise", entonces label_binary debe ser necesariamente "argumentative".
"""

ANNOTATION_PROMPT = """
You are an expert argumentation annotator being evaluated for strict adherence to formal annotation guidelines.

This task is part of a controlled benchmark. Any deviation from the guidelines, label constraints, or output format will be considered an error.

---

## Guías de anotación
{guidelines}

---

## Contexto del artículo

**Titular:** {headline}

**Contenido completo:**
{content}

---

## Contexto inmediato

**Párrafo:**
{paragraph_text}

---

## Tarea

Anota la siguiente oración (marcada con >> <<) siguiendo estrictamente las guías de anotación.

Párrafo con oración marcada:
{annotated_paragraph}

---

## Árbol de decisión (OBLIGATORIO)

Aplica los siguientes pasos EN ORDEN.  
No omitas pasos.  
No uses intuición global.  
La decisión debe basarse únicamente en el contexto local disponible.

{decision_tree}

---

## Criterios operativos obligatorios

- La anotación se realiza a nivel de oración individual.
- Evalúa la FUNCIÓN argumentativa, no la forma gramatical.
- Solo se permite reconstrucción mínima si la inferencia está fuertemente respaldada por el contexto inmediato.
- No infieras la tesis global del autor.
- Si la función argumentativa no puede determinarse razonablemente desde el contexto local, etiqueta como "non-argumentative".

---

## Instrucciones de output (FORMATO ESTRICTO)

Responde ÚNICAMENTE con un objeto JSON válido con exactamente estas claves:

- "label_binary": "argumentative" o "non-argumentative"
- "label_component": "claim", "premise" o "none"
- "confidence_binary": número entre 0.0 y 1.0
- "confidence_component": número entre 0.0 y 1.0
- "reasoning": justificación breve en español (máximo 2 oraciones) basada explícitamente en las guías

Restricciones obligatorias:
- Si "label_binary" es "non-argumentative", entonces "label_component" debe ser "none".
- No uses valores distintos a los permitidos.
- No añadas texto fuera del JSON.
- No incluyas explicaciones adicionales.
- El JSON debe ser estrictamente válido.

Calibración de confianza:
- >0.85 solo si la función es claramente inequívoca según las guías.
- 0.60–0.85 si existe leve ambigüedad contextual.
- <0.60 si la decisión depende de inferencia limitada o ambigüedad significativa.

Si violas el formato o las restricciones, la evaluación fallará.

Ejemplo de output válido:
{{"label_binary": "argumentative", "label_component": "claim", "confidence_binary": 0.91, "confidence_component": 0.85, "reasoning": "La oración expresa un juicio evaluativo que puede ser cuestionado, cumpliendo el Challenge Test."}}
"""

# Functions
def load_input(path: str, skip: bool = False) -> pd.DataFrame:
    """Load input CSV and validate required columns."""
    log.info(f"Loading input from: {path}")
    
    df = pd.read_csv(path,
                    sep=";", 
                    decimal=".",
                    na_values="NA", 
                    quotechar='"', 
                    encoding="utf-8")

    if skip:
        REQUIRED_COLUMNS = {"headline", "content", "newspaper", "date", 
                          "article_id", "paragraph_i", "paragraph_text"}
    else:
        REQUIRED_COLUMNS = {"headline", "content", "newspaper", "date"}
        
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    if not skip:
        KEEP_COLUMNS = ["headline", "content", "newspaper", "date"]
        df = df[KEEP_COLUMNS].copy() # Discard unnecessary columns early to reduce memory fragmentation

    original_len = len(df)
    df = df.dropna(subset=["content", "headline"])
    df = df[df["content"].str.strip().str.len() > 0]
    log.info(f"Loaded {original_len} articles, {len(df)} after dropping empty content.")

    # Normalize date to year for stratification
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["year"] = df["date"].dt.year

    # Assign stable article_id
    if not skip:
        df = df.reset_index(drop=True)
        df["article_id"] = df.index.map(lambda i: f"ART_{i:06d}")

    return df

def load_guidelines(path: str) -> str:
    """Load annotation guidelines from a markdown file."""
    log.info(f"Loading annotation guidelines from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        guidelines = f.read()
    log.info(f"Guidelines loaded: {len(guidelines)} characters.")
    return guidelines

def build_paragrapher_prompts(df: pd.DataFrame, model_name: str) -> list[str]:    
    if "mistral" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            fix_mistral_regex=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []
    for _, row in df.iterrows():
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a text segmentation assistant specialized in "
                    "Peruvian Spanish journalistic articles. You always respond "
                    "with valid JSON only, no additional text."
                )
            },
            {
                "role": "user",
                "content": PARAGRAPHER_PROMPT.format(content=row["content"])
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    
    return prompts

def annotate_target_sentence(paragraph_text: str, sentence_text: str) -> str:
    """Mark the target sentence within its paragraph context with >> <<."""
    if sentence_text in paragraph_text:
        return paragraph_text.replace(sentence_text, f">> {sentence_text} <<", 1)
    # Fallback: append marked sentence if not found
    log.warning(f"Sentence not found in paragraph, appending marked version.")
    return f"{paragraph_text}\n\n>> {sentence_text} <<"

def build_annotation_prompts(df: pd.DataFrame, guidelines: str, model_name: str,) -> list[str]:
    if "mistral" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            fix_mistral_regex=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []
    for _, row in df.iterrows():
        annotated_paragraph = annotate_target_sentence(
            str(row["paragraph_text"]),
            str(row["sentence_text"]),
        )
        user_content = ANNOTATION_PROMPT.format(
            guidelines=guidelines,
            headline=row["headline"],
            content=row["content"],
            paragraph_text=row["paragraph_text"],
            annotated_paragraph=annotated_paragraph,
            decision_tree=DECISION_TREE,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un experto en minería de argumentos (Argument Mining) especializado en textos periodísticos en español peruano sobre economía informal."
                    "Tu tarea es anotar oraciones siguiendo unas guías de anotación específicas."
                    "Siempre respondes ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin explicaciones fuera del JSON, sin bloques de código markdown."
                )
            },
            {
                "role": "user", 
                "content": user_content
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    return prompts

def load_spacy_model(preferred: str = "es_dep_news_trf", fallback_lang: str = "es") -> tuple:
    """
    Load spaCy model for sentence segmentation.
    Returns (nlp, model_used) where model_used is a string for logging.
    """
    try:
        nlp = spacy.load(preferred)
        log.info(f"Loaded spaCy model: {preferred}")
        return nlp, preferred
    except Exception:
        log.warning(
            f"Could not load {preferred}. "
            f"Install with: python3 -m spacy download {preferred}. "
            f"Falling back to spacy_udpipe ({fallback_lang})."
        )
        try:
            nlp = spacy_udpipe.load(fallback_lang)
        except Exception:
            log.info(f"Downloading spacy_udpipe model: {fallback_lang}")
            spacy_udpipe.download(fallback_lang)
            nlp = spacy_udpipe.load(fallback_lang)
        log.info(f"Loaded spacy_udpipe model: {fallback_lang}")
        return nlp, f"udpipe:{fallback_lang}"

def report_distribution(df: pd.DataFrame, strat_cols: list[str]) -> None:
    """Log stratum sizes for a given stratification."""
    dist = df.groupby(strat_cols)["paragraph_i"].nunique().reset_index()
    dist.columns = strat_cols + ["n_paragraphs"]
    log.info(f"\nParagraph distribution by {strat_cols}:\n{dist.to_string(index=False)}")

def stratified_sample(
    df: pd.DataFrame,
    n_samples: int = 600,
    min_stratum_size: int = 10,
    min_sentences: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n_samples paragraphs with stratification.

    Strategy:
        1. Filter out paragraphs with fewer than min_sentences detected sentences.
        2. Try stratification by newspaper + year.
        3. If any stratum has fewer than min_stratum_size paragraphs,
           fall back to stratification by newspaper only.
        4. Allocate samples proportionally to stratum size.
        5. Report final distribution.

    Sampling is at the paragraph level (one row per paragraph,
    all sentences of sampled paragraphs are included).
    """
    # Work at paragraph level — count sentences per paragraph first
    para_df = (
        df.groupby(["article_id", "paragraph_i"])
        .agg(
            newspaper=("newspaper", "first"),
            year=("year", "first"),
            paragraph_text=("paragraph_text", "first"),
            n_sentences=("sentence_j", "count"),
        )
        .reset_index()
    )

    # --- Filter: keep only paragraphs with at least min_sentences ---
    before = len(para_df)
    para_df = para_df[para_df["n_sentences"] >= min_sentences].copy()
    dropped = before - len(para_df)
    log.info(
        f"Paragraph filter: kept {len(para_df)}/{before} paragraphs "
        f"with >= {min_sentences} sentences ({dropped} dropped)."
    )

    # --- Attempt 1: stratify by newspaper + year ---
    strat_cols = ["newspaper", "year"]
    stratum_sizes = para_df.groupby(strat_cols).size()
    small_strata = (stratum_sizes < min_stratum_size).sum()

    if small_strata > 0:
        log.warning(
            f"{small_strata} strata have fewer than {min_stratum_size} paragraphs "
            f"under newspaper+year stratification. Falling back to newspaper only."
        )
        strat_cols = ["newspaper"]
        stratum_sizes = para_df.groupby(strat_cols).size()

    report_distribution(df, strat_cols)

    # --- Proportional allocation ---
    total_paragraphs = stratum_sizes.sum()
    allocations = (stratum_sizes / total_paragraphs * n_samples).round().astype(int)

    # Adjust rounding errors to hit exactly n_samples
    diff = n_samples - allocations.sum()
    if diff != 0:
        largest = allocations.idxmax()
        allocations[largest] += diff

    log.info(f"\nSample allocation per stratum:\n{allocations.to_string()}")

    # --- Sample ---
    sampled_parts = []
    for stratum_key, n_alloc in allocations.items():
        if isinstance(stratum_key, str):
            stratum_key = (stratum_key,)
        mask = pd.Series([True] * len(para_df))
        for col, val in zip(strat_cols, stratum_key):
            mask &= para_df[col] == val
        stratum_df = para_df[mask]

        if len(stratum_df) < n_alloc:
            log.warning(
                f"Stratum {stratum_key} has only {len(stratum_df)} paragraphs, "
                f"requested {n_alloc}. Sampling with replacement."
            )
            sampled = stratum_df.sample(n=n_alloc, replace=True, random_state=seed)
        else:
            sampled = stratum_df.sample(n=n_alloc, replace=False, random_state=seed)

        sampled_parts.append(sampled)

    sampled_paras = pd.concat(sampled_parts, ignore_index=True)
    sampled_keys = set(
        zip(sampled_paras["article_id"], sampled_paras["paragraph_i"])
    )

    # Retrieve all sentence rows for sampled paragraphs
    mask = df.apply(
        lambda r: (r["article_id"], r["paragraph_i"]) in sampled_keys, axis=1
    )
    result = df[mask].copy()

    log.info(
        f"\nFinal sample: {len(sampled_paras)} paragraphs, "
        f"{len(result)} sentence rows."
    )
    return result

def clear_gpu_memory():
    """Comprehensive GPU memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()
    else:
        print("⚠️ No GPU available, skipping CUDA cleanup")