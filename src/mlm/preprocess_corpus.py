"""
preprocess_corpus.py
--------------------
Preprocessing pipeline for the OCR-extracted CGEC13-20 corpus,
preparing it for BETO domain-adaptive MLM pre-training.

Expected input structure:
    corpus_root/
        {newspaper}/
            {year}/
                {month}/
                    {day}/
                        *.txt

Output:
    A single corpus_clean.txt file with one paragraph per line,
    separated by blank lines (document boundaries).
    Ready for HuggingFace datasets.load_dataset("text").

Usage:
    python preprocess_corpus.py --input_dir /path/to/cgec1320 --output_file corpus_clean.txt
"""

import re
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Cleaning parameters
# ─────────────────────────────────────────────

MIN_LINE_CHARS = 25          # Back towards original — avoids discarding valid short sentences
MIN_PARAGRAPH_TOKENS = 4     # Slightly more permissive
MAX_GARBAGE_RATIO = 0.3      # Back to original — less aggressive
MAX_SHORT_TOKEN_RATIO = 0.5  # More permissive — allows siglas, cifras, etc.
MIN_REAL_WORD_RATIO = 0.35   # Much more permissive — allows PBI, INEI, 3.2%, etc.

# Regex for a "real" Spanish word token
REAL_WORD_RE = re.compile(r"^[a-záéíóúüñA-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ\-\']{2,}$")

# Patterns for meaningless ALL-CAPS fragments (newspaper column titles broken by OCR)
# e.g. "SILVA, SE ESTRENA HOY EN ee" or "N VALENTÍN E MabDuro vano"
CAPS_FRAGMENT_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ\s\d,.\-]{10,}$")


# ─────────────────────────────────────────────
# Cleaning functions
# ─────────────────────────────────────────────

def is_garbage_line(line: str) -> bool:
    """
    Detects lines that are primarily OCR noise using multiple heuristics:
      1. Too short
      2. Too many non-alphanumeric characters
      3. Too many very short tokens (single letters, 2-char fragments) → layout noise
      4. Too few real Spanish word tokens
      5. Fully uppercase fragment → broken newspaper column header
    """
    if len(line) < MIN_LINE_CHARS:
        return True

    # Heuristic 1: non-alphanumeric ratio
    non_alpha = sum(1 for c in line if not (c.isalnum() or c.isspace()))
    if non_alpha / len(line) > MAX_GARBAGE_RATIO:
        return True

    tokens = line.split()
    if not tokens:
        return True

    # Heuristic 2: ratio of very short tokens (<=2 chars)
    short_tokens = sum(1 for t in tokens if len(t) <= 2)
    if short_tokens / len(tokens) > MAX_SHORT_TOKEN_RATIO:
        return True

    # Heuristic 3: ratio of real word tokens
    real_words = sum(1 for t in tokens if REAL_WORD_RE.match(t))
    if real_words / len(tokens) < MIN_REAL_WORD_RATIO:
        return True

    # Heuristic 4: all-caps fragment (broken column header)
    if CAPS_FRAGMENT_RE.match(line.strip()):
        return True

    return False


def clean_line(line: str) -> str:
    """Basic cleaning of a single line."""
    line = line.strip()
    # Normalize whitespace
    line = re.sub(r"\s+", " ", line)
    # Normalize typographic dashes and quotes
    line = line.replace("\u2013", "-").replace("\u2014", "-")
    line = line.replace("\u201c", '"').replace("\u201d", '"')
    line = line.replace("\u2018", "'").replace("\u2019", "'")
    # Remove excessive dots (common OCR artifact)
    line = re.sub(r"\.{4,}", "...", line)
    # Remove spaces before punctuation
    line = re.sub(r"\s([.,;:!?])", r"\1", line)
    # Remove trailing OCR junk tokens (short non-word sequences at end of line)
    # e.g. "...embajador en Canadá, rxss.v1s"
    line = re.sub(r"\s+\S{1,6}$", lambda m: m.group(0) if re.search(r"[a-záéíóúüñ]{3,}", m.group(0), re.I) else "", line)
    return line.strip()


def join_broken_lines(lines: list) -> list:
    """
    Rejoins lines broken by the OCR pipeline.

    Heuristic: if a line does not end with sentence-final punctuation
    and the next line starts with a lowercase letter, they likely
    belong to the same sentence and are merged.
    """
    SENTENCE_END = re.compile(r"[.!?\"]\s*$")
    joined = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            # Blank line = paragraph boundary
            if buffer:
                joined.append(buffer)
                buffer = ""
            continue

        if buffer:
            if not SENTENCE_END.search(buffer) and line and line[0].islower():
                # Continuation of broken sentence
                buffer = buffer + " " + line
            else:
                joined.append(buffer)
                buffer = line
        else:
            buffer = line

    if buffer:
        joined.append(buffer)

    return joined


def is_valid_paragraph(paragraph: str) -> bool:
    """Filters out paragraphs that are too short or too noisy."""
    tokens = paragraph.split()
    if len(tokens) < MIN_PARAGRAPH_TOKENS:
        return False
    if is_garbage_line(paragraph):
        return False
    return True


def process_file(filepath: Path) -> list:
    """Reads a single OCR .txt file and returns a list of clean paragraphs."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
    except Exception as e:
        log.warning(f"Could not read {filepath}: {e}")
        return []

    # Clean each line individually, discard garbage lines early
    cleaned_lines = []
    for line in raw_lines:
        cl = clean_line(line)
        if cl and not is_garbage_line(cl):
            cleaned_lines.append(cl)

    # Rejoin OCR-broken lines into paragraphs
    paragraphs = join_broken_lines(cleaned_lines)

    # Final paragraph-level validation
    valid = [p for p in paragraphs if is_valid_paragraph(p)]

    return valid


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def collect_txt_files(root: Path) -> list:
    """Recursively collects all .txt files under root."""
    files = list(root.rglob("*.txt"))
    log.info(f"Found {len(files):,} .txt files under {root}")
    return files


def run(input_dir: str, output_file: str):
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    txt_files = collect_txt_files(root)
    if not txt_files:
        raise ValueError("No .txt files found.")

    total_paragraphs = 0
    total_files = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for filepath in tqdm(txt_files, desc="Processing files"):
            paragraphs = process_file(filepath)
            if not paragraphs:
                continue

            total_paragraphs += len(paragraphs)
            total_files += 1

            for paragraph in paragraphs:
                out.write(paragraph + "\n")

            # Blank line = document boundary for HuggingFace datasets
            out.write("\n")

    log.info(f"Done.")
    log.info(f"  Files processed        : {total_files:,}")
    log.info(f"  Valid paragraphs saved : {total_paragraphs:,}")
    log.info(f"  Output file            : {output_file}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess OCR corpus for BETO MLM domain-adaptive pre-training"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root folder containing the .txt files (searched recursively)"
    )
    parser.add_argument(
        "--output_file",
        default="corpus_clean.txt",
        help="Output file path (default: corpus_clean.txt)"
    )
    args = parser.parse_args()

    run(args.input_dir, args.output_file)