"""
parse_tei.py — TEI XML Argumentative Corpus Parser

Converts TEI-annotated XML thesis files into the project's unified schema.
Accepts a single .xml file or a folder of .xml files, producing a single
merged output CSV.

TEI structure:
    <div type="thesis">
        <div type="section">
            <div type="paragraph_anotation">
                <p>
                    <seg type="conclusion|premise" function="..." rend="...">text</seg>
                    non-argumentative text...
                </p>
                <desc type="argument_level">...</desc>
                <desc type="argument_type">...</desc>
                <desc type="relation">...</desc>
            </div>
        </div>
    </div>

Mapping:
    seg type="conclusion" → claim
    seg type="premise"    → premise
    text outside <seg>    → non-argumentative

Output schema:
    title | level | section | filename | arg_level | arg_type | relation |
    text_type | seg_function | seg_rend | text

Usage:
    # Single file
    python parse_tei.py \\
        --input  data/external/corpus.xml \\
        --output data/adapted/tei_adapted.csv

    # Folder of .xml files
    python parse_tei.py \\
        --input  data/external/tei/ \\
        --output data/adapted/tei_adapted.csv

Requirements:
    pip install pandas
"""

import argparse
import csv
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATASET_NAME = "tei_xml_corpus"
NS = "http://www.tei-c.org/ns/1.0"

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

SEG_TYPE_MAP = {
    "conclusion": ("argumentative", "claim"),
    "premise":    ("argumentative", "premise"),
}

FIELDNAMES = ["sentence_text", "label_binary", "label_component"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def tag(name: str) -> str:
    return f"{{{NS}}}{name}"


def find_desc(elem, type_val: str) -> str:
    for desc in elem.findall(f".//{{{NS}}}desc"):
        if desc.get("type") == type_val:
            return desc.text.strip() if desc.text else ""
    return ""


def get_text_outside_segs(p_elem) -> str:
    parts = []
    if p_elem.text and p_elem.text.strip():
        parts.append(p_elem.text.strip())
    for seg in p_elem:
        if seg.tail and seg.tail.strip():
            parts.append(seg.tail.strip())
    return " ".join(parts)


# ------------------------------------------------------------------
# Core parsing
# ------------------------------------------------------------------

def parse_xml(path: Path) -> list[dict]:
    """Parse a single TEI XML file into a list of row dicts."""
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        log.warning(f"  [{path.name}] Failed to parse XML: {e} — skipping.")
        return []

    root = tree.getroot()
    records = []
    n_paragraphs = 0
    n_errors = 0

    for thesis in root.iter(tag("div")):
        if thesis.get("type") != "thesis":
            continue

        title_elem = thesis.find(f".//{{{NS}}}title")
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        level = find_desc(thesis.find(f".//{{{NS}}}head") or thesis, "level")

        for section in thesis.iter(tag("div")):
            if section.get("type") != "section":
                continue

            section_name = find_desc(section, "section")
            filename     = find_desc(section, "filename")

            for annotation in section.iter(tag("div")):
                if annotation.get("type") != "paragraph_anotation":
                    continue

                n_paragraphs += 1
                p_elem = annotation.find(f".//{{{NS}}}p")
                if p_elem is None:
                    log.warning(f"  [{path.name}] Annotation in '{title}' has no <p> — skipping.")
                    n_errors += 1
                    continue

                arg_level = find_desc(annotation, "argument_level")
                arg_type  = find_desc(annotation, "argument_type")
                relation  = find_desc(annotation, "relation")

                base = dict(title=title, level=level, section=section_name,
                            filename=filename, arg_level=arg_level,
                            arg_type=arg_type, relation=relation)

                # Non-argumentative text (outside <seg>)
                non_arg = get_text_outside_segs(p_elem)
                if non_arg:
                    records.append({
                        "sentence_text":   non_arg,
                        "label_binary":    "non-argumentative",
                        "label_component": "none",
                    })

                # Claims y premises
                for seg in p_elem.findall(f"{{{NS}}}seg"):
                    seg_type = seg.get("type", "")
                    text = seg.text.strip() if seg.text else ""

                    if not text:
                        log.warning(f"  [{path.name}] Empty <seg type='{seg_type}'> in '{title}' — skipping.")
                        continue

                    if seg_type not in SEG_TYPE_MAP:
                        log.warning(f"  [{path.name}] Unknown seg type '{seg_type}' in '{title}' — skipping.")
                        continue

                    label_binary, label_component = SEG_TYPE_MAP[seg_type]
                    records.append({
                        "sentence_text":   text,
                        "label_binary":    label_binary,
                        "label_component": label_component,
                    })

    log.info(
        f"  {path.name}: {n_paragraphs} paragraphs, {n_errors} errors → "
        f"{len(records)} segments"
    )
    return records


# ------------------------------------------------------------------
# Input resolution
# ------------------------------------------------------------------

def resolve_input_files(input_path: str, ext: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob(f"*{ext}"))
        if not files:
            raise FileNotFoundError(
                f"No '{ext}' files found in directory: {p}\n"
                f"Use --ext to specify a different extension."
            )
        return files
    raise FileNotFoundError(f"Input path does not exist: {p}")


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    valid_binary    = {"argumentative", "non-argumentative"}
    valid_component = {"claim", "premise", "none"}
    bad_b = ~df["label_binary"].isin(valid_binary)
    bad_c = ~df["label_component"].isin(valid_component)
    if bad_b.any():
        raise ValueError(f"Invalid label_binary values: {df.loc[bad_b, 'label_binary'].unique()}")
    if bad_c.any():
        raise ValueError(f"Invalid label_component values: {df.loc[bad_c, 'label_component'].unique()}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse TEI XML file(s) into project schema CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single .xml file OR a folder containing multiple .xml files."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the merged output CSV."
    )
    parser.add_argument(
        "--ext", default=".xml",
        help="File extension to glob when --input is a folder (default: .xml)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    files = resolve_input_files(args.input, args.ext)
    log.info(f"Found {len(files)} file(s) to process:")
    for f in files:
        log.info(f"  {f}")

    all_records = []
    for f in files:
        all_records.extend(parse_xml(f))

    if not all_records:
        log.error("No segments extracted. Check file format and structure.")
        sys.exit(1)

    df = pd.DataFrame(all_records).reset_index(drop=True)
    validate(df)

    log.info(
        f"\nAdapted dataset summary ({DATASET_NAME}):\n"
        f"  Files processed : {len(files)}\n"
        f"  Total segments  : {len(df)}\n\n"
        f"  label_binary:\n{df['label_binary'].value_counts().to_string()}\n\n"
        f"  label_component:\n{df['label_component'].value_counts().to_string()}"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, **CSV_OPTS)
    log.info(f"\nSaved → {args.output}  ({len(df)} rows × {len(df.columns)} columns)")


if __name__ == "__main__":
    main()