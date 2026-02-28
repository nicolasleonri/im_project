"""
parse_vivesdebate.py — VivesDebate CSV Parser

Converts VivesDebate annotated CSV files into the project's unified schema.
Accepts a single .csv file or a folder of .csv files, producing a single
merged output CSV.

Input structure:
    ID | TYPE | ARGUMENT NUMBER | TEAM STANCE | RELATED ID |
    ARGUMENTAL RELATION TYPE | ADU_CAT | ADU_ES | ADU_EN

Logic:
    1) ADU without RELATED ID and nobody points to it → non-argumentative
    2) ADU without RELATED ID and somebody points to it (any relation) →
       introduction, fused with the claim that references it
    3) ADU that points to an introduction (any relation) → CLAIM
       (text = introduction + claim fused)
    4) ADUs that point to the CLAIM with RA (transitively) → PREMISE

Output schema:
    sentence_text | label_binary | label_component

Usage:
    # Single file
    python parse_vivesdebate.py \\
        --input  data/external/debate.csv \\
        --output data/adapted/vivesdebate_adapted.csv

    # Folder of .csv files
    python parse_vivesdebate.py \\
        --input  data/external/vivesdebate/ \\
        --output data/adapted/vivesdebate_adapted.csv

Requirements:
    pip install pandas
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATASET_NAME = "vivesdebate"

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


# ------------------------------------------------------------------
# Core parsing
# ------------------------------------------------------------------

def parse_csv(path: Path) -> list[dict]:
    """Parse a single VivesDebate CSV into a list of segment dicts."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        log.warning(f"  [{path.name}] Failed to read CSV: {e} — skipping.")
        return []

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Detect all (RELATED ID, RELATION TYPE) column pairs dynamically
    related_id_cols = [c for c in df.columns if c.startswith("RELATED ID")]
    rel_type_cols   = [c for c in df.columns if c.startswith("ARGUMENTAL RELATION")]

    if len(related_id_cols) != len(rel_type_cols):
        log.warning(f"  [{path.name}] Mismatched RELATED ID / RELATION TYPE columns — skipping.")
        return []

    relation_pairs = list(zip(related_id_cols, rel_type_cols))

    # Build ID → text lookup
    id_to_text = dict(zip(df["ID (CHRONOLOGICAL)"], df["ADU_ES"]))

    # ------------------------------------------------------------------
    # Step 1: Build relation graph
    # all_relations: id → list of (related_id, rel_type) for ANY relation
    # ra_relations:  id → list of related_id for RA only
    # referenced_by: related_id → list of ids that point to it (any relation)
    # ------------------------------------------------------------------
    all_relations = {row["ID (CHRONOLOGICAL)"]: [] for _, row in df.iterrows()}
    ra_targets    = {row["ID (CHRONOLOGICAL)"]: [] for _, row in df.iterrows()}
    referenced_by = {row["ID (CHRONOLOGICAL)"]: [] for _, row in df.iterrows()}

    for _, row in df.iterrows():
        current_id = row["ID (CHRONOLOGICAL)"]
        for rel_id_col, rel_type_col in relation_pairs:
            rel_type    = str(row.get(rel_type_col, "")).strip().upper()
            related_raw = str(row.get(rel_id_col,   "")).strip()

            if not related_raw or related_raw == "NAN":
                continue

            try:
                related_ids = [int(x.strip()) for x in related_raw.split(";")]
            except ValueError:
                log.warning(f"  [{path.name}] ID {current_id}: could not parse RELATED ID '{related_raw}' — skipping.")
                continue

            for rid in related_ids:
                all_relations[current_id].append((rid, rel_type))
                referenced_by[rid].append(current_id)
                if rel_type == "RA":
                    ra_targets[current_id].append(rid)

    # ------------------------------------------------------------------
    # Step 2: Identify introductions, claims, premises, non-argumentative
    # ------------------------------------------------------------------
    roles     = {}  # id → "claim" | "premise" | "non-argumentative"
    fuse_with = {}  # claim_id → intro_id to fuse text with

    for _, row in df.iterrows():
        adu_id = row["ID (CHRONOLOGICAL)"]

        has_outgoing = len(all_relations[adu_id]) > 0
        has_incoming = len(referenced_by[adu_id]) > 0

        if not has_outgoing and not has_incoming:
            # Nobody references it and it references nobody → non-argumentative
            roles[adu_id] = "non-argumentative"

        elif not has_outgoing and has_incoming:
            # Introduction: someone points to it, it points to nobody
            # Will be fused with the claim that references it — skip for now
            roles[adu_id] = "introduction"

    # Identify claims: ADUs that point to an introduction (any relation)
    for adu_id, relations in all_relations.items():
        for rid, rel_type in relations:
            if roles.get(rid) == "introduction":
                roles[adu_id] = "claim"
                fuse_with[adu_id] = rid
                break

    # ------------------------------------------------------------------
    # Step 3: Identify premises transitively (BFS over RA edges toward claims)
    # ------------------------------------------------------------------
    claim_ids = {adu_id for adu_id, role in roles.items() if role == "claim"}

    # Seed: ADUs that point via RA to a claim
    queue = []
    for adu_id, ra_list in ra_targets.items():
        if roles.get(adu_id) in ("claim", "introduction", "non-argumentative"):
            continue
        for rid in ra_list:
            if rid in claim_ids:
                roles[adu_id] = "premise"
                queue.append(adu_id)
                break

    # BFS: ADUs that point via RA to a premise are also premises
    visited = set(queue)
    while queue:
        current = queue.pop(0)
        for adu_id, ra_list in ra_targets.items():
            if adu_id in visited:
                continue
            if roles.get(adu_id) in ("claim", "introduction", "non-argumentative"):
                continue
            if current in ra_list:
                roles[adu_id] = "premise"
                visited.add(adu_id)
                queue.append(adu_id)

    # ------------------------------------------------------------------
    # Step 4: Build records
    # ------------------------------------------------------------------
    records = []
    n_discarded = 0

    for _, row in df.iterrows():
        adu_id = row["ID (CHRONOLOGICAL)"]
        role   = roles.get(adu_id)

        if role == "introduction":
            # Will be fused into the claim — skip standalone
            continue

        if role is None:
            # Has relations but didn't fit any category — discard
            n_discarded += 1
            continue

        text = str(id_to_text.get(adu_id, "")).strip()
        if not text or text == "nan":
            log.warning(f"  [{path.name}] ID {adu_id}: empty ADU_ES — skipping.")
            continue

        if role == "claim":
            # Fuse with introduction if present
            intro_id = fuse_with.get(adu_id)
            if intro_id is not None:
                intro_text = str(id_to_text.get(intro_id, "")).strip()
                if intro_text and intro_text != "nan":
                    text = intro_text + " " + text

            records.append({
                "sentence_text":   text,
                "label_binary":    "argumentative",
                "label_component": "claim",
            })

        elif role == "premise":
            records.append({
                "sentence_text":   text,
                "label_binary":    "argumentative",
                "label_component": "premise",
            })

        elif role == "non-argumentative":
            records.append({
                "sentence_text":   text,
                "label_binary":    "non-argumentative",
                "label_component": "none",
            })

    log.info(
        f"  {path.name}: {len(df)} ADUs, {n_discarded} unclassified → "
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
        description="Parse VivesDebate CSV file(s) into project schema CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single .csv file OR a folder containing multiple .csv files."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the merged output CSV."
    )
    parser.add_argument(
        "--ext", default=".csv",
        help="File extension to glob when --input is a folder (default: .csv)."
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
        all_records.extend(parse_csv(f))

    if not all_records:
        log.error("No segments extracted. Check file format and structure.")
        sys.exit(1)

    df = pd.DataFrame(all_records).reset_index(drop=True)
    validate(df)

    log.info(
        f"\nAdapted dataset summary ({DATASET_NAME}):\n"
        f"  Files processed  : {len(files)}\n"
        f"  Total segments   : {len(df)}\n\n"
        f"  label_binary:\n{df['label_binary'].value_counts().to_string()}\n\n"
        f"  label_component:\n{df['label_component'].value_counts().to_string()}"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, **CSV_OPTS)
    log.info(f"\nSaved → {args.output}  ({len(df)} rows × {len(df.columns)} columns)")


if __name__ == "__main__":
    main()