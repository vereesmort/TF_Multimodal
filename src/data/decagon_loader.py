"""
Decagon data loader.

Downloads and processes the Decagon dataset from the Stanford SNAP repository:
  http://snap.stanford.edu/decagon/

Produces:
  - Drug-target (drug -> protein) edges
  - Protein-protein interaction (PPI) edges
  - Polypharmacy side effect triples (drug, side_effect, drug)
  - Monopharmacy side effect matrix (drug x side_effect binary matrix)
  - SMILES strings per drug (from STITCH / PubChem mapping)
"""

from __future__ import annotations

import logging
import os
import urllib.request
import gzip
import tarfile
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SNAP_BASE = "http://snap.stanford.edu/decagon/"

# Each entry: csv_filename -> tar.gz archive name on SNAP
SNAP_FILES = {
    "ppi":      ("bio-decagon-ppi.csv",              "bio-decagon-ppi.tar.gz"),
    "targets":  ("bio-decagon-targets.csv",          "bio-decagon-targets.tar.gz"),
    "combo":    ("bio-decagon-combo.csv",            "bio-decagon-combo.tar.gz"),
    "mono":     ("bio-decagon-mono.csv",             "bio-decagon-mono.tar.gz"),
    "se_names": ("bio-decagon-effectcategories.csv", "bio-decagon-effectcategories.tar.gz"),
}


@dataclass
class DecagonData:
    """Container for all Decagon graph components."""

    # Entity id mappings
    drug_to_id: Dict[str, int] = field(default_factory=dict)
    protein_to_id: Dict[str, int] = field(default_factory=dict)
    se_pair_to_id: Dict[str, int] = field(default_factory=dict)  # polypharmacy SEs
    se_mono_to_id: Dict[str, int] = field(default_factory=dict)  # monopharmacy SEs

    # Raw edge lists (string identifiers)
    ppi_edges: List[Tuple[str, str]] = field(default_factory=list)
    drug_target_edges: List[Tuple[str, str]] = field(default_factory=list)
    poly_triples: List[Tuple[str, str, str]] = field(default_factory=list)  # (drugA, SE, drugB)

    # Monopharmacy side effect matrix (drugs x mono_SEs), binary
    # Row order = sorted(drug_to_id.keys()), col order = sorted(se_mono_to_id.keys())
    mono_se_matrix: Optional[np.ndarray] = None

    # SMILES strings per drug CID
    drug_smiles: Dict[str, str] = field(default_factory=dict)

    # Drug target sets (drug_id -> set of protein_ids)
    drug_targets: Dict[str, Set[str]] = field(default_factory=dict)


def download_decagon(raw_dir: str = "data/raw") -> Path:
    """Download all Decagon files to raw_dir if not already present.

    Files are distributed as .tar.gz archives on SNAP; this function
    downloads each archive, extracts the CSV, and removes the archive.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    for key, (csv_fname, tar_fname) in SNAP_FILES.items():
        dest_csv = raw_path / csv_fname
        if dest_csv.exists():
            logger.info(f"  {csv_fname} already present, skipping.")
            continue

        url = SNAP_BASE + tar_fname
        dest_tar = raw_path / tar_fname
        logger.info(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, dest_tar)
            logger.info(f"  Extracting {tar_fname} ...")
            with tarfile.open(dest_tar, "r:gz") as tf:
                # Extract only the CSV member (ignore any directory prefix)
                for member in tf.getmembers():
                    if member.name.endswith(".csv"):
                        member.name = csv_fname  # flatten to raw_dir
                        tf.extract(member, path=raw_path)
                        break
            dest_tar.unlink()
            logger.info(f"  Saved to {dest_csv}")
        except Exception as e:
            logger.warning(f"  Failed to download/extract {tar_fname}: {e}")
            if dest_tar.exists():
                dest_tar.unlink()

    return raw_path


def load_decagon(raw_dir: str = "data/raw") -> DecagonData:
    """Load Decagon dataset from raw CSV files."""
    raw_path = Path(raw_dir)
    data = DecagonData()

    # Decagon CSVs are comma-separated and use Latin-1 encoding
    # (side effect names contain special characters like °).
    READ_OPTS = dict(sep=",", encoding="latin-1")

    # -- PPI edges --
    ppi_file = raw_path / SNAP_FILES["ppi"][0]
    if ppi_file.exists():
        df = pd.read_csv(ppi_file, **READ_OPTS)
        # columns: Gene 1, Gene 2
        cols = df.columns.tolist()
        for _, row in df.iterrows():
            g1, g2 = str(row[cols[0]]), str(row[cols[1]])
            data.ppi_edges.append((g1, g2))
            data.protein_to_id.setdefault(g1, len(data.protein_to_id))
            data.protein_to_id.setdefault(g2, len(data.protein_to_id))
        logger.info(f"Loaded {len(data.ppi_edges)} PPI edges, {len(data.protein_to_id)} proteins")

    # -- Drug-target edges --
    target_file = raw_path / SNAP_FILES["targets"][0]
    if target_file.exists():
        df = pd.read_csv(target_file, **READ_OPTS)
        cols = df.columns.tolist()
        for _, row in df.iterrows():
            drug, protein = str(row[cols[0]]), str(row[cols[1]])
            data.drug_target_edges.append((drug, protein))
            data.drug_to_id.setdefault(drug, len(data.drug_to_id))
            data.protein_to_id.setdefault(protein, len(data.protein_to_id))
            data.drug_targets.setdefault(drug, set()).add(protein)
        logger.info(f"Loaded {len(data.drug_target_edges)} drug-target edges")

    # -- Polypharmacy combo edges --
    combo_file = raw_path / SNAP_FILES["combo"][0]
    if combo_file.exists():
        df = pd.read_csv(combo_file, **READ_OPTS)
        # columns: STITCH 1, STITCH 2, Polypharmacy Side Effect, Side Effect Name
        for _, row in df.iterrows():
            d1, d2 = str(row.iloc[0]), str(row.iloc[1])
            se = str(row.iloc[2])
            data.poly_triples.append((d1, se, d2))
            data.drug_to_id.setdefault(d1, len(data.drug_to_id))
            data.drug_to_id.setdefault(d2, len(data.drug_to_id))
            data.se_pair_to_id.setdefault(se, len(data.se_pair_to_id))
        logger.info(f"Loaded {len(data.poly_triples)} polypharmacy triples, "
                    f"{len(data.se_pair_to_id)} side effect types")

    # -- Monopharmacy side effects --
    mono_file = raw_path / SNAP_FILES["mono"][0]
    if mono_file.exists():
        df = pd.read_csv(mono_file, **READ_OPTS)
        # columns: STITCH, Side Effect Name, Umls concept id
        drug_col = df.columns[0]
        se_col = df.columns[1]
        # Build drug x mono_SE binary matrix
        drugs_in_mono = df[drug_col].unique()
        ses_in_mono = df[se_col].unique()

        for d in drugs_in_mono:
            data.drug_to_id.setdefault(str(d), len(data.drug_to_id))
        for se in ses_in_mono:
            data.se_mono_to_id.setdefault(str(se), len(data.se_mono_to_id))

        n_drugs = len(data.drug_to_id)
        n_mono = len(data.se_mono_to_id)
        # Build sparse indicator (only for drugs that appear in mono data)
        drug_idx_map = {d: data.drug_to_id[str(d)] for d in drugs_in_mono}
        se_idx_map = {se: data.se_mono_to_id[str(se)] for se in ses_in_mono}

        mat = np.zeros((n_drugs, n_mono), dtype=np.float32)
        for _, row in df.iterrows():
            d_idx = drug_idx_map.get(row[drug_col])
            se_idx = se_idx_map.get(row[se_col])
            if d_idx is not None and se_idx is not None:
                mat[d_idx, se_idx] = 1.0
        data.mono_se_matrix = mat
        logger.info(f"Monopharmacy matrix: {mat.shape}, "
                    f"density={mat.mean():.4f}")

    return data


def build_pykeen_triples(data: DecagonData, split: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Build PyKEEN-compatible triple factories from DecagonData.

    Returns:
        train_tf, valid_tf, test_tf: PyKEEN TriplesFactory objects
        entity_to_id: merged drug+protein entity mapping
        relation_to_id: polypharmacy SE relation mapping
    """
    from pykeen.triples import TriplesFactory

    # Merge entity spaces: drugs first, then proteins
    entity_to_id: Dict[str, int] = {}
    for drug, idx in data.drug_to_id.items():
        entity_to_id[f"drug:{drug}"] = idx
    offset = len(data.drug_to_id)
    for protein, idx in data.protein_to_id.items():
        entity_to_id[f"protein:{protein}"] = offset + idx

    relation_to_id = {f"SE:{se}": idx for se, idx in data.se_pair_to_id.items()}

    # Add drug-protein target relation
    relation_to_id["drug_targets"] = len(relation_to_id)
    # Add PPI relation
    relation_to_id["ppi"] = len(relation_to_id)

    # Build raw triple list: [head_label, relation_label, tail_label]
    triples = []
    for d1, se, d2 in data.poly_triples:
        triples.append([f"drug:{d1}", f"SE:{se}", f"drug:{d2}"])
    for drug, protein in data.drug_target_edges:
        triples.append([f"drug:{drug}", "drug_targets", f"protein:{protein}"])
    for p1, p2 in data.ppi_edges:
        triples.append([f"protein:{p1}", "ppi", f"protein:{p2}"])

    triples_arr = np.array(triples)

    # Shuffle and split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(triples_arr))
    n = len(idx)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]

    # Only poly triples go to validation/test (biological evaluation convention)
    poly_mask = np.array([t[1].startswith("SE:") for t in triples_arr])
    poly_indices = np.where(poly_mask)[0]
    non_poly_indices = np.where(~poly_mask)[0]

    rng2 = np.random.default_rng(42)
    poly_perm = rng2.permutation(poly_indices)
    n_poly = len(poly_perm)
    n_pv = int(n_poly * split[1])
    n_pt = int(n_poly * split[2])

    poly_valid = poly_perm[:n_pv]
    poly_test = poly_perm[n_pv:n_pv + n_pt]
    poly_train = poly_perm[n_pv + n_pt:]

    train_triples = np.concatenate([
        triples_arr[non_poly_indices],
        triples_arr[poly_train]
    ])
    valid_triples = triples_arr[poly_valid]
    test_triples = triples_arr[poly_test]

    tf_all = TriplesFactory.from_labeled_triples(
        triples=triples_arr,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    train_tf = TriplesFactory.from_labeled_triples(
        triples=train_triples,
        entity_to_id=tf_all.entity_to_id,
        relation_to_id=tf_all.relation_to_id,
    )
    valid_tf = TriplesFactory.from_labeled_triples(
        triples=valid_triples,
        entity_to_id=tf_all.entity_to_id,
        relation_to_id=tf_all.relation_to_id,
    )
    test_tf = TriplesFactory.from_labeled_triples(
        triples=test_triples,
        entity_to_id=tf_all.entity_to_id,
        relation_to_id=tf_all.relation_to_id,
    )
    logger.info(f"Triples — train: {len(train_triples)}, "
                f"valid: {len(valid_triples)}, test: {len(test_triples)}")
    return train_tf, valid_tf, test_tf, tf_all.entity_to_id, tf_all.relation_to_id
