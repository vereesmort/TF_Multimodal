"""
ablation_track1_ml.py
---------------------
Track 1 ablation study using tree-based classifiers (XGBoost / CatBoost).

Goal
----
Determine which embedding sources contribute to polypharmacy side effect
prediction BEFORE committing to the full KGE training loop.

Tree-based models are used here because:
  - Fast to train (seconds per condition vs hours for KGE)
  - Naturally handle feature importance
  - Make the contribution of each embedding source directly inspectable
  - No gradient flow complications — each condition is cleanly isolated

Conditions
----------
  A  [ ChemBERTa || drug-via-ESM ]   (hstack, D_d + D_p e.g. 128+128=256)
  B  [ 0^{D_d} || drug-via-ESM ]     (protein signal only, drug slot zero)
  C  [ ChemBERTa || 0^{D_p} ]        (drug signal only, protein slot zero)
  D  [ Xavier_d || Xavier_p ]        (independent random per-drug slots)
  E  [ (ChemBERTa+mono)/2 || (drug-via-ESM + drug-via-PPI)/2 ]  (hstack; each
     branch can be mean-fused internally when two tensors share a modality)

Feature construction for a drug pair (drug_A, drug_B, side_effect_r)
---------------------------------------------------------------------
Tree models cannot directly use the SimplE scoring function.
Each drug has a **fused** vector of size ``D_d + D_p`` (e.g. 256) built by
**concatenating** the drug branch (width ``D_d``) and the protein branch
(width ``D_p``). The protein branch is **target-mean** ESM-2 (and for E, also
PPI) from ``bio-decagon-targets.csv`` (drug→gene→embedding row). The ``+`` in
older docs meant element-wise sum; the default now is **hstack** (``torch.cat``).

Combination strategies:
  - sum:    [e_A + e_B]                  (dim, order-invariant)
  - concatenate:   [e_A || e_B]                  (2×dim features)
  - elementwise:   [e_A * e_B]                   (dim features, like DistMult)
  - absolute diff: [|e_A - e_B|]                 (dim features, symmetric)
  - all three:     [e_A * e_B || |e_A - e_B| || e_A + e_B]   (3×dim)

We use elementwise product + absolute difference as default — this is the
standard approach in KGE-to-classifier transfer (Nickel et al., Hamilton et al.)
and is invariant to drug ordering (symmetric), which is correct for PSE.
`sym` follows a common KGE-to-classifier transfer pattern (Nickel et al.,
Hamilton et al.). `concat` keeps branch-wise identity information; `sum` is a
compact symmetric baseline.


Negative sampling
-----------------
The polypharmacy graph only contains positive pairs. We generate negatives
by corrupting one drug in each positive pair (standard KGE negative sampling).
Ratio: 1 positive : 1 negative (balanced). Seed-controlled for reproducibility.

Memory / scaling
----------------
Rough size of the feature matrix alone: ``n_samples × (8 × (D_d + D_p))`` bytes
(float32 product + float32 abs-diff over fused length ``D_d + D_p``).
Example: 5M samples and ``D_d+D_p`` = 256 → ~10 GB for ``X`` alone.
The pipeline deletes ``X`` after taking train/test slices. Per-condition results
are appended to CSV as each model finishes (no giant list of DataFrames).

Use ``--n_se_sample`` with ``--se_offset`` to process side effects in batches
(resume later without ``--max_pos_edges``). Put ``--output`` and
``--dataset_cache_dir`` on Colab VM local storage (e.g. ``/content/...``) to
avoid Google Drive sync latency; Drive mounts add network I/O.

Output
------
  ablation_results.csv     — per-condition AUROC, AUPRC, AP@50 per SE tier
  ablation_summary.csv     — aggregate metrics across conditions
  feature_importance/      — per-condition feature importance plots (XGBoost)

Usage
-----
  # Protein tensors can be either:
  #   (a) protein-level (n_proteins, D) -> pass --targets for aggregation
  #   (b) pre-aggregated drug-level (n_drugs, D) -> --targets optional
  python ablation_track1_ml.py \\
    --combo    bio-decagon-combo.csv \\
    --targets  bio-decagon-targets.csv \\
    --drug_emb_chemberta  data/cache/ablation/drug_emb_chemberta_256.pt \\
    --drug_emb_mono       data/cache/ablation/drug_emb_mono_256.pt \\
    --prot_emb_esm2       data/cache/ablation/protein_emb_esm2_only_dim256_esm2_t30_150M_UR50D.pt \\
    --prot_emb_ppi        data/cache/ablation/protein_emb_ppi_neighbour_dim256_esm2_t30_150M_UR50D.pt \\
    --output   ./ablation_results \\
    --seed     42

  Build the *_chemberta_*, *_mono_*, and protein_* tensors with:
    python scripts/precompute_embeddings_ablation.py --output_dir data/cache/ablation

  Minimal run (conditions A–D only, no mono/PPI):
  python ablation_track1_ml.py \\
    --combo    bio-decagon-combo.csv \\
    --targets  bio-decagon-targets.csv \\
    --drug_emb_chemberta  data/cache/ablation/drug_emb_chemberta_256.pt \\
    --prot_emb_esm2       data/cache/ablation/protein_emb_esm2_only_dim256_esm2_t30_150M_UR50D.pt \\
    --output   ./ablation_results

  Same run but using pre-aggregated drug-level protein tensors:
  python ablation_track1_ml.py \\
    --combo    bio-decagon-combo.csv \\
    --drug_emb_chemberta  data/cache/ablation/drug_emb_chemberta_256.pt \\
    --prot_emb_esm2       data/cache/ablation/drug_via_mean_esm2.pt \\
    --output   ./ablation_results

  With Weights & Biases (requires `pip install wandb` and `wandb login`):
  python ablation_track1_ml.py ... --wandb --wandb_project my-project --wandb_tags track1,ablation

  Colab: use VM-local paths (fast): ``--output /content/ablation_out``
  ``--dataset_cache_dir /content/ablation_cache``. Avoid ``/content/drive/MyDrive/...``
  for heavy .npz I/O unless you need persistence across runtime disconnects.

  Batched SEs (resume without max_pos_edges): SEs are ordered by decreasing
  edge count. Run batch 0: ``--n_se_sample 100 --se_offset 0``; batch 1:
  ``--n_se_sample 100 --se_offset 100 --append_se_results`` (same ``--output``).

  ``--append_se_results``: when ``ablation_results_per_se.csv`` already exists
  under ``--output``, append new rows (no duplicate header). Omit on batch 0;
  add on batch 1, 2, … so later runs do not overwrite earlier SE batches.
  ``ablation_summary.csv`` is recomputed at the end of each run from the full CSV.
"""

import argparse
import gc
import hashlib
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--combo",              required=True)
    p.add_argument("--drug_emb_chemberta", required=True,
                   help="(n_drugs, 256) tensor — ChemBERTa drug embeddings")
    p.add_argument("--drug_emb_mono",      default=None,
                   help="(n_drugs, 256) tensor — TF-IDF/CUR mono SE embeddings")
    p.add_argument("--prot_emb_esm2",      required=True,
                   help="(n_proteins, 256) tensor — ESM-2 protein embeddings")
    p.add_argument("--prot_emb_ppi",       default=None,
                   help="(n_proteins, 256) tensor — PPI mean-pool embeddings")
    p.add_argument("--entity_to_id",       default=None,
                   help=".json mapping entity string → row index in embedding tensors")
    p.add_argument("--targets",            default=None,
                   help="bio-decagon-targets.csv (STITCH, Gene). Required for conditions A, B, E.")
    p.add_argument("--prot_gene_idx",      default=None,
                   help="JSON mapping Gene string → row index in --prot_emb_esm2 / --prot_emb_ppi. "
                        "If omitted, rows are assumed 0..G-1 for Genes sorted unique in targets (fragile).")
    p.add_argument("--target_impute",      default="zero",
                   choices=["zero", "mean"],
                   help="Drugs with no resolvable targets: zero vector or global mean protein embedding.")
    p.add_argument("--pair_repr",          default="sym",
                   choices=["sym", "concat", "sum"],
                   help="Pair representation: "
                        "'sym' = [e_A * e_B || |e_A - e_B|] (order-invariant), "
                        "'concat' = [e_A || e_B] with canonical STITCH ordering, "
                        "'sum' = [e_A + e_B] (order-invariant).")
    p.add_argument("--drug_fusion",        default="hstack",
                   choices=["hstack"],
                   help="How to combine drug/protein branches per drug. "
                        "Currently only 'hstack' = torch.cat([drug_branch, prot_branch], dim=1).")
    p.add_argument("--output",             default="./ablation_results")
    p.add_argument("--seed",               type=int,  default=42)
    p.add_argument("--neg_ratio",          type=int,  default=1,
                   help="negatives per positive (default 1 = balanced)")
    p.add_argument("--test_frac",          type=float, default=0.2)
    p.add_argument("--model",              default="xgboost",
                   choices=["xgboost", "catboost", "both"])
    p.add_argument("--min_edges",          type=int,  default=500,
                   help="Lloyd's threshold — 963 SEs")
    p.add_argument("--n_se_sample",        type=int,  default=None,
                   help="If set, only use this many side effects per run. Combine with "
                        "--se_offset to resume the next chunk (SEs ordered by decreasing edge count).")
    p.add_argument("--se_offset",          type=int,  default=0,
                   help="Skip the first N side effects in the canonical frequency-sorted list "
                        "(used with --n_se_sample for batched / resume runs).")
    p.add_argument("--append_se_results", action="store_true",
                   help="If ablation_results_per_se.csv exists under --output, append rows "
                        "(no header). Use from batch 2 onward when merging --n_se_sample windows "
                        "(--se_offset 100, 200, …); omit on first batch or the file is truncated.")
    p.add_argument("--max_pos_edges",      type=int,  default=None,
                   help="Cap positive edges after filtering (random subsample, seed-controlled). "
                        "Cuts RAM ~linearly in this cap.")
    p.add_argument("--dataset_cache_dir",  default=None,
                   help="If set, save each condition's built arrays under this directory as .npz "
                        "(see --reuse_dataset_cache). Uses disk; lowers repeated-run RAM if you "
                        "resume without rebuilding.")
    p.add_argument("--reuse_dataset_cache", action="store_true",
                   help="If --dataset_cache_dir is set and a cache file exists for a condition, "
                        "load it instead of calling build_dataset (same CLI args must apply).")
    # Weights & Biases (optional)
    p.add_argument("--wandb", action="store_true",
                   help="Log metrics and config to Weights & Biases")
    p.add_argument("--wandb_project", default="tf-multimodal-track1-ablation",
                   help="W&B project name")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity (team or username); None uses default account")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (default: auto-generated)")
    p.add_argument("--wandb_tags", default=None,
                   help="Comma-separated W&B run tags")
    p.add_argument("--wandb_offline", action="store_true",
                   help="Use W&B offline mode (sync later with wandb sync)")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_combo(path: str, min_edges: int):
    df = pd.read_csv(path)
    se_counts = (df.groupby("Polypharmacy Side Effect")
                   .size().reset_index(name="edge_count"))
    valid_ses = se_counts[se_counts["edge_count"] >= min_edges][
        "Polypharmacy Side Effect"].tolist()
    df = df[df["Polypharmacy Side Effect"].isin(valid_ses)].copy()
    print(f"Loaded {len(df):,} positive pairs across {df['Polypharmacy Side Effect'].nunique()} SEs")
    return df


def load_tensor(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        t = torch.stack(list(t.values()))
    print(f"  Loaded {path}: shape {t.shape}")
    return t.float()


def make_entity_index(df: pd.DataFrame):
    """
    Build drug→row_index and protein→row_index mappings.
    Assumes embeddings are stacked: drugs first (rows 0..n_drugs-1),
    proteins next (rows n_drugs..n_drugs+n_proteins-1).
    Adjust if your entity_to_id JSON uses a different ordering.
    """
    drugs = pd.concat([df["STITCH 1"], df["STITCH 2"]]).unique()
    drug_to_idx = {d: i for i, d in enumerate(sorted(drugs))}
    return drug_to_idx


def load_targets_table(path: str) -> pd.DataFrame:
    """bio-decagon-targets.csv with columns STITCH, Gene."""
    t = pd.read_csv(path)
    if "STITCH" not in t.columns or "Gene" not in t.columns:
        raise ValueError(f"{path}: expected columns STITCH and Gene")
    t = t.copy()
    t["STITCH"] = t["STITCH"].astype(str)
    t["Gene"] = t["Gene"].astype(str)
    return t


def resolve_gene_to_row(
    n_prot_rows: int,
    targets: pd.DataFrame,
    prot_gene_idx: str | None,
) -> dict[str, int]:
    """Map Gene ID → row index into prot_emb tensor."""
    if prot_gene_idx:
        with open(prot_gene_idx) as f:
            raw = json.load(f)
        return {str(k): int(v) for k, v in raw.items()}
    genes = sorted(targets["Gene"].unique())
    if len(genes) > n_prot_rows:
        raise ValueError(
            f"{len(genes)} unique Genes in targets but protein embedding has only "
            f"{n_prot_rows} rows — pass --prot_gene_idx with Gene→row indices."
        )
    print(
        "WARNING: Using Gene order = sorted unique Genes in targets → rows 0..G-1 "
        "of prot_emb. If your ESM tensor uses a different ordering, pass --prot_gene_idx."
    )
    return {g: i for i, g in enumerate(genes)}


def compute_drug_via_targets(
    prot_emb: torch.Tensor,
    targets: pd.DataFrame,
    drug_to_idx: dict[str, int],
    gene_to_row: dict[str, int],
    impute: str,
) -> torch.Tensor:
    """
    Mean protein embedding over target genes per drug (aligned with drug_to_idx rows).
    """
    dim = prot_emb.shape[1]
    n_drugs = len(drug_to_idx)
    drug_targets: dict[str, list[str]] = defaultdict(list)
    for _, row in targets.iterrows():
        drug_targets[row["STITCH"]].append(row["Gene"])

    if impute == "zero":
        fallback = torch.zeros(dim, dtype=prot_emb.dtype, device=prot_emb.device)
    else:
        fallback = prot_emb.mean(dim=0)

    out = torch.zeros(n_drugs, dim, dtype=prot_emb.dtype, device=prot_emb.device)
    for drug, idx in drug_to_idx.items():
        genes = drug_targets.get(drug, [])
        ok = [g for g in genes if g in gene_to_row]
        if ok:
            rows = []
            for g in ok:
                r = gene_to_row[g]
                if r < 0 or r >= prot_emb.shape[0]:
                    raise IndexError(
                        f"Gene {g} maps to row {r}; prot_emb has shape {tuple(prot_emb.shape)}"
                    )
                rows.append(r)
            out[idx] = prot_emb[rows].mean(dim=0)
        else:
            out[idx] = fallback
    return out


def build_fused_drug_embedding_matrix(
    condition: str,
    n_drugs: int,
    drug_dim: int,
    prot_dim: int,
    drug_fusion: str,
    seed: int,
    chemberta: torch.Tensor | None,
    mono: torch.Tensor | None,
    via_esm: torch.Tensor | None,
    via_ppi: torch.Tensor | None,
) -> torch.Tensor:
    """
    Per-drug vector for pair features: ``torch.cat([drug_branch, prot_branch], dim=1)``.

    Shapes: ``(n_drugs, drug_dim + prot_dim)``. Drug width ``drug_dim`` comes from
    ChemBERTa (and mono for E); protein width ``prot_dim`` from ``--prot_emb_esm2``.
    """
    if drug_fusion != "hstack":
        raise ValueError(f"Unsupported --drug_fusion {drug_fusion!r}; use 'hstack'.")

    drug_branch = build_drug_embeddings(
        condition, n_drugs, chemberta, mono, drug_dim, seed)

    def _hstack(drug_b: torch.Tensor, prot_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([drug_b, prot_b], dim=1)

    if condition == "A":
        if via_esm is None:
            raise ValueError("Condition A requires --targets and ESM protein embeddings.")
        return _hstack(drug_branch, via_esm)

    if condition == "B":
        if via_esm is None:
            raise ValueError("Condition B requires --targets and ESM protein embeddings.")
        return _hstack(drug_branch, via_esm)

    if condition == "C":
        return _hstack(drug_branch, zero_tensor(n_drugs, prot_dim))

    if condition == "D":
        prot_slot = xavier_tensor(n_drugs, prot_dim, seed + 1)
        return _hstack(drug_branch, prot_slot)

    if condition == "E":
        if via_esm is None or via_ppi is None:
            raise ValueError(
                "Condition E requires --targets, --prot_emb_esm2, and --prot_emb_ppi."
            )
        prot_branch = (via_esm + via_ppi) / 2.0
        return _hstack(drug_branch, prot_branch)

    raise ValueError(f"Unknown condition: {condition}")

def zero_tensor(n_entities: int, dim: int) -> torch.Tensor:
    return torch.zeros(n_entities, dim)


def xavier_tensor(n_entities: int, dim: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    t = torch.empty(n_entities, dim)
    torch.nn.init.xavier_uniform_(t)
    return t


def build_drug_embeddings(condition: str,
                          n_drugs: int,
                          chemberta: torch.Tensor | None,
                          mono: torch.Tensor | None,
                          dim: int,
                          seed: int) -> torch.Tensor:
    """
    Return (n_drugs, dim) drug embedding matrix for a given condition.

    Condition controls the drug side of the embedding:
      A, C, E  →  ChemBERTa (+ mono for E)
      B        →  zeros
      D        →  Xavier random
    """
    if condition in ("A", "C"):
        assert chemberta is not None, "ChemBERTa embeddings required for condition A/C"
        return chemberta[:n_drugs]
    elif condition == "E":
        assert chemberta is not None and mono is not None, \
            "Both ChemBERTa and mono embeddings required for condition E"
        # Simple mean fusion — same dim assumed
        return (chemberta[:n_drugs] + mono[:n_drugs]) / 2.0
    elif condition == "B":
        return zero_tensor(n_drugs, dim)
    elif condition == "D":
        return xavier_tensor(n_drugs, dim, seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def build_protein_embeddings(condition: str,
                             n_proteins: int,
                             esm2: torch.Tensor | None,
                             ppi: torch.Tensor | None,
                             dim: int,
                             seed: int) -> torch.Tensor:
    """
    Legacy: per-residue/gene protein stacks — not used once fused drug matrices
    (build_fused_drug_embedding_matrix + compute_drug_via_targets) are enabled.
    """
    if condition in ("A", "B"):
        assert esm2 is not None, "ESM-2 embeddings required for condition A/B"
        return esm2[:n_proteins]
    elif condition == "E":
        assert esm2 is not None and ppi is not None, \
            "Both ESM-2 and PPI embeddings required for condition E"
        return (esm2[:n_proteins] + ppi[:n_proteins]) / 2.0
    elif condition == "C":
        return zero_tensor(n_proteins, dim)
    elif condition == "D":
        return xavier_tensor(n_proteins, dim, seed + 1)  # different seed from drugs
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ── Feature construction ───────────────────────────────────────────────────────

def pair_features(e_a: np.ndarray, e_b: np.ndarray, pair_repr: str = "sym") -> np.ndarray:
    """
    Construct symmetric feature vector for a drug pair.

    Uses elementwise product + absolute difference:
      [e_A * e_B  ||  |e_A - e_B|]

    This is:
      - Symmetric (order-invariant, correct for undirected PSE pairs)
      - Equivalent to the DistMult scoring signal (product term)
      - Sensitive to dissimilarity (diff term)
      - 2×dim features total
    """
    if pair_repr == "sym":
        return np.concatenate([e_a * e_b, np.abs(e_a - e_b)], axis=-1)
    if pair_repr == "concat":
        return np.concatenate([e_a, e_b], axis=-1)
    if pair_repr == "sum":
        return e_a + e_b
    raise ValueError(f"Unknown pair_repr: {pair_repr}")


def build_dataset(df: pd.DataFrame,
                  drug_emb: torch.Tensor,
                  drug_to_idx: dict,
                  neg_ratio: int,
                  seed: int,
                  se_sample: list | None = None,
                  max_pos_edges: int | None = None,
                  pair_repr: str = "sym"):
    """
    Build (X, y, se_labels) arrays for binary classification.

    Positives: real drug pairs from combo CSV.
    Negatives: corrupt one drug per positive (standard KGE negative sampling).

    Memory: one preallocated float32 matrix (no Python lists of row arrays).

    Returns
    -------
    X         : (n_samples, 2*dim) float array
    y         : (n_samples,) binary labels
    se_labels : (n_samples,) SE CUI code for each sample (for per-SE eval)
    degrees   : (n_samples,) degree product for bias analysis
    """
    rng = np.random.RandomState(seed)
    all_drug_ids = list(drug_to_idx.keys())
    n_drugs = len(all_drug_ids)

    emb = np.ascontiguousarray(
        drug_emb.detach().cpu().numpy(), dtype=np.float32)
    dim = emb.shape[1]
    if pair_repr not in {"sym", "concat", "sum"}:
        raise ValueError(f"Unknown pair_repr: {pair_repr}")
    feat_dim = dim if pair_repr == "sum" else 2 * dim

    # Degree product (bias metric): counts over the FULL combo passed in, before
    # --n_se_sample filtering. Subsetting SEs only would omit drugs that appear
    # in other SEs → KeyError when negatives sample those drugs as corrupt endpoints.
    stacked_all = pd.concat([df["STITCH 1"], df["STITCH 2"]], ignore_index=True)
    degree = stacked_all.value_counts().to_dict()

    def deg_prod(x, y) -> float:
        return float(degree.get(x, 1) * degree.get(y, 1))

    def write_concat_pair(row_idx: int,
                          left_id: str,
                          left_vec: np.ndarray,
                          right_id: str,
                          right_vec: np.ndarray) -> None:
        """
        Canonicalize undirected pairs for concat: lower STITCH id goes first.
        Prevents direction leakage from arbitrary input order.
        """
        if left_id <= right_id:
            X[row_idx, :dim] = left_vec
            X[row_idx, dim:] = right_vec
        else:
            X[row_idx, :dim] = right_vec
            X[row_idx, dim:] = left_vec

    if se_sample is not None:
        df = df[df["Polypharmacy Side Effect"].isin(se_sample)]

    mask = df["STITCH 1"].isin(drug_to_idx) & df["STITCH 2"].isin(drug_to_idx)
    df = df.loc[mask]
    if len(df) == 0:
        return (
            np.empty((0, feat_dim), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=object),
            np.empty(0, dtype=np.float32),
        )

    if max_pos_edges is not None and len(df) > max_pos_edges:
        df = df.sample(n=max_pos_edges, random_state=seed)
        print(f"  Subsampled positives to --max_pos_edges={max_pos_edges:,}")

    n_pos = len(df)
    n_samples = n_pos * (1 + neg_ratio)

    s1 = df["STITCH 1"].to_numpy()
    s2 = df["STITCH 2"].to_numpy()
    se_col = df["Polypharmacy Side Effect"].to_numpy()

    X = np.empty((n_samples, feat_dim), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int32)
    se_labels = np.empty(n_samples, dtype=object)
    degrees = np.empty(n_samples, dtype=np.float32)

    row = 0
    for i in range(n_pos):
        a, b, se = s1[i], s2[i], se_col[i]
        ia, ib = drug_to_idx[a], drug_to_idx[b]
        ea, eb = emb[ia], emb[ib]
        if pair_repr == "sym":
            X[row, :dim] = ea * eb
            X[row, dim:] = np.abs(ea - eb)
        elif pair_repr == "concat":
            write_concat_pair(row, a, ea, b, eb)
        else:  # pair_repr == "sum"
            X[row, :] = ea + eb
        y[row] = 1
        se_labels[row] = se
        degrees[row] = np.float32(deg_prod(a, b))
        row += 1

        for _ in range(neg_ratio):
            if rng.rand() < 0.5:
                neg_drug = all_drug_ids[rng.randint(n_drugs)]
                ic = drug_to_idx[neg_drug]
                ec = emb[ic]
                if pair_repr == "sym":
                    X[row, :dim] = ec * eb
                    X[row, dim:] = np.abs(ec - eb)
                elif pair_repr == "concat":
                    write_concat_pair(row, neg_drug, ec, b, eb)
                else:  # pair_repr == "sum"
                    X[row, :] = ec + eb
                degrees[row] = np.float32(deg_prod(neg_drug, b))
            else:
                neg_drug = all_drug_ids[rng.randint(n_drugs)]
                ic = drug_to_idx[neg_drug]
                ec = emb[ic]
                if pair_repr == "sym":
                    X[row, :dim] = ea * ec
                    X[row, dim:] = np.abs(ea - ec)
                elif pair_repr == "concat":
                    write_concat_pair(row, a, ea, neg_drug, ec)
                else:  # pair_repr == "sum"
                    X[row, :] = ea + ec
                degrees[row] = np.float32(deg_prod(a, neg_drug))
            y[row] = 0
            se_labels[row] = se
            row += 1

    assert row == n_samples
    return X, y, se_labels, degrees


def _dataset_cache_signature(args: argparse.Namespace,
                             se_sample: list | None) -> str:
    meta = {
        "combo": Path(args.combo).name,
        "seed": args.seed,
        "neg_ratio": args.neg_ratio,
        "max_pos_edges": args.max_pos_edges,
        "min_edges": args.min_edges,
        "se_offset": args.se_offset,
        "n_se_sample": sorted(se_sample) if se_sample else None,
        "targets": Path(args.targets).name if args.targets else None,
        "target_impute": args.target_impute,
        "prot_gene_idx": Path(args.prot_gene_idx).name if args.prot_gene_idx else None,
        "pair_repr": args.pair_repr,
        "drug_fusion": args.drug_fusion,
        "fusion": "hstack_drug_prot",
    }
    blob = json.dumps(meta, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def save_dataset_npz(path: Path, X, y, se_labels, degrees,
                     meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        se_labels=se_labels,
        degrees=degrees,
        meta_json=json.dumps(meta),
    )


def load_dataset_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    X = np.ascontiguousarray(z["X"], dtype=np.float32)
    y = np.ascontiguousarray(z["y"])
    se_labels = z["se_labels"]
    degrees = np.ascontiguousarray(z["degrees"], dtype=np.float32)
    z.close()
    return X, y, se_labels, degrees


def append_per_se_results(path: Path, df: pd.DataFrame, is_first: bool) -> bool:
    """Write or append rows to ablation_results_per_se.csv. is_first: first write this run."""
    df.to_csv(path, mode="w" if is_first else "a", index=False, header=is_first)
    return False


# ── Evaluation ────────────────────────────────────────────────────────────────

def assign_tiers(se_counts: pd.Series, n_tiers: int = 5) -> dict:
    """Assign frequency tier label to each SE CUI."""
    sorted_ses = se_counts.sort_values()
    tier_size = len(sorted_ses) // n_tiers
    tier_map = {}
    labels = [f"T{n_tiers}_rare", f"T{n_tiers-1}", "T3", "T2", "T1_frequent"]
    for i, (se, _) in enumerate(sorted_ses.items()):
        tier_idx = min(i // tier_size, n_tiers - 1)
        tier_map[se] = labels[tier_idx]
    return tier_map


def evaluate(y_true: np.ndarray,
             y_score: np.ndarray,
             se_labels: np.ndarray,
             degrees: np.ndarray,
             tier_map: dict,
             condition: str) -> pd.DataFrame:
    """
    Compute per-SE and per-tier AUROC, AUPRC, and degree-score correlation.

    Degree-score correlation measures bias:
      High correlation → model is scoring pairs by popularity, not biology
      Low correlation  → model is using embedding content, not graph structure
    """
    results = []
    unique_ses = np.unique(se_labels)

    # Overall degree-score correlation (bias metric)
    pos_mask = y_true == 1
    deg_corr = np.corrcoef(degrees[pos_mask],
                           y_score[pos_mask])[0, 1]

    for se in unique_ses:
        mask = se_labels == se
        yt = y_true[mask]
        ys = y_score[mask]

        if yt.sum() == 0 or yt.sum() == mask.sum():
            continue  # skip degenerate SE

        auroc = roc_auc_score(yt, ys)
        auprc = average_precision_score(yt, ys)

        # AP@50: precision at top 50 positive predictions
        top50_idx = np.argsort(ys)[::-1][:50]
        ap50 = yt[top50_idx].sum() / 50.0

        results.append({
            "condition": condition,
            "se": se,
            "tier": tier_map.get(se, "unknown"),
            "auroc": auroc,
            "auprc": auprc,
            "ap50": ap50,
            "n_pos": int(yt.sum()),
            "n_total": int(mask.sum()),
            "degree_score_corr": deg_corr,
        })

    return pd.DataFrame(results)


# ── Model training ────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_train, y_train, verbose=False)
    return clf, clf.predict_proba(X_test)[:, 1]


def train_catboost(X_train, y_train, X_test, seed: int):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("Install catboost: pip install catboost")

    clf = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        random_seed=seed,
        verbose=0,
        task_type="CPU",
    )
    clf.fit(X_train, y_train)
    return clf, clf.predict_proba(X_test)[:, 1]


def train_model(model_name: str, X_train, y_train, X_test, seed: int):
    if model_name == "xgboost":
        return train_xgboost(X_train, y_train, X_test, seed)
    elif model_name == "catboost":
        return train_catboost(X_train, y_train, X_test, seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Feature importance ────────────────────────────────────────────────────────

def save_feature_importance(clf, condition: str, dim: int, output_dir: Path,
                             model_name: str, pair_repr: str):
    """
    Save feature importance for a trained XGBoost/CatBoost model.

    Aggregate importance by pair representation:
      sym    -> product vs absolute_difference
      concat -> first_drug_slot vs second_drug_slot
      sum    -> summed_pair_features
    """
    fi_dir = output_dir / "feature_importance"
    fi_dir.mkdir(exist_ok=True)

    try:
        if model_name == "xgboost":
            imp = clf.feature_importances_
        else:
            imp = clf.get_feature_importance()

        groups = []
        if pair_repr == "sym":
            groups = [
                ("elementwise_product (e_A * e_B)", imp[:dim].sum()),
                ("absolute_difference (|e_A - e_B|)", imp[dim:].sum()),
            ]
        elif pair_repr == "concat":
            groups = [
                ("first_drug_slot", imp[:dim].sum()),
                ("second_drug_slot", imp[dim:].sum()),
            ]
        elif pair_repr == "sum":
            groups = [("summed_pair_features (e_A + e_B)", imp.sum())]
        else:
            groups = [("all_features", imp.sum())]

        total = sum(v for _, v in groups)
        if total <= 0:
            total = 1.0
        summary = pd.DataFrame({
            "feature_group": [g for g, _ in groups],
            "total_importance": [v for _, v in groups],
            "fraction": [v / total for _, v in groups],
        })
        summary.to_csv(fi_dir / f"importance_{condition}_{model_name}.csv", index=False)
        print(f"  Feature importance saved for condition {condition}")
    except Exception as e:
        print(f"  Could not save feature importance: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Track 1 Ablation — tree-based classifiers")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_combo(args.combo, args.min_edges)

    print("\nLoading embeddings...")
    chemberta = load_tensor(args.drug_emb_chemberta)
    mono      = load_tensor(args.drug_emb_mono)
    esm2      = load_tensor(args.prot_emb_esm2)
    ppi       = load_tensor(args.prot_emb_ppi)

    drug_dim = chemberta.shape[1]
    prot_dim = esm2.shape[1]
    fused_dim = drug_dim + prot_dim
    print(f"Drug branch dim: {drug_dim}, protein branch dim: {prot_dim}, fused: {fused_dim}")

    drug_to_idx = make_entity_index(df)
    n_drugs     = len(drug_to_idx)
    n_proteins  = esm2.shape[0] if esm2 is not None else 0

    # SE frequency tiers for stratified reporting
    se_counts = df.groupby("Polypharmacy Side Effect").size()
    tier_map  = assign_tiers(se_counts)

    # Side-effect subsampling: canonical order = decreasing edge count (stable resume batches)
    ordered_se = se_counts.sort_values(ascending=False).index.tolist()
    n_se_total = len(ordered_se)
    se_sample = None
    if args.n_se_sample is not None:
        start = max(0, args.se_offset)
        end = min(start + args.n_se_sample, n_se_total)
        if start >= n_se_total:
            raise RuntimeError(
                f"--se_offset={start} is past the {n_se_total} side effects "
                f"(after --min_edges filter)."
            )
        se_sample = ordered_se[start:end]
        if len(se_sample) == 0:
            raise RuntimeError(
                "Empty SE slice; increase --n_se_sample or lower --se_offset."
            )
        print(
            f"SE slice [{start}:{end}] of {n_se_total} (by decreasing edge count): "
            f"{len(se_sample)} side effects"
        )
    elif args.se_offset != 0:
        raise ValueError("--se_offset is only meaningful together with --n_se_sample")

    # Determine which conditions to run
    conditions = ["A", "B", "C", "D"]
    if mono is not None and ppi is not None:
        conditions.append("E")
    else:
        print("Skipping condition E (mono or PPI embeddings not provided)")

    models_to_run = (["xgboost", "catboost"] if args.model == "both"
                     else [args.model])

    need_targets = bool(set(conditions) & {"A", "B", "E"})
    via_esm = None
    via_ppi = None
    targets_df = None

    if need_targets:
        # Support both:
        #  1) pre-aggregated drug-level tensors: (n_drugs, D)
        #  2) protein-level tensors:            (n_proteins, D) + --targets mapping
        if esm2.shape[0] == n_drugs:
            via_esm = esm2[:n_drugs]
            print(f"  Using pre-aggregated drug-level --prot_emb_esm2: {tuple(via_esm.shape)}")
        else:
            if not args.targets:
                raise ValueError(
                    "--prot_emb_esm2 appears protein-level (rows != n_drugs). "
                    "Provide --targets to aggregate protein→drug, or pass a pre-aggregated "
                    "drug-level tensor."
                )
            targets_df = load_targets_table(args.targets)
            gene_to_row = resolve_gene_to_row(esm2.shape[0], targets_df, args.prot_gene_idx)
            via_esm = compute_drug_via_targets(
                esm2, targets_df, drug_to_idx, gene_to_row, args.target_impute
            )
            print(f"  drug-via-ESM (mean over targets): {tuple(via_esm.shape)}")

        if "E" in conditions:
            if ppi is None:
                raise ValueError("Condition E requires --prot_emb_ppi.")
            if ppi.shape[0] == n_drugs:
                via_ppi = ppi[:n_drugs]
                print(f"  Using pre-aggregated drug-level --prot_emb_ppi: {tuple(via_ppi.shape)}")
            else:
                if targets_df is None:
                    if not args.targets:
                        raise ValueError(
                            "--prot_emb_ppi appears protein-level (rows != n_drugs). "
                            "Provide --targets to aggregate protein→drug, or pass a "
                            "pre-aggregated drug-level tensor."
                        )
                    targets_df = load_targets_table(args.targets)
                gene_to_row_ppi = resolve_gene_to_row(
                    ppi.shape[0], targets_df, args.prot_gene_idx
                )
                via_ppi = compute_drug_via_targets(
                    ppi, targets_df, drug_to_idx, gene_to_row_ppi, args.target_impute
                )
                print(f"  drug-via-PPI (mean over targets): {tuple(via_ppi.shape)}")

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "Weights & Biases requires the wandb package: pip install wandb"
            ) from e
        tags = None
        if args.wandb_tags:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            if not tags:
                tags = None
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=tags,
            mode="offline" if args.wandb_offline else "online",
            config={
                "combo": Path(args.combo).name,
                "output_dir": str(output_dir.resolve()),
                "seed": args.seed,
                "neg_ratio": args.neg_ratio,
                "test_frac": args.test_frac,
                "model_policy": args.model,
                "min_edges": args.min_edges,
                "n_se_sample": args.n_se_sample,
                "se_offset": args.se_offset,
                "append_se_results": args.append_se_results,
                "max_pos_edges": args.max_pos_edges,
                "dataset_cache_dir": args.dataset_cache_dir,
                "reuse_dataset_cache": args.reuse_dataset_cache,
                "targets": args.targets,
                "target_impute": args.target_impute,
                "pair_repr": args.pair_repr,
                "drug_fusion": args.drug_fusion,
                "embedding_dim": fused_dim,
                "drug_dim": drug_dim,
                "prot_dim": prot_dim,
                "n_drugs": n_drugs,
                "n_proteins": n_proteins,
                "conditions": conditions,
                "models_to_run": models_to_run,
            },
        )

    cache_sig = _dataset_cache_signature(args, se_sample)
    results_per_se_path = output_dir / "ablation_results_per_se.csv"
    if args.append_se_results and results_per_se_path.is_file():
        per_se_first = False
        print(f"Appending per-SE results to existing file: {results_per_se_path}")
    else:
        per_se_first = True
    wandb_step = 0

    for condition in conditions:
        print(f"\n{'─'*40}")
        print(f"Condition {condition}")

        cache_npz = None
        if args.dataset_cache_dir:
            cache_npz = Path(args.dataset_cache_dir).resolve() / f"{condition}_{cache_sig}.npz"

        if (
            args.dataset_cache_dir
            and args.reuse_dataset_cache
            and cache_npz is not None
            and cache_npz.is_file()
        ):
            X, y, se_labels, degrees = load_dataset_npz(cache_npz)
            print(f"  Dataset: loaded cache {cache_npz.name} (skipping drug_emb + build_dataset)")
        else:
            drug_emb = build_fused_drug_embedding_matrix(
                condition, n_drugs, drug_dim, prot_dim, args.drug_fusion, args.seed,
                chemberta, mono, via_esm, via_ppi,
            )

            print(f"  Fused per-drug embedding ({args.drug_fusion}): {drug_emb.shape}")

            print(f"  Building dataset...")
            X, y, se_labels, degrees = build_dataset(
                df, drug_emb, drug_to_idx,
                neg_ratio=args.neg_ratio,
                seed=args.seed,
                se_sample=se_sample,
                max_pos_edges=args.max_pos_edges,
                pair_repr=args.pair_repr,
            )
            del drug_emb
            gc.collect()

            if args.dataset_cache_dir and cache_npz is not None:
                meta = {
                    "condition": condition,
                    "signature": cache_sig,
                    "combo": Path(args.combo).name,
                    "seed": args.seed,
                    "neg_ratio": args.neg_ratio,
                    "max_pos_edges": args.max_pos_edges,
                }
                save_dataset_npz(cache_npz, X, y, se_labels, degrees, meta)
                print(f"  Wrote dataset cache → {cache_npz}")

        if len(y) == 0:
            raise RuntimeError(
                "No samples after dataset load/build; check --combo, --min_edges, cache, "
                "and entity coverage."
            )

        print(f"  Samples: {len(y):,} ({y.sum():,} pos, {(1-y).sum():,} neg)")

        idx = np.arange(len(y), dtype=np.int64)
        train_idx, test_idx = train_test_split(
            idx,
            test_size=args.test_frac,
            stratify=y,
            random_state=args.seed,
        )
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        se_train = se_labels[train_idx]
        se_test = se_labels[test_idx]
        deg_train = degrees[train_idx]
        deg_test = degrees[test_idx]
        del X
        gc.collect()

        for model_name in models_to_run:
            print(f"  Training {model_name}...")
            clf, y_score = train_model(model_name, X_train, y_train, X_test, args.seed)

            results_df = evaluate(
                y_test, y_score, se_test, deg_test, tier_map,
                condition=f"{condition}_{model_name}"
            )
            per_se_first = append_per_se_results(
                results_per_se_path, results_df, per_se_first)

            save_feature_importance(
                clf, condition, fused_dim, output_dir, model_name, args.pair_repr
            )

            overall_auroc = roc_auc_score(y_test, y_score)
            overall_auprc = average_precision_score(y_test, y_score)

            print(f"  {model_name} — overall AUROC: "
                  f"{overall_auroc:.4f}  "
                  f"AUPRC: {overall_auprc:.4f}")

            if wandb_run is not None:
                import wandb

                log_payload = {
                    "overall/auroc": overall_auroc,
                    "overall/auprc": overall_auprc,
                    "per_se/median_auroc": float(results_df["auroc"].median()),
                    "per_se/median_auprc": float(results_df["auprc"].median()),
                    "bias/degree_score_corr": float(
                        results_df["degree_score_corr"].iloc[0]),
                    "samples/n_test": int(len(y_test)),
                    "samples/n_pos_test": int(y_test.sum()),
                    "meta/condition": condition,
                    "meta/model": model_name,
                }
                for tier, val in results_df.groupby("tier")["auprc"].median().items():
                    log_payload[f"tier/median_auprc/{tier}"] = float(val)

                wandb.log(log_payload, step=wandb_step)
                wandb_step += 1

            del clf, y_score
            gc.collect()

        del X_train, X_test, y_train, y_test, se_train, se_test, deg_train, deg_test
        gc.collect()

    # ── Aggregate results ──────────────────────────────────────────────────
    all_df = pd.read_csv(results_per_se_path)

    # Summary: median per condition per tier
    summary = (all_df.groupby(["condition", "tier"])
                     .agg(
                         median_auroc=("auroc", "median"),
                         median_auprc=("auprc", "median"),
                         median_ap50=("ap50", "median"),
                         median_deg_corr=("degree_score_corr", "median"),
                         n_ses=("se", "count"),
                     )
                     .reset_index())
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("SUMMARY — median AUPRC by condition and tier")
    print("=" * 60)
    pivot = summary.pivot_table(
        index="tier", columns="condition", values="median_auprc"
    ).round(4)
    print(pivot.to_string())

    print("\nDegree-score correlation (bias metric — lower is better):")
    bias = (all_df.groupby("condition")["degree_score_corr"]
                  .median().round(4).reset_index())
    print(bias.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")

    if wandb_run is not None:
        import wandb

        wandb.log({
            "tables/ablation_summary": wandb.Table(dataframe=summary),
            "tables/bias_by_condition": wandb.Table(dataframe=bias),
        })
        for _, row in summary.iterrows():
            key = f"final/median_auprc/{row['condition']}/{row['tier']}"
            wandb.summary[key] = float(row["median_auprc"])
        wandb.finish()


if __name__ == "__main__":
    main()
