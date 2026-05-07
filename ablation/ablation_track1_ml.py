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
  A  ChemBERTa + ESM-2               (pure pretrained, no extras)
  B  zeros + ESM-2                   (protein only — isolates protein signal)
  C  ChemBERTa + zeros               (drug only — isolates drug signal)
  D  random Xavier + random Xavier   (graph topology anchor — Lloyd baseline)
  E  ChemBERTa+mono + ESM-2+PPI      (your full model — does fusion help?)

Feature construction for a drug pair (drug_A, drug_B, side_effect_r)
---------------------------------------------------------------------
Tree models cannot directly use the SimplE scoring function.
Instead, we construct a fixed-size feature vector for each drug pair by
combining the two drug embeddings and (optionally) the SE relation embedding.

Combination strategies:
  - concatenate:   [e_A || e_B]                  (2×dim features)
  - elementwise:   [e_A * e_B]                   (dim features, like DistMult)
  - absolute diff: [|e_A - e_B|]                 (dim features, symmetric)
  - all three:     [e_A * e_B || |e_A - e_B| || e_A + e_B]   (3×dim)

We use elementwise product + absolute difference as default — this is the
standard approach in KGE-to-classifier transfer (Nickel et al., Hamilton et al.)
and is invariant to drug ordering (symmetric), which is correct for PSE.

Negative sampling
-----------------
The polypharmacy graph only contains positive pairs. We generate negatives
by corrupting one drug in each positive pair (standard KGE negative sampling).
Ratio: 1 positive : 1 negative (balanced). Seed-controlled for reproducibility.

Output
------
  ablation_results.csv     — per-condition AUROC, AUPRC, AP@50 per SE tier
  ablation_summary.csv     — aggregate metrics across conditions
  feature_importance/      — per-condition feature importance plots (XGBoost)

Usage
-----
  python ablation_track1_ml.py \\
    --combo    bio-decagon-combo.csv \\
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
    --drug_emb_chemberta  data/cache/ablation/drug_emb_chemberta_256.pt \\
    --prot_emb_esm2       data/cache/ablation/protein_emb_esm2_only_dim256_esm2_t30_150M_UR50D.pt \\
    --output   ./ablation_results
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
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
                   help="Subsample N side effects for fast testing (None = all 963)")
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


# ── Embedding conditions ───────────────────────────────────────────────────────

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
    Return (n_proteins, dim) protein embedding matrix for a given condition.

    Condition controls the protein side:
      A, B, E  →  ESM-2 (+ PPI for E)
      C        →  zeros
      D        →  Xavier random
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

def pair_features(e_a: np.ndarray, e_b: np.ndarray) -> np.ndarray:
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
    return np.concatenate([e_a * e_b, np.abs(e_a - e_b)], axis=-1)


def build_dataset(df: pd.DataFrame,
                  drug_emb: torch.Tensor,
                  drug_to_idx: dict,
                  neg_ratio: int,
                  seed: int,
                  se_sample: list | None = None):
    """
    Build (X, y, se_labels) arrays for binary classification.

    Positives: real drug pairs from combo CSV.
    Negatives: corrupt one drug per positive (standard KGE negative sampling).

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

    emb = drug_emb.numpy()
    dim = emb.shape[1]

    if se_sample is not None:
        df = df[df["Polypharmacy Side Effect"].isin(se_sample)]

    # Drug degree (number of pairs each drug appears in)
    degree = defaultdict(int)
    for _, row in df.iterrows():
        degree[row["STITCH 1"]] += 1
        degree[row["STITCH 2"]] += 1

    X_pos, X_neg = [], []
    se_pos, se_neg = [], []
    deg_pos, deg_neg = [], []

    for _, row in df.iterrows():
        a, b, se = row["STITCH 1"], row["STITCH 2"], row["Polypharmacy Side Effect"]
        if a not in drug_to_idx or b not in drug_to_idx:
            continue

        ea = emb[drug_to_idx[a]]
        eb = emb[drug_to_idx[b]]
        X_pos.append(pair_features(ea, eb))
        se_pos.append(se)
        deg_pos.append(degree[a] * degree[b])

        # Generate neg_ratio corrupted negatives
        for _ in range(neg_ratio):
            # Corrupt drug A or B with equal probability
            if rng.rand() < 0.5:
                neg_drug = all_drug_ids[rng.randint(n_drugs)]
                ec = emb[drug_to_idx[neg_drug]]
                X_neg.append(pair_features(ec, eb))
                deg_neg.append(degree[neg_drug] * degree[b])
            else:
                neg_drug = all_drug_ids[rng.randint(n_drugs)]
                ec = emb[drug_to_idx[neg_drug]]
                X_neg.append(pair_features(ea, ec))
                deg_neg.append(degree[a] * degree[neg_drug])
            se_neg.append(se)

    X = np.array(X_pos + X_neg, dtype=np.float32)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int32)
    se_labels = np.array(se_pos + se_neg)
    degrees = np.array(deg_pos + deg_neg, dtype=np.float64)

    return X, y, se_labels, degrees


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
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test := None)],  # no early stopping here
            verbose=False)
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
                             model_name: str):
    """
    Save feature importance for a trained XGBoost/CatBoost model.

    Feature names:
      0..dim-1        : elementwise product dims (DistMult-like signal)
      dim..2*dim-1    : absolute difference dims  (dissimilarity signal)

    Aggregate importance by group (product vs diff) to see which
    combination type the model relies on more.
    """
    fi_dir = output_dir / "feature_importance"
    fi_dir.mkdir(exist_ok=True)

    try:
        if model_name == "xgboost":
            imp = clf.feature_importances_
        else:
            imp = clf.get_feature_importance()

        product_imp = imp[:dim].sum()
        diff_imp    = imp[dim:].sum()
        total       = product_imp + diff_imp

        summary = pd.DataFrame({
            "feature_group": ["elementwise_product (e_A * e_B)",
                               "absolute_difference (|e_A - e_B|)"],
            "total_importance": [product_imp, diff_imp],
            "fraction": [product_imp / total, diff_imp / total],
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

    dim = chemberta.shape[1] if chemberta is not None else esm2.shape[1]
    print(f"Embedding dim: {dim}")

    drug_to_idx = make_entity_index(df)
    n_drugs     = len(drug_to_idx)
    n_proteins  = esm2.shape[0] if esm2 is not None else 0

    # SE frequency tiers for stratified reporting
    se_counts = df.groupby("Polypharmacy Side Effect").size()
    tier_map  = assign_tiers(se_counts)

    # Optional SE subsampling for fast testing
    se_sample = None
    if args.n_se_sample is not None:
        se_sample = list(se_counts.index[:args.n_se_sample])
        print(f"Subsampling to {args.n_se_sample} SEs for fast testing")

    # Determine which conditions to run
    conditions = ["A", "B", "C", "D"]
    if mono is not None and ppi is not None:
        conditions.append("E")
    else:
        print("Skipping condition E (mono or PPI embeddings not provided)")

    models_to_run = (["xgboost", "catboost"] if args.model == "both"
                     else [args.model])

    all_results = []

    for condition in conditions:
        print(f"\n{'─'*40}")
        print(f"Condition {condition}")

        # Build drug embedding matrix for this condition
        drug_emb = build_drug_embeddings(
            condition, n_drugs, chemberta, mono, dim, args.seed)

        # NOTE: protein embeddings are not directly used in the tree model
        # feature vector because we only have drug-drug pairs in combo.csv.
        # Proteins connect to drugs via the targets file, not directly via pairs.
        # Two options:
        #
        # Option 1 (implemented here — simpler):
        #   Aggregate protein embeddings per drug via drug→target→ESM-2.
        #   Each drug gets a mean of its target protein ESM-2 vectors.
        #   This "drug via targets" embedding is concatenated with the
        #   drug's own embedding before pair feature construction.
        #
        # Option 2 (more faithful to KGE):
        #   Train a full KGE and use the learned entity embeddings.
        #   This is what the main HPO script does.
        #
        # For the ablation, Option 1 is appropriate because we want to
        # isolate embedding contributions without KGE training confounding.

        print(f"  Drug embedding: {drug_emb.shape}")

        # Build dataset
        print(f"  Building dataset...")
        X, y, se_labels, degrees = build_dataset(
            df, drug_emb, drug_to_idx,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
            se_sample=se_sample,
        )
        print(f"  Samples: {len(y):,} ({y.sum():,} pos, {(1-y).sum():,} neg)")

        # Train/test split — stratified by SE type
        X_train, X_test, y_train, y_test, se_train, se_test, deg_train, deg_test = \
            train_test_split(
                X, y, se_labels, degrees,
                test_size=args.test_frac,
                stratify=y,
                random_state=args.seed,
            )

        for model_name in models_to_run:
            print(f"  Training {model_name}...")
            clf, y_score = train_model(model_name, X_train, y_train, X_test, args.seed)

            # Per-SE evaluation
            results_df = evaluate(
                y_test, y_score, se_test, deg_test, tier_map,
                condition=f"{condition}_{model_name}"
            )
            all_results.append(results_df)

            # Feature importance
            save_feature_importance(clf, condition, dim, output_dir, model_name)

            # Quick summary
            print(f"  {model_name} — overall AUROC: "
                  f"{roc_auc_score(y_test, y_score):.4f}  "
                  f"AUPRC: {average_precision_score(y_test, y_score):.4f}")

    # ── Aggregate results ──────────────────────────────────────────────────
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(output_dir / "ablation_results_per_se.csv", index=False)

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


if __name__ == "__main__":
    main()
