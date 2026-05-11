"""
hpo_track1_optuna.py
--------------------
Hyperparameter optimisation for the Track 1 ablation XGBoost classifier
using Optuna, evaluated on a single user-specified side effect (SE).

Why tune on one SE first?
-------------------------
Running HPO across all 963 SEs would take hours. A single SE gives a fast
proxy: find the best hyperparameters, then plug them into ablation_track1_ml.py
via --xgb_params_json for the full run. Choose a representative SE — ideally
one from T3 (median frequency, 455–802 positive pairs) so the tuned params
are not overfit to either extreme (T1 has too many pairs, T5 too few).

Evaluation strategy: stratified 5-fold cross-validation on the training split.
The held-out test set (20%) is never touched during HPO — it is only used once
at the end to report the final performance of the best model.

Optuna sampler: TPE (Tree-structured Parzen Estimator) by default.
Pruner: MedianPruner — kills unpromising trials early after the first CV fold.

Search space
------------
Parameter          Type      Range / Choices
-----------        ----      ---------------
n_estimators       int       100 – 1000  (step 50)
max_depth          int       3 – 10
learning_rate      float     0.005 – 0.3  (log scale)
subsample          float     0.5 – 1.0
colsample_bytree   float     0.4 – 1.0
min_child_weight   int       1 – 20
gamma              float     0.0 – 5.0
reg_alpha          float     1e-8 – 10.0 (log scale)  L1
reg_lambda         float     1e-8 – 10.0 (log scale)  L2
scale_pos_weight   float     0.5 – 5.0   (class imbalance; 1.0 = balanced)
neg_ratio          int       1 – 5       (negatives per positive, dataset-level)

Outputs
-------
  hpo_results/
  ├── best_params.json          best hyperparameters (drop into ablation_track1_ml.py)
  ├── optuna_trials.csv         all trial results
  ├── optuna_study.pkl          serialised Optuna study (for resuming / plotting)
  └── hpo_report.txt            human-readable summary

Usage
-----
  # Minimal (already drug-level protein tensors) — tune on a specific SE CUI
  python hpo_track1_optuna.py \\
    --combo              bio-decagon-combo.csv \\
    --drug_emb_chemberta drug_emb_chemberta_256.pt \\
    --prot_emb_esm2      drug_via_targets_256.pt \\
    --se_id              C0015230 \\
    --n_trials           100 \\
    --output             ./hpo_results

  # Full (already drug-level protein tensors) — tune all five conditions
  python hpo_track1_optuna.py \\
    --combo              bio-decagon-combo.csv \\
    --drug_emb_chemberta drug_emb_chemberta_256.pt \\
    --drug_emb_mono      drug_emb_mono_256.pt \\
    --prot_emb_esm2      drug_via_targets_256.pt \\
    --prot_emb_ppi       drug_via_targets_ppi_256.pt \\
    --se_id              C0015230 \\
    --conditions         A B C D E \\
    --n_trials           200 \\
    --cv_folds           5 \\
    --output             ./hpo_results \\
    --seed               42

  # Raw protein tensors (n_proteins x D) — script auto-aggregates to drug level
  # via --targets; pass --prot_gene_idx if row order is not sorted unique Gene IDs
  python hpo_track1_optuna.py \\
    --combo              bio-decagon-combo.csv \\
    --targets            bio-decagon-targets.csv \\
    --prot_gene_idx      gene_to_row.json \\
    --drug_emb_chemberta drug_emb_chemberta_256.pt \\
    --drug_emb_mono      drug_emb_mono_256.pt \\
    --prot_emb_esm2      protein_emb_esm2_only_dim256.pt \\
    --prot_emb_ppi       protein_emb_ppi_neighbour_dim256.pt \\
    --se_id              C0015230 \\
    --conditions         A B C D E \\
    --n_trials           200 \\
    --cv_folds           5 \\
    --output             ./hpo_results \\
    --seed               42

  # If you do not know the SE CUI, use --list_ses to print all valid SEs
  # sorted by edge count, with their tier label:
  python hpo_track1_optuna.py \\
    --combo bio-decagon-combo.csv --list_ses

  # Resume an interrupted study (Optuna reads from the .pkl):
  python hpo_track1_optuna.py ... --resume

Dependencies
------------
  pip install torch pandas numpy scikit-learn xgboost optuna
"""

import argparse
import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna HPO for Track 1 ablation XGBoost on a single SE."
    )
    # Data inputs — same interface as ablation_track1_ml.py
    p.add_argument("--combo",              required=False,
                   help="bio-decagon-combo.csv")
    p.add_argument("--drug_emb_chemberta", default=None,
                   help="(n_drugs, D) ChemBERTa drug embedding tensor")
    p.add_argument("--drug_emb_mono",      default=None,
                   help="(n_drugs, D) mono side-effect embedding tensor (optional, for condition E)")
    p.add_argument("--prot_emb_esm2",      default=None,
                   help="(n_drugs, D) drug-via-targets ESM-2 tensor (output of build_drug_protein_features.py)")
    p.add_argument("--prot_emb_ppi",       default=None,
                   help="(n_drugs, D) drug-via-targets PPI tensor (optional, for condition E)")
    p.add_argument("--targets",            default=None,
                   help="bio-decagon-targets.csv. Required when --prot_emb_esm2/--prot_emb_ppi "
                        "are protein-level tensors (n_proteins x D) and must be aggregated "
                        "to drug level.")
    p.add_argument("--prot_gene_idx",      default=None,
                   help="Optional JSON mapping Gene string -> row index in --prot_emb_esm2 / "
                        "--prot_emb_ppi when those tensors are protein-level (n_proteins x D).")
    p.add_argument("--target_impute",      default="zero",
                   choices=["zero", "mean"],
                   help="How to fill drugs without resolvable targets when converting protein-level "
                        "tensors to drug-level: zero or global mean.")

    # SE selection
    p.add_argument("--se_id",     default=None,
                   help="CUI of the side effect to tune on (e.g. C0015230). "
                        "If omitted, the script picks the SE closest to the median edge count "
                        "(i.e. a typical T3 SE).")
    p.add_argument("--list_ses",  action="store_true",
                   help="Print all valid SEs with edge counts and tier labels, then exit.")

    # Conditions
    p.add_argument("--conditions", nargs="+", default=["A", "D"],
                   choices=["A", "B", "C", "D", "E"],
                   help="Which ablation conditions to tune. Separate Optuna studies are run "
                        "per condition. Default: A D (the comparison of most interest).")

    # HPO settings
    p.add_argument("--n_trials",   type=int, default=100,
                   help="Number of Optuna trials per condition (default 100).")
    p.add_argument("--cv_folds",   type=int, default=5,
                   help="Number of CV folds on the training split (default 5).")
    p.add_argument("--timeout",    type=int, default=None,
                   help="Stop HPO after this many seconds (per condition), regardless of n_trials.")
    p.add_argument("--metric",     default="auroc",
                   choices=["auroc", "auprc"],
                   help="Metric to optimise (default auroc).")
    p.add_argument("--tune_neg_ratio", action="store_true",
                   help="Also search over neg_ratio (1–5). Rebuilds the dataset inside each trial "
                        "— slower but important if you suspect class balance matters.")

    # Dataset settings — match ablation_track1_ml.py defaults
    p.add_argument("--neg_ratio",  type=int, default=1,
                   help="Negatives per positive. Used as the fixed ratio unless --tune_neg_ratio.")
    p.add_argument("--test_frac",  type=float, default=0.2,
                   help="Fraction of SE data held out as final test set (default 0.2).")
    p.add_argument("--min_edges",  type=int, default=500,
                   help="Minimum edges to include an SE (Lloyd threshold, default 500).")
    p.add_argument("--seed",       type=int, default=42)

    # Output
    p.add_argument("--output",  default="./hpo_results",
                   help="Directory for all outputs.")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from an existing optuna_study_<condition>.pkl if present.")
    p.add_argument("--n_jobs",  type=int, default=1,
                   help="Parallel Optuna workers per study (default 1). "
                        "Values >1 require a shared storage backend (e.g. SQLite); "
                        "with default in-memory storage, keep this at 1.")

    return p.parse_args()


# ── Data loading (reused from ablation_track1_ml.py) ─────────────────────────

def load_combo(path: str, min_edges: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    se_counts = df.groupby("Polypharmacy Side Effect").size()
    valid = se_counts[se_counts >= min_edges].index
    df = df[df["Polypharmacy Side Effect"].isin(valid)].copy()
    print(f"Loaded {len(df):,} positive pairs across {df['Polypharmacy Side Effect'].nunique()} SEs "
          f"(min_edges={min_edges})")
    return df


def load_tensor(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        t = torch.stack(list(t.values()))
    return t.float()


def make_drug_to_idx(df: pd.DataFrame) -> dict:
    drugs = sorted(pd.concat([df["STITCH 1"], df["STITCH 2"]]).unique())
    return {d: i for i, d in enumerate(drugs)}


def assign_tiers(se_counts: pd.Series, n_tiers: int = 5) -> dict:
    sorted_ses = se_counts.sort_values()
    tier_size = len(sorted_ses) // n_tiers
    labels = [f"T{n_tiers}_rare", f"T{n_tiers-1}", "T3", "T2", "T1_frequent"]
    tier_map = {}
    for i, (se, _) in enumerate(sorted_ses.items()):
        tier_map[se] = labels[min(i // tier_size, n_tiers - 1)]
    return tier_map


def load_targets_table(path: str) -> pd.DataFrame:
    """Load targets table with required columns STITCH and Gene."""
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
    """Resolve Gene -> embedding-row index."""
    if prot_gene_idx:
        with open(prot_gene_idx) as f:
            raw = json.load(f)
        return {str(k): int(v) for k, v in raw.items()}

    genes = sorted(targets["Gene"].unique())
    if len(genes) > n_prot_rows:
        raise ValueError(
            f"{len(genes)} unique Genes in targets but embedding has {n_prot_rows} rows. "
            "Pass --prot_gene_idx with explicit Gene->row mapping."
        )
    print("WARNING: assuming protein embedding rows map to sorted unique Gene IDs from targets.")
    return {g: i for i, g in enumerate(genes)}


def compute_drug_via_targets(
    prot_emb: torch.Tensor,
    targets: pd.DataFrame,
    drug_to_idx: dict[str, int],
    gene_to_row: dict[str, int],
    impute: str,
) -> torch.Tensor:
    """Aggregate protein embeddings to drug-level by averaging target proteins."""
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
                        f"Gene {g} maps to row {r}; embedding shape is {tuple(prot_emb.shape)}"
                    )
                rows.append(r)
            out[idx] = prot_emb[rows].mean(dim=0)
        else:
            out[idx] = fallback
    return out


# ── Embedding construction (mirrors ablation_track1_ml.py exactly) ───────────

def xavier_tensor(n: int, d: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    t = torch.empty(n, d)
    torch.nn.init.xavier_uniform_(t)
    return t


def build_fused_embedding(
    condition: str,
    n_drugs: int,
    drug_dim: int,
    prot_dim: int,
    seed: int,
    chemberta: torch.Tensor | None,
    mono: torch.Tensor | None,
    via_esm: torch.Tensor | None,
    via_ppi: torch.Tensor | None,
) -> torch.Tensor:
    """
    Returns (n_drugs, drug_dim + prot_dim) fused embedding matrix.
    Logic is identical to build_fused_drug_embedding_matrix() in ablation_track1_ml.py.
    """
    # ── Drug branch ───────────────────────────────────────────────────────
    if condition in ("A", "C"):
        drug_branch = chemberta[:n_drugs]
    elif condition == "E":
        drug_branch = (chemberta[:n_drugs] + mono[:n_drugs]) / 2.0
    elif condition == "B":
        drug_branch = torch.zeros(n_drugs, drug_dim)
    else:  # D
        drug_branch = xavier_tensor(n_drugs, drug_dim, seed)

    # ── Protein branch ────────────────────────────────────────────────────
    if condition in ("A", "B"):
        prot_branch = via_esm
    elif condition == "E":
        prot_branch = (via_esm + via_ppi) / 2.0
    elif condition == "C":
        prot_branch = torch.zeros(n_drugs, prot_dim)
    else:  # D
        prot_branch = xavier_tensor(n_drugs, prot_dim, seed + 1)

    return torch.cat([drug_branch, prot_branch], dim=1)


# ── Dataset builder for a single SE ──────────────────────────────────────────

def build_se_dataset(
    df_se: pd.DataFrame,
    df_full: pd.DataFrame,
    emb: np.ndarray,
    drug_to_idx: dict,
    neg_ratio: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, degrees) for a single SE.

    Degree product is computed over the FULL combo (df_full), not just df_se,
    matching the behaviour in ablation_track1_ml.py build_dataset().

    Parameters
    ----------
    df_se    : rows of the combo file for the chosen SE only
    df_full  : all combo rows (for global degree computation)
    emb      : (n_drugs, dim) float32 numpy array
    drug_to_idx : drug string → row index
    neg_ratio : negatives per positive
    seed     : RNG seed

    Returns
    -------
    X        : (n_samples, 2*dim) float32
    y        : (n_samples,) int32   1=positive, 0=negative
    degrees  : (n_samples,) float32 degree product
    """
    rng = np.random.RandomState(seed)
    all_drug_ids = list(drug_to_idx.keys())
    n_drugs = len(all_drug_ids)
    dim = emb.shape[1]
    feat_dim = 2 * dim

    # Global degree (full combo)
    stacked = pd.concat([df_full["STITCH 1"], df_full["STITCH 2"]], ignore_index=True)
    degree = stacked.value_counts().to_dict()

    def deg_prod(a, b):
        return float(degree.get(a, 1) * degree.get(b, 1))

    # Filter to drugs in the embedding index
    mask = df_se["STITCH 1"].isin(drug_to_idx) & df_se["STITCH 2"].isin(drug_to_idx)
    df_se = df_se.loc[mask]
    if len(df_se) == 0:
        raise RuntimeError("No pairs remain after filtering to indexed drugs for this SE.")

    s1 = df_se["STITCH 1"].to_numpy()
    s2 = df_se["STITCH 2"].to_numpy()
    n_pos = len(df_se)
    n_samples = n_pos * (1 + neg_ratio)

    X = np.empty((n_samples, feat_dim), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int32)
    degrees = np.empty(n_samples, dtype=np.float32)

    row = 0
    for i in range(n_pos):
        a, b = s1[i], s2[i]
        ea = emb[drug_to_idx[a]]
        eb = emb[drug_to_idx[b]]
        X[row, :dim] = ea * eb
        X[row, dim:] = np.abs(ea - eb)
        y[row] = 1
        degrees[row] = deg_prod(a, b)
        row += 1

        for _ in range(neg_ratio):
            if rng.rand() < 0.5:
                neg = all_drug_ids[rng.randint(n_drugs)]
                ec = emb[drug_to_idx[neg]]
                X[row, :dim] = ec * eb
                X[row, dim:] = np.abs(ec - eb)
                degrees[row] = deg_prod(neg, b)
            else:
                neg = all_drug_ids[rng.randint(n_drugs)]
                ec = emb[drug_to_idx[neg]]
                X[row, :dim] = ea * ec
                X[row, dim:] = np.abs(ea - ec)
                degrees[row] = deg_prod(a, neg)
            y[row] = 0
            row += 1

    return X, y, degrees


# ── XGBoost with arbitrary hyperparameters ────────────────────────────────────

def train_xgboost_hpo(X_tr, y_tr, X_val, params: dict, seed: int):
    """
    Train XGBoost with the given hyperparameter dict and return (clf, y_score).
    params must contain the keys suggested by the Optuna objective below.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("pip install xgboost")

    clf = XGBClassifier(
        n_estimators       = params["n_estimators"],
        max_depth          = params["max_depth"],
        learning_rate      = params["learning_rate"],
        subsample          = params["subsample"],
        colsample_bytree   = params["colsample_bytree"],
        min_child_weight   = params["min_child_weight"],
        gamma              = params["gamma"],
        reg_alpha          = params["reg_alpha"],
        reg_lambda         = params["reg_lambda"],
        scale_pos_weight   = params["scale_pos_weight"],
        use_label_encoder  = False,
        eval_metric        = "logloss",
        random_state       = seed,
        n_jobs             = -1,
        verbosity          = 0,
    )
    clf.fit(X_tr, y_tr, verbose=False)
    return clf, clf.predict_proba(X_val)[:, 1]


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    df_full: pd.DataFrame,
    emb_fn,               # callable(neg_ratio) → np.ndarray (only used if tune_neg_ratio)
    drug_to_idx: dict,
    df_se: pd.DataFrame,
    seed: int,
    cv_folds: int,
    metric: str,
    tune_neg_ratio: bool,
    base_neg_ratio: int,
    test_frac: float,
):
    """
    Returns an Optuna objective function.

    If tune_neg_ratio=True, neg_ratio is included in the search space and the
    dataset is rebuilt inside each trial. This is slower but necessary if you
    want to optimise class balance jointly with model hyperparameters.

    If tune_neg_ratio=False, X_train/y_train are built once outside the
    objective and reused across all trials (much faster).
    """
    def objective(trial):
        # ── Hyperparameter suggestions ────────────────────────────────────
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth"        : trial.suggest_int("max_depth", 3, 10),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 20),
            "gamma"            : trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        }

        # ── Optionally tune neg_ratio (rebuilds dataset) ──────────────────
        if tune_neg_ratio:
            neg_ratio = trial.suggest_int("neg_ratio", 1, 5)
            emb = emb_fn(neg_ratio)
            X, y, _ = build_se_dataset(df_se, df_full, emb, drug_to_idx, neg_ratio, seed)
            # re-split to match test_frac held out in main
            X_tr, _, y_tr, _ = train_test_split(
                X, y, test_size=test_frac, stratify=y, random_state=seed
            )
        else:
            neg_ratio = base_neg_ratio
            X_tr, y_tr = X_train, y_train

        # ── Stratified k-fold CV on the training split ────────────────────
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            X_f_tr = X_tr[tr_idx]
            y_f_tr = y_tr[tr_idx]
            X_f_val = X_tr[val_idx]
            y_f_val = y_tr[val_idx]

            _, y_score = train_xgboost_hpo(X_f_tr, y_f_tr, X_f_val, params, seed)

            if metric == "auroc":
                score = roc_auc_score(y_f_val, y_score)
            else:
                score = average_precision_score(y_f_val, y_score)

            fold_scores.append(score)

            # Pruning: report intermediate value after each fold
            # MedianPruner will kill the trial if it is below median of completed trials
            trial.report(np.mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                import optuna
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ── Final evaluation on held-out test set ────────────────────────────────────

def final_eval(X_train, y_train, X_test, y_test, best_params, seed, degrees_test):
    """Train once on full training set with best params, evaluate on test set."""
    _, y_score = train_xgboost_hpo(X_train, y_train, X_test, best_params, seed)

    auroc = roc_auc_score(y_test, y_score)
    auprc = average_precision_score(y_test, y_score)

    top50_idx = np.argsort(y_score)[::-1][:50]
    ap50 = y_test[top50_idx].sum() / 50.0

    pos_mask = y_test == 1
    if pos_mask.sum() > 1:
        deg_corr = float(np.corrcoef(degrees_test[pos_mask], y_score[pos_mask])[0, 1])
    else:
        deg_corr = float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "ap50": float(ap50),
        "degree_score_corr": deg_corr,
        "n_pos_test": int(y_test.sum()),
        "n_total_test": int(len(y_test)),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def save_report(output_dir: Path, condition: str, se_id: str, study,
                best_params: dict, test_metrics: dict, metric: str,
                default_params: dict):
    """Write a human-readable summary to hpo_report_<condition>.txt."""
    lines = []
    lines.append("=" * 64)
    lines.append(f"HPO REPORT — Condition {condition}  |  SE: {se_id}")
    lines.append("=" * 64)
    lines.append(f"\nOptimisation metric : {metric.upper()}")
    lines.append(f"Completed trials    : {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
    lines.append(f"Pruned trials       : {len([t for t in study.trials if t.state.name == 'PRUNED'])}")
    lines.append(f"Best CV {metric.upper():5s}       : {study.best_value:.4f}")

    lines.append("\n── Best hyperparameters ──────────────────────────────────")
    lines.append(f"{'Parameter':<22} {'Best value':>14}  {'Default':>14}")
    lines.append("-" * 54)
    for k, v in best_params.items():
        default = default_params.get(k, "—")
        if isinstance(v, float):
            lines.append(f"  {k:<20} {v:>14.5f}  {str(default):>14}")
        else:
            lines.append(f"  {k:<20} {v:>14}  {str(default):>14}")

    lines.append("\n── Final test-set metrics (best params, full train set) ──")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<28} {v:.4f}")
        else:
            lines.append(f"  {k:<28} {v}")

    lines.append("\n── Top 10 trials by CV score ─────────────────────────────")
    trials_df = study.trials_dataframe()
    if "value" in trials_df.columns:
        top10 = trials_df.dropna(subset=["value"]).nlargest(10, "value")
        for _, row in top10.iterrows():
            lines.append(f"  Trial {int(row['number']):4d}  CV {metric.upper()}={row['value']:.4f}")

    lines.append("\n── How to use these params in ablation_track1_ml.py ─────")
    lines.append("  Add --xgb_params_json path/to/best_params.json")
    lines.append("  (The main ablation script reads this file if the flag is present.)")
    lines.append("  Note: ablation_track1_ml.py does not yet have --xgb_params_json;")
    lines.append("  you need to add a json.load() call in its train_xgboost() function.")

    report_path = output_dir / f"hpo_report_{condition}.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Guard: check optuna is installed ──────────────────────────────────
    try:
        import optuna
    except ImportError:
        print("ERROR: optuna is not installed. Run:  pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Load combo ────────────────────────────────────────────────────────
    df_full = load_combo(args.combo, args.min_edges)
    se_counts = df_full.groupby("Polypharmacy Side Effect").size()
    tier_map = assign_tiers(se_counts)

    # ── --list_ses mode ───────────────────────────────────────────────────
    if args.list_ses:
        rows = (
            se_counts.reset_index()
                     .rename(columns={"Polypharmacy Side Effect": "se", 0: "n_pos"})
        )
        rows["tier"] = rows["se"].map(tier_map)
        rows = rows.sort_values("n_pos", ascending=False)
        print(f"\n{'SE CUI':<15} {'n_pos':>8}  {'tier':<14}")
        print("-" * 42)
        for _, row in rows.iterrows():
            print(f"{row['se']:<15} {row['n_pos']:>8}  {row['tier']:<14}")
        print(f"\nTotal: {len(rows)} SEs")
        sys.exit(0)

    # ── Select SE ─────────────────────────────────────────────────────────
    if args.se_id is None:
        # Default: SE closest to median edge count (most representative T3)
        median_count = se_counts.median()
        se_id = (se_counts - median_count).abs().idxmin()
        print(f"\nNo --se_id given. Auto-selected SE closest to median edge count:")
        print(f"  SE={se_id}, n_pos={se_counts[se_id]}, tier={tier_map.get(se_id,'?')}")
    else:
        se_id = args.se_id
        if se_id not in se_counts.index:
            print(f"ERROR: SE '{se_id}' not found in combo after filtering. "
                  f"Use --list_ses to see valid SEs.")
            sys.exit(1)
        print(f"\nTuning on SE={se_id}, n_pos={se_counts[se_id]}, "
              f"tier={tier_map.get(se_id,'?')}")

    df_se = df_full[df_full["Polypharmacy Side Effect"] == se_id].copy()

    # ── Load embeddings ───────────────────────────────────────────────────
    print("\nLoading embeddings...")
    chemberta = load_tensor(args.drug_emb_chemberta)
    mono      = load_tensor(args.drug_emb_mono)
    via_esm   = load_tensor(args.prot_emb_esm2)
    via_ppi   = load_tensor(args.prot_emb_ppi)

    drug_to_idx = make_drug_to_idx(df_full)
    n_drugs  = len(drug_to_idx)
    # If protein tensors are protein-level (n_proteins x D), convert to drug-level
    # (n_drugs x D) via targets, matching ablation_track1_ml expectations.
    need_conversion = (
        (via_esm is not None and via_esm.shape[0] != n_drugs)
        or (via_ppi is not None and via_ppi.shape[0] != n_drugs)
    )
    if need_conversion:
        if not args.targets:
            raise ValueError(
                "Received protein-level embeddings (rows != n_drugs). "
                "Provide --targets to aggregate protein->drug (drug-via-targets), "
                "or pass pre-aggregated drug-level tensors."
            )
        targets_df = load_targets_table(args.targets)
        base_rows = via_esm.shape[0] if via_esm is not None else via_ppi.shape[0]
        gene_to_row = resolve_gene_to_row(base_rows, targets_df, args.prot_gene_idx)

        if via_esm is not None and via_esm.shape[0] != n_drugs:
            via_esm = compute_drug_via_targets(
                via_esm, targets_df, drug_to_idx, gene_to_row, args.target_impute
            )
            print(f"  Converted --prot_emb_esm2 to drug-level via targets: {tuple(via_esm.shape)}")
        if via_ppi is not None and via_ppi.shape[0] != n_drugs:
            via_ppi = compute_drug_via_targets(
                via_ppi, targets_df, drug_to_idx, gene_to_row, args.target_impute
            )
            print(f"  Converted --prot_emb_ppi to drug-level via targets: {tuple(via_ppi.shape)}")

    drug_dim = chemberta.shape[1] if chemberta is not None else (via_esm.shape[1] if via_esm is not None else 256)
    prot_dim = via_esm.shape[1]   if via_esm   is not None else drug_dim

    # Default XGBoost params (from ablation_track1_ml.py) — used in report
    DEFAULT_PARAMS = {
        "n_estimators"    : 300,
        "max_depth"       : 6,
        "learning_rate"   : 0.1,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma"           : 0.0,
        "reg_alpha"       : 0.0,
        "reg_lambda"      : 1.0,
        "scale_pos_weight": 1.0,
        "neg_ratio"       : args.neg_ratio,
    }

    all_best = {}  # condition → best_params

    # ── Per-condition HPO loop ────────────────────────────────────────────
    for condition in args.conditions:
        print(f"\n{'─'*60}")
        print(f"Condition {condition}  —  {args.n_trials} trials  "
              f"x  {args.cv_folds}-fold CV  (metric={args.metric})")
        print(f"{'─'*60}")

        # Validate embedding availability
        if condition in ("A", "B", "E") and via_esm is None:
            print(f"  SKIP: condition {condition} requires --prot_emb_esm2.")
            continue
        if condition in ("A", "C", "E") and chemberta is None:
            print(f"  SKIP: condition {condition} requires --drug_emb_chemberta.")
            continue
        if condition == "E" and (mono is None or via_ppi is None):
            print(f"  SKIP: condition E requires --drug_emb_mono and --prot_emb_ppi.")
            continue

        # Build the fused embedding for this condition
        # (for --tune_neg_ratio this is rebuilt inside the objective)
        def make_emb(neg_ratio_unused=None):
            return np.ascontiguousarray(
                build_fused_embedding(
                    condition, n_drugs, drug_dim, prot_dim, args.seed,
                    chemberta, mono, via_esm, via_ppi
                ).detach().cpu().numpy(),
                dtype=np.float32
            )

        emb = make_emb()

        # Build full SE dataset once (unless tune_neg_ratio rebuilds it per trial)
        if not args.tune_neg_ratio:
            print("  Building dataset...")
            X_all, y_all, degrees_all = build_se_dataset(
                df_se, df_full, emb, drug_to_idx, args.neg_ratio, args.seed
            )
            print(f"  Samples: {len(y_all):,}  "
                  f"({y_all.sum():,} pos, {(1-y_all).sum():,} neg)")

            # Hold out test set — never touched during HPO
            X_train, X_test, y_train, y_test, deg_train, deg_test = train_test_split(
                X_all, y_all, degrees_all,
                test_size=args.test_frac,
                stratify=y_all,
                random_state=args.seed,
            )
        else:
            # When tuning neg_ratio, we can't pre-build the dataset.
            # Build a base dataset for the test split only (neg_ratio=1 placeholder),
            # then the objective rebuilds with the trial's neg_ratio for training.
            X_all_base, y_all_base, deg_all_base = build_se_dataset(
                df_se, df_full, emb, drug_to_idx, 1, args.seed
            )
            _, X_test, _, y_test, _, deg_test = train_test_split(
                X_all_base, y_all_base, deg_all_base,
                test_size=args.test_frac,
                stratify=y_all_base,
                random_state=args.seed,
            )
            X_train, y_train = None, None  # will be rebuilt inside objective

        # ── Build or resume Optuna study ──────────────────────────────────
        study_path = output_dir / f"optuna_study_{condition}.pkl"

        if args.resume and study_path.exists():
            with open(study_path, "rb") as f:
                study = pickle.load(f)
            print(f"  Resumed study with {len(study.trials)} existing trials.")
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,   # don't prune until 10 trials complete
                    n_warmup_steps=1,      # don't prune on the very first fold
                    interval_steps=1,
                ),
            )

        objective_fn = make_objective(
            X_train=X_train,
            y_train=y_train,
            df_full=df_full,
            emb_fn=make_emb,
            drug_to_idx=drug_to_idx,
            df_se=df_se,
            seed=args.seed,
            cv_folds=args.cv_folds,
            metric=args.metric,
            tune_neg_ratio=args.tune_neg_ratio,
            base_neg_ratio=args.neg_ratio,
            test_frac=args.test_frac,
        )

        # ── Run optimisation ──────────────────────────────────────────────
        study.optimize(
            objective_fn,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # ── Save study ────────────────────────────────────────────────────
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        print(f"\n  Study saved → {study_path}")

        # ── Save trials CSV ───────────────────────────────────────────────
        trials_df = study.trials_dataframe()
        trials_path = output_dir / f"optuna_trials_{condition}.csv"
        trials_df.to_csv(trials_path, index=False)
        print(f"  Trials CSV → {trials_path}")

        # ── Best params ───────────────────────────────────────────────────
        best_params = study.best_params.copy()
        # Ensure neg_ratio is present (either tuned or fixed)
        if "neg_ratio" not in best_params:
            best_params["neg_ratio"] = args.neg_ratio
        all_best[condition] = best_params

        params_path = output_dir / f"best_params_{condition}.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"  Best params → {params_path}")

        # ── Final evaluation on held-out test set ─────────────────────────
        print("\n  Evaluating best model on held-out test set...")

        # If neg_ratio was tuned, rebuild training data with the best neg_ratio
        if args.tune_neg_ratio:
            best_nr = best_params.get("neg_ratio", args.neg_ratio)
            X_all_best, y_all_best, deg_all_best = build_se_dataset(
                df_se, df_full, emb, drug_to_idx, best_nr, args.seed
            )
            X_train, _, y_train, _, _, _ = train_test_split(
                X_all_best, y_all_best, deg_all_best,
                test_size=args.test_frac,
                stratify=y_all_best,
                random_state=args.seed,
            )

        # Model params only (exclude neg_ratio, which is a dataset param)
        model_params = {k: v for k, v in best_params.items() if k != "neg_ratio"}
        test_metrics = final_eval(
            X_train, y_train, X_test, y_test, model_params, args.seed, deg_test
        )

        print(f"\n  Test AUROC  : {test_metrics['auroc']:.4f}")
        print(f"  Test AUPRC  : {test_metrics['auprc']:.4f}")
        print(f"  Test AP@50  : {test_metrics['ap50']:.4f}")
        print(f"  Degree corr : {test_metrics['degree_score_corr']:.4f}")
        print(f"  n_pos test  : {test_metrics['n_pos_test']}")

        # ── Save report ───────────────────────────────────────────────────
        save_report(output_dir, condition, se_id, study,
                    best_params, test_metrics, args.metric, DEFAULT_PARAMS)

    # ── Cross-condition summary ───────────────────────────────────────────
    if len(all_best) > 1:
        print(f"\n{'='*60}")
        print("CROSS-CONDITION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Condition':<12} {'n_est':>7} {'depth':>6} {'lr':>8} "
              f"{'sub':>6} {'col':>6} {'min_cw':>7} {'gamma':>7} "
              f"{'alpha':>8} {'lambda':>8} {'spw':>6}")
        print("-" * 85)
        for cond, bp in all_best.items():
            print(
                f"  {cond:<10} "
                f"{bp.get('n_estimators','-'):>7} "
                f"{bp.get('max_depth','-'):>6} "
                f"{bp.get('learning_rate',0):>8.4f} "
                f"{bp.get('subsample',0):>6.3f} "
                f"{bp.get('colsample_bytree',0):>6.3f} "
                f"{bp.get('min_child_weight','-'):>7} "
                f"{bp.get('gamma',0):>7.3f} "
                f"{bp.get('reg_alpha',0):>8.5f} "
                f"{bp.get('reg_lambda',0):>8.5f} "
                f"{bp.get('scale_pos_weight',0):>6.3f}"
            )

        summary_path = output_dir / "best_params_all_conditions.json"
        with open(summary_path, "w") as f:
            json.dump(all_best, f, indent=2)
        print(f"\nAll best params saved → {summary_path}")

    print(f"\nAll outputs in: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
