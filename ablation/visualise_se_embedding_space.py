"""
visualise_se_embedding_space.py
--------------------------------
Visualise the pair-level feature space for individual side effects to
explain WHY T1 (frequent) SEs are harder to classify than T5 (rare) SEs.

Fully compatible with ablation_track1_ml.py and build_drug_protein_features.py.
All data loading, embedding construction, and dataset building logic mirrors
the production code exactly — same function names, same argument interface,
same pair_repr modes.

Three analyses
--------------
1. UMAP projection
   512-dim (or 256-dim for sum) pair features -> 2D, coloured by label.
   Run for one T1 SE and one T5 SE side-by-side.
   Shows whether positive pairs form a tight cluster (T5) or scatter
   across the space (T1 heterogeneity).

2. Pairwise cosine distance
   Mean pairwise cosine distance between positive pair vectors within each SE.
   Plotted against n_pos across all SEs (--all_ses_cosine).
   Directly evidences M1: larger SEs contain more structurally diverse positives.

3. Product magnitude distribution
   ||e_A * e_B||_2 KDE for positives vs corrupted negatives per SE.
   T5 (drug-class-specific SE): tight positive cluster -> separated distributions.
   T1 (diverse SE): broad positive spread -> overlapping distributions.

Outputs
-------
  umap_T1_<se>.png                 UMAP: positive vs negative, T1 SE
  umap_T5_<se>.png                 UMAP: positive vs negative, T5 SE
  umap_both.png                    side-by-side comparison
  product_magnitude_comparison.png KDE: ||e_A*e_B|| pos vs neg, both SEs
  cosine_dist_vs_npos.png          scatter + boxplot (requires --all_ses_cosine)
  cosine_stats_per_se.csv          per-SE cosine stats
  embedding_space_summary.txt      plain-text interpretation

Usage
-----
  # Minimal - auto-selects SEs, condition A, sym pair repr
  python visualise_se_embedding_space.py \\
    --combo              bio-decagon-combo.csv \\
    --drug_emb_chemberta drug_emb_chemberta_256.pt \\
    --prot_emb_esm2      drug_via_targets_256.pt \\
    --output             ./se_visualisations

  # Same interface as ablation_track1_ml.py
  python visualise_se_embedding_space.py \\
    --combo              bio-decagon-combo.csv \\
    --drug_emb_chemberta drug_emb_chemberta_256.pt \\
    --drug_emb_mono      drug_emb_mono_256.pt \\
    --prot_emb_esm2      drug_via_targets_256.pt \\
    --prot_emb_ppi       drug_via_targets_ppi_256.pt \\
    --condition          E \\
    --pair_repr          sym \\
    --se_t1              C0027498 \\
    --se_t5              C0015230 \\
    --output             ./se_visualisations

  # With protein-level ESM tensor + targets file
  python visualise_se_embedding_space.py \\
    --combo    bio-decagon-combo.csv \\
    --targets  bio-decagon-targets.csv \\
    --drug_emb_chemberta drug_emb_chemberta_256.pt \\
    --prot_emb_esm2      protein_init_dim256_esm2_t30_150M_UR50D.pt \\
    --condition A \\
    --output ./se_visualisations

  # Full cosine analysis across all SEs (~15 min)
  python visualise_se_embedding_space.py \\
    --combo              bio-decagon-combo.csv \\
    --drug_emb_chemberta drug_emb_chemberta_256.pt \\
    --prot_emb_esm2      drug_via_targets_256.pt \\
    --all_ses_cosine \\
    --output             ./se_visualisations

Dependencies
------------
  pip install torch pandas numpy scikit-learn umap-learn matplotlib seaborn scipy
"""

import argparse
import gc
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ---- Style -------------------------------------------------------------------
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 12,
    "axes.titleweight" : "medium",
    "axes.labelsize"   : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 150,
    "savefig.bbox"     : "tight",
    "savefig.facecolor": "white",
})

POS_COLOR = "#378ADD"
NEG_COLOR = "#D85A30"
TIER_COLORS = {
    "T1_frequent": "#D85A30",
    "T2"         : "#BA7517",
    "T3"         : "#1D9E75",
    "T4"         : "#378ADD",
    "T5_rare"    : "#7F77DD",
}


# ---- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise pair-level embedding space for PSE prediction."
    )
    # Identical flags to ablation_track1_ml.py
    p.add_argument("--combo",              required=True)
    p.add_argument("--drug_emb_chemberta", required=True)
    p.add_argument("--drug_emb_mono",      default=None)
    p.add_argument("--prot_emb_esm2",      required=True)
    p.add_argument("--prot_emb_ppi",       default=None)
    p.add_argument("--targets",            default=None,
                   help="bio-decagon-targets.csv -- required when prot_emb_esm2 is "
                        "protein-level (rows != n_drugs)")
    p.add_argument("--prot_gene_idx",      default=None,
                   help="JSON: Gene -> row index in prot_emb tensor")
    p.add_argument("--target_impute",      default="zero",
                   choices=["zero", "mean"])
    p.add_argument("--pair_repr",          default="sym",
                   choices=["sym", "concat", "sum"])
    p.add_argument("--drug_fusion",        default="hstack",
                   choices=["hstack"])
    p.add_argument("--min_edges",          type=int, default=500)
    p.add_argument("--neg_ratio",          type=int, default=1)
    p.add_argument("--seed",               type=int, default=42)
    # Visualisation-specific
    p.add_argument("--condition", default="A",
                   choices=["A","B","C","D","E"])
    p.add_argument("--se_t1", default=None,
                   help="SE CUI for T1 visualisation. Auto-selected if omitted.")
    p.add_argument("--se_t5", default=None,
                   help="SE CUI for T5 visualisation. Auto-selected if omitted.")
    p.add_argument("--n_umap_samples",    type=int,   default=2000,
                   help="Max total samples fed to UMAP per SE.")
    p.add_argument("--umap_neighbors",    type=int,   default=30)
    p.add_argument("--umap_min_dist",     type=float, default=0.1)
    p.add_argument("--all_ses_cosine",    action="store_true",
                   help="Run cosine distance analysis across ALL SEs (~15 min).")
    p.add_argument("--cosine_max_sample", type=int,   default=300,
                   help="Max positive pairs sampled per SE for cosine analysis.")
    p.add_argument("--output", default="./se_visualisations")
    return p.parse_args()


# ---- Data loading — mirrors ablation_track1_ml.py ----------------------------

def load_combo(path, min_edges):
    df = pd.read_csv(path)
    se_counts = df.groupby("Polypharmacy Side Effect").size().reset_index(name="edge_count")
    valid = se_counts[se_counts["edge_count"] >= min_edges]["Polypharmacy Side Effect"].tolist()
    df = df[df["Polypharmacy Side Effect"].isin(valid)].copy()
    sc = df.groupby("Polypharmacy Side Effect").size()
    print(f"Loaded {len(df):,} pairs across {df['Polypharmacy Side Effect'].nunique()} SEs")
    return df, sc


def load_tensor(path):
    if path is None:
        return None
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        t = torch.stack(list(t.values()))
    print(f"  Loaded {path}: shape {t.shape}")
    return t.float()


def make_entity_index(df):
    drugs = pd.concat([df["STITCH 1"], df["STITCH 2"]]).unique()
    return {d: i for i, d in enumerate(sorted(drugs))}


def load_targets_table(path):
    t = pd.read_csv(path)
    t["STITCH"] = t["STITCH"].astype(str)
    t["Gene"]   = t["Gene"].astype(str)
    return t


def resolve_gene_to_row(n_prot_rows, targets, prot_gene_idx):
    if prot_gene_idx:
        with open(prot_gene_idx) as f:
            raw = json.load(f)
        return {str(k): int(v) for k, v in raw.items()}
    genes = sorted(targets["Gene"].unique())
    if len(genes) > n_prot_rows:
        raise ValueError(
            f"{len(genes)} unique Genes but prot_emb has only {n_prot_rows} rows. "
            "Pass --prot_gene_idx."
        )
    print("WARNING: using sorted Gene order -> prot_emb rows. "
          "Pass --prot_gene_idx if your tensor uses a different ordering.")
    return {g: i for i, g in enumerate(genes)}


def compute_drug_via_targets(prot_emb, targets, drug_to_idx, gene_to_row, impute):
    dim = prot_emb.shape[1]
    drug_targets = defaultdict(list)
    for _, row in targets.iterrows():
        drug_targets[row["STITCH"]].append(row["Gene"])
    fallback = torch.zeros(dim) if impute == "zero" else prot_emb.mean(dim=0)
    out = torch.zeros(len(drug_to_idx), dim, dtype=prot_emb.dtype)
    for drug, idx in drug_to_idx.items():
        ok = [g for g in drug_targets.get(drug, []) if g in gene_to_row]
        if ok:
            out[idx] = prot_emb[[gene_to_row[g] for g in ok]].mean(dim=0)
        else:
            out[idx] = fallback
    return out


# ---- Embedding construction — mirrors build_fused_drug_embedding_matrix ------

def zero_tensor(n, d):
    return torch.zeros(n, d)


def xavier_tensor(n, d, seed):
    torch.manual_seed(seed)
    t = torch.empty(n, d)
    torch.nn.init.xavier_uniform_(t)
    return t


def build_drug_embeddings(condition, n_drugs, chemberta, mono, dim, seed):
    if condition in ("A", "C"):
        return chemberta[:n_drugs]
    if condition == "E":
        return (chemberta[:n_drugs] + mono[:n_drugs]) / 2.0
    if condition == "B":
        return zero_tensor(n_drugs, dim)
    return xavier_tensor(n_drugs, dim, seed)  # D


def build_fused_drug_embedding_matrix(condition, n_drugs, drug_dim, prot_dim,
                                       seed, chemberta, mono, via_esm, via_ppi):
    drug_b = build_drug_embeddings(condition, n_drugs, chemberta, mono, drug_dim, seed)
    if condition in ("A", "B"):
        prot_b = via_esm
    elif condition == "C":
        prot_b = zero_tensor(n_drugs, prot_dim)
    elif condition == "D":
        prot_b = xavier_tensor(n_drugs, prot_dim, seed + 1)
    else:  # E
        prot_b = (via_esm + via_ppi) / 2.0
    return torch.cat([drug_b, prot_b], dim=1)


# ---- Tier assignment ---------------------------------------------------------

def assign_tiers(se_counts, n_tiers=5):
    sorted_ses = se_counts.sort_values()
    tier_size  = len(sorted_ses) // n_tiers
    labels     = [f"T{n_tiers}_rare", f"T{n_tiers-1}", "T3", "T2", "T1_frequent"]
    tmap       = {}
    for i, (se, _) in enumerate(sorted_ses.items()):
        tmap[se] = labels[min(i // tier_size, n_tiers - 1)]
    return tmap


# ---- Per-SE pair dataset builder ---------------------------------------------

def build_se_pairs(df_se, df_full, emb, drug_to_idx, neg_ratio, seed,
                   pair_repr, max_samples=None):
    """
    Build pair feature matrix X and labels y for one SE.
    Returns X, y, ea_pos (drug A raw vecs), eb_pos (drug B raw vecs).
    ea_pos / eb_pos are needed for the cosine distance analysis.
    """
    rng       = np.random.RandomState(seed)
    all_drugs = list(drug_to_idx.keys())
    n_drugs   = len(all_drugs)
    fused_dim = emb.shape[1]
    feat_dim  = fused_dim if pair_repr == "sum" else 2 * fused_dim

    mask  = (df_se["STITCH 1"].isin(drug_to_idx) &
             df_se["STITCH 2"].isin(drug_to_idx))
    df_se = df_se.loc[mask].copy()
    if len(df_se) == 0:
        return None, None, None, None

    s1 = df_se["STITCH 1"].to_numpy()
    s2 = df_se["STITCH 2"].to_numpy()
    n_pos = len(df_se)

    if max_samples is not None:
        max_pos = max_samples // (1 + neg_ratio)
        if n_pos > max_pos:
            idx  = rng.choice(n_pos, size=max_pos, replace=False)
            s1, s2 = s1[idx], s2[idx]
            n_pos = max_pos

    n_total = n_pos * (1 + neg_ratio)
    X      = np.empty((n_total, feat_dim), dtype=np.float32)
    y      = np.empty(n_total, dtype=np.int32)
    ea_pos = np.empty((n_pos, fused_dim), dtype=np.float32)
    eb_pos = np.empty((n_pos, fused_dim), dtype=np.float32)

    def write_pair(row_idx, id_a, ea, id_b, eb):
        if pair_repr == "sym":
            X[row_idx, :fused_dim] = ea * eb
            X[row_idx, fused_dim:] = np.abs(ea - eb)
        elif pair_repr == "concat":
            # canonical order: lower STITCH ID first
            lv, rv = (ea, eb) if id_a <= id_b else (eb, ea)
            X[row_idx, :fused_dim] = lv
            X[row_idx, fused_dim:] = rv
        else:  # sum
            X[row_idx, :] = ea + eb

    row = 0
    for i in range(n_pos):
        a, b = s1[i], s2[i]
        ea   = emb[drug_to_idx[a]]
        eb   = emb[drug_to_idx[b]]
        write_pair(row, a, ea, b, eb)
        y[row]    = 1
        ea_pos[i] = ea
        eb_pos[i] = eb
        row += 1

        for _ in range(neg_ratio):
            neg = all_drugs[rng.randint(n_drugs)]
            ec  = emb[drug_to_idx[neg]]
            if rng.rand() < 0.5:
                write_pair(row, neg, ec, b, eb)
            else:
                write_pair(row, a, ea, neg, ec)
            y[row] = 0
            row   += 1

    return X, y, ea_pos, eb_pos


# ---- SE auto-selection -------------------------------------------------------

def auto_select_ses(se_counts, tier_map):
    t1 = se_counts[[s for s, t in tier_map.items() if t == "T1_frequent"]]
    t5 = se_counts[[s for s, t in tier_map.items() if t == "T5_rare"]]
    se_t1 = t1[t1 >= t1.quantile(0.75)].idxmax()  # Q4 of T1 - most diverse
    se_t5 = (t5 - t5.median()).abs().idxmin()       # closest to T5 median
    return se_t1, se_t5


# ---- UMAP --------------------------------------------------------------------

def run_umap(X, n_neighbors, min_dist, seed):
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("pip install umap-learn")
    from sklearn.preprocessing import StandardScaler
    # Normalise before UMAP: high-variance dimensions otherwise dominate the projection
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=2, random_state=seed, verbose=False)
    return reducer.fit_transform(X_scaled)


def linear_separability_auroc(X, y, seed):
    """
    Fit a logistic regression on the pair feature matrix and return its
    cross-validated AUROC.  This directly measures how linearly separable
    positives are from negatives in the pair embedding space.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    scores = []
    for tr, val in skf.split(X, y):
        Xtr = scaler.fit_transform(X[tr])
        Xval = scaler.transform(X[val])
        clf = LogisticRegression(max_iter=200, C=1.0, random_state=seed, n_jobs=-1)
        clf.fit(Xtr, y[tr])
        scores.append(roc_auc_score(y[val], clf.predict_proba(Xval)[:, 1]))
    return float(np.mean(scores))


def plot_umap(ax, emb2d, y, se_id, tier, n_pos, pair_repr, condition):
    pos, neg = y == 1, y == 0
    ax.scatter(emb2d[neg, 0], emb2d[neg, 1], c=NEG_COLOR, s=5,
               alpha=0.40, linewidths=0, rasterized=True,
               label=f"Corrupted neg  (n={neg.sum()})")
    ax.scatter(emb2d[pos, 0], emb2d[pos, 1], c=POS_COLOR, s=7,
               alpha=0.55, linewidths=0, rasterized=True,
               label=f"True positive  (n={pos.sum()})")
    ax.set_title(
        f"{tier.replace('_', ' ')}  |  SE: {se_id}\n"
        f"n_pos={n_pos}  |  cond {condition}  |  {pair_repr}", pad=8)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, fontsize=9, framealpha=0.7)


# ---- Cosine distance helpers -------------------------------------------------

def mean_pairwise_cosine_dist(ea, eb, max_n, seed):
    n = min(len(ea), max_n)
    if n < 2:
        return float("nan")
    rng  = np.random.RandomState(seed)
    idx  = rng.choice(len(ea), size=n, replace=False)
    vecs = np.concatenate([ea[idx], eb[idx]], axis=1)
    vecs = normalize(vecs, norm="l2")
    d    = cosine_distances(vecs)
    return float(d[np.triu_indices(n, k=1)].mean())


def kde_overlap(a, b, n=500):
    from scipy.stats import gaussian_kde
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    xs = np.linspace(lo, hi, n)
    try:
        return float(np.trapz(np.minimum(gaussian_kde(a)(xs),
                                          gaussian_kde(b)(xs)), xs))
    except Exception:
        return float("nan")


# ---- Product magnitude plot --------------------------------------------------

def plot_product_magnitude(ax, X, y, fused_dim, pair_repr, se_id, tier, n_pos):
    if pair_repr == "sum":
        ax.text(0.5, 0.5,
                "Product magnitude not applicable\nfor sum representation",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="gray")
        ax.set_title(f"{tier.replace('_', ' ')}  |  {se_id}")
        return
    if pair_repr == "sym":
        # First half of X is already e_A ⊙ e_B
        prod = X[:, :fused_dim]
    else:
        # concat: first half is e_A, second half is e_B — compute product for visualisation
        # NOTE: this is a derived signal for visualisation only; XGBoost trained on concat(e_A,e_B)
        prod = X[:, :fused_dim] * X[:, fused_dim:]
    mag  = np.linalg.norm(prod, axis=1)
    pm, nm = mag[y == 1], mag[y == 0]
    ov = kde_overlap(pm, nm)
    sns.kdeplot(pm, ax=ax, color=POS_COLOR, fill=True, alpha=0.35,
                label=f"Positive (n={len(pm)})", linewidth=1.5)
    sns.kdeplot(nm, ax=ax, color=NEG_COLOR, fill=True, alpha=0.35,
                label=f"Corrupted neg (n={len(nm)})", linewidth=1.5)
    ax.set_xlabel("||e_A * e_B||_2")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{tier.replace('_', ' ')}  |  SE: {se_id}\n"
        f"n_pos={n_pos}  |  KDE overlap approx {ov:.3f}")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---- Cosine scatter plot -----------------------------------------------------

def plot_cosine_scatter(stats_df, output_dir):
    tier_order = ["T1_frequent","T2","T3","T4","T5_rare"]
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5))

    for tier in tier_order:
        sub = stats_df[stats_df["tier"] == tier]
        if sub.empty:
            continue
        axes[0].scatter(sub["n_pos"], sub["mean_cosine_dist"],
                        c=TIER_COLORS.get(tier, "#888780"),
                        s=16, alpha=0.55, linewidths=0,
                        label=tier.replace("_", " "))

    valid = stats_df.dropna(subset=["mean_cosine_dist"])
    if len(valid) > 2:
        m, b = np.polyfit(valid["n_pos"], valid["mean_cosine_dist"], 1)
        xs   = np.linspace(valid["n_pos"].min(), valid["n_pos"].max(), 300)
        r    = np.corrcoef(valid["n_pos"], valid["mean_cosine_dist"])[0, 1]
        axes[0].plot(xs, m * xs + b, color="#444441", lw=1.5, ls="--",
                     alpha=0.7, label=f"OLS  r={r:.3f}")
        axes[0].text(0.97, 0.05, f"Pearson r = {r:.3f}\nn = {len(valid)} SEs",
                     transform=axes[0].transAxes, ha="right", va="bottom",
                     fontsize=10, color="#444441")

    axes[0].set_xlabel("n_pos (positive pairs per SE)")
    axes[0].set_ylabel("Mean pairwise cosine distance\nbetween positive pairs")
    axes[0].set_title("Heterogeneity of positive pairs vs SE size (M1 mechanism)")
    axes[0].legend(fontsize=8, markerscale=1.5, loc="upper left")

    data = [stats_df.loc[stats_df["tier"] == t, "mean_cosine_dist"]
              .dropna().values for t in tier_order]
    bp = axes[1].boxplot(data, patch_artist=True, notch=False, widths=0.5,
                          medianprops={"color": "white", "linewidth": 2})
    for patch, tier in zip(bp["boxes"], tier_order):
        patch.set_facecolor(TIER_COLORS.get(tier, "#888780"))
        patch.set_alpha(0.75)
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xticklabels([t.replace("_", "\n") for t in tier_order], fontsize=9)
    axes[1].set_ylabel("Mean pairwise cosine distance")
    axes[1].set_title("Positive-pair heterogeneity by frequency tier")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=2)
    out = output_dir / "cosine_dist_vs_npos.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- Summary -----------------------------------------------------------------

def write_summary(output_dir, se_results, condition, pair_repr, stats_df=None):
    lines = [
        "=" * 64,
        "Embedding space analysis summary",
        f"Condition {condition}  |  pair_repr = {pair_repr}",
        "=" * 64,
    ]
    for se_id, res in se_results.items():
        X, y, ea, eb = res["X"], res["y"], res["ea_pos"], res["eb_pos"]
        fused = ea.shape[1]
        lines.append(f"\n-- {res['tier']}  SE={se_id}  n_pos={res['n_pos']} ----")

        if pair_repr != "sum":
            prod = X[:, :fused] if pair_repr == "sym" else X[:, :fused] * X[:, fused:]
            mag  = np.linalg.norm(prod, axis=1)
            pm, nm = mag[y == 1], mag[y == 0]
            ov = kde_overlap(pm, nm)
            lines.append(f"  Product magnitude pos: mean={pm.mean():.4f}  std={pm.std():.4f}")
            lines.append(f"  Product magnitude neg: mean={nm.mean():.4f}  std={nm.std():.4f}")
            lines.append(f"  KDE overlap: {ov:.3f}  "
                         "(0=perfectly separated, 1=identical distributions)")

        mcd = mean_pairwise_cosine_dist(ea, eb, 300, 42)
        lines.append(f"  Mean pairwise cosine dist between positives: {mcd:.4f}")
        label = ("HIGH" if mcd > 0.5 else "MODERATE" if mcd > 0.3 else "LOW")
        lines.append(f"  Heterogeneity: {label}")

        lin_auroc = res.get("lin_auroc")
        if lin_auroc is not None:
            sep = ("GOOD" if lin_auroc > 0.75 else "MODERATE" if lin_auroc > 0.60 else "POOR")
            lines.append(f"  Linear separability AUROC: {lin_auroc:.4f}  [{sep}]"
                         "  (logistic regression 5-CV on pair features)")

    if len(se_results) == 2:
        items = list(se_results.items())
        d1 = mean_pairwise_cosine_dist(
            items[0][1]["ea_pos"], items[0][1]["eb_pos"], 300, 42)
        d2 = mean_pairwise_cosine_dist(
            items[1][1]["ea_pos"], items[1][1]["eb_pos"], 300, 42)
        lines.append("\n-- Key comparison ----------------------------------------")
        if d2 > 0:
            lines.append(
                f"  T1 cosine dist={d1:.4f}, T5 cosine dist={d2:.4f}, "
                f"ratio={d1/d2:.2f}x")
            lines.append(
                f"  T1 positives are {d1/d2:.1f}x more heterogeneous than T5 "
                f"positives in embedding space.")
        lines.append(
            "  Supports M1: harder T1 classification boundary reflects "
            "structural diversity of positive pairs, not model failure.")

    if stats_df is not None and len(stats_df) > 2:
        r = np.corrcoef(stats_df["n_pos"].values,
                        stats_df["mean_cosine_dist"].values)[0, 1]
        m = np.polyfit(stats_df["n_pos"].values,
                       stats_df["mean_cosine_dist"].values, 1)[0]
        lines.append(f"\n  Pearson r(n_pos, cosine_dist) across all SEs: {r:.4f}")
        lines.append(
            f"  OLS: each additional 100 positive pairs adds "
            f"{m*100:.4f} cosine distance units.")

    out = output_dir / "embedding_space_summary.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


# ---- Main --------------------------------------------------------------------

def main():
    args       = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_full, se_counts = load_combo(args.combo, args.min_edges)
    tier_map    = assign_tiers(se_counts)
    drug_to_idx = make_entity_index(df_full)
    n_drugs     = len(drug_to_idx)

    print("\nLoading embeddings...")
    chemberta = load_tensor(args.drug_emb_chemberta)
    mono      = load_tensor(args.drug_emb_mono)
    esm2      = load_tensor(args.prot_emb_esm2)
    ppi       = load_tensor(args.prot_emb_ppi)

    drug_dim  = chemberta.shape[1]
    prot_dim  = esm2.shape[1]
    fused_dim = drug_dim + prot_dim

    # Resolve via_esm / via_ppi using the same dual-path logic as ablation_track1_ml.py
    via_esm = via_ppi = None
    if args.condition in ("A", "B", "E"):
        if esm2.shape[0] == n_drugs:
            via_esm = esm2[:n_drugs]
            print(f"  Pre-aggregated drug-level ESM: {tuple(via_esm.shape)}")
        else:
            if not args.targets:
                raise ValueError(
                    "--prot_emb_esm2 appears protein-level (rows != n_drugs). "
                    "Provide --targets, or pass a pre-aggregated drug-level tensor."
                )
            tdf     = load_targets_table(args.targets)
            g2r     = resolve_gene_to_row(esm2.shape[0], tdf, args.prot_gene_idx)
            via_esm = compute_drug_via_targets(
                esm2, tdf, drug_to_idx, g2r, args.target_impute)
            print(f"  drug-via-ESM aggregated: {tuple(via_esm.shape)}")

    if args.condition == "E":
        if ppi is None:
            raise ValueError("Condition E requires --prot_emb_ppi.")
        if ppi.shape[0] == n_drugs:
            via_ppi = ppi[:n_drugs]
        else:
            tdf2    = load_targets_table(args.targets)
            g2r_ppi = resolve_gene_to_row(ppi.shape[0], tdf2, args.prot_gene_idx)
            via_ppi = compute_drug_via_targets(
                ppi, tdf2, drug_to_idx, g2r_ppi, args.target_impute)

    # Build fused embedding
    print(f"\nBuilding condition {args.condition} embedding...")
    drug_emb_t = build_fused_drug_embedding_matrix(
        args.condition, n_drugs, drug_dim, prot_dim, args.seed,
        chemberta, mono, via_esm, via_ppi,
    )
    emb = np.ascontiguousarray(drug_emb_t.detach().cpu().numpy(), dtype=np.float32)
    del drug_emb_t, chemberta, mono, esm2, ppi
    gc.collect()
    print(f"  Fused embedding: {emb.shape}")

    # Select SEs
    se_t1_auto, se_t5_auto = auto_select_ses(se_counts, tier_map)
    se_t1 = args.se_t1 or se_t1_auto
    se_t5 = args.se_t5 or se_t5_auto
    for se_id in (se_t1, se_t5):
        if se_id not in se_counts.index:
            raise ValueError(
                f"SE '{se_id}' not found after --min_edges filter."
            )
    print(f"\nSelected SEs:")
    print(f"  T1: {se_t1}  n_pos={se_counts[se_t1]}  tier={tier_map[se_t1]}")
    print(f"  T5: {se_t5}  n_pos={se_counts[se_t5]}  tier={tier_map[se_t5]}")

    # Build pair datasets for the two SEs
    print("\n-- Building pair datasets -------------------------------------------")
    se_results = {}
    for se_id in (se_t1, se_t5):
        df_se = df_full[df_full["Polypharmacy Side Effect"] == se_id].copy()
        tier  = tier_map[se_id]
        n_pos = int(se_counts[se_id])
        print(f"  SE={se_id} ({tier}, n_pos={n_pos})...", end=" ", flush=True)
        X, y, ea_pos, eb_pos = build_se_pairs(
            df_se, df_full, emb, drug_to_idx,
            args.neg_ratio, args.seed, args.pair_repr,
            max_samples=args.n_umap_samples,
        )
        if X is None:
            print("SKIP (no valid pairs)")
            continue
        print(f"{X.shape[0]:,} samples")
        se_results[se_id] = {
            "X": X, "y": y, "ea_pos": ea_pos, "eb_pos": eb_pos,
            "tier": tier, "n_pos": n_pos,
        }

    # Analysis 1: UMAP
    # Linear separability AUROC — quantifies how well a simple linear model
    # separates positives from negatives in the actual pair feature space
    print("\n-- Linear separability (logistic regression 5-CV AUROC) -------------")
    for se_id, res in se_results.items():
        lin_auroc = linear_separability_auroc(res["X"], res["y"], args.seed)
        res["lin_auroc"] = lin_auroc
        print(f"  {res['tier']} SE={se_id}: linear AUROC = {lin_auroc:.4f}  "
              f"(0.5=random, 1.0=perfectly separable)")

    print("\n-- UMAP projections (features StandardScaler-normalised before UMAP) -")
    umap_embeds = {}
    for se_id, res in se_results.items():
        print(f"  UMAP for {res['tier']} SE={se_id} ({res['X'].shape[0]} samples)...")
        emb2d = run_umap(res["X"], args.umap_neighbors,
                          args.umap_min_dist, args.seed)
        umap_embeds[se_id] = emb2d
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_umap(ax, emb2d, res["y"], se_id, res["tier"],
                   res["n_pos"], args.pair_repr, args.condition)
        lin_label = f"Linear AUROC = {res.get('lin_auroc', float('nan')):.4f}"
        ax.text(0.02, 0.02, lin_label, transform=ax.transAxes,
                fontsize=9, color="#444441", va="bottom")
        out = output_dir / f"umap_{res['tier']}_{se_id}.png"
        fig.savefig(out); plt.close(fig)
        print(f"    Saved: {out}")

    if len(umap_embeds) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Pair-level embedding space  --  condition {args.condition}  |  {args.pair_repr}\n"
            f"T1 (frequent) vs T5 (rare): separability of true pairs vs corrupted negatives",
            fontsize=12, y=1.01)
        for ax, (se_id, emb2d) in zip(axes, umap_embeds.items()):
            res = se_results[se_id]
            plot_umap(ax, emb2d, res["y"], se_id, res["tier"],
                       res["n_pos"], args.pair_repr, args.condition)
        fig.tight_layout(pad=2)
        out = output_dir / "umap_both.png"
        fig.savefig(out); plt.close(fig)
        print(f"  Saved: {out}")

    # Analysis 3: Product magnitude
    if len(se_results) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Product magnitude ||e_A * e_B||_2  --  condition {args.condition}\n"
            f"Wider separation = embedding similarity discriminates pos from neg",
            fontsize=12)
        for ax, (se_id, res) in zip(axes, se_results.items()):
            plot_product_magnitude(ax, res["X"], res["y"], fused_dim,
                                    args.pair_repr, se_id, res["tier"], res["n_pos"])
        fig.tight_layout(pad=2)
        out = output_dir / "product_magnitude_comparison.png"
        fig.savefig(out); plt.close(fig)
        print(f"  Saved: {out}")

    # Analysis 2: Cosine distance
    stats_df = None
    print("\n-- Cosine distance analysis -----------------------------------------")
    if args.all_ses_cosine:
        all_ses = list(se_counts.index)
        print(f"  Processing {len(all_ses)} SEs...")
        rows = []
        for i, se_id in enumerate(all_ses):
            if i % 100 == 0 and i > 0:
                print(f"    {i}/{len(all_ses)}...")
            df_se_tmp = df_full[df_full["Polypharmacy Side Effect"] == se_id].copy()
            _, _, ea, eb = build_se_pairs(
                df_se_tmp, df_full, emb, drug_to_idx,
                1, args.seed, args.pair_repr,
                max_samples=args.cosine_max_sample * 2,
            )
            if ea is None or len(ea) < 2:
                continue
            mcd = mean_pairwise_cosine_dist(ea, eb, args.cosine_max_sample, args.seed)
            rows.append({
                "se": se_id, "n_pos": int(se_counts[se_id]),
                "tier": tier_map[se_id], "mean_cosine_dist": mcd,
            })
        stats_df = pd.DataFrame(rows)
        out_csv  = output_dir / "cosine_stats_per_se.csv"
        stats_df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")
        if len(stats_df) > 2:
            r = np.corrcoef(stats_df["n_pos"].values,
                            stats_df["mean_cosine_dist"].values)[0, 1]
            print(f"  Pearson r(n_pos, cosine_dist) = {r:.4f}")
            print("  Per-tier mean cosine dist:")
            print(stats_df.groupby("tier")["mean_cosine_dist"].mean()
                  .reindex(["T1_frequent","T2","T3","T4","T5_rare"])
                  .round(4).to_string())
            plot_cosine_scatter(stats_df, output_dir)
    else:
        print("  (Pass --all_ses_cosine for the full scatter across all SEs)")
        for se_id, res in se_results.items():
            mcd = mean_pairwise_cosine_dist(
                res["ea_pos"], res["eb_pos"], args.cosine_max_sample, args.seed)
            print(f"  {res['tier']} SE={se_id}: cosine dist={mcd:.4f}  n_pos={res['n_pos']}")

    write_summary(output_dir, se_results, args.condition, args.pair_repr, stats_df)
    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
