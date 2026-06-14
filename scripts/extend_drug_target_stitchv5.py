#!/usr/bin/env python
"""
Extend Decagon drug-target annotations with STITCH v5 human targets.

This script converts the workflow from
feature_engineering/protein_coverage/protein_STITCH_v5.ipynb into a reusable
CLI. It:

1. Loads Decagon combo drugs and original Decagon targets.
2. Downloads STITCH v5 human protein-chemical links and STRING protein aliases
   if needed.
3. Maps STITCH chemicals to Decagon STITCH drug IDs via PubChem CID.
4. Maps STITCH/STRING proteins (9606.ENSP...) to Entrez Gene IDs.
5. Writes expanded target files compatible with the existing pipeline:
   STITCH,Gene

The STITCH detailed links file has per-channel scores (0-1000):
  - experimental : binding assays, ChEMBL Ki/IC50, PDB, PDSP, kinase screens
  - database     : manually curated databases (DrugBank, TTD, KEGG, Reactome…)
  - textmining   : co-occurrence / NLP over MEDLINE + PubMed Central
  - prediction   : structure-based computational predictions
  - combined_score: Bayesian combination of all channels

Use --experimental_only to keep only rows where experimental > 0
(closest to the Decagon paper's "experimentally verified" criterion).
Use --min_experimental N to additionally require a minimum experimental
channel score (e.g. 400 for medium confidence in that channel alone).
Use --min_score N to fall back to filtering on combined_score only.

Example — replicate Decagon's experimental-only criterion:
    python scripts/extend_drug_target_stitchv5.py --experimental_only

Example — experimental + high-confidence combined score:
    python scripts/extend_drug_target_stitchv5.py --experimental_only --min_score 700

Example — original combined_score-only mode:
    python scripts/extend_drug_target_stitchv5.py --min_score 700

Then rerun precompute with the generated target CSV:
    python scripts/precompute_embeddings_ablation_mlp.py \
      --targets data/cache/protein_coverage/bio-decagon-targets-expanded-stitch-v5-experimental.csv
"""

from __future__ import annotations

import argparse
import re
import shutil
import urllib.request
from pathlib import Path

import pandas as pd


STITCH_LINKS_URLS = [
    "http://stitch.embl.de/download/protein_chemical.links.detailed.v5.0/"
    "9606.protein_chemical.links.detailed.v5.0.tsv.gz",
    "http://stitch-db.org/download/protein_chemical.links.detailed.v5.0/"
    "9606.protein_chemical.links.detailed.v5.0.tsv.gz",
    "https://github.com/MaastrichtU-IDS/data2services-download/raw/master/"
    "datasets/stitch-sample/9606.protein_chemical.links.detailed.v5.0.tsv.gz",
]

STRING_ALIASES_URLS = [
    "https://stringdb-static.org/download/protein.aliases.v11.5/"
    "9606.protein.aliases.v11.5.txt.gz",
    "https://stringdb-static.org/download/protein.aliases.v11.0/"
    "9606.protein.aliases.v11.0.txt.gz",
]


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Expand Decagon STITCH,Gene targets with STITCH v5 human protein targets."
    )
    parser.add_argument("--root", type=Path, default=root_default, help="Repository root.")
    parser.add_argument(
        "--combo",
        type=Path,
        default=None,
        help="Path to bio-decagon-combo.csv. Defaults to root/data/raw/bio-decagon-combo.csv.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="Path to bio-decagon-targets.csv. Defaults to root/data/raw/bio-decagon-targets.csv.",
    )
    parser.add_argument(
        "--external_dir",
        type=Path,
        default=None,
        help="Directory for downloaded STITCH/STRING files. Defaults to root/data/external/stitch_v5.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory for expanded targets. Defaults to root/data/cache/protein_coverage.",
    )
    parser.add_argument(
        "--min_score",
        type=int,
        default=0,
        help=(
            "Minimum STITCH combined_score to keep (0-1000). "
            "Default is 0 (no combined_score filter). "
            "When used with --experimental_only the combined_score filter is "
            "applied on top of the experimental-channel filter."
        ),
    )
    parser.add_argument(
        "--experimental_only",
        action="store_true",
        help=(
            "Keep only rows where the STITCH 'experimental' channel score > 0. "
            "This is the closest approximation to the Decagon paper's criterion of "
            "'experimentally verified' interactions (ChEMBL, PDB, PDSP Ki Database, "
            "kinase screens). Mutually compatible with --min_experimental and --min_score."
        ),
    )
    parser.add_argument(
        "--min_experimental",
        type=int,
        default=0,
        help=(
            "Minimum STITCH 'experimental' channel score to keep (0-1000). "
            "Only meaningful with --experimental_only. "
            "E.g. --min_experimental 400 keeps medium-confidence experimental evidence. "
            "Default 0 means: keep any row with experimental > 0."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per chunk when reading the large STITCH links file.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Do not download files; require them to already exist.",
    )
    parser.add_argument(
        "--stitch_links_gz",
        type=Path,
        default=None,
        help="Optional explicit path to 9606.protein_chemical.links.detailed.v5.0.tsv.gz.",
    )
    parser.add_argument(
        "--string_aliases_gz",
        type=Path,
        default=None,
        help="Optional explicit path to 9606.protein.aliases.*.txt.gz.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.root.resolve()
    external_dir = args.external_dir or root / "data/external/stitch_v5"
    out_dir = args.out_dir or root / "data/cache/protein_coverage"
    external_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "combo": args.combo or root / "data/raw/bio-decagon-combo.csv",
        "targets": args.targets or root / "data/raw/bio-decagon-targets.csv",
        "external_dir": external_dir,
        "out_dir": out_dir,
        "stitch_links_gz": args.stitch_links_gz
        or external_dir / "9606.protein_chemical.links.detailed.v5.0.tsv.gz",
        "string_aliases_gz": args.string_aliases_gz
        or external_dir / "9606.protein.aliases.v11.5.txt.gz",
    }


def download_if_missing(urls: str | list[str], path: Path) -> None:
    """Download a URL once with browser-like headers and fallback URLs."""
    if path.exists() and path.stat().st_size > 0:
        print(f"Exists: {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return

    if isinstance(urls, str):
        urls = [urls]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/gzip, application/octet-stream, */*",
    }

    last_error: Exception | None = None
    tmp_path = path.with_suffix(path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    for url in urls:
        try:
            print(f"Downloading:\n  {url}\n-> {path}")
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(tmp_path, "wb") as out:
                    shutil.copyfileobj(response, out)
            tmp_path.replace(path)
            print(f"Done: {path} ({path.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as exc:  # noqa: BLE001 - report all fallback failures
            last_error = exc
            tmp_path.unlink(missing_ok=True)
            print(f"  Failed: {type(exc).__name__}: {exc}")

    print("\nAll Python download attempts failed.")
    print("Try one of these shell commands in the project root, then rerun with --skip_download:")
    for url in urls:
        print(f'  curl -L -A "Mozilla/5.0" -o "{path}" "{url}"')
    if last_error is not None:
        raise last_error


def require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {label}: {path}")


def decagon_stitch_to_pubchem_cid(stitch_id: str) -> int:
    """Convert Decagon/STITCH drug ID like CID000003488 to PubChem CID."""
    digits = str(stitch_id).replace("CID", "")
    return int(digits[1:])  # first digit is STITCH flat/stereo flag


def stitch_chemical_to_pubchem_cid(chemical_id: str) -> int | None:
    """Convert STITCH chemical IDs such as CIDm000003488/CIDs000003488 to PubChem CID."""
    if pd.isna(chemical_id):
        return None
    match = re.search(r"CID[ms](\d+)", str(chemical_id))
    if not match:
        return None
    return int(match.group(1))


def coverage_report(combo_df: pd.DataFrame, target_df: pd.DataFrame, label: str) -> dict:
    combo_drugs = set(pd.concat([combo_df["STITCH 1"], combo_df["STITCH 2"]]).astype(str))
    target_drugs = set(target_df["STITCH"].astype(str))
    covered = combo_drugs & target_drugs
    out = {
        "label": label,
        "combo_drugs": len(combo_drugs),
        "target_drugs_total": len(target_drugs),
        "covered_combo_drugs": len(covered),
        "missing_combo_drugs": len(combo_drugs - target_drugs),
        "coverage_pct": 100 * len(covered) / len(combo_drugs),
        "target_rows": len(target_df),
        "unique_genes": target_df["Gene"].astype(str).nunique(),
    }
    print(
        f"{label}: {out['covered_combo_drugs']}/{out['combo_drugs']} combo drugs covered "
        f"({out['coverage_pct']:.1f}%), missing={out['missing_combo_drugs']}, "
        f"rows={out['target_rows']:,}, genes={out['unique_genes']:,}"
    )
    return out


def load_decagon_inputs(combo_path: Path, targets_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading combo: {combo_path}")
    combo = pd.read_csv(combo_path, dtype=str)
    print(f"Loading Decagon targets: {targets_path}")
    decagon_targets = pd.read_csv(targets_path, dtype=str)

    required_combo = {"STITCH 1", "STITCH 2"}
    required_targets = {"STITCH", "Gene"}
    if not required_combo.issubset(combo.columns):
        raise ValueError(f"Combo missing columns {required_combo - set(combo.columns)}")
    if not required_targets.issubset(decagon_targets.columns):
        raise ValueError(f"Targets missing columns {required_targets - set(decagon_targets.columns)}")

    combo["STITCH 1"] = combo["STITCH 1"].astype(str)
    combo["STITCH 2"] = combo["STITCH 2"].astype(str)
    decagon_targets["STITCH"] = decagon_targets["STITCH"].astype(str)
    decagon_targets["Gene"] = decagon_targets["Gene"].astype(str)
    return combo, decagon_targets


def build_combo_drug_map(combo: pd.DataFrame) -> pd.DataFrame:
    combo_drugs = sorted(set(pd.concat([combo["STITCH 1"], combo["STITCH 2"]]).astype(str)))
    return pd.DataFrame(
        {
            "STITCH": combo_drugs,
            "pubchem_cid": [decagon_stitch_to_pubchem_cid(drug) for drug in combo_drugs],
        }
    )


def build_protein_to_entrez(string_aliases_gz: Path) -> pd.DataFrame:
    print(f"Reading STRING aliases: {string_aliases_gz}")
    alias_cols = pd.read_csv(string_aliases_gz, sep="\t", nrows=0, compression="gzip").columns
    print("STRING alias columns:", list(alias_cols))

    aliases = pd.read_csv(
        string_aliases_gz,
        sep="\t",
        compression="gzip",
        dtype=str,
        usecols=lambda c: c in {"#string_protein_id", "string_protein_id", "protein", "alias", "source"},
    )
    protein_col = next(
        c for c in ["#string_protein_id", "string_protein_id", "protein"] if c in aliases.columns
    )
    aliases = aliases.rename(columns={protein_col: "protein"})
    aliases["alias"] = aliases["alias"].astype(str)
    aliases["source"] = aliases.get("source", "").astype(str)

    entrez_aliases = aliases[
        aliases["alias"].str.fullmatch(r"\d+")
        & aliases["source"].str.contains("entrez|geneid|ncbi", case=False, na=False)
    ].copy()

    if entrez_aliases.empty:
        print("No Entrez-like source labels found; falling back to all numeric aliases.")
        entrez_aliases = aliases[aliases["alias"].str.fullmatch(r"\d+")].copy()

    protein_to_entrez = (
        entrez_aliases[["protein", "alias"]].rename(columns={"alias": "Gene"}).drop_duplicates()
    )
    print(f"Protein->Entrez rows: {len(protein_to_entrez):,}")
    print(f"Unique proteins mapped: {protein_to_entrez['protein'].nunique():,}")
    print(f"Unique Entrez genes: {protein_to_entrez['Gene'].nunique():,}")
    return protein_to_entrez


def read_stitch_links(
    stitch_links_gz: Path,
    combo_drug_map: pd.DataFrame,
    min_score: int,
    chunksize: int,
    experimental_only: bool = False,
    min_experimental: int = 0,
) -> pd.DataFrame:
    """Read and filter STITCH detailed links.

    The detailed links file has these per-channel scores (0-1000):
      experimental  – binding assays (ChEMBL, PDB, PDSP Ki, kinase screens)
      database      – manually curated DBs (DrugBank, TTD, KEGG, Reactome …)
      textmining    – co-occurrence / NLP over MEDLINE + PubMed Central
      prediction    – structure-based computational predictions
      combined_score – Bayesian combination of all channels

    With experimental_only=True only rows with experimental > 0 (or >=
    min_experimental if that is > 0) are kept.  This best approximates the
    Decagon paper's "experimentally verified" criterion.
    min_score applies to combined_score and is layered on top.
    """
    combo_pubchem = set(combo_drug_map["pubchem_cid"])
    chunks: list[pd.DataFrame] = []
    rows_seen = 0
    rows_kept = 0

    header = pd.read_csv(stitch_links_gz, sep="\t", nrows=0, compression="gzip").columns.tolist()
    print("STITCH link columns:", header)

    has_experimental = "experimental" in header
    if experimental_only and not has_experimental:
        raise ValueError(
            "--experimental_only requested but 'experimental' column not found in STITCH file. "
            f"Available columns: {header}. "
            "Make sure you are using the *detailed* links file "
            "(9606.protein_chemical.links.detailed.v5.0.tsv.gz), not the plain links file."
        )

    for chunk in pd.read_csv(stitch_links_gz, sep="\t", compression="gzip", dtype=str, chunksize=chunksize):
        rows_seen += len(chunk)
        chunk = chunk.rename(
            columns={
                "#chemical": "chemical",
                "chemical_id": "chemical",
                "protein_id": "protein",
            }
        )
        if "chemical" not in chunk.columns or "protein" not in chunk.columns:
            raise ValueError(f"Expected chemical/protein columns, got: {chunk.columns.tolist()}")

        score_col = "combined_score" if "combined_score" in chunk.columns else None
        if score_col is None:
            possible_scores = [c for c in chunk.columns if c.endswith("score") or c == "score"]
            if not possible_scores:
                raise ValueError(f"No score column found in columns: {chunk.columns.tolist()}")
            score_col = possible_scores[-1]

        # Always carry experimental score through when present, for provenance
        read_cols = ["chemical", "protein", score_col]
        if has_experimental:
            read_cols.append("experimental")

        keep = chunk[read_cols].copy()
        keep = keep.rename(columns={score_col: "combined_score"})
        keep["combined_score"] = pd.to_numeric(keep["combined_score"], errors="coerce")

        if has_experimental:
            keep["experimental"] = pd.to_numeric(keep["experimental"], errors="coerce").fillna(0)

        # --- Apply filters ---
        # 1. Experimental channel (closest to Decagon's "experimentally verified")
        if experimental_only and has_experimental:
            threshold = max(1, min_experimental)  # at minimum experimental > 0
            keep = keep[keep["experimental"] >= threshold]

        # 2. Combined score filter (applied on top, or standalone)
        if min_score > 0:
            keep = keep[keep["combined_score"] >= min_score]

        # 3. Restrict to combo drugs
        keep["pubchem_cid"] = keep["chemical"].map(stitch_chemical_to_pubchem_cid)
        keep = keep[keep["pubchem_cid"].isin(combo_pubchem)]

        if len(keep):
            rows_kept += len(keep)
            chunks.append(keep)

    filter_desc = []
    if experimental_only:
        filter_desc.append(f"experimental>={max(1, min_experimental)}")
    if min_score > 0:
        filter_desc.append(f"combined_score>={min_score}")
    filter_str = " AND ".join(filter_desc) if filter_desc else "no score filter"

    print(f"Rows scanned: {rows_seen:,}")
    print(f"Rows kept ({filter_str}, combo drugs): {rows_kept:,}")

    empty_cols = ["chemical", "protein", "combined_score", "pubchem_cid"]
    if has_experimental:
        empty_cols.append("experimental")
    if not chunks:
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(chunks, ignore_index=True)


def build_stitch_targets(
    stitch_links: pd.DataFrame,
    combo_drug_map: pd.DataFrame,
    protein_to_entrez: pd.DataFrame,
    min_score: int,
    experimental_only: bool = False,
    min_experimental: int = 0,
) -> pd.DataFrame:
    stitch_targets = (
        stitch_links.merge(combo_drug_map, on="pubchem_cid", how="inner")
        .merge(protein_to_entrez, on="protein", how="inner")
    )

    # Base columns always present
    out_cols = ["STITCH", "Gene", "combined_score", "chemical", "protein"]
    # Include experimental score in output when available, for provenance
    if "experimental" in stitch_targets.columns:
        out_cols.append("experimental")

    out = (
        stitch_targets[out_cols]
        .assign(
            source="stitch_v5",
            min_stitch_score=min_score,
            experimental_only=experimental_only,
            min_experimental_score=min_experimental if experimental_only else pd.NA,
        )
        .drop_duplicates(["STITCH", "Gene"])
        .sort_values(["STITCH", "Gene"])
        .reset_index(drop=True)
    )
    print(f"STITCH-derived target rows: {len(out):,}")
    print(f"Drugs covered by STITCH-derived targets: {out['STITCH'].nunique():,}")
    print(f"Unique Entrez genes from STITCH: {out['Gene'].nunique():,}")
    if "experimental" in out.columns:
        n_exp = (out["experimental"] > 0).sum()
        print(f"Rows with experimental evidence > 0: {n_exp:,} ({100*n_exp/len(out):.1f}%)")
    return out


def merge_targets(decagon_targets: pd.DataFrame, stitch_targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decagon_with_source = decagon_targets[["STITCH", "Gene"]].copy()
    decagon_with_source["source"] = "decagon"
    decagon_with_source["combined_score"] = pd.NA
    decagon_with_source["experimental"] = pd.NA
    decagon_with_source["chemical"] = pd.NA
    decagon_with_source["protein"] = pd.NA
    decagon_with_source["min_stitch_score"] = pd.NA
    decagon_with_source["experimental_only"] = pd.NA
    decagon_with_source["min_experimental_score"] = pd.NA

    # Carry all provenance columns from stitch_targets (some may not exist depending on flags)
    stitch_cols = ["STITCH", "Gene", "source", "combined_score", "chemical", "protein",
                   "min_stitch_score", "experimental_only", "min_experimental_score"]
    if "experimental" in stitch_targets.columns:
        stitch_cols.insert(4, "experimental")  # keep column order sensible

    expanded_with_source = (
        pd.concat(
            [
                decagon_with_source,
                stitch_targets[[c for c in stitch_cols if c in stitch_targets.columns]],
            ],
            ignore_index=True,
        )
        .assign(STITCH=lambda d: d["STITCH"].astype(str), Gene=lambda d: d["Gene"].astype(str))
        .drop_duplicates(["STITCH", "Gene"])
        .sort_values(["STITCH", "Gene"])
        .reset_index(drop=True)
    )
    expanded_targets = expanded_with_source[["STITCH", "Gene"]].copy()
    return expanded_targets, expanded_with_source


def save_outputs(
    out_dir: Path,
    min_score: int,
    expanded_targets: pd.DataFrame,
    expanded_with_source: pd.DataFrame,
    stitch_targets: pd.DataFrame,
    coverage_df: pd.DataFrame,
    experimental_only: bool = False,
    min_experimental: int = 0,
) -> dict[str, Path]:
    # Build a descriptive tag so filenames reflect exactly what filters were used
    tag_parts = []
    if experimental_only:
        exp_thresh = max(1, min_experimental)
        tag_parts.append(f"experimental{exp_thresh}")
    if min_score > 0:
        tag_parts.append(f"score{min_score}")
    if not tag_parts:
        tag_parts.append("all")
    score_tag = "-".join(tag_parts)

    paths = {
        "expanded_two_col": out_dir / f"bio-decagon-targets-expanded-stitch-v5-{score_tag}.csv",
        "expanded_with_source": out_dir / f"bio-decagon-targets-expanded-stitch-v5-{score_tag}-with-source.csv",
        "stitch_only": out_dir / f"stitch-v5-derived-targets-{score_tag}.csv",
        "coverage": out_dir / f"target-coverage-decagon-vs-stitch-v5-{score_tag}.csv",
    }
    expanded_targets.to_csv(paths["expanded_two_col"], index=False)
    expanded_with_source.to_csv(paths["expanded_with_source"], index=False)
    stitch_targets.to_csv(paths["stitch_only"], index=False)
    coverage_df.to_csv(paths["coverage"], index=False)
    print("Saved:")
    for path in paths.values():
        print(f"  {path}")
    return paths


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    print("Repository root:", paths["root"])
    print("Combo:", paths["combo"])
    print("Decagon targets:", paths["targets"])
    print("External data dir:", paths["external_dir"])
    print("Output dir:", paths["out_dir"])
    print("Minimum STITCH score:", args.min_score)
    print("Experimental only:", args.experimental_only)
    if args.experimental_only:
        print("Minimum experimental score:", max(1, args.min_experimental))

    if not args.skip_download:
        download_if_missing(STITCH_LINKS_URLS, paths["stitch_links_gz"])
        download_if_missing(STRING_ALIASES_URLS, paths["string_aliases_gz"])
    else:
        print("Skipping downloads; using existing files.")

    require_file(paths["stitch_links_gz"], "STITCH links gz")
    require_file(paths["string_aliases_gz"], "STRING aliases gz")

    combo, decagon_targets = load_decagon_inputs(paths["combo"], paths["targets"])
    combo_drug_map = build_combo_drug_map(combo)

    current_report = coverage_report(combo, decagon_targets, "Decagon original")
    missing_before = sorted(set(combo_drug_map["STITCH"]) - set(decagon_targets["STITCH"]))
    print("Example missing drugs:", missing_before[:10])

    protein_to_entrez = build_protein_to_entrez(paths["string_aliases_gz"])
    stitch_links = read_stitch_links(
        paths["stitch_links_gz"],
        combo_drug_map,
        min_score=args.min_score,
        chunksize=args.chunksize,
        experimental_only=args.experimental_only,
        min_experimental=args.min_experimental,
    )
    stitch_targets = build_stitch_targets(
        stitch_links,
        combo_drug_map,
        protein_to_entrez,
        min_score=args.min_score,
        experimental_only=args.experimental_only,
        min_experimental=args.min_experimental,
    )
    expanded_targets, expanded_with_source = merge_targets(decagon_targets, stitch_targets)

    label_parts = []
    if args.experimental_only:
        label_parts.append(f"experimental>={max(1, args.min_experimental)}")
    if args.min_score > 0:
        label_parts.append(f"combined_score>={args.min_score}")
    label = "Decagon + STITCH v5 " + (" AND ".join(label_parts) if label_parts else "all")
    expanded_report = coverage_report(combo, expanded_targets, label)
    coverage_df = pd.DataFrame([current_report, expanded_report])
    save_outputs(
        paths["out_dir"],
        args.min_score,
        expanded_targets,
        expanded_with_source,
        stitch_targets,
        coverage_df,
        experimental_only=args.experimental_only,
        min_experimental=args.min_experimental,
    )

    tag_parts = []
    if args.experimental_only:
        tag_parts.append(f"experimental{max(1, args.min_experimental)}")
    if args.min_score > 0:
        tag_parts.append(f"score{args.min_score}")
    if not tag_parts:
        tag_parts.append("all")
    out_tag = "-".join(tag_parts)
    print("\nUse this expanded target file with precompute:")
    print("  --targets " + str(paths["out_dir"] / f"bio-decagon-targets-expanded-stitch-v5-{out_tag}.csv"))


if __name__ == "__main__":
    main()
