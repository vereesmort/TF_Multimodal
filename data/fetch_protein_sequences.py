"""
fetch_protein_sequences.py

Fetch canonical amino acid sequences for all Decagon proteins from UniProt.

Gene IDs are read from data/raw/bio-decagon-targets.csv (the 'Gene' column
contains Entrez/NCBI gene IDs).  UniProt is queried via the REST API using
the NCBI GeneID cross-reference to map gene IDs to UniProtKB entries and
retrieve the canonical sequence.

Strategy:
    1. Collect all unique Entrez gene IDs from bio-decagon-targets.csv.
    2. Batch-query UniProt (up to 500 IDs per request) using the
       /search endpoint with the 'xref_geneid' field.
    3. For each hit, record the canonical sequence for the reviewed
       (Swiss-Prot) entry if available, otherwise unreviewed (TrEMBL).
    4. Save progress after every batch — safe to interrupt and resume.

Outputs:
    data/raw/protein_sequences.json    {entrez_gene_id_str: amino_acid_sequence}

Usage:
    python data/fetch_protein_sequences.py
    python data/fetch_protein_sequences.py --raw_dir data/raw --batch_size 500
"""

import argparse
import csv
import json
import time
from pathlib import Path

import requests

RAW           = Path("data/raw")
DATA          = Path("data")
PROGRESS_FILE = DATA / "protein_sequences_progress.json"

BATCH_SIZE  = 500    # UniProt handles large batches well
SLEEP_BATCH = 0.5    # seconds between batch requests
MAX_RETRIES = 3
RETRY_WAIT  = 5      # seconds, doubled on each retry

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
FIELDS         = "gene_names,sequence,reviewed,organism_name"


# ── Helpers ───────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
})


def make_request(url: str, params: dict) -> dict | None:
    """GET with exponential backoff on rate-limit."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503) and attempt < MAX_RETRIES:
                wait = RETRY_WAIT * (2 ** attempt)
                print(f"    Rate limit (HTTP {r.status_code}) — waiting {wait}s ...")
                time.sleep(wait)
            else:
                print(f"    HTTP {r.status_code} for batch — skipping.")
                return None
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
            else:
                print(f"    Request error: {e}")
                return None
    return None


def fetch_batch(gene_ids: list[str]) -> dict[str, str]:
    """
    Query UniProt for a batch of Entrez gene IDs.

    Returns {gene_id_str: sequence} for all hits.  When multiple UniProt
    entries map to the same gene, the reviewed (Swiss-Prot) entry is
    preferred; ties are broken by sequence length (longest wins).
    """
    query = " OR ".join(f"xref_geneid:{g}" for g in gene_ids)
    params = {
        "query":    query,
        "fields":   FIELDS,
        "format":   "json",
        "size":     len(gene_ids) * 3,  # allow multiple hits per gene
    }

    data = make_request(UNIPROT_SEARCH, params)
    if not data:
        return {}

    results: dict[str, tuple[bool, int, str]] = {}
    # {gene_id: (is_reviewed, seq_len, sequence)}

    for entry in data.get("results", []):
        seq = (entry.get("sequence") or {}).get("value")
        if not seq:
            continue

        reviewed = entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"

        # Extract gene IDs from cross-references
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") != "GeneID":
                continue
            gid = xref.get("id", "").strip()
            if gid not in gene_ids:
                continue
            existing = results.get(gid)
            if existing is None or (reviewed and not existing[0]) or \
               (reviewed == existing[0] and len(seq) > existing[1]):
                results[gid] = (reviewed, len(seq), seq)

    return {gid: v[2] for gid, v in results.items()}


def check_connectivity() -> bool:
    """Verify UniProt is reachable with a single known gene (TP53 = 7157)."""
    print("Testing UniProt connectivity (TP53 gene ID 7157) ...")
    result = fetch_batch(["7157"])
    if result:
        seq_preview = list(result.values())[0][:40]
        print(f"  OK — accessible. TP53 sequence starts: {seq_preview}...")
        return True
    print("  FAILED — UniProt not reachable.")
    print("  Try: curl 'https://rest.uniprot.org/uniprotkb/search?query=xref_geneid:7157&fields=sequence&format=json'")
    return False


def load_gene_ids(raw_dir: Path) -> list[str]:
    """Read unique Entrez gene IDs from bio-decagon-targets.csv."""
    targets_file = raw_dir / "bio-decagon-targets.csv"
    if not targets_file.exists():
        raise FileNotFoundError(
            f"{targets_file} not found. "
            "Run: bash data/raw/download_unpack.sh"
        )
    gene_ids = set()
    with open(targets_file, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = row.get("Gene", "").strip()
            if gid:
                gene_ids.add(gid)
    return sorted(gene_ids, key=lambda x: int(x))


def save_outputs(sequences: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sequences, f, indent=2)
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",    type=str, default="data/raw")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    raw_dir  = Path(args.raw_dir)
    out_path = raw_dir / "protein_sequences.json"

    DATA.mkdir(exist_ok=True)

    # Resume from interrupted run
    sequences: dict[str, str] = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            sequences = json.load(f)
        print(f"Resuming: {len(sequences)} sequences already fetched.")

    all_gene_ids = load_gene_ids(raw_dir)
    remaining    = [g for g in all_gene_ids if g not in sequences]

    print(f"Total gene IDs in targets file : {len(all_gene_ids)}")
    print(f"Already fetched                : {len(sequences)}")
    print(f"To fetch                       : {len(remaining)}")

    if not remaining:
        print("All sequences fetched — nothing to do.")
        save_outputs(sequences, out_path)
        return

    if not check_connectivity():
        return

    print(f"\nFetching sequences in batches of {args.batch_size} ...")
    failed_ids = []

    for i in range(0, len(remaining), args.batch_size):
        batch  = remaining[i : i + args.batch_size]
        result = fetch_batch(batch)

        for gid in batch:
            if gid in result:
                sequences[gid] = result[gid]
            else:
                failed_ids.append(gid)

        done = min(i + args.batch_size, len(remaining))
        print(f"  {done}/{len(remaining)}  "
              f"(fetched: {len(sequences)}, not found so far: {len(failed_ids)})")

        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(sequences, f)

        time.sleep(args.batch_size / 1000 + SLEEP_BATCH)

    if failed_ids:
        print(f"\n{len(failed_ids)} gene IDs had no UniProt entry "
              f"(obsolete IDs or genes without a protein product):")
        for g in failed_ids[:20]:
            print(f"  GeneID {g}")
        if len(failed_ids) > 20:
            print(f"  ... and {len(failed_ids) - 20} more")

    save_outputs(sequences, out_path)

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    total = len(all_gene_ids)
    found = len(sequences)
    print(f"\nResult: {found}/{total} gene IDs have sequences "
          f"({found / total * 100:.1f}% coverage)")


if __name__ == "__main__":
    main()
