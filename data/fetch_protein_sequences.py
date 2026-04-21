"""
fetch_protein_sequences.py

Fetch canonical amino acid sequences for all Decagon proteins from UniProt
using the UniProt ID Mapping service (NCBI GeneID → UniProtKB).

Workflow:
    1. Collect unique Entrez gene IDs from data/raw/bio-decagon-targets.csv.
    2. Submit batches to the UniProt /idmapping/run endpoint.
    3. Poll until the job is complete, then page through results.
    4. Prefer Swiss-Prot (reviewed) entries; fall back to TrEMBL (unreviewed).
    5. Save progress after every batch — safe to interrupt and resume.

Outputs:
    data/raw/protein_sequences.json    {"gene_id_str": "amino_acid_sequence", ...}

Usage:
    python data/fetch_protein_sequences.py
    python data/fetch_protein_sequences.py --raw_dir data/raw --batch_size 2000
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

BATCH_SIZE  = 2000   # UniProt ID Mapping handles up to ~100k per job
POLL_INTERVAL = 3    # seconds between status polls
MAX_POLLS   = 60     # give up after ~3 minutes per batch
SLEEP_BATCH = 1.0    # seconds between batch submissions
PAGE_SIZE   = 500    # results per page when downloading

IDMAP_RUN     = "https://rest.uniprot.org/idmapping/run"
IDMAP_STATUS  = "https://rest.uniprot.org/idmapping/status/{jobId}"
IDMAP_RESULTS = "https://rest.uniprot.org/idmapping/uniprotkb/results/{jobId}"


# ── Helpers ───────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


def submit_job(gene_ids: list[str]) -> str | None:
    """Submit an ID mapping job; returns jobId or None on failure."""
    r = SESSION.post(IDMAP_RUN, data={
        "from": "GeneID",
        "to":   "UniProtKB",
        "ids":  ",".join(gene_ids),
    }, timeout=30)
    if r.status_code == 200:
        return r.json().get("jobId")
    print(f"  Job submission failed: HTTP {r.status_code} — {r.text[:200]}")
    return None


def poll_job(job_id: str) -> bool:
    """Poll until job is finished. Returns True on success."""
    url = IDMAP_STATUS.format(jobId=job_id)
    for _ in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        r = SESSION.get(url, timeout=15)
        if r.status_code != 200:
            continue
        body = r.json()
        status = body.get("jobStatus", "")
        if status == "FINISHED" or "results" in body or "failedIds" in body:
            return True
        if status == "ERROR":
            print(f"  Job {job_id} errored: {body}")
            return False
    print(f"  Job {job_id} timed out after {MAX_POLLS * POLL_INTERVAL}s")
    return False


def download_results(job_id: str) -> dict[str, str]:
    """
    Page through all results for a finished job.

    Returns {gene_id: sequence}, preferring Swiss-Prot over TrEMBL entries.
    When multiple UniProt entries map to the same gene ID, the reviewed
    (Swiss-Prot) entry is kept; ties broken by longest sequence.
    """
    url = IDMAP_RESULTS.format(jobId=job_id)
    params = {"fields": "sequence,reviewed", "format": "json", "size": PAGE_SIZE}

    # {gene_id: (is_reviewed, seq_len, sequence)}
    best: dict[str, tuple[bool, int, str]] = {}

    while url:
        r = SESSION.get(url, params=params, timeout=60)
        params = None  # only use params on first request; subsequent URLs are full
        if r.status_code != 200:
            print(f"  Results fetch failed: HTTP {r.status_code}")
            break

        body = r.json()
        for entry in body.get("results", []):
            # The "from" field holds the original GeneID we submitted
            gene_id = str(entry.get("from", "")).strip()
            seq     = (entry.get("to", {}).get("sequence") or {}).get("value")
            if not gene_id or not seq:
                continue

            reviewed = (
                entry.get("to", {}).get("entryType", "")
                == "UniProtKB reviewed (Swiss-Prot)"
            )
            existing = best.get(gene_id)
            if existing is None \
               or (reviewed and not existing[0]) \
               or (reviewed == existing[0] and len(seq) > existing[1]):
                best[gene_id] = (reviewed, len(seq), seq)

        # Follow pagination via Link header
        link = r.headers.get("Link", "")
        if 'rel="next"' in link:
            url = link.split("<")[1].split(">")[0]
        else:
            url = None

    return {gid: v[2] for gid, v in best.items()}


def check_connectivity() -> bool:
    """Verify UniProt ID Mapping is reachable using TP53 (GeneID 7157)."""
    print("Testing UniProt ID Mapping (TP53 GeneID 7157) ...")
    job_id = submit_job(["7157"])
    if not job_id:
        print("  FAILED — could not submit job.")
        return False
    if not poll_job(job_id):
        print("  FAILED — job did not complete.")
        return False
    results = download_results(job_id)
    if results:
        seq = list(results.values())[0]
        print(f"  OK — TP53 sequence starts: {seq[:40]}...")
        return True
    print("  FAILED — no results returned.")
    return False


def load_gene_ids(raw_dir: Path) -> list[str]:
    """Read unique Entrez gene IDs from bio-decagon-targets.csv."""
    targets_file = raw_dir / "bio-decagon-targets.csv"
    if not targets_file.exists():
        raise FileNotFoundError(
            f"{targets_file} not found. "
            "Run: bash data/raw/download_unpack.sh"
        )
    gene_ids: set[str] = set()
    with open(targets_file, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = row.get("Gene", "").strip()
            if gid:
                gene_ids.add(gid)
    return sorted(gene_ids, key=lambda x: int(x))


def save_output(sequences: dict, out_path: Path):
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
        save_output(sequences, out_path)
        return

    if not check_connectivity():
        return

    not_found: list[str] = []

    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i : i + args.batch_size]
        done  = min(i + args.batch_size, len(remaining))
        print(f"\nBatch {i // args.batch_size + 1}: "
              f"genes {i + 1}–{done} of {len(remaining)} ...")

        job_id = submit_job(batch)
        if not job_id:
            print("  Skipping batch — submission failed.")
            not_found.extend(batch)
            continue

        print(f"  Job ID: {job_id}  (polling ...)")
        if not poll_job(job_id):
            not_found.extend(batch)
            continue

        result = download_results(job_id)
        for gid in batch:
            if gid in result:
                sequences[gid] = result[gid]
            else:
                not_found.append(gid)

        print(f"  Fetched: {len(result)}/{len(batch)}  "
              f"(total so far: {len(sequences)}, not found: {len(not_found)})")

        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(sequences, f)

        time.sleep(SLEEP_BATCH)

    if not_found:
        print(f"\n{len(not_found)} gene IDs had no UniProt entry "
              "(obsolete IDs or non-coding genes):")
        for g in not_found[:20]:
            print(f"  GeneID {g}")
        if len(not_found) > 20:
            print(f"  ... and {len(not_found) - 20} more")

    save_output(sequences, out_path)

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    total = len(all_gene_ids)
    found = len(sequences)
    print(f"\nDone: {found}/{total} gene IDs mapped to sequences "
          f"({found / total * 100:.1f}% coverage)")


if __name__ == "__main__":
    main()
