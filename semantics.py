import os
import json
import glob
import time
import requests
import urllib.parse
from pathlib import Path

INPUT_DIR   = "data/processed_graphs/"
RESULTS_DIR = "results/"
RESULTS_OUT = "results/phantom_analysis.json"

RATE_LIMIT_SLEEP = 3


def check_semantic_scholar(raw_text):
    """
    Pings Semantic Scholar API.
    Returns True if found, False if phantom, None if API error (treat as unknown).
    """
    clean_query = urllib.parse.quote(raw_text[:100].strip())
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={clean_query}&limit=1"

    try:
        headers = {"User-Agent": "AuthentiGraph-Research-Script/1.0"}
        resp = requests.get(url, headers=headers, timeout=5)

        if resp.status_code == 429:
            print("    [!] Rate limited. Sleeping 30s...")
            time.sleep(30)
            return check_semantic_scholar(raw_text)

        if resp.status_code != 200:
            return None

        data = resp.json()
        return len(data.get("data", [])) > 0

    except Exception as e:
        print(f"    [!] API Error: {e}")
        return None


def atomic_save(filepath, data):
    tmp_path = filepath + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, filepath)


def process_file(filepath):
    """
    Verifies all unverified citations in one paper graph.
    Returns (total, phantom_count, skipped_count).
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    bibliography = data.get("paper_DOM", {}).get("bibliography", [])
    if not bibliography:
        return 0, 0, 0

    total    = len(bibliography)
    phantoms = 0
    skipped  = 0
    changed  = False

    for citation in bibliography:
        current = citation.get("semantic_scholar_verified")

        if current is not None:
            if current is False:
                phantoms += 1
            skipped += 1
            continue

        raw_text = citation.get("raw_text", "")
        if len(raw_text.strip()) < 10:
            citation["semantic_scholar_verified"] = None
            continue

        is_real = check_semantic_scholar(raw_text)
        citation["semantic_scholar_verified"] = is_real
        changed = True

        if is_real is False:
            phantoms += 1

        time.sleep(RATE_LIMIT_SLEEP)

    verified_total = sum(
        1 for c in bibliography if c.get("semantic_scholar_verified") is not None
    )
    phantom_count = sum(
        1 for c in bibliography if c.get("semantic_scholar_verified") is False
    )
    phantom_rate = round(phantom_count / verified_total, 4) if verified_total > 0 else 0.0

    data["paper_metadata"]["phantom_rate"]      = phantom_rate
    data["paper_metadata"]["total_citations"]   = total
    data["paper_metadata"]["phantom_count"]     = phantom_count
    changed = True

    if changed:
        atomic_save(filepath, data)

    return total, phantoms, skipped


def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    print(f"Found {len(json_files)} papers for citation verification.\n")

    all_results   = []
    grand_total   = 0
    grand_phantom = 0
    grand_skipped = 0

    for idx, filepath in enumerate(json_files):
        paper_id = Path(filepath).stem.replace("_base_graph", "")
        print(f"[{idx+1}/{len(json_files)}] {paper_id}", end=" ... ", flush=True)

        try:
            total, phantom, skipped = process_file(filepath)
            grand_total   += total
            grand_phantom += phantom
            grand_skipped += skipped

            rate = round(phantom / total, 4) if total > 0 else 0.0
            print(f"{total} citations | {phantom} phantom ({rate:.1%})")

            all_results.append({
                "paper_id":     paper_id,
                "total":        total,
                "phantom":      phantom,
                "phantom_rate": rate,
            })

        except Exception as e:
            print(f"ERROR: {e}")

    overall_rate = round(grand_phantom / grand_total, 4) if grand_total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Total citations checked : {grand_total}")
    print(f"Already verified (skip) : {grand_skipped}")
    print(f"Phantom citations found : {grand_phantom}")
    print(f"Overall phantom rate    : {overall_rate:.2%}")

    summary = {
        "total_citations": grand_total,
        "phantom_count":   grand_phantom,
        "phantom_rate":    overall_rate,
        "per_paper":       all_results,
    }
    with open(RESULTS_OUT, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_OUT}")


if __name__ == "__main__":
    main()