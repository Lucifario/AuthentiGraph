import os
import json
import glob
import time
import re
import random
import requests
import openreview
from pathlib import Path
from dom_parser_v2 import process_pdf_with_grobid, parse_tei_xml
from dotenv import load_dotenv
load_dotenv()

PEERREAD_DIR          = "data/PeerRead/data/"
OPENREVIEW_DIR        = "data/openreview_pdfs/"
OUTPUT_DIR            = "data/processed_graphs/"
FAILED_LOG            = "data/failed_papers.log"
TARGET_PAPER_COUNT    = 1500
MIN_SENTENCES_PER_REVIEW = 3
MIN_REVIEWS_PER_PAPER    = 2
LIANG_ALPHA_THRESHOLD    = 0.95

OPENREVIEW_SEARCH_QUERIES = [
    {"year": "2018", "venue_tag": "ICLR 2018 Poster",    "api": "v1"},
    {"year": "2018", "venue_tag": "ICLR 2018 Oral",      "api": "v1"},
    {"year": "2019", "venue_tag": "ICLR 2019 Poster",    "api": "v1"},
    {"year": "2019", "venue_tag": "ICLR 2019 Oral",      "api": "v1"},
    {"year": "2020", "venue_tag": "ICLR 2020 Poster",    "api": "v1"},
    {"year": "2020", "venue_tag": "ICLR 2020 Spotlight", "api": "v1"},
    {"year": "2020", "venue_tag": "ICLR 2020 Oral",      "api": "v1"},
    {"year": "2021", "venue_tag": "ICLR 2021 Poster",    "api": "v2"},
    {"year": "2021", "venue_tag": "ICLR 2021 Spotlight", "api": "v2"},
    {"year": "2021", "venue_tag": "ICLR 2021 Oral",      "api": "v2"},
]

def compute_liang_alpha(text):
    """
    Fast statistical proxy for LLM influence via sentence length variance.
    Human writing = HIGH variance. LLM writing = LOW variance (uniform sentences).
    Returns 0.0 (clean) to 1.0 (likely AI-influenced).
    Replace with GPT-2 perplexity scorer in a post-processing Colab pass
    before final training.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                 if len(s.strip()) > 5]
    if len(sentences) < 5:
        return 0.0

    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.0

    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    burstiness = (variance ** 0.5) / mean_len

    alpha = max(0.0, min(1.0, (0.65 - burstiness) / 0.35))
    return round(alpha, 4)

def naive_sentence_split(text):
    """
    Sentence splitter that respects common academic abbreviations.
    Swap for nltk.sent_tokenize() in production if needed.
    """
    abbreviations = (
        r'(?<!\bFig)(?<!\bvs)(?<!\bet al)(?<!\be\.g)'
        r'(?<!\bi\.e)(?<!\bDr)(?<!\bProf)(?<!\bEq)'
    )
    pattern = abbreviations + r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def already_processed(paper_id):
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{paper_id}_base_graph.json"))


def log_failure(paper_id, reason):
    with open(FAILED_LOG, 'a') as f:
        f.write(f"{paper_id}\t{reason}\n")


def build_and_save(paper_id, source, year, decision,
                   human_reviews, pdf_path, total_reviews_available=None):
    """
    Shared save step for both PeerRead and OpenReview paths.
    Runs GROBID, assembles the master graph, writes to disk.
    Deletes the PDF afterwards for OpenReview papers to save space.
    """
    try:
        xml_content = process_pdf_with_grobid(pdf_path)
        sections, bibliography, hetero_edges = parse_tei_xml(xml_content)
    except Exception as e:
        log_failure(paper_id, f"GROBID Crash: {str(e)}")
        print(f"  -> GROBID crashed. Skipping.")
        return False

    if not sections:
        log_failure(paper_id, "GROBID returned no sections")
        print("  -> GROBID returned no sections. Skipping.")
        return False

    master_graph = {
        "paper_metadata": {
            "paper_id":                paper_id,
            "source":                  source,
            "year":                    year,
            "decision":                decision,
            "total_reviews_available": total_reviews_available or len(human_reviews),
        },
        "paper_DOM": {
            "sections":     sections,
            "bibliography": bibliography,
        },
        "human_ground_truth": human_reviews,
        "adversarial_reviews": [],
        "heterogeneous_edges": hetero_edges,
    }

    out_file = os.path.join(OUTPUT_DIR, f"{paper_id}_base_graph.json")
    with open(out_file, 'w') as f:
        json.dump(master_graph, f, indent=4)

    if source.startswith("OpenReview") and os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"  -> PDF deleted after parsing.")

    print(f"  -> Saved. {len(sections)} sections | "
          f"{len(bibliography)} citations | {len(hetero_edges)} edges.")
    return True

def load_peerread_human_reviews(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    human_reviews = []
    for idx, rev in enumerate(data.get('reviews', [])):
        raw_text = rev.get('comments', '')

        sentences = naive_sentence_split(raw_text)
        if len(sentences) < MIN_SENTENCES_PER_REVIEW:
            print(f"    -> Review {idx} dropped "
                  f"(Too short: {len(sentences)} sentences)")
            continue

        alpha = compute_liang_alpha(raw_text)
        if alpha > LIANG_ALPHA_THRESHOLD:
            print(f"    -> Review {idx} dropped "
                  f"(Burstiness Alpha: {alpha:.2f} > {LIANG_ALPHA_THRESHOLD})")
            continue

        review_obj = {
            "review_id":         f"human_rev_{idx}",
            "type":              "Human",
            "liang_alpha_score": alpha,
            "aspect_scores": {
                "originality": rev.get('ORIGINALITY', "Not Discussed"),
                "substance":   rev.get('SUBSTANCE',   "Not Discussed"),
                "clarity":     rev.get('CLARITY',     "Not Discussed"),
                "impact":      rev.get('IMPACT',      "Not Discussed"),
            },
            "review_sentences": [
                {
                    "sentence_id":  f"human_rev_{idx}_s_{s_idx}",
                    "text":         sentence,
                    "author_label": "HUMAN",
                }
                for s_idx, sentence in enumerate(sentences)
            ]
        }
        human_reviews.append(review_obj)

    return human_reviews, data.get('accepted', False)


def get_pdf_path_from_json(json_path):
    """Derives PDF path from PeerRead JSON path using pathlib (not string replace)."""
    p = Path(json_path)
    if p.parent.name != 'reviews':
        return None
    return str(p.parent.parent / 'pdfs' / p.with_suffix('.pdf').name)


def ingest_peerread(counter, target):
    review_jsons = glob.glob(f"{PEERREAD_DIR}/**/*.json", recursive=True)
    review_jsons = [p for p in review_jsons if Path(p).parent.name == 'reviews']
    print(f"PeerRead: {len(review_jsons)} JSONs found.")

    for json_path in review_jsons:
        if counter[0] >= target:
            break

        paper_id = Path(json_path).stem

        if already_processed(paper_id):
            counter[0] += 1
            continue

        pdf_path = get_pdf_path_from_json(json_path)
        if not pdf_path or not os.path.exists(pdf_path):
            log_failure(paper_id, "PDF missing")
            continue

        print(f"\n[{counter[0]+1}/{target}] PeerRead | {paper_id}")
        try:
            human_reviews, is_accepted = load_peerread_human_reviews(json_path)
        except Exception as e:
            print(f"  -> FAILED: {e}")
            log_failure(paper_id, str(e))
            continue

        if not human_reviews:
            log_failure(paper_id, "No human reviews passed filter")
            continue

        ok = build_and_save(
            paper_id                = paper_id,
            source                  = "PeerRead",
            year                    = "Pre-2022",
            decision                = "Accepted" if is_accepted else "Rejected",
            human_reviews           = human_reviews,
            pdf_path                = pdf_path,
            total_reviews_available = len(human_reviews),
        )
        if ok:
            counter[0] += 1

def get_or_client(api_version):
    """Returns an openreview-py guest client for the correct API version."""
    if api_version == "v1":
        return openreview.Client(baseurl="https://api.openreview.net")
    else:
        return openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")


def fetch_openreview_submissions(query_config, max_papers=600):
    venue_tag = query_config['venue_tag']
    
    try:
        if query_config['api'] == 'v2':
            client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
        else:
            client = openreview.Client(baseurl='https://api.openreview.net')
        notes = client.get_all_notes(content={"venue": venue_tag})
        return notes[:max_papers]
        
    except Exception as e:
        print(f"  [Submission fetch error]: {e}")
        return []


def parse_openreview_reviews(notes):
    """
    Parses a list of openreview-py Note objects into our review schema.
    Applies sentence length and Liang alpha filters.
    """
    human_reviews = []
    for idx, note in enumerate(notes):
        invitations = note.invitations or []
        if not any("Official_Review" in inv for inv in invitations):
            continue

        content = note.content or {}

        def get_val(field):
            v = content.get(field, {})
            return v.get("value", v) if isinstance(v, dict) else v

        text_parts = []
        for field in ["summary", "strengths", "weaknesses", "questions",
                      "limitations", "review", "comment"]:
            val = get_val(field)
            if isinstance(val, str) and len(val) > 20:
                text_parts.append(val)

        full_text = "\n\n".join(text_parts)
        if not full_text.strip():
            continue

        sentences = naive_sentence_split(full_text)
        if len(sentences) < MIN_SENTENCES_PER_REVIEW:
            print(f"    -> Review {idx} dropped "
                  f"(Too short: {len(sentences)} sentences)")
            continue

        alpha = compute_liang_alpha(full_text)
        if alpha > LIANG_ALPHA_THRESHOLD:
            print(f"    -> Review {idx} dropped "
                  f"(Burstiness Alpha: {alpha:.2f} > {LIANG_ALPHA_THRESHOLD})")
            continue

        review_obj = {
            "review_id":         f"human_rev_{idx}",
            "type":              "Human",
            "liang_alpha_score": alpha,
            "aspect_scores": {
                "originality": get_val("originality")  or "Not Discussed",
                "substance":   get_val("soundness")    or "Not Discussed",
                "clarity":     get_val("presentation") or "Not Discussed",
                "impact":      get_val("contribution") or "Not Discussed",
            },
            "review_sentences": [
                {
                    "sentence_id":  f"human_rev_{idx}_s_{s_idx}",
                    "text":         sentence,
                    "author_label": "HUMAN",
                }
                for s_idx, sentence in enumerate(sentences)
            ]
        }
        human_reviews.append(review_obj)

    return human_reviews


def fetch_reviews_for_paper(forum_id, api_version):
    """Fetches and parses all Official Reviews for a single paper."""
    try:
        client = get_or_client(api_version)
        notes = client.get_all_notes(forum=forum_id)
        return parse_openreview_reviews(notes)
    except Exception as e:
        print(f"  [Review fetch error {forum_id}]: {e}")
        return []


def download_pdf(url, dest_path):
    if os.path.exists(dest_path):
        return True
    try:
        resp = requests.get(url, stream=True, timeout=20,
                            headers={"User-Agent": "AuthentiGraph/1.0"})
        if resp.status_code != 200:
            return False
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception:
        return False

def ingest_openreview(counter, target):
    os.makedirs(OPENREVIEW_DIR, exist_ok=True)
    print("\nInitializing OpenReview V1 Client...")
    client = openreview.Client(
        baseurl='https://api.openreview.net', 
        username=os.getenv('OPENREVIEW_USERNAME'),
        password=os.getenv('OPENREVIEW_PASSWORD')
    )

    venues = [
        ("2018", "ICLR.cc/2018/Conference"),
        ("2019", "ICLR.cc/2019/Conference"),
        ("2020", "ICLR.cc/2020/Conference"),
        ("2021", "ICLR.cc/2021/Conference") 
    ]

    for year, venue_id in venues:
        if counter[0] >= target: break
        print(f"\n=== OpenReview: {venue_id} ===")
        invitation = f'{venue_id}/-/Blind_Submission'
        try:
            submissions = client.get_all_notes(invitation=invitation, details='directReplies')
            print(f"  Successfully fetched {len(submissions)} submissions.")
        except Exception as e:
            print(f"  [Submission fetch error]: {e}")
            continue
        for sub in submissions:
            if counter[0] >= target: break
            paper_id = sub.id
            if already_processed(paper_id):
                counter[0] += 1
                continue
            print(f"\n[{counter[0]+1}/{target}] OpenReview | {paper_id}")
            direct_replies = sub.details.get('directReplies', [])
            raw_reviews = [reply for reply in direct_replies if 'Official_Review' in reply['invitation']]
            
            if not raw_reviews:
                print("    -> Skipped: 0 Official_Reviews found in payload.")
                log_failure(paper_id, "No Official_Reviews found in directReplies")
                continue
            decision = "Unknown"
            for reply in direct_replies:
                if 'Decision' in reply['invitation']:
                    dec_val = reply.get('content', {}).get('decision', '')
                    decision = "Accepted" if "Accept" in str(dec_val) else "Rejected"
                    break
            human_reviews = []
            for idx, rev in enumerate(raw_reviews):
                content = rev.get('content', {})
                def get_val(field):
                    v = content.get(field, {})
                    return v.get("value", v) if isinstance(v, dict) else v

                text_parts = []
                for field in ["summary", "strengths", "weaknesses", "questions", "limitations", "review", "comment"]:
                    val = get_val(field)
                    if isinstance(val, str) and len(val) > 20:
                        text_parts.append(val)

                full_text = "\n\n".join(text_parts)
                if not full_text.strip(): continue
                sentences = naive_sentence_split(full_text)
                if len(sentences) < MIN_SENTENCES_PER_REVIEW:
                    print(f"    -> Review {idx} dropped (Too short: {len(sentences)} sentences)")
                    continue

                alpha = compute_liang_alpha(full_text)
                if alpha > LIANG_ALPHA_THRESHOLD:
                    print(f"    -> Review {idx} dropped (Burstiness: {alpha:.2f} > {LIANG_ALPHA_THRESHOLD})")
                    continue

                review_obj = {
                    "review_id": f"human_rev_{idx}",
                    "type": "Human",
                    "liang_alpha_score": alpha,
                    "aspect_scores": {
                        "originality": get_val("originality")  or "Not Discussed",
                        "substance":   get_val("soundness")    or "Not Discussed",
                        "clarity":     get_val("presentation") or "Not Discussed",
                        "impact":      get_val("contribution") or "Not Discussed",
                    },
                    "review_sentences": []
                }

                for s_idx, sentence in enumerate(sentences):
                    review_obj["review_sentences"].append({
                        "sentence_id":  f"human_rev_{idx}_s_{s_idx}",
                        "text":         sentence,
                        "author_label": "HUMAN",
                    })

                human_reviews.append(review_obj)
            if len(human_reviews) < MIN_REVIEWS_PER_PAPER:
                print(f"    -> Skipped: Only {len(human_reviews)} valid reviews left (Need {MIN_REVIEWS_PER_PAPER}).")
                log_failure(paper_id, f"Not enough reviews ({len(human_reviews)} < {MIN_REVIEWS_PER_PAPER})")
                continue
            print("    -> Downloading PDF via OpenReview V1 Client...")
            pdf_path = os.path.join(OPENREVIEW_DIR, f"{paper_id}.pdf")
            
            if not os.path.exists(pdf_path):
                try:
                    pdf_binary = client.get_pdf(id=paper_id)
                    with open(pdf_path, 'wb') as f:
                        f.write(pdf_binary)
                    time.sleep(2.5)
                except Exception as e:
                    print(f"    -> Skipped: PDF Download Failed via Client API ({e})")
                    log_failure(paper_id, f"PDF download failed: {e}")
                    continue
            print("    -> Running GROBID and Saving Graph...")
            ok = build_and_save(
                paper_id                = paper_id,
                source                  = f"OpenReview_ICLR_{year}",
                year                    = year,
                decision                = decision,
                human_reviews           = human_reviews,
                pdf_path                = pdf_path,
                total_reviews_available = len(human_reviews),
            )
            
            if ok: 
                counter[0] += 1

def build_dataset_loop():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Path(FAILED_LOG).touch(exist_ok=True)
    counter = [0]

    print("\n=== PHASE 1: PeerRead (local, no API) ===")
    ingest_peerread(counter, TARGET_PAPER_COUNT)

    if counter[0] < TARGET_PAPER_COUNT:
        remaining = TARGET_PAPER_COUNT - counter[0]
        print(f"\n=== PHASE 2: OpenReview (need {remaining} more papers) ===")
        ingest_openreview(counter, TARGET_PAPER_COUNT)

    print(f"\nDone. {counter[0]} papers processed -> {OUTPUT_DIR}")


if __name__ == "__main__":
    print("Starting AuthentiGraph Ingestion Pipeline...")
    build_dataset_loop()