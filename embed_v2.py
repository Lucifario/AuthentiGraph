"""
embed_v2.py — Stage 4: Sentence-Level Embedding + Cosine Edge Routing

Input:  data/processed_graphs/*.json  (or a single file for testing)
Output: data/embedded_graphs/*.pt     — PyG HeteroData objects

Key upgrades over original embed.py:
  - Embeds at SENTENCE level (one vector per sentence, not per paragraph)
  - Top-k cosine edge routing: each review sentence connects to its k=3
    nearest paper blocks — guarantees connectivity, no fixed threshold
  - All 7 adversarial modes embedded alongside human reviews
  - Stores cosine weight on each CRITIQUES_BLOCK edge

Node types:
  paper_block      — paragraph blocks from the paper DOM
  review_sentence  — sentences from human or adversarial reviews

Edge types:
  (paper_block,      CONTAINS,         paper_block)     — section hierarchy
  (paper_block,      CITES,            paper_block)     — inline citation
  (review_sentence,  CRITIQUES_BLOCK,  paper_block)     — cosine-routed
  (review_sentence,  SAME_REVIEW,      review_sentence) — intra-review

Labels:
  y=0  human review sentence
  y=1  AI-generated review sentence
  y=2  AI_INSERTED (spliced) — hybrid ground truth

Usage:
  # Single file test:
  python embed_v2.py --input QDER_test_full_graph.json --output test_embedded.pt

  # Full dataset:
  python embed_v2.py --input_dir data/processed_graphs/ --output_dir data/embedded_graphs/
"""

import os
import json
import glob
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 16
MAX_LENGTH = 128
TOP_K = 3

LABEL_MAP = {
    "HUMAN":        0,
    "AI_GENERATED": 1,
    "AI_INSERTED":  2,
}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def load_specter():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tok, model

@torch.no_grad()
def embed_texts(texts, tok, model):
    """Embeds a list of strings -> (N, 768) float32 tensor."""
    if not texts:
        return torch.zeros((0, 768), dtype=torch.float32)
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        out = model(**enc)
        vecs = out.last_hidden_state[:, 0, :]
        all_vecs.append(vecs.cpu())
    return torch.cat(all_vecs, dim=0).float()

def cosine_top_k_edges(review_embs, block_embs, k=TOP_K):
    """
    For each review sentence, finds the top-k most similar paper blocks.
    Returns (src_indices, dst_indices, weights) all as lists.
    Guarantees connectivity: every review sentence gets exactly k edges.
    """
    if review_embs.shape[0] == 0 or block_embs.shape[0] == 0:
        return [], [], []
    r_norm = F.normalize(review_embs, dim=1)
    b_norm = F.normalize(block_embs, dim=1)
    sim    = torch.mm(r_norm, b_norm.T)
    actual_k  = min(k, block_embs.shape[0])
    top_vals, top_idx = sim.topk(actual_k, dim=1)
    src, dst, weights = [], [], []
    for r in range(review_embs.shape[0]):
        for j in range(actual_k):
            src.append(r)
            dst.append(top_idx[r, j].item())
            weights.append(top_vals[r, j].item())
    return src, dst, weights

def build_intra_review_edges(n_sentences):
    """Connects consecutive sentences within a review (chain graph)."""
    if n_sentences < 2:
        return [], []
    src = list(range(n_sentences - 1))
    dst = list(range(1, n_sentences))
    return src + dst, dst + src

def process_graph(graph_dict, tok, model):
    """
    Converts one paper JSON into a PyG HeteroData object.
    Returns a list of HeteroData — one per review (human + adversarial).
    Each graph represents: paper blocks + ONE review's sentences.
    """
    paper_id  = graph_dict["paper_metadata"]["paper_id"]
    paper_dom = graph_dict["paper_DOM"]
    block_ids   = []
    block_texts = []
    block_id_to_local = {}
    for section in paper_dom.get("sections", []):
        for block in section.get("blocks", []):
            bid  = block["block_id"]
            text = block.get("text", "").strip()
            if not text:
                continue
            block_id_to_local[bid] = len(block_ids)
            block_ids.append(bid)
            block_texts.append(text)
    if not block_ids:
        return []
    print(f"  Embedding {len(block_texts)} paper blocks...", end=" ", flush=True)
    block_embs = embed_texts(block_texts, tok, model)
    print("done.")
    struct_src, struct_dst = [], []
    cite_src,   cite_dst   = [], []
    for edge in graph_dict.get("heterogeneous_edges", []):
        rel = edge["relation"]
        s   = edge["source"]
        d   = edge["target"]
        if rel == "CONTAINS_BLOCK":
            if s in block_id_to_local and d in block_id_to_local:
                struct_src.append(block_id_to_local[s])
                struct_dst.append(block_id_to_local[d])
        elif rel == "CITES_INLINE":
            if s in block_id_to_local and d in block_id_to_local:
                cite_src.append(block_id_to_local[s])
                cite_dst.append(block_id_to_local[d])
    all_reviews = (
        [(r, 0) for r in graph_dict.get("human_ground_truth", [])] +
        [(r, 1) for r in graph_dict.get("adversarial_reviews", [])]
    )
    hetero_graphs = []
    for review, is_ai in all_reviews:
        sentences  = review.get("review_sentences", [])
        mode       = review.get("mode", "Human")
        review_id  = review.get("review_id", "unknown")
        gen_model  = review.get("generation_model", "Human")
        if len(sentences) < 3:
            continue
        sent_texts  = [s["text"] for s in sentences]
        sent_labels = [LABEL_MAP.get(s.get("author_label", "HUMAN"), 0) for s in sentences]
        sent_embs = embed_texts(sent_texts, tok, model)
        cs, cd, cw = cosine_top_k_edges(sent_embs, block_embs)
        rs, rd = build_intra_review_edges(len(sentences))
        data = HeteroData()
        data["paper_block"].x     = block_embs
        data["review_sentence"].x = sent_embs
        data["review_sentence"].y = torch.tensor(sent_labels, dtype=torch.long)
        data.y         = torch.tensor([is_ai], dtype=torch.long)
        data.paper_id  = paper_id
        data.review_id = review_id
        data.mode      = mode
        data.gen_model = gen_model
        if struct_src:
            data["paper_block", "CONTAINS", "paper_block"].edge_index = \
                torch.tensor([struct_src, struct_dst], dtype=torch.long)
        if cite_src:
            data["paper_block", "CITES", "paper_block"].edge_index = \
                torch.tensor([cite_src, cite_dst], dtype=torch.long)
        if cs:
            data["review_sentence", "CRITIQUES_BLOCK", "paper_block"].edge_index = \
                torch.tensor([cs, cd], dtype=torch.long)
            data["review_sentence", "CRITIQUES_BLOCK", "paper_block"].edge_weight = \
                torch.tensor(cw, dtype=torch.float)
        if rs:
            data["review_sentence", "SAME_REVIEW", "review_sentence"].edge_index = \
                torch.tensor([rs, rd], dtype=torch.long)
        hetero_graphs.append(data)
        print(f"    [{mode}] {len(sentences)} sentences | " f"{len(cs)} CRITIQUES_BLOCK edges | label={is_ai}")
    return hetero_graphs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      help="Single JSON file (test mode)")
    parser.add_argument("--output",     help="Single .pt output file (test mode)")
    parser.add_argument("--input_dir",  default="data/processed_graphs/")
    parser.add_argument("--output_dir", default="data/embedded_graphs/")
    args = parser.parse_args()
    tok, model = load_specter()
    if args.input:
        print(f"\nProcessing: {args.input}")
        with open(args.input) as f:
            graph_dict = json.load(f)
        graphs = process_graph(graph_dict, tok, model)
        out    = args.output or "test_embedded.pt"
        torch.save(graphs, out)
        print(f"\nSaved {len(graphs)} graphs -> {out}")
        for g in graphs:
            print(f"\n  [{g.mode}] paper_id={g.paper_id}")
            print(f"    paper_block nodes  : {g['paper_block'].x.shape}")
            print(f"    review_sent nodes  : {g['review_sentence'].x.shape}")
            print(f"    sent labels        : {g['review_sentence'].y.tolist()[:5]}...")
            print(f"    graph label (y)    : {g.y.item()}")
    else:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
        print(f"Found {len(json_files)} graphs.")
        all_graphs = []
        for idx, fp in enumerate(json_files):
            print(f"\n[{idx+1}/{len(json_files)}] {Path(fp).stem}")
            with open(fp) as f:
                gd = json.load(f)
            graphs = process_graph(gd, tok, model)
            all_graphs.extend(graphs)
        out = os.path.join(args.output_dir, "all_graphs.pt")
        torch.save(all_graphs, out)
        print(f"\nTotal: {len(all_graphs)} graphs saved -> {out}")

if __name__ == "__main__":
    main()