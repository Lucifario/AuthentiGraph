import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
try:
    from torch_geometric.nn import HGTConv, Linear
    HGT_AVAILABLE = True
except ImportError:
    HGT_AVAILABLE = False
    print("Warning: torch_geometric not installed. Training mode unavailable.")
    print("Install: pip install torch_geometric")

HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 60
BATCH_SIZE = 8
CONTRASTIVE_WEIGHT = 0.1
SEED = 42

NODE_TYPES = ["paper_block", "review_sentence"]
EDGE_TYPES = [
    ("paper_block",      "CONTAINS",        "paper_block"),
    ("paper_block",      "CITES",           "paper_block"),
    ("review_sentence",  "CRITIQUES_BLOCK", "paper_block"),
    ("review_sentence",  "SAME_REVIEW",     "review_sentence"),
]

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

class AuthentiGraphHGT(nn.Module):
    def __init__(self, in_dim=768, hidden=128, heads=4, layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.proj = nn.ModuleDict({
            nt: Linear(in_dim, hidden) for nt in NODE_TYPES
        })
        meta = (NODE_TYPES, [(s, r, d) for s, r, d in EDGE_TYPES])
        self.convs = nn.ModuleList([
            HGTConv(hidden, hidden, metadata=meta, heads=heads)
            for _ in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64)
        )
    def encode(self, x_dict, edge_index_dict):
        h = {}
        for nt in NODE_TYPES:
            if nt in x_dict and x_dict[nt].shape[0] > 0:
                h[nt] = F.relu(self.proj[nt](x_dict[nt]))
            else:
                ref = next(iter(h.values())) if h else None
                dim = ref.shape[1] if ref is not None else HIDDEN_DIM
                h[nt] = torch.zeros((1, dim), device=DEVICE)
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            h = {
                nt: self.norm(self.dropout(h_new[nt]) + h[nt])
                if nt in h_new else h[nt]
                for nt in h
            }
        rev_h = h.get("review_sentence", torch.zeros(1, HIDDEN_DIM, device=DEVICE))
        graph_emb = rev_h.mean(dim=0, keepdim=True)
        return graph_emb
    def forward(self, x_dict, edge_index_dict):
        graph_emb = self.encode(x_dict, edge_index_dict)
        logits = self.mlp(graph_emb)
        return logits, graph_emb

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature
    def forward(self, features, labels):
        device = features.device
        N = features.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=device)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        dot = torch.matmul(features, features.T) / self.T
        logits = dot - dot.detach().max(dim=1, keepdim=True).values
        self_m = torch.eye(N, dtype=torch.bool, device=device)
        exp_log = torch.exp(logits).masked_fill(self_m, 0.0)
        log_prob = logits - torch.log(exp_log.sum(dim=1, keepdim=True) + 1e-8)
        pos_mask = mask.clone()
        pos_mask[self_m] = 0
        n_pos = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        loss = -mean_log_prob[n_pos > 0].mean()
        return loss

def graph_to_device(data, device):
    x_dict, ei_dict = {}, {}
    for nt in NODE_TYPES:
        try:
            x_dict[nt] = data[nt].x.to(device)
        except Exception:
            pass
    for (s, r, d) in EDGE_TYPES:
        try:
            ei = data[s, r, d].edge_index.to(device)
        except Exception:
            ei = torch.zeros((2, 0), dtype=torch.long, device=device)
        ei_dict[s, r, d] = ei
    return x_dict, ei_dict


def train_epoch(model, graphs, optimizer, ce_fn, con_fn, device):
    import random
    model.train()
    random.shuffle(graphs)
    total_loss, correct, n = 0, 0, 0
    for i in range(0, len(graphs), BATCH_SIZE):
        batch = graphs[i:i+BATCH_SIZE]
        optimizer.zero_grad()
        logits_list, emb_list, label_list = [], [], []
        for g in batch:
            xd, eid = graph_to_device(g, device)
            logits, emb = model(xd, eid)
            logits_list.append(logits)
            emb_list.append(emb)
            label_list.append(g.y)
        logits_cat = torch.cat(logits_list, dim=0)
        labels_cat = torch.stack(label_list).squeeze().to(device)
        embs_cat   = torch.cat(emb_list, dim=0)
        ce  = ce_fn(logits_cat, labels_cat)
        proj = F.normalize(model.proj_head(embs_cat), dim=1)
        con  = con_fn(proj, labels_cat)
        loss = ce + CONTRASTIVE_WEIGHT * con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch)
        preds = logits_cat.argmax(dim=1)
        correct += (preds == labels_cat).sum().item()
        n += len(batch)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, graphs, device):
    model.eval()
    preds, labels, probs = [], [], []
    for g in graphs:
        xd, eid = graph_to_device(g, device)
        logits, _ = model(xd, eid)
        p = logits.argmax(dim=1).item()
        prob = F.softmax(logits, dim=1)[0, 1].item()
        preds.append(p)
        labels.append(g.y.item())
        probs.append(prob)
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return acc, preds, labels, probs

def run_training(graphs):
    if not HGT_AVAILABLE:
        print("torch_geometric required for training.")
        return
    set_seed(SEED)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    paper_ids = sorted(list(set(g.paper_id for g in graphs)))
    
    import random
    rng = random.Random(SEED)
    rng.shuffle(paper_ids)
    
    n = len(paper_ids)
    train_ids = set(paper_ids[:int(0.7*n)])
    val_ids = set(paper_ids[int(0.7*n):int(0.85*n)])
    test_ids = set(paper_ids[int(0.85*n):])
    
    train_g = [g for g in graphs if g.paper_id in train_ids]
    val_g = [g for g in graphs if g.paper_id in val_ids]
    test_g = [g for g in graphs if g.paper_id in test_ids]
    print(f"Train: {len(train_g)} | Val: {len(val_g)} | Test: {len(test_g)}")
    
    n_human = sum(1 for g in train_g if g.y.item() == 0)
    n_ai    = sum(1 for g in train_g if g.y.item() == 1)
    w = torch.tensor([n_ai / (n_human + 1e-8), 1.0], dtype=torch.float).to(DEVICE)
    
    model = AuthentiGraphHGT(hidden=HIDDEN_DIM, heads=NUM_HEADS, layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    ce_fn = nn.CrossEntropyLoss(weight=w)
    con_fn = SupConLoss()
    best_val = 0.0
    MODEL_OUT = "models/authentigraph_hgt.pt"
    
    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_g, optimizer, ce_fn, con_fn, DEVICE)
        val_acc, _, _, _ = evaluate(model, val_g, DEVICE)
        scheduler.step()
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Loss {tr_loss:.4f} | " f"Train {tr_acc:.3f} | Val {val_acc:.3f} | Best {best_val:.3f}")
            
    model.load_state_dict(torch.load(MODEL_OUT))
    test_acc, preds, truths, probs = evaluate(model, test_g, DEVICE)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(classification_report(truths, preds, target_names=["Human", "AI"], digits=3))
    try:
        print(f"ROC-AUC: {roc_auc_score(truths, probs):.4f}")
    except Exception:
        pass
    print("Confusion Matrix:")
    print(confusion_matrix(truths, preds))
    print("\nPer-mode breakdown:")
    modes = defaultdict(lambda: {"correct": 0, "total": 0})
    for g, p in zip(test_g, preds):
        m = g.mode
        modes[m]["total"] += 1
        if p == g.y.item():
            modes[m]["correct"] += 1
    for m, v in sorted(modes.items()):
        print(f"  {m:<25}: {v['correct']/v['total']:.3f} ({v['correct']}/{v['total']})")

def run_meta_graph(graphs):
    """
    Builds a consensus meta-graph per paper:
      - Each review = one node (mean of its sentence embeddings)
      - Edge weight = cosine similarity between review embeddings
      - Reports: intra-human vs human-AI similarity, density gap
    """
    Path("results").mkdir(exist_ok=True)
    by_paper = defaultdict(list)
    for g in graphs:
        by_paper[g.paper_id].append(g)
    print(f"\n{'='*60}")
    print("Stage 7: Consensus Meta-Graph Analysis")
    print(f"{'='*60}")
    print(f"Papers: {len(by_paper)}\n")
    all_results = []
    for paper_id, paper_graphs in by_paper.items():
        print(f"Paper: {paper_id}")
        review_nodes = []
        for g in paper_graphs:
            sent_emb = g["review_sentence"].x
            review_emb = sent_emb.mean(dim=0)
            review_nodes.append({
                "mode":      g.mode,
                "label":     g.y.item(),
                "embedding": review_emb,
                "gen_model": g.gen_model,
            })
        if len(review_nodes) < 2:
            print("  -> Not enough reviews for meta-graph. Skipping.\n")
            continue
        embs = torch.stack([r["embedding"] for r in review_nodes])
        embs_norm = F.normalize(embs, dim=1)
        sim_matrix = torch.mm(embs_norm, embs_norm.T)

        N = len(review_nodes)
        hh_sims, ha_sims, aa_sims = [], [], []
        for i in range(N):
            for j in range(i+1, N):
                s = sim_matrix[i, j].item()
                li, lj = review_nodes[i]["label"], review_nodes[j]["label"]
                if li == 0 and lj == 0:
                    hh_sims.append(s)
                elif li == 1 and lj == 1:
                    aa_sims.append(s)
                else:
                    ha_sims.append(s)
        hh_mean = np.mean(hh_sims) if hh_sims else 0.0
        ha_mean = np.mean(ha_sims) if ha_sims else 0.0
        aa_mean = np.mean(aa_sims) if aa_sims else 0.0
        print(f"  Reviews: {N} ({sum(1 for r in review_nodes if r['label']==0)} human, " f"{sum(1 for r in review_nodes if r['label']==1)} AI)")
        print(f"  Human-Human similarity  : {hh_mean:.4f}")
        print(f"  Human-AI similarity     : {ha_mean:.4f}")
        print(f"  AI-AI similarity        : {aa_mean:.4f}")
        print(f"  Consensus gap (HH-HA)   : {hh_mean - ha_mean:+.4f}")
        print()
        print("  Similarity matrix (rows/cols = review modes):")
        modes = [r["mode"][:12] for r in review_nodes]
        header = f"  {'':14}" + "".join(f"{m:>14}" for m in modes)
        print(header)
        for i, row_mode in enumerate(modes):
            row = f"  {row_mode:<14}"
            for j in range(N):
                row += f"{sim_matrix[i,j].item():>14.3f}"
            print(row)
        print()
        all_results.append({
            "paper_id":       paper_id,
            "n_reviews":      N,
            "hh_similarity":  round(hh_mean, 4),
            "ha_similarity":  round(ha_mean, 4),
            "aa_similarity":  round(aa_mean, 4),
            "consensus_gap":  round(hh_mean - ha_mean, 4),
        })
    if all_results:
        avg_gap = np.mean([r["consensus_gap"] for r in all_results])
        print(f"Mean consensus gap (HH-HA) across papers: {avg_gap:+.4f}")
        print("Positive gap = human reviews cluster more tightly than human-AI pairs.")
        print("This is the expected signal AuthentiGraph's graph topology should capture.\n")
    with open("results/consensus_meta.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Meta-graph results saved -> results/consensus_meta.json")

def show_architecture(graphs):
    if not HGT_AVAILABLE:
        print("torch_geometric not available — skipping architecture summary.")
        return
    model = AuthentiGraphHGT(hidden=HIDDEN_DIM, heads=NUM_HEADS, layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print("Stage 6: HGT Architecture Summary")
    print(f"{'='*60}")
    print(f"Node types       : {NODE_TYPES}")
    print(f"Edge types       : {len(EDGE_TYPES)}")
    print(f"Hidden dim       : {HIDDEN_DIM}")
    print(f"Attention heads  : {NUM_HEADS}")
    print(f"HGT layers       : {NUM_LAYERS}")
    print(f"Total parameters : {n_params:,}")
    print(f"Device           : {DEVICE}")
    print()
    g = graphs[0]
    xd, eid = graph_to_device(g, DEVICE)
    with torch.no_grad():
        logits, emb = model(xd, eid)
    print(f"Test forward pass ({g.mode}):")
    print(f"  paper_block nodes  : {g['paper_block'].x.shape}")
    print(f"  review_sent nodes  : {g['review_sentence'].x.shape}")
    print(f"  graph embedding    : {emb.shape}")
    print(f"  logits             : {logits.shape} -> {logits}")
    print(f"  predicted label    : {logits.argmax(dim=1).item()} (true: {g.y.item()})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   default="test", choices=["test", "train", "meta", "all"])
    parser.add_argument("--graphs", default="test_embedded.pt", help="Path to .pt file of HeteroData graphs")
    args = parser.parse_args()
    print(f"Loading graphs from {args.graphs}...")
    graphs = torch.load(args.graphs, weights_only=False)
    print(f"Loaded {len(graphs)} graphs.\n")
    if args.mode in ("test", "all"):
        show_architecture(graphs)
    if args.mode in ("meta", "all", "test"):
        run_meta_graph(graphs)
    if args.mode in ("train", "all"):
        if len(graphs) < 10:
            print("\nNote: Training with < 10 graphs is only for pipeline verification.")
            print("Results will not be meaningful. Use full dataset for real training.")
        run_training(graphs)

if __name__ == "__main__":
    main()