import os
import glob
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils import resample
from train_and_meta import AuthentiGraphHGT, graph_to_device, DEVICE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT

JSON_DIR = "data/processed_graphs/"
GRAPHS_PATH = "data/embedded_graphs/all_graphs.pt"
MODEL_WEIGHTS = "models/authentigraph_hgt.pt"
N_BOOTSTRAP = 2000

def bootstrap_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, alpha=0.05):
    """95% CI on ROC-AUC via bootstrap resampling."""
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    boot_aucs = []
    idx = np.arange(len(y_true))
    for _ in range(n_boot):
        sample_idx = resample(idx)
        yt, yp = y_true[sample_idx], y_prob[sample_idx]
        if len(set(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, yp))
    if not boot_aucs:
        return (float("nan"), float("nan"))
    lo = np.percentile(boot_aucs, 100 * alpha / 2)
    hi = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return lo, hi

def map_alpha_scores(json_dir):
    """Extracts Liang alpha scores from JSONs and maps them to review IDs."""
    alpha_map = {}
    total_human = 0
    for filepath in glob.glob(os.path.join(json_dir, "*.json")):
        with open(filepath, 'r') as f:
            data = json.load(f)
        pid = data.get("paper_metadata", {}).get("paper_id", "")
        for rev in data.get("human_ground_truth", []):
            total_human += 1
            rid = rev.get("review_id", "")
            alpha_map[f"{pid}::{rid}"] = rev.get("liang_alpha_score", 0.0)
    return alpha_map, total_human

def evaluate_hgt(graphs, model):
    """Evaluates the PyTorch model on a specific list of graphs."""
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for g in graphs:
            xd, eid = graph_to_device(g, DEVICE)
            logits, _ = model(xd, eid)
            prob_ai = F.softmax(logits, dim=1).squeeze()[1].item()
            labels.append(g.y.item())
            preds.append(int(prob_ai > 0.5))
            probs.append(prob_ai)  
    labels, preds, probs = np.array(labels), np.array(preds), np.array(probs)
    if len(set(labels)) < 2:
        return {"n": len(labels), "roc_auc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "fpr": float("nan"), "human_acc": float("nan")}
    roc_auc = roc_auc_score(labels, probs)
    ci_lo, ci_hi = bootstrap_ci(labels, probs)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / max(1, (fp + tn))
    human_mask = labels == 0
    human_acc = (preds[human_mask] == labels[human_mask]).mean() if human_mask.any() else float("nan")
    return {
        "n": len(labels),
        "n_human": int((labels == 0).sum()),
        "n_ai": int((labels == 1).sum()),
        "roc_auc": round(roc_auc, 3),
        "ci_lo": round(ci_lo, 3),
        "ci_hi": round(ci_hi, 3),
        "fpr": round(fpr, 3),
        "human_acc": round(human_acc, 3),
    }

def run_ablation_study():
    print(f"Loading Pre-Embedded Graphs from {GRAPHS_PATH}...")
    all_graphs = torch.load(GRAPHS_PATH, weights_only=False)
    alpha_map, total_human_available = map_alpha_scores(JSON_DIR)
    model = AuthentiGraphHGT(hidden=HIDDEN_DIM, heads=NUM_HEADS, layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    alpha_thresholds = [1.00, 0.95, 0.85, 0.75, 0.00]
    results = []
    for alpha in alpha_thresholds:
        filtered_graphs = []
        retained_human = 0
        for g in all_graphs:
            if g.y.item() == 1: 
                filtered_graphs.append(g)
            else:
                key = f"{g.paper_id}::{g.review_id}"
                score = alpha_map.get(key, 0.0)
                if score <= alpha:
                    retained_human += 1
                    filtered_graphs.append(g)
        metrics = evaluate_hgt(filtered_graphs, model)
        metrics["alpha_threshold"] = alpha
        metrics["retention_pct"] = round(100 * retained_human / max(1, total_human_available), 1)
        results.append(metrics)
        print(f"--> alpha <= {alpha:.2f}: {retained_human}/{total_human_available} human reviews retained")
    df = pd.DataFrame(results)
    cols = ["alpha_threshold", "retention_pct", "n", "n_human", "n_ai", "roc_auc", "ci_lo", "ci_hi", "fpr", "human_acc"]
    print("\n=== REAL ABLATION RESULTS ===")
    print(df[cols].to_string(index=False))
    return df

if __name__ == "__main__":
    df = run_ablation_study()