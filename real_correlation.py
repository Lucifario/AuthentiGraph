import json
import glob
import numpy as np
from scipy.stats import pearsonr, spearmanr
from spectral import extract_features

INPUT_DIR = "data/processed_graphs/"
alphas, hub_focus, spread_ratio, relation_entropy, spectral_gap = [], [], [], [], []

print("Extracting features for correlation check...")
for filepath in glob.glob(f"{INPUT_DIR}/*.json"):
    with open(filepath, 'r') as f:
        data = json.load(f)
    for rev in data.get("human_ground_truth", []):
        feats = extract_features(rev, data)
        if feats is not None:
            alphas.append(rev.get("liang_alpha_score", 0.0))
            spread_ratio.append(feats[0])
            hub_focus.append(feats[2])
            relation_entropy.append(feats[3])
            spectral_gap.append(feats[5])

print("\nCorrelation between Liang alpha and structural features (Human reviews only):\n")
for name, vals in [("hub_focus", hub_focus), ("spread_ratio", spread_ratio), 
                   ("relation_entropy", relation_entropy), ("spectral_gap", spectral_gap)]:
    r, p = pearsonr(alphas, vals)
    rho, p_rho = spearmanr(alphas, vals)
    print(f"  {name:18s}  Pearson r={r:+.3f} (p={p:.3f})   Spearman rho={rho:+.3f} (p={p_rho:.3f})")