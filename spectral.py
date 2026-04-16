import os
import json
import glob
import math
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

INPUT_DIR   = "data/processed_graphs/"
RESULTS_OUT = "results/spectral_baseline.json"

FEATURE_NAMES = [
    "spread_ratio", "phantom_rate", "hub_focus", "relation_entropy",
    "review_length", "liang_alpha", "spectral_gap", "clustering_coeff",
]

def spectral_gap(G):
    """Fiedler value of the largest connected component. Returns 0.0 if < 4 nodes."""
    if G.number_of_nodes() < 4:
        return 0.0
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = G.subgraph(components[0])
    if lcc.number_of_nodes() < 4:
        return 0.0
    L = nx.laplacian_matrix(lcc).todense()
    eigvals = np.sort(np.real(np.linalg.eigvals(L)))
    return round(float(eigvals[1]), 5)

def shannon_entropy(values):
    """Shannon entropy over a list of numeric values via histogram."""
    if not values:
        return 0.0
    counts = Counter(values)
    total  = sum(counts.values())
    probs  = [v / total for v in counts.values()]
    return round(-sum(p * math.log2(p) for p in probs if p > 0), 4)

def cohens_d(a, b):
    """Cohen's d effect size between two arrays."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = math.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return round((np.mean(a) - np.mean(b)) / pooled_std, 4) if pooled_std > 0 else 0.0


def build_block_text_map(paper_dom):
    """Maps block_id -> text for specificity and edge heuristics."""
    block_map = {}
    for section in paper_dom.get("sections", []):
        for block in section.get("blocks", []):
            bid = block.get("block_id", "")
            txt = block.get("text", "")
            if bid:
                block_map[bid] = txt
    return block_map

def extract_features(review, paper_data):
    """
    Extracts structural features for one review.
    Returns a feature vector (list) or None if review is too small.
    """
    sentences   = review.get("review_sentences", [])
    n_sentences = len(sentences)
    if n_sentences < 3:
        return None

    paper_dom  = paper_data.get("paper_DOM", {})
    paper_meta = paper_data.get("paper_metadata", {})
    block_map  = build_block_text_map(paper_dom)
    G = nx.Graph()
    target_hits = Counter()
    tgt_nodes   = set()

    for s in sentences:
        sid    = s.get("sentence_id", "")
        s_text = s.get("text", "").lower()
        words  = {w for w in s_text.split() if len(w) > 5}
        G.add_node(sid)

        for bid, btxt in block_map.items():
            overlap = len(words & {w for w in btxt.lower().split() if len(w) > 5})
            if overlap >= 3:
                G.add_edge(sid, bid)
                tgt_nodes.add(bid)
                target_hits[bid] += 1

    unique_targets = len(tgt_nodes)
    max_hits       = max(target_hits.values(), default=0)
    spread_ratio = round(unique_targets / n_sentences, 4)
    phantom_rate = paper_meta.get("phantom_rate", 0.0) or 0.0
    hub_focus = round(max_hits / n_sentences, 4)
    lengths          = [len(s.get("text", "").split()) for s in sentences]
    relation_entropy = shannon_entropy(lengths)
    review_length = n_sentences
    alphas = [
        s.get("liang_alpha_score") for s in sentences
        if s.get("liang_alpha_score") is not None
    ]
    liang_alpha = round(np.mean(alphas), 4) if alphas else (
        0.05 if review.get("type") == "Human" else 1.0
    )
    gap = spectral_gap(G)
    clust = round(nx.average_clustering(G), 5)
    return [
        spread_ratio, phantom_rate, hub_focus, relation_entropy,
        review_length, liang_alpha, gap, clust,
    ]

def main():
    Path("results").mkdir(exist_ok=True)
    json_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    print(f"Found {len(json_files)} graphs.\n")
    X, y, modes = [], [], []
    print("Extracting features...")
    for filepath in json_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        for rev in data.get("human_ground_truth", []):
            feats = extract_features(rev, data)
            if feats is not None:
                X.append(feats)
                y.append(0)
                modes.append("Human")
        for rev in data.get("adversarial_reviews", []):
            feats = extract_features(rev, data)
            if feats is not None:
                X.append(feats)
                y.append(1)
                modes.append(rev.get("mode", "Unknown"))

    if not X:
        print("No features extracted. Check INPUT_DIR and JSON structure.")
        return

    X      = np.array(X)
    y      = np.array(y)
    modes  = np.array(modes)

    print(f"Dataset: {len(y)} reviews total")
    print(f"  Human : {sum(y==0)}")
    print(f"  AI    : {sum(y==1)}")
    human_X = X[y == 0]
    ai_X    = X[y == 1]

    if len(ai_X) > 0:
        print(f"\n{'FEATURE':<20} {'Human μ':<12} {'AI μ':<12} {'Δ':<10} {'Cohen d':<10} Effect")
        print("-" * 70)

        separation_stats = {}
        for i, name in enumerate(FEATURE_NAMES):
            h_vals = human_X[:, i]
            a_vals = ai_X[:, i]
            h_mean = np.mean(h_vals)
            a_mean = np.mean(a_vals)
            delta  = a_mean - h_mean
            d      = cohens_d(a_vals, h_vals)
            effect = "LARGE" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
            direction = "AI↑" if delta > 0 else "HU↑"
            print(f"{name:<20} {h_mean:<12.4f} {a_mean:<12.4f} "
                  f"{delta:+<10.4f} {d:<10.4f} {effect} {direction}")
            separation_stats[name] = {
                "human_mean": round(h_mean, 4),
                "ai_mean":    round(a_mean, 4),
                "delta":      round(delta, 4),
                "cohens_d":   d,
                "effect":     effect,
            }
        print(f"\n{'─'*70}")
        print("Per-mode Cohen's d vs Human:")
        print(f"\n{'MODE':<22}", end="")
        for name in FEATURE_NAMES:
            print(f"{name[:9]:<11}", end="")
        print()
        print("-" * (22 + 11 * len(FEATURE_NAMES)))

        mode_stats = {}
        for mode in sorted(set(modes)):
            if mode == "Human":
                continue
            mask = modes == mode
            mode_X = X[mask]
            print(f"{mode:<22}", end="")
            mode_stats[mode] = {}
            for i, name in enumerate(FEATURE_NAMES):
                d = cohens_d(mode_X[:, i], human_X[:, i])
                print(f"{d:<11.3f}", end="")
                mode_stats[mode][name] = d
            print()

        print("\nCohen's d: >0.8 large | >0.5 medium | <0.2 small")
        print(f"\n{'='*50}")
        print("Random Forest Baseline Classifier")
        print(f"{'='*50}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred, target_names=["Human", "AI"], digits=3))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

        print("\nFeature Importances (Random Forest):")
        importances = sorted(
            zip(FEATURE_NAMES, clf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for name, imp in importances:
            bar = "█" * int(imp * 40)
            print(f"  {name:<20} {imp:.4f}  {bar}")

    else:
        print("\nNo adversarial reviews found — saving human-only stats.")
        print("Run generator_pass1.py and generator_pass2.py first for AI class.")
        separation_stats = {}
        mode_stats = {}
    results = {
        "n_human":          int(sum(y == 0)),
        "n_ai":             int(sum(y == 1)),
        "separation_stats": separation_stats if len(ai_X) > 0 else {},
        "mode_stats":       mode_stats if len(ai_X) > 0 else {},
        "feature_names":    FEATURE_NAMES,
    }
    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_OUT}")

if __name__ == "__main__":
    main()