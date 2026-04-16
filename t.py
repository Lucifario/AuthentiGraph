import json
with open("data/processed_graphs/400_base_graph.json") as f:
    d = json.load(f)
from generator_pass1 import extract_paper_text
print(extract_paper_text(d["paper_DOM"])[:500])