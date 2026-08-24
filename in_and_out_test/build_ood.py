import os
import json
import re
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dom_parser_v2 import parse_tei_xml

def naive_sentence_split(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    return sentences

def build_ood():
    os.makedirs("ood_processed_graphs", exist_ok=True)
    
    with open("reviews_metadata_100.json", "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    print(f"Stitching {len(papers)} OOD papers...")
    success_count = 0
    
    for row in papers:
        pid = row["paper_id"]
        xml_path = f"parsed_xml/{pid}.tei.xml"
        
        if not os.path.exists(xml_path):
            continue
            
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_content = f.read()
            
        try:
            sections, bibliography, hetero_edges = parse_tei_xml(xml_content)
        except Exception as e:
            print(f"Skipping {pid} (XML parse error: {e})")
            continue
            
        human_text = row.get("human_review", "") or ""
        ai_text = row.get("gpt4o_ai_review", "") or ""
        
        human_sents = naive_sentence_split(human_text)
        ai_sents = naive_sentence_split(ai_text)
        
        if len(human_sents) < 3 or len(ai_sents) < 3:
            continue
            
        graph = {
            "paper_metadata": {
                "paper_id": pid, 
                "source": "Gen-Review-OOD"
            },
            "paper_DOM": {
                "sections": sections, 
                "bibliography": bibliography
            },
            "heterogeneous_edges": hetero_edges,
            "human_ground_truth": [{
                "review_id": f"human_{pid}", 
                "type": "Human", 
                "review_sentences": [
                    {"sentence_id": f"h_{i}", "text": s, "author_label": "HUMAN"} 
                    for i, s in enumerate(human_sents)
                ]
            }],
            "adversarial_reviews": [{
                "mode": "Gen-Review-GPT4o", 
                "generation_model": "GPT-4o",
                "review_sentences": [
                    {"sentence_id": f"ai_{i}", "text": s, "author_label": "AI_GENERATED"} 
                    for i, s in enumerate(ai_sents)
                ]
            }]
        }
        
        with open(f"ood_processed_graphs/{pid}_base_graph.json", "w") as out:
            json.dump(graph, out, indent=4)
        success_count += 1
            
    print(f"Done! Created {success_count} OOD base graphs in `ood_processed_graphs/`.")

if __name__ == "__main__":
    build_ood()