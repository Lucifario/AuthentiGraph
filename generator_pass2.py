import os
import json
import glob
import re
import random
import copy
import ollama

INPUT_DIR = "data/processed_graphs/"
OLLAMA_MODEL = "llama3"


def naive_sentence_split(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sentences if sentences else [text.strip()]


def ask_ollama(prompt, system_prompt="You are an expert AI researcher."):
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': prompt}
    ])
    return response['message']['content']


def atomic_save(filepath, data):
    """Write to a temp file then rename — prevents corruption on crash."""
    tmp_path = filepath + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, filepath)


def process_pass_2():
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} graphs to process.")

    for idx, filepath in enumerate(json_files):
        print(f"\n[{idx+1}/{len(json_files)}] Pass 2: {os.path.basename(filepath)}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        if "adversarial_reviews" not in data:
            print("  -> No adversarial_reviews field. Run Pass 1 first.")
            continue

        human_reviews = data.get("human_ground_truth", [])
        if not human_reviews:
            print("  -> No human reviews found. Skipping.")
            continue

        existing_modes = {r.get("mode") for r in data["adversarial_reviews"]}
        changed = False

        base_human = human_reviews[0]
        human_text = " ".join(
            s["text"] for s in base_human.get("review_sentences", [])
        )

        if len(human_text.strip()) < 100:
            print("  -> Human review text too short. Skipping.")
            continue

        try:
            if "Style-Assisted" not in existing_modes:
                print("  -> Generating Mode 3: Style-Assisted...")

                extract_prompt = (
                    f"Extract the key technical points from this peer review as a "
                    f"numbered list. Each point should be one sentence only. "
                    f"Return only the numbered list, nothing else.\n\n"
                    f"Review:\n{human_text}"
                )
                bullet_points = ask_ollama(
                    extract_prompt,
                    "You return only a numbered list of technical points."
                )

                expand_prompt = (
                    f"You are writing a peer review for an AI conference paper. "
                    f"Convert these rough reviewer notes into a complete, formal, "
                    f"professional peer review with fluent academic prose. "
                    f"Preserve all technical points.\n\n"
                    f"Notes:\n{bullet_points}"
                )
                m3_text = ask_ollama(expand_prompt)

                data["adversarial_reviews"].append({
                    "mode":             "Style-Assisted",
                    "generation_model": OLLAMA_MODEL,
                    "generation_role":  "style_assistant",
                    "review_sentences": [
                        {
                            "sentence_id":  f"ai_m3_s_{i}",
                            "text":         s,
                            "author_label": "AI_GENERATED"
                        }
                        for i, s in enumerate(naive_sentence_split(m3_text))
                    ]
                })
                changed = True
            else:
                print("  -> Mode 3 already exists. Skipping.")

            if "Spliced" not in existing_modes:
                print("  -> Generating Mode 4: Spliced...")

                splice_prompt = (
                    f"Here is a peer review:\n{human_text}\n\n"
                    f"Write exactly 3 new, highly critical sentences that point out "
                    f"a flaw or missing experiment NOT mentioned in this review. "
                    f"Return ONLY the 3 sentences, one per line, nothing else."
                )
                m4_text = ask_ollama(
                    splice_prompt,
                    "You return exactly 3 sentences and nothing else."
                )
                ai_sentences = naive_sentence_split(m4_text)[:3]

                spliced_sentences = copy.deepcopy(base_human["review_sentences"])

                insert_idx = random.randint(1, max(1, len(spliced_sentences) - 2))
                for i, ai_s in enumerate(ai_sentences):
                    spliced_sentences.insert(insert_idx + i, {
                        "sentence_id":  f"spliced_ai_{i}",
                        "text":         ai_s,
                        "author_label": "AI_INSERTED"
                    })

                data["adversarial_reviews"].append({
                    "mode":             "Spliced",
                    "generation_model": "Hybrid",
                    "generation_role":  "splicer",
                    "splice_insert_idx": insert_idx,
                    "review_sentences": spliced_sentences
                })
                changed = True
            else:
                print("  -> Mode 4 already exists. Skipping.")

            if "Polarity-Flipped" not in existing_modes:
                print("  -> Generating Mode 5: Polarity-Flipped...")

                flip_prompt = (
                    f"Here is a peer review:\n{human_text}\n\n"
                    f"Rewrite this entire review to have the OPPOSITE overall "
                    f"recommendation. If the review recommends acceptance, rewrite "
                    f"it to recommend rejection, and vice versa. Keep all the same "
                    f"technical concepts and paper details, but invert the "
                    f"conclusions, tone, and final recommendation."
                )
                m5_text = ask_ollama(flip_prompt)

                data["adversarial_reviews"].append({
                    "mode":             "Polarity-Flipped",
                    "generation_model": OLLAMA_MODEL,
                    "generation_role":  "polarity_flipper",
                    "review_sentences": [
                        {
                            "sentence_id":  f"ai_m5_s_{i}",
                            "text":         s,
                            "author_label": "AI_GENERATED"
                        }
                        for i, s in enumerate(naive_sentence_split(m5_text))
                    ]
                })
                changed = True
            else:
                print("  -> Mode 5 already exists. Skipping.")

        except Exception as e:
            print(f"  -> Error: {e}")

        if changed:
            atomic_save(filepath, data)
            print("  -> Saved.")


if __name__ == "__main__":
    process_pass_2()