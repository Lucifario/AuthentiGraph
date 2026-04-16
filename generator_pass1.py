import os
import json
import glob
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_DIR = "data/processed_graphs/"
MODEL_ID  = "maxidl/Llama-OpenReviewer-8B"
DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
).to(DEVICE)
model.eval()

REVIEW_FIELDS = """## Summary
Briefly summarize the paper and its contributions.

## Soundness
4: excellent | 3: good | 2: fair | 1: poor

## Presentation
4: excellent | 3: good | 2: fair | 1: poor

## Contribution
4: excellent | 3: good | 2: fair | 1: poor

## Strengths
A substantive assessment of the strengths of the paper.

## Weaknesses
A substantive assessment of the weaknesses of the paper.

## Questions
Questions and suggestions for the authors.

## Rating
1: strong reject | 3: reject, not good enough | 5: marginally below acceptance |
6: marginally above acceptance | 8: accept, good paper | 10: strong accept"""

SYSTEM_PROMPT = f"""You are an expert reviewer for AI conferences. You follow best practices \
and review papers according to the reviewer guidelines.

Reviewer guidelines:
1. Read the paper carefully and consider objective, strong points, and weak points.
2. Answer key questions to make a recommendation to Accept or Reject.
3. Write your review including a summary, strengths, weaknesses, questions, and a rating.

Your reviews contain the following sections:

# Review

{REVIEW_FIELDS}

Your response must only contain the review in markdown format with sections as defined above."""


def naive_sentence_split(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sentences if sentences else [text.strip()]


def extract_paper_text(dom_data):
    """
    Extracts abstract and introduction text from the paper DOM.
    FIX 1: Uses 'title' not 'heading' to match our actual schema.
    """
    text_blocks = []
    for section in dom_data.get("sections", []):
        header = section.get("title", "").lower()
        if any(kw in header for kw in ["abstract", "introduction", "background"]):
            for block in section.get("blocks", []):
                text = block.get("text", "").strip()
                if text:
                    text_blocks.append(text)

    if not text_blocks:
        for section in dom_data.get("sections", [])[:3]:
            for block in section.get("blocks", []):
                text = block.get("text", "").strip()
                if text:
                    text_blocks.append(text)

    return "\n\n".join(text_blocks)[:6000]


def generate_text(messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=800,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    )


def atomic_save(filepath, data):
    """Write to a temp file then rename — prevents corruption on crash."""
    tmp_path = filepath + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, filepath)


def process_pass_1():
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} graphs to process.")

    for idx, filepath in enumerate(json_files):
        print(f"\n[{idx+1}/{len(json_files)}] Pass 1: {os.path.basename(filepath)}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        paper_text = extract_paper_text(data.get("paper_DOM", {}))
        if len(paper_text) < 500:
            print("  -> Skipped: Not enough paper text extracted.")
            continue

        if "adversarial_reviews" not in data:
            data["adversarial_reviews"] = []

        existing_modes = {r.get("mode") for r in data["adversarial_reviews"]}
        changed = False

        try:
            if "Zero-Shot" not in existing_modes:
                print("  -> Generating Mode 1: Zero-Shot...")
                zero_shot_text = generate_text([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Review the following paper:\n\n{paper_text}"}
                ])
                data["adversarial_reviews"].append({
                    "mode":             "Zero-Shot",
                    "generation_model": MODEL_ID,
                    "generation_role":  "zero_shot_reviewer",
                    "review_sentences": [
                        {
                            "sentence_id":  f"ai_m1_s_{i}",
                            "text":         s,
                            "author_label": "AI_GENERATED"
                        }
                        for i, s in enumerate(naive_sentence_split(zero_shot_text))
                    ]
                })
                existing_modes.add("Zero-Shot")
                changed = True
            else:
                print("  -> Mode 1 already exists. Skipping.")
                zero_shot_text = " ".join(
                    s["text"] for r in data["adversarial_reviews"]
                    if r.get("mode") == "Zero-Shot"
                    for s in r.get("review_sentences", [])
                )

            if "Paraphrased" not in existing_modes:
                print("  -> Generating Mode 2: Paraphrased...")
                paraphrased_text = generate_text([
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the provided peer review. Keep the exact same "
                            "technical meaning and structure, but change the vocabulary "
                            "and phrasing substantially."
                        )
                    },
                    {"role": "user", "content": f"Rewrite this review:\n\n{zero_shot_text}"}
                ])
                data["adversarial_reviews"].append({
                    "mode":             "Paraphrased",
                    "generation_model": MODEL_ID,
                    "generation_role":  "paraphraser",
                    "review_sentences": [
                        {
                            "sentence_id":  f"ai_m2_s_{i}",
                            "text":         s,
                            "author_label": "AI_GENERATED"
                        }
                        for i, s in enumerate(naive_sentence_split(paraphrased_text))
                    ]
                })
                changed = True
            else:
                print("  -> Mode 2 already exists. Skipping.")

        except Exception as e:
            print(f"  -> Error: {e}")

        if changed:
            atomic_save(filepath, data)
            print("  -> Saved.")


if __name__ == "__main__":
    process_pass_1()