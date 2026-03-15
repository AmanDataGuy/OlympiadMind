"""
describe_figures.py
-------------------
Step 2.3b — Figure Description with Qwen2-VL

Second pass — reads saved figure images from extract_figures_fast.py
and adds structured descriptions using Qwen2-VL 2B.

Runs selectively — only on olympiad problems by default, not solutions.
Can be run on any subset by changing SOURCE_FILTER and DOCTYPE_FILTER.

Output: updates existing {doc_id}_figures.json with description field filled
"""

import json
import torch
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ── config ────────────────────────────────────────────────────────────────────

FIGURES_DIR = Path("data/extracted/figures")
IMAGES_DIR  = FIGURES_DIR / "images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# only describe figures from these sources — skip solutions to save time
SOURCE_FILTER  = {"olympiad", "cbse", "science_bowl"}
DOCTYPE_FILTER = {"problem", "assertion_reason", "mcq"}

# only describe figures larger than this — skip small decorative images
MIN_DESCRIBE_SIZE = 150


# ── prompts ───────────────────────────────────────────────────────────────────

PROMPTS = {
    "force diagram with arrows": """Describe this physics force diagram in JSON:
{"diagram_type":"force diagram","physics_concept":"","objects":[],"forces":[],"description":""}
Return ONLY valid JSON.""",

    "electric circuit diagram": """Describe this electric circuit in JSON:
{"diagram_type":"circuit diagram","physics_concept":"","components":[],"topology":"","description":""}
Return ONLY valid JSON.""",

    "graph or plot with axes": """Describe this physics graph in JSON:
{"diagram_type":"graph","physics_concept":"","x_axis":"","y_axis":"","curve_behavior":"","description":""}
Return ONLY valid JSON.""",

    "default": """Describe this physics diagram in JSON:
{"diagram_type":"","physics_concept":"","axes_labels":[],"annotations":[],"description":""}
Return ONLY valid JSON.""",
}


# ── qwen2-vl ──────────────────────────────────────────────────────────────────

def load_qwen():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor


def describe_figure(qwen_model, processor, image: Image.Image, diagram_type: str) -> dict:
    prompt = PROMPTS.get(diagram_type, PROMPTS["default"])

    # resize large images — prevents Qwen2-VL from hanging on high-res figures
    if image.width > 1000 or image.height > 1000:
        image = image.resize((1000, 1000), Image.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    from qwen_vl_utils import process_vision_info
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response  = processor.batch_decode(generated, skip_special_tokens=True)[0]

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "diagram_type":    diagram_type,
            "physics_concept": "unknown",
            "description":     response[:200],
            "parse_error":     True,
        }


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading Qwen2-VL 2B on {DEVICE}...\n")
    qwen_model, qwen_processor = load_qwen()
    print("Model loaded. Starting description...\n")

    json_files = list(FIGURES_DIR.glob("*.json"))

    # filter to target sources and doc types
    target_files = []
    for f in json_files:
        data = json.load(open(f, encoding="utf-8"))
        if data.get("source") not in SOURCE_FILTER:
            continue
        target_files.append(f)

    print(f"Files to describe: {len(target_files)}\n")

    total_described = 0
    failed          = 0

    for json_path in tqdm(target_files, desc="Describing figures"):
        data    = json.load(open(json_path, encoding="utf-8"))
        figures = data.get("figures", [])
        updated = False

        for fig in figures:
            # skip if already described
            if fig.get("description") is not None:
                continue

            # skip small figures
            if fig.get("width", 0) < MIN_DESCRIBE_SIZE or fig.get("height", 0) < MIN_DESCRIBE_SIZE:
                continue

            image_path = Path(fig["image_path"])
            if not image_path.exists():
                continue

            try:
                image       = Image.open(image_path).convert("RGB")
                description = describe_figure(
                    qwen_model, qwen_processor, image, fig["diagram_type"]
                )
                fig.update(description)
                total_described += 1
                updated = True
            except Exception as e:
                fig["description"]  = f"failed: {str(e)}"
                fig["parse_error"]  = True
                failed += 1

        # write back only if something changed
        if updated:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Described : {total_described}")
    print(f"  Failed    : {failed}")


if __name__ == "__main__":
    main()