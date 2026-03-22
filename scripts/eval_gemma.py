import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID   = "google/gemma-2-2b-it"
PEFT_PATH  = "models/gemma-physics"

# load test data
test_data = [json.loads(l) for l in Path("data/finetune/test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

# load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(PEFT_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(model, PEFT_PATH, device_map={"": 0})
model.eval()

def extract_concepts(problem_text):
    prompt = f"<start_of_turn>user\nList the physics concepts required to solve this problem.\n\n{problem_text[:400]}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# evaluate on test set
correct = 0
results = []
for i, example in enumerate(test_data[:20]):
    pred    = extract_concepts(example["input"])
    gold    = json.loads(example["output"])
    pred_concepts = set(c.lower().strip() for c in (json.loads(pred) if pred.startswith("[") else [pred]))
    gold_concepts = set(c.lower().strip() for c in gold)
    overlap = len(pred_concepts & gold_concepts)
    precision = overlap / len(pred_concepts) if pred_concepts else 0
    recall    = overlap / len(gold_concepts) if gold_concepts else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    results.append({"f1": f1, "pred": pred, "gold": example["output"]})
    print(f"[{i+1}] F1={f1:.2f} | gold={example['output'][:60]} | pred={pred[:60]}")

avg_f1 = sum(r["f1"] for r in results) / len(results)
print(f"\nAvg F1: {avg_f1:.3f} over {len(results)} test examples")