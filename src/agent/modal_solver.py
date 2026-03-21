import modal
import sys

app = modal.App("olympiadmind-solver")

image = modal.Image.debian_slim().pip_install("transformers", "torch", "accelerate")

@app.function(image=image, gpu="A10G", timeout=120)
def run_mistral(prompt: str) -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id  = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    formatted = f"[INST] {prompt} [/INST]"
    inputs    = tokenizer(formatted, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

@app.local_entrypoint()
def main(prompt_path: str):
    with open(prompt_path, encoding="utf-8") as f:
        prompt = f.read()
    result = run_mistral.remote(prompt)
    print(result)