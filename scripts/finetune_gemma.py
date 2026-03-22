import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from huggingface_hub import login
import os
login(token=os.getenv("HF_TOKEN"))

# ── config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "google/gemma-2-2b-it"
OUTPUT_DIR = "models/gemma-physics"
MAX_SEQ    = 512

# ── load data ─────────────────────────────────────────────────────────────────
def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

train_data = load_jsonl("data/finetune/train.jsonl")
val_data   = load_jsonl("data/finetune/val.jsonl")

def format_prompt(example):
    return {
        "text": f"<start_of_turn>user\n{example['instruction']}\n\n{example['input']}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
    }

train_dataset = Dataset.from_list(train_data).map(format_prompt)
val_dataset   = Dataset.from_list(val_data).map(format_prompt)

# ── quantization ──────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)

# ── model + tokenizer ─────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config = bnb_config,
    device_map          = "auto",
)
model = prepare_model_for_kbit_training(model)

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── training args ─────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir               = OUTPUT_DIR,
    num_train_epochs         = 3,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate            = 2e-4,
    bf16                     = True,
    logging_steps            = 10,
    eval_strategy            = "no",
    save_strategy            = "epoch",
    report_to                = "none",
)

# ── trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model           = model,
    args            = args,
    train_dataset   = train_dataset,
    formatting_func = lambda x: x["text"],
)

print("Starting training...")
trainer.train()

print("Saving model...")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")