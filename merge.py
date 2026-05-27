import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
ADAPTER_DIR = "./tinyllama-alpaca-sft"
MERGED_DIR = "./tinyllama-alpaca-merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging adapter into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR}...")
model.save_pretrained(MERGED_DIR)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("Done.")
