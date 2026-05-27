import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# Ada Lovelace (RTX 40xx) performance flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUTPUT_DIR = "./tinyllama-alpaca-sft"

dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_sample(sample):
    if sample.get("input", "").strip():
        text = (
            f"Below is an instruction that describes a task, paired with an input. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    else:
        text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    return {"text": text}

dataset = dataset.map(format_sample)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config — required for RTX 4050 (6GB VRAM). Full fine-tune will OOM.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,      # Small batch for 6GB VRAM
    gradient_accumulation_steps=16,     # Effective batch = 32
    learning_rate=2e-4,                 # Higher LR for LoRA
    lr_scheduler_type="cosine",
    warmup_steps=50,
    fp16=True,                            # Match checkpoint (was saved with fp16)
    tf32=True,                            # TF32 matmul — free speedup on RTX 40xx
    optim="adamw_torch",                  # Compatible with fp16 + checkpoint
    dataloader_num_workers=2,             # Parallel data prefetch
    dataloader_pin_memory=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=3,
    max_length=512,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=BASE_MODEL,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

print(f"Fine-tuning with LoRA on {len(dataset)} samples...")
trainer.train(resume_from_checkpoint=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Model saved to {OUTPUT_DIR}")
