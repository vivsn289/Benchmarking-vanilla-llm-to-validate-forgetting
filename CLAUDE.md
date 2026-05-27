# CLAUDE.md

## What This Project Does

Measures catastrophic forgetting caused by Self-Instruct style instruction fine-tuning.
Benchmarks TinyLlama 1.1B before and after fine-tuning on Alpaca 52K data.
Produces per-domain forgetting scores to determine if forgetting is negligible, uniform, or domain-specific.

## Hardware

NVIDIA RTX 4050 (6GB VRAM). This is tight for even TinyLlama 1.1B during training.
Key constraints this imposes on every step:

- Benchmarks (Steps 2, 4): Use `dtype=float16` with `--batch_size 1`. Should fit in ~3GB.
- Fine-tuning (Step 3): Use LoRA (not full fine-tune). Full SFT on 6GB will OOM. LoRA uses ~4GB.
- If anything OOMs: add `load_in_4bit=True` to model_args. This cuts memory in half.

## Steps

Execute these in order. Each step must complete before the next begins.

### Step 1: Install Dependencies

```bash
pip install torch transformers accelerate datasets trl peft lm-eval rouge-score matplotlib pandas tabulate bitsandbytes --quiet
```

Verify GPU:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 2: Benchmark Base Model

Run lm-eval on the base TinyLlama model. Save results to results/base/.

```bash
mkdir -p results/base results/instruct

lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T,dtype=float16 \
    --tasks arc_challenge,hellaswag,boolq,mmlu,winogrande \
    --batch_size 1 \
    --output_path results/base/
```

If GPU runs out of memory, change model_args to:
`pretrained=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T,load_in_4bit=True`
If a specific task fails, remove it from the list and continue with the rest.

This step takes 60-120 minutes on RTX 4050 with batch_size 1.

### Step 3: Fine-tune on Alpaca 52K

Create and run `finetune.py` with this exact content:

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

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
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    max_seq_length=512,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=BASE_MODEL,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

print(f"Fine-tuning with LoRA on {len(dataset)} samples...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Model saved to {OUTPUT_DIR}")
```

Run:
```bash
python finetune.py
```

This takes 2-3 hours on RTX 4050 with LoRA. Uses ~4GB VRAM.

Note: LoRA fine-tunes adapter weights only, not the full model. This is a limitation —
full fine-tune would show more forgetting. Document this in your findings. If you get
access to a bigger GPU later (Colab A100, university cluster), rerun with full fine-tune
by removing the peft_config and LoRA import.

### Step 4: Benchmark Fine-tuned Model

```bash
lm_eval --model hf \
    --model_args pretrained=./tinyllama-alpaca-sft,dtype=float16 \
    --tasks arc_challenge,hellaswag,boolq,mmlu,winogrande \
    --batch_size 1 \
    --output_path results/instruct/
```

Same tasks, same settings as Step 2.

### Step 5: Compare Results

Create and run `compare.py`:

```python
import json
import glob
import os

def load_scores(results_dir):
    scores = {}
    for jf in glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True):
        with open(jf) as f:
            data = json.load(f)
        if "results" in data:
            for task, metrics in data["results"].items():
                for key in ["acc_norm,none", "acc,none", "exact_match,none"]:
                    if key in metrics:
                        scores[task] = round(metrics[key] * 100, 2)
                        break
    return scores

base = load_scores("results/base")
inst = load_scores("results/instruct")

print(f"\n{'Task':<25} {'Base':>8} {'Instruct':>10} {'Delta':>8} {'Forgetting':>12}")
print("-" * 65)

for task in sorted(base.keys()):
    if task in inst:
        b = base[task]
        i = inst[task]
        delta = round(i - b, 2)
        fgt = round((b - i) / b * 100, 2) if b > 0 else 0
        flag = " <<<" if fgt > 5 else (" ++" if fgt < -5 else "")
        print(f"{task:<25} {b:>8} {i:>10} {delta:>+8} {fgt:>10}%{flag}")

print("\n<<< = significant forgetting (>5%)")
print(" ++ = significant improvement (>5%)")
```

Run:
```bash
python compare.py
```

This prints the forgetting table instantly.

### Step 6: Generate Visualizations

Create and run `visualize.py`:

```python
import json
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def load_scores(results_dir):
    scores = {}
    for jf in glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True):
        with open(jf) as f:
            data = json.load(f)
        if "results" in data:
            for task, metrics in data["results"].items():
                for key in ["acc_norm,none", "acc,none", "exact_match,none"]:
                    if key in metrics:
                        scores[task] = round(metrics[key] * 100, 2)
                        break
    return scores

base = load_scores("results/base")
inst = load_scores("results/instruct")

tasks = sorted([t for t in base if t in inst])
base_vals = [base[t] for t in tasks]
inst_vals = [inst[t] for t in tasks]
deltas = [inst[t] - base[t] for t in tasks]
forgetting = [(base[t] - inst[t]) / base[t] * 100 if base[t] > 0 else 0 for t in tasks]

# Clean up task names for display
labels = [t.replace("_", " ").title() for t in tasks]

os.makedirs("results/figures", exist_ok=True)

# ── FIGURE 1: Side-by-side comparison ──
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(tasks))
w = 0.35
bars1 = ax.bar([i - w/2 for i in x], base_vals, w, label="Base (TinyLlama)", color="#3274A1")
bars2 = ax.bar([i + w/2 for i in x], inst_vals, w, label="Instruction-Tuned (Alpaca SFT)", color="#E1812C")
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Base vs Instruction-Tuned: Per-Benchmark Performance", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(base_vals), max(inst_vals)) * 1.15)
# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/comparison.png", dpi=150)
plt.close()
print("Saved: results/figures/comparison.png")

# ── FIGURE 2: Forgetting scores (horizontal bar) ──
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#E74C3C" if f > 5 else ("#27AE60" if f < -5 else "#95A5A6") for f in forgetting]
bars = ax.barh(labels, forgetting, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Forgetting Score (%)  —  Positive = Lost  |  Negative = Gained", fontsize=11)
ax.set_title("Per-Domain Forgetting After Instruction Tuning", fontsize=14, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.8)
ax.axvline(x=5, color="#E74C3C", linewidth=0.8, linestyle="--", alpha=0.4)
ax.axvline(x=-5, color="#27AE60", linewidth=0.8, linestyle="--", alpha=0.4)
# Value labels
for bar, val in zip(bars, forgetting):
    xpos = bar.get_width() + 0.3 if bar.get_width() >= 0 else bar.get_width() - 0.3
    ha = "left" if bar.get_width() >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", ha=ha, va="center", fontsize=9)
red_patch = mpatches.Patch(color="#E74C3C", label="Significant forgetting (>5%)")
green_patch = mpatches.Patch(color="#27AE60", label="Significant improvement (>5%)")
gray_patch = mpatches.Patch(color="#95A5A6", label="Stable (±5%)")
ax.legend(handles=[red_patch, gray_patch, green_patch], loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/forgetting.png", dpi=150)
plt.close()
print("Saved: results/figures/forgetting.png")

# ── FIGURE 3: Delta waterfall chart ──
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in deltas]
ax.bar(labels, deltas, color=colors, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Accuracy Change (percentage points)", fontsize=11)
ax.set_title("Impact of Instruction Tuning: Per-Benchmark Accuracy Change", fontsize=14, fontweight="bold")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
for i, (d, label) in enumerate(zip(deltas, labels)):
    ax.text(i, d + (0.3 if d >= 0 else -0.3), f"{d:+.1f}", ha="center",
            va="bottom" if d >= 0 else "top", fontsize=9, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/delta_waterfall.png", dpi=150)
plt.close()
print("Saved: results/figures/delta_waterfall.png")

# ── Summary stats ──
avg_fgt = sum(forgetting) / len(forgetting)
max_fgt_task = tasks[forgetting.index(max(forgetting))]
min_fgt_task = tasks[forgetting.index(min(forgetting))]
severe = [t for t, f in zip(labels, forgetting) if f > 5]
improved = [t for t, f in zip(labels, forgetting) if f < -5]

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"Average forgetting:    {avg_fgt:.2f}%")
print(f"Worst forgetting:      {max(forgetting):.2f}% ({max_fgt_task})")
print(f"Best improvement:      {min(forgetting):.2f}% ({min_fgt_task})")
print(f"Domains with severe forgetting (>5%): {severe if severe else 'None'}")
print(f"Domains that improved (>5%): {improved if improved else 'None'}")
print(f"Forgetting spread (max - min): {max(forgetting) - min(forgetting):.2f}%")
print(f"\nVERDICT:", end=" ")
spread = max(forgetting) - min(forgetting)
if max(forgetting) < 3:
    print("Minimal forgetting. Adaptive rehearsal has WEAK motivation.")
elif spread < 5:
    print("Uniform moderate forgetting. Simple data mixing may suffice. Adaptive rehearsal has MODERATE motivation.")
else:
    print("Domain-specific uneven forgetting. Adaptive self-rehearsal has STRONG motivation.")
print(f"{'='*50}")
```

Run:
```bash
python visualize.py
```

This creates three publication-ready figures in results/figures/:
- comparison.png — side-by-side bars of base vs instruct scores
- forgetting.png — horizontal bar chart of per-domain forgetting scores with color coding
- delta_waterfall.png — waterfall chart showing accuracy gain/loss per benchmark

It also prints a verdict on whether the forgetting pattern supports adaptive self-rehearsal.

## Important Notes

- RTX 4050 has 6GB VRAM. LoRA is mandatory for fine-tuning. Benchmarks use batch_size=1.
- LoRA fine-tunes adapter weights only. This UNDERESTIMATES forgetting compared to full SFT. Note this when presenting results. If your professor wants full SFT numbers, use a university GPU cluster or Google Colab A100.
- Steps 2 and 4 (benchmarks): ~60-120 min each on RTX 4050.
- Step 3 (LoRA fine-tuning): ~2-3 hours on RTX 4050.
- Steps 5 and 6: seconds.
- Total wall time: ~5-7 hours. Your active time: ~20 minutes.
- Execute step by step: tell Claude Code "execute Step N" one at a time. Wait for each to complete.
- If Claude Code times out on a long command, ask it to write the command as a bash script, then run it yourself in tmux.
