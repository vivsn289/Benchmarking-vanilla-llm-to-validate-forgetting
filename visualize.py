import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_scores(results_dir):
    scores = {}
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname.startswith("results") and fname.endswith(".json"):
                with open(os.path.join(root, fname)) as f:
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

# Top-level tasks only (exclude mmlu subtasks)
TOP_TASKS = ["arc_challenge", "hellaswag", "boolq", "mmlu", "winogrande"]
tasks = [t for t in TOP_TASKS if t in base and t in inst]
base_vals = [base[t] for t in tasks]
inst_vals = [inst[t] for t in tasks]
deltas = [inst[t] - base[t] for t in tasks]
forgetting = [(base[t] - inst[t]) / base[t] * 100 if base[t] > 0 else 0 for t in tasks]

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
severe = [labels[i] for i, f in enumerate(forgetting) if f > 5]
improved = [labels[i] for i, f in enumerate(forgetting) if f < -5]

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
