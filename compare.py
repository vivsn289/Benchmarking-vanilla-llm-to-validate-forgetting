import json
import glob
import os

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
