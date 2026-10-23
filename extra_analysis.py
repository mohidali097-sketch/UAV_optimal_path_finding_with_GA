"""Extra analysis used in the report (weight test, random-search breakdown, solution uniqueness).

    python extra_analysis.py        # takes about 5 minutes

Uses the same seeds (1 to 5) as experiments.py, so the GA numbers match it exactly.
Saves results/extra_results.json and results/uniqueness.png
"""
import itertools
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from src.grid import Environment
from src.ga import GeneticAlgorithm
from src.random_search import RandomSearch
from src import visualize as viz

SEEDS = [1, 2, 3, 4, 5]

WEIGHT_SETS = {
    "A balanced (default)": {"coverage": 0.35, "detection": 0.25, "energy": 0.15, "time": 0.10, "hazard": 0.15},
    "B mission-first":      {"coverage": 0.50, "detection": 0.30, "energy": 0.05, "time": 0.05, "hazard": 0.10},
    "C cost and safety-first": {"coverage": 0.20, "detection": 0.15, "energy": 0.25, "time": 0.15, "hazard": 0.25},
}


def clean(parts):
    return {k: round(float(v), 4) for k, v in parts.items()}


def mean_parts(list_of_parts):
    return {k: round(float(np.mean([p[k] for p in list_of_parts])), 4) for k in list_of_parts[0]}


def jaccard(a, b):
    """How much two covered areas overlap (1 = identical, 0 = nothing in common)."""
    a, b = a > 0, b > 0
    return float((a & b).sum() / (a | b).sum())


def main():
    os.makedirs("results", exist_ok=True)
    env = Environment()
    out = {}

    # 1. GA with default settings, 5 seeds
    print("1/4  GA default settings...")
    ga_runs = []
    for s in SEEDS:
        ga = GeneticAlgorithm(env, seed=s)
        r = ga.run()
        ev = ga.evaluator.evaluate(r["best"])
        ga_runs.append({"seed": s, "chrom": r["best"], "score": ev["score"],
                        "parts": clean(ev["parts"]), "detect": ev["detect"],
                        "evals": r["evaluations"]})
        print(f"   seed {s}: {ev['score']:.4f}")
    out["ga_scores"] = [round(float(r["score"]), 4) for r in ga_runs]
    out["ga_parts_mean"] = mean_parts([r["parts"] for r in ga_runs])

    # 2. Random search with the same budget
    print("2/4  Random search...")
    rs_runs = []
    for r in ga_runs:
        rs = RandomSearch(env, budget=r["evals"], seed=r["seed"]).run()
        ev = GeneticAlgorithm(env, seed=0).evaluator.evaluate(rs["best"])
        rs_runs.append({"seed": r["seed"], "chrom": rs["best"], "score": rs["best_score"],
                        "parts": clean(ev["parts"]), "detect": ev["detect"],
                        "n_above_zero": int(sum(x > 0 for x in rs["all_scores"])),
                        "n_plans": len(rs["all_scores"])})
    out["rs_scores"] = [round(float(r["score"]), 4) for r in rs_runs]
    out["rs_parts_mean"] = mean_parts([r["parts"] for r in rs_runs])
    out["rs_plans_above_zero"] = int(sum(r["n_above_zero"] for r in rs_runs))
    out["rs_plans_total"] = int(sum(r["n_plans"] for r in rs_runs))

    # 3. Uniqueness: how different are the final plans from each other?
    print("3/4  Uniqueness...")
    ga_j = [jaccard(a["detect"], b["detect"]) for a, b in itertools.combinations(ga_runs, 2)]
    rs_j = [jaccard(a["detect"], b["detect"]) for a, b in itertools.combinations(rs_runs, 2)]
    out["jaccard_ga_mean"] = round(float(np.mean(ga_j)), 3)
    out["jaccard_ga_min"] = round(float(np.min(ga_j)), 3)
    out["jaccard_ga_max"] = round(float(np.max(ga_j)), 3)
    out["jaccard_rs_mean"] = round(float(np.mean(rs_j)), 3)

    evaluator = GeneticAlgorithm(env, seed=0).evaluator
    panels = [("Random search best (seed 1)", rs_runs[0]["chrom"], rs_runs[0]["score"])]
    panels += [(f"GA best (seed {r['seed']})", r["chrom"], r["score"]) for r in ga_runs[:3]]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.9))
    for ax, (title, chrom, score) in zip(axes, panels):
        viz.draw_map(ax, env)
        viz.draw_paths(ax, chrom)
        ax.set_title(f"{title}\nfitness {score:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig("results/uniqueness.png", dpi=150)
    plt.close(fig)

    # 4. Weight sensitivity + one combined "tuned" setting
    print("4/4  Weight test and combined setting...")
    out["weights"] = {}
    for name, w in WEIGHT_SETS.items():
        parts_list = []
        for s in SEEDS:
            ga = GeneticAlgorithm(env, seed=s, weights=w)
            r = ga.run()
            parts_list.append(clean(ga.evaluator.evaluate(r["best"])["parts"]))
        out["weights"][name] = mean_parts(parts_list)
        print("   ", name, out["weights"][name])

    tuned = []
    for s in SEEDS:
        r = GeneticAlgorithm(env, seed=s, mutation_rate=0.01, tournament_size=5, adaptive=False).run()
        tuned.append(r["history"]["best"][-1])
    out["tuned_scores"] = [round(float(x), 4) for x in tuned]
    out["tuned_mean"] = round(float(np.mean(tuned)), 4)
    out["tuned_std"] = round(float(np.std(tuned)), 4)

    with open("results/extra_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
