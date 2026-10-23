"""Hyperparameter tests and GA vs random search over several seeds.

    python experiments.py            # full run (about 10 minutes)
    python experiments.py --quick    # fewer seeds and generations (about 2 minutes)
"""
import argparse
import csv
import os
import numpy as np

from src.grid import Environment
from src.ga import GeneticAlgorithm
from src.random_search import RandomSearch
from src import visualize as viz


def run_setting(env, seeds, gens, **params):
    curves, finals, evals = [], [], []
    for seed in seeds:
        r = GeneticAlgorithm(env, generations=gens, seed=seed, **params).run()
        best = r["history"]["best"]
        best = best + [best[-1]] * (gens + 1 - len(best))     # pad if stopped early
        curves.append(best)
        finals.append(best[-1])
        evals.append(r["evaluations"])
    return np.array(curves), finals, evals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = list(range(1, 4)) if args.quick else list(range(1, 6))
    gens = 50 if args.quick else 100
    os.makedirs("results", exist_ok=True)
    env = Environment()

    sweeps = {
        "population size": ("pop_size", [20, 50, 100]),
        "mutation rate": ("mutation_rate", [0.01, 0.05, 0.2]),
        "tournament size": ("tournament_size", [2, 3, 5]),
    }
    rows = [["experiment", "value", "mean_final_fitness", "std_final_fitness", "mean_evaluations"]]

    for name, (param, values) in sweeps.items():
        print(f"\nTesting {name}: {values}")
        labels, curves, finals = [], [], []
        for v in values:
            c, f, e = run_setting(env, seeds, gens, **{param: v})
            labels.append(v)
            curves.append(c)
            finals.append(f)
            print(f"  {param}={v}: final fitness {np.mean(f):.3f} +- {np.std(f):.3f}")
            rows.append([name, v, round(np.mean(f), 4), round(np.std(f), 4), int(np.mean(e))])
        fname = name.replace(" ", "_")
        viz.plot_sweep(name, labels, curves, finals, f"results/sweep_{fname}.png")

    # adaptive mutation on / off
    print("\nTesting adaptive mutation: on vs off")
    labels, curves, finals = ["on", "off"], [], []
    for flag in (True, False):
        c, f, e = run_setting(env, seeds, gens, adaptive=flag)
        curves.append(c)
        finals.append(f)
        print(f"  adaptive={flag}: final fitness {np.mean(f):.3f} +- {np.std(f):.3f}")
        rows.append(["adaptive mutation", flag, round(np.mean(f), 4), round(np.std(f), 4), int(np.mean(e))])
    viz.plot_sweep("adaptive mutation", labels, curves, finals, "results/sweep_adaptive_mutation.png")

    # GA vs random search over seeds
    print("\nGA vs random search over seeds")
    ga_scores, rs_scores = [], []
    for seed in seeds:
        r = GeneticAlgorithm(env, generations=gens, seed=seed).run()
        rs = RandomSearch(env, budget=r["evaluations"], seed=seed).run()
        ga_scores.append(r["history"]["best"][-1])
        rs_scores.append(rs["best_score"])
        print(f"  seed {seed}: GA {ga_scores[-1]:.3f} | random {rs_scores[-1]:.3f}")
    viz.plot_boxplot_vs_random(ga_scores, rs_scores, "results/ga_vs_random_boxplot.png")
    rows.append(["GA", "-", round(np.mean(ga_scores), 4), round(np.std(ga_scores), 4), ""])
    rows.append(["random search", "-", round(np.mean(rs_scores), 4), round(np.std(rs_scores), 4), ""])

    with open("results/experiment_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("\nSaved all graphs and results/experiment_results.csv")


if __name__ == "__main__":
    main()
