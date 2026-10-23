"""Run one full simulation: GA + random search + all main graphs.

    python main.py             # saves graphs in results/
    python main.py --show      # also pops up the graph windows
"""
import argparse
import os

from src.grid import Environment
from src.ga import GeneticAlgorithm
from src.random_search import RandomSearch
from src import visualize as viz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--mut", type=float, default=0.05)
    parser.add_argument("--tour", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    env = Environment()

    ga = GeneticAlgorithm(env, pop_size=args.pop, generations=args.gens,
                          mutation_rate=args.mut, tournament_size=args.tour, seed=args.seed)
    print("Running genetic algorithm...")
    result = ga.run()
    h = result["history"]
    print(f"  stopped after {result['generations_run']} generations, "
          f"{result['evaluations']} evaluations")
    print(f"  best fitness: {h['best'][0]:.3f} (generation 0) -> {h['best'][-1]:.3f} (final)")

    print("Running random search with the same budget...")
    rs = RandomSearch(env, budget=result["evaluations"], seed=args.seed).run()
    print(f"  random search best fitness: {rs['best_score']:.3f}")

    print("Saving graphs to results/ ...")
    viz.plot_convergence(h, "results/convergence.png", args.show)
    viz.plot_generation_snapshots(env, ga.evaluator, result["best_chroms"],
                                  "results/paths_and_coverage_over_generations.png", args.show)
    final = viz.plot_final_solution(env, ga.evaluator, result["best"],
                                    "results/final_solution.png", args.show)
    viz.plot_ga_vs_random(h, rs, "results/ga_vs_random.png", args.show)

    print("\nFinal solution breakdown:")
    for key, value in final["parts"].items():
        print(f"  {key:10s} {value:.3f}")
    print(f"  total fitness {final['score']:.3f}")


if __name__ == "__main__":
    main()
