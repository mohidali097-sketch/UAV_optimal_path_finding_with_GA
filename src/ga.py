"""The genetic algorithm loop."""
import numpy as np
from src.chromosome import Chromosome
from src.fitness import FitnessEvaluator
from src.operators import GeneticOperators


class GeneticAlgorithm:
    def __init__(self, env, pop_size=50, generations=100, crossover_rate=0.9,
                 mutation_rate=0.05, tournament_size=3, elite=2, patience=30,
                 adaptive=True, diversity_floor=4.0, seed=0, weights=None):
        self.env = env
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite = elite
        self.patience = patience
        self.adaptive = adaptive                # boost mutation when population gets too similar
        self.diversity_floor = diversity_floor
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.evaluator = FitnessEvaluator(env, weights)
        self.ops = GeneticOperators(env, self.rng)

    def _new_individual(self):
        return self.ops.repair(Chromosome.random(self.rng, self.env.size))

    def run(self):
        pop = [self._new_individual() for _ in range(self.pop_size)]
        fits = np.array([self.evaluator.evaluate(c)["score"] for c in pop])
        evals = self.pop_size

        hist = {"best": [], "avg": [], "diversity": [], "evals": [], "mut_rate": []}
        best_chroms = []
        best_so_far, last_improve = -np.inf, 0

        for gen in range(self.generations + 1):
            diversity = self.ops.diversity(pop)
            rate = self.mutation_rate
            if self.adaptive and diversity < self.diversity_floor:
                rate = min(self.mutation_rate * 3, 0.5)

            i_best = int(np.argmax(fits))
            hist["best"].append(float(fits[i_best]))
            hist["avg"].append(float(fits.mean()))
            hist["diversity"].append(diversity)
            hist["evals"].append(evals)
            hist["mut_rate"].append(rate)
            best_chroms.append(pop[i_best].copy())

            if fits[i_best] > best_so_far + 1e-4:
                best_so_far, last_improve = fits[i_best], gen
            if gen == self.generations or gen - last_improve >= self.patience:
                break

            # ---- build next generation ----
            order = np.argsort(fits)[::-1]
            new_pop = [pop[i].copy() for i in order[:self.elite]]     # elitism
            new_fits = [fits[i] for i in order[:self.elite]]
            fresh = []
            while len(new_pop) + len(fresh) < self.pop_size:
                p1 = self.ops.tournament(pop, fits, self.tournament_size)
                p2 = self.ops.tournament(pop, fits, self.tournament_size)
                if self.rng.random() < self.crossover_rate:
                    c1, c2 = self.ops.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for child in (c1, c2):
                    self.ops.mutate(child, rate)
                    self.ops.repair(child)
                    fresh.append(child)
            fresh = fresh[:self.pop_size - len(new_pop)]
            fresh_fits = [self.evaluator.evaluate(c)["score"] for c in fresh]
            evals += len(fresh)

            pop = new_pop + fresh
            fits = np.array(new_fits + fresh_fits)

        final = max(best_chroms, key=lambda c: self.evaluator.evaluate(c)["score"])
        return {"best": final, "history": hist, "best_chroms": best_chroms,
                "evaluations": evals, "generations_run": len(hist["best"]) - 1}
