"""Baseline: just try random plans and keep the best one."""
import numpy as np
from src.chromosome import Chromosome
from src.fitness import FitnessEvaluator
from src.operators import GeneticOperators


class RandomSearch:
    def __init__(self, env, budget, seed=0, record_every=50, weights=None):
        self.env = env
        self.budget = budget                  # same number of evaluations as the GA
        self.record_every = record_every
        self.rng = np.random.default_rng(seed)
        self.evaluator = FitnessEvaluator(env, weights)
        self.ops = GeneticOperators(env, self.rng)

    def run(self):
        best, best_score = None, -np.inf
        evals_hist, best_hist = [], []
        scores = []
        for i in range(1, self.budget + 1):
            chrom = self.ops.repair(Chromosome.random(self.rng, self.env.size))
            score = self.evaluator.evaluate(chrom)["score"]
            scores.append(score)
            if score > best_score:
                best, best_score = chrom, score
            if i % self.record_every == 0 or i == self.budget:
                evals_hist.append(i)
                best_hist.append(best_score)
        return {"best": best, "best_score": best_score, "evals": evals_hist,
                "history": best_hist, "all_scores": scores}
