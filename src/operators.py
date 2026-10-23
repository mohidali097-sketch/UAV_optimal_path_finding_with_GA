"""Selection, crossover, mutation and repair."""
import numpy as np
from src import config


class GeneticOperators:
    def __init__(self, env, rng):
        self.env = env
        self.rng = rng

    # ---------- selection ----------
    def tournament(self, population, fitnesses, k):
        """Pick k random plans and return the best one. Bigger k = more pressure."""
        picks = self.rng.integers(0, len(population), k)
        return population[picks[np.argmax(fitnesses[picks])]]

    # ---------- crossover ----------
    def crossover(self, p1, p2):
        """Mix two parents into two children using one of 3 methods."""
        a, b = p1.genes.copy(), p2.genes.copy()
        D, W, _ = a.shape
        kind = self.rng.integers(3)

        if kind == 0:        # swap whole drone routes
            for d in range(D):
                if self.rng.random() < 0.5:
                    a[d], b[d] = b[d].copy(), a[d].copy()
        elif kind == 1:      # swap a piece of each route
            for d in range(D):
                i, j = sorted(self.rng.choice(W + 1, 2, replace=False))
                a[d, i:j], b[d, i:j] = b[d, i:j].copy(), a[d, i:j].copy()
        else:                # geometric blend: children sit between the parents
            alpha = self.rng.random((D, W, 1))
            a, b = alpha * p1.genes + (1 - alpha) * p2.genes, \
                   (1 - alpha) * p1.genes + alpha * p2.genes

        return type(p1)(a), type(p1)(b)

    # ---------- mutation ----------
    def mutate(self, chrom, rate):
        """rate = chance that each waypoint is changed."""
        g = chrom.genes
        D, W, _ = g.shape
        hit = self.rng.random((D, W)) < rate
        nudge = hit & (self.rng.random((D, W)) < 0.8)    # small move
        reset = hit & ~nudge                             # brand new waypoint

        n = nudge.sum()
        g[..., 0:2][nudge] += self.rng.normal(0, 3.0, (n, 2))   # move up to a few cells
        g[..., 2][nudge] += self.rng.normal(0, 0.1, n)
        g[..., 3][nudge] += self.rng.normal(0, 0.1, n)

        m = reset.sum()
        if m:
            g[reset] = np.column_stack([
                self.rng.uniform(0, self.env.size - 1, m),
                self.rng.uniform(0, self.env.size - 1, m),
                self.rng.uniform(*config.SPEED_RANGE, m),
                self.rng.uniform(*config.SENSOR_RANGE, m)])

        # order mutation: swap two waypoints inside one route
        if self.rng.random() < 0.1:
            d = self.rng.integers(D)
            i, j = self.rng.choice(W, 2, replace=False)
            g[d, [i, j]] = g[d, [j, i]]
        return chrom

    # ---------- repair ----------
    def repair(self, chrom):
        """Keep values legal. Move waypoints out of no-fly zones."""
        g, env = chrom.genes, self.env
        g[..., 0:2] = np.clip(g[..., 0:2], 0, env.size - 1)
        g[..., 2] = np.clip(g[..., 2], *config.SPEED_RANGE)
        g[..., 3] = np.clip(g[..., 3], *config.SENSOR_RANGE)

        r = np.rint(g[..., 1]).astype(int)
        c = np.rint(g[..., 0]).astype(int)
        blocked = env.no_fly[r, c] == 1
        if blocked.any():
            g[..., 1][blocked] = env.nearest_row[r, c][blocked]
            g[..., 0][blocked] = env.nearest_col[r, c][blocked]
        return chrom

    # ---------- diversity ----------
    @staticmethod
    def diversity(population):
        """Average spread of waypoint positions. Near 0 = everyone is the same."""
        xy = np.stack([c.genes[..., 0:2] for c in population])
        return float(xy.std(axis=0).mean())
