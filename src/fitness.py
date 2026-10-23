"""Scores a chromosome. Higher score = better plan."""
import numpy as np
from src import config


class FitnessEvaluator:
    SAMPLES = 30      # points checked along each flight segment
    MAX_RADIUS = 2    # biggest sensor reach in cells

    def __init__(self, env, weights=None):
        self.env = env
        self.weights = weights or config.WEIGHTS
        self.t = np.linspace(0, 1, self.SAMPLES)
        k = np.arange(-self.MAX_RADIUS, self.MAX_RADIUS + 1)
        dy, dx = np.meshgrid(k, k, indexing="ij")
        self.dy, self.dx = dy.ravel(), dx.ravel()

    def evaluate(self, chrom):
        env, size = self.env, self.env.size
        rows, cols, sens = [], [], []
        energy = longest_time = range_excess = hazard_exposure = violation = 0.0

        for drone in chrom.genes:
            x, y, speed_all, sensor_all = drone[:, 0], drone[:, 1], drone[:, 2], drone[:, 3]
            d = np.hypot(np.diff(x), np.diff(y))          # length of each segment
            speed, sensor = speed_all[:-1], sensor_all[:-1]
            seg_time = d / speed

            energy += (d * speed ** 2).sum() + (0.5 * sensor * seg_time).sum()
            longest_time = max(longest_time, seg_time.sum())
            range_excess += max(0.0, d.sum() - config.MAX_RANGE)

            # points along every segment
            xs = x[:-1, None] + np.diff(x)[:, None] * self.t[None, :]
            ys = y[:-1, None] + np.diff(y)[:, None] * self.t[None, :]
            r = np.clip(np.rint(ys).astype(int), 0, size - 1)
            c = np.clip(np.rint(xs).astype(int), 0, size - 1)

            hazard_exposure += (env.hazard[r, c].mean(axis=1) * d).sum()
            violation += (env.no_fly[r, c].mean(axis=1) * d).sum()

            rows.append(r.ravel())
            cols.append(c.ravel())
            sens.append(np.repeat(sensor, self.SAMPLES))

        detect = self._detection_map(np.concatenate(rows), np.concatenate(cols),
                                     np.concatenate(sens))

        total_priority = env.priority.sum()
        parts = {
            "coverage": ((detect > 0) * env.priority).sum() / total_priority,
            "detection": (detect * env.priority).sum() / total_priority,
            "energy": min(energy / config.NORM["energy"], 1.5),
            "time": min(longest_time / config.NORM["time"], 1.5),
            "hazard": min(hazard_exposure / config.NORM["hazard"], 1.5),
        }
        w = self.weights
        score = (w["coverage"] * parts["coverage"] + w["detection"] * parts["detection"]
                 - w["energy"] * parts["energy"] - w["time"] * parts["time"]
                 - w["hazard"] * parts["hazard"])

        # Constraint handling by penalty: no-fly flying and range limit.
        penalty = config.PENALTY_WEIGHT * (violation / config.NORM["violation"]
                                           + range_excess / config.NORM["range"])
        parts["penalty"] = penalty
        return {"score": score - penalty, "parts": parts, "detect": detect}

    def _detection_map(self, R, C, S):
        """Mark every cell a sensor can see. Stronger sensors see further."""
        size = self.env.size
        radius = np.where(S >= 0.6, 2, 1)                     # weak sensor sees 1 cell around, strong sees 2
        rr = R[:, None] + self.dy[None, :]
        cc = C[:, None] + self.dx[None, :]
        ok = ((np.abs(self.dy)[None, :] <= radius[:, None])
              & (np.abs(self.dx)[None, :] <= radius[:, None])
              & (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size))
        values = np.broadcast_to(S[:, None], rr.shape)
        detect = np.zeros((size, size))
        np.maximum.at(detect, (rr[ok], cc[ok]), values[ok])   # keep strongest reading
        return detect
