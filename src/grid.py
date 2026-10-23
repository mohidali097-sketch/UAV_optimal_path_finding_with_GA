"""The map the drones fly over."""
import numpy as np


class Environment:
    """A square grid with no-fly zones, hazard zones and priority zones.

    Always index the grid as grid[row, col] = grid[y, x].
    """

    def __init__(self, size=50):
        self.size = size
        self.no_fly = np.zeros((size, size))     # 1 = blocked
        self.hazard = np.zeros((size, size))     # 0..1 = how dangerous
        self.priority = np.ones((size, size))    # 1..5 = how important
        self._build_map()
        self._build_repair_table()

    def _build_map(self):
        # Fixed map so every run and every team member gets the same world.
        self.no_fly[10:18, 20:28] = 1
        self.no_fly[38:44, 28:36] = 1
        self.no_fly[26:30, 5:12] = 1

        self.hazard[25:35, 36:46] = 0.8
        self.hazard[5:12, 38:48] = 0.5
        self.hazard[20:28, 14:22] = 0.6

        self.priority[32:42, 8:20] = 5
        self.priority[4:14, 4:14] = 4
        self.priority[20:30, 24:34] = 3

    def _build_repair_table(self):
        """For every blocked cell, remember the closest free cell."""
        n = self.size
        self.nearest_row, self.nearest_col = np.indices((n, n))
        blocked = np.argwhere(self.no_fly == 1)
        free = np.argwhere(self.no_fly == 0)
        for r, c in blocked:
            dist = (free[:, 0] - r) ** 2 + (free[:, 1] - c) ** 2
            best = free[np.argmin(dist)]
            self.nearest_row[r, c] = best[0]
            self.nearest_col[r, c] = best[1]
