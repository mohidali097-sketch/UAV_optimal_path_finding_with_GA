"""One chromosome = one complete plan for the whole fleet."""
import numpy as np
from src import config


class Chromosome:
    """genes has shape (drones, waypoints, 4).

    The 4 numbers of every waypoint are: x, y, speed, sensor power.
    """

    def __init__(self, genes):
        self.genes = genes

    @classmethod
    def random(cls, rng, size=config.GRID_SIZE,
               num_drones=config.NUM_DRONES, num_waypoints=config.NUM_WAYPOINTS):
        shape = (num_drones, num_waypoints)
        genes = np.zeros(shape + (4,))
        genes[:, :, 0] = rng.uniform(0, size - 1, shape)               # x
        genes[:, :, 1] = rng.uniform(0, size - 1, shape)               # y
        genes[:, :, 2] = rng.uniform(*config.SPEED_RANGE, shape)       # speed
        genes[:, :, 3] = rng.uniform(*config.SENSOR_RANGE, shape)      # sensor power
        return cls(genes)

    def copy(self):
        return Chromosome(self.genes.copy())
