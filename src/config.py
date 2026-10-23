"""All the settings in one place. Change numbers here, not inside other files."""

# ---- World ----
GRID_SIZE = 50          # grid is 50 x 50 cells

# ---- Fleet ----
NUM_DRONES = 3
NUM_WAYPOINTS = 15      # waypoints per drone
MAX_RANGE = 200.0       # max distance one drone may fly (hard limit)
SPEED_RANGE = (0.5, 1.5)
SENSOR_RANGE = (0.2, 1.0)

# ---- Fitness weights (good things positive, bad things negative) ----
WEIGHTS = {
    "coverage": 0.35,   # how much priority area was seen
    "detection": 0.25,  # how strongly it was seen (sensor power)
    "energy": 0.15,     # battery used
    "time": 0.10,       # slowest drone's flight time
    "hazard": 0.15,     # time spent in dangerous cells
}

# ---- Used to squeeze each part into roughly 0..1 ----
NORM = {
    "energy": 1500.0,
    "time": 500.0,
    "hazard": 150.0,
    "violation": 40.0,   # path length inside no-fly zones
    "range": 150.0,      # distance flown beyond MAX_RANGE
}
PENALTY_WEIGHT = 1.0    # how harshly rule-breaking is punished
