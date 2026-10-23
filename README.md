# UAV Fleet Strategy Engine (Genetic Algorithm)

A Genetic Algorithm that plans waypoints, speeds and sensor power for a fleet of
drones over a 50x50 hazard grid. It balances coverage, detection, energy, time and
hazard exposure while respecting no-fly zones and a maximum flight range.

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

## 2. Run

```bash
python main.py                 # one full run, saves graphs to results/
python main.py --show          # same, and opens the graph windows
python main.py --pop 100 --gens 150 --mut 0.1 --tour 5 --seed 7

python experiments.py          # full hyperparameter tests (about 10 minutes)
python experiments.py --quick  # faster test (fewer seeds and generations)
```

## 3. Project structure

| File | What it does |
|------|--------------|
| `src/config.py` | All settings: drones, waypoints, weights, limits |
| `src/grid.py` | `Environment`: map with no-fly, hazard and priority zones |
| `src/chromosome.py` | `Chromosome`: one fleet plan, shape (drones, waypoints, 4) |
| `src/fitness.py` | `FitnessEvaluator`: scores a plan |
| `src/operators.py` | `GeneticOperators`: selection, crossover, mutation, repair |
| `src/ga.py` | `GeneticAlgorithm`: the main evolution loop |
| `src/random_search.py` | `RandomSearch`: baseline for comparison |
| `src/visualize.py` | All plots |
| `main.py` | Runs one simulation |
| `experiments.py` | Runs the sensitivity tests |

## 4. Design summary

**Chromosome:** 3 drones x 15 waypoints x 4 values (x, y, speed, sensor power), all real numbers.

**Fitness:**
`score = 0.35*coverage + 0.25*detection - 0.15*energy - 0.10*time - 0.15*hazard - penalty`
Every part is scaled to about 0..1 first so no single part dominates.

**Constraints:**
- No-fly zones: waypoints are *repaired* (moved to the nearest free cell); paths that still cross a zone are *penalised*.
- Max flight range: *penalised* by the distance over the limit.

**Operators:**
- Selection: tournament.
- Crossover (one of three, chosen at random): swap whole drone routes, swap route segments, geometric blend.
- Mutation: small move, full waypoint reset, and swapping two waypoints in a route.
- Diversity: the mutation rate is tripled when the population becomes too similar (adaptive mutation). Top 2 plans are always kept (elitism).

**Stopping:** max generations, or no improvement for 30 generations.

## 5. Outputs (in `results/`)

- `convergence.png`: best vs average fitness, plus diversity
- `paths_and_coverage_over_generations.png`: paths and coverage heatmaps at 4 stages
- `final_solution.png`: final paths and coverage heatmap
- `ga_vs_random.png`: GA vs random search
- `sweep_*.png`, `ga_vs_random_boxplot.png`, `experiment_results.csv`: from `experiments.py`
