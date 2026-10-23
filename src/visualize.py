"""All the graphs. Each function saves a PNG and can optionally show it."""
import numpy as np
import matplotlib.pyplot as plt

COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:cyan", "tab:brown"]


def _finish(fig, path, show):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def draw_map(ax, env, heat=None):
    """Priority zones (orange), hazards (red tint), no-fly zones (black)."""
    if heat is None:
        ax.imshow(env.priority, cmap="YlOrBr", origin="lower", vmin=1, vmax=6)
    else:
        ax.imshow(heat, cmap="viridis", origin="lower", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(env.hazard == 0, env.hazard), cmap="Reds",
              origin="lower", alpha=0.45, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(env.no_fly == 0, env.no_fly), cmap="gray_r",
              origin="lower", alpha=0.95, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_paths(ax, chrom):
    for d, drone in enumerate(chrom.genes):
        col = COLORS[d % len(COLORS)]
        ax.plot(drone[:, 0], drone[:, 1], "-o", color=col, ms=3, lw=1.4,
                label=f"Drone {d + 1}")
        ax.plot(drone[0, 0], drone[0, 1], "s", color="white", mec=col, ms=7)  # start


def plot_convergence(history, path, show=False):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    gens = range(len(history["best"]))
    a1.plot(gens, history["best"], label="Best fitness", lw=2)
    a1.plot(gens, history["avg"], label="Average fitness", lw=2)
    a1.set_xlabel("Generation")
    a1.set_ylabel("Fitness (higher is better)")
    a1.set_title("Fitness convergence")
    a1.legend()
    a1.grid(alpha=0.3)

    a2.plot(gens, history["diversity"], color="tab:red", lw=2)
    a2.set_xlabel("Generation")
    a2.set_ylabel("Population diversity (cells)")
    a2.set_title("Diversity over generations")
    a2.grid(alpha=0.3)
    _finish(fig, path, show)


def plot_generation_snapshots(env, evaluator, best_chroms, path, show=False):
    """Top row: flight paths. Bottom row: coverage heatmap. At 4 stages."""
    n = len(best_chroms) - 1
    picks = sorted(set([0, n // 3, 2 * n // 3, n]))
    fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 8), squeeze=False)
    for col, g in enumerate(picks):
        res = evaluator.evaluate(best_chroms[g])
        draw_map(axes[0, col], env)
        draw_paths(axes[0, col], best_chroms[g])
        axes[0, col].set_title(f"Generation {g}\nfitness {res['score']:.2f}")
        draw_map(axes[1, col], env, heat=res["detect"])
        axes[1, col].set_title(f"Coverage {res['parts']['coverage'] * 100:.0f}%")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    _finish(fig, path, show)


def plot_final_solution(env, evaluator, chrom, path, show=False):
    res = evaluator.evaluate(chrom)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    draw_map(a1, env)
    draw_paths(a1, chrom)
    a1.set_title("Final flight paths\n(orange = priority, red = hazard, black = no-fly)")
    a1.legend(loc="upper right", fontsize=7)
    draw_map(a2, env, heat=res["detect"])
    draw_paths(a2, chrom)
    a2.set_title("Final coverage heatmap\n(brighter = stronger detection)")
    _finish(fig, path, show)
    return res


def plot_ga_vs_random(ga_history, rs_result, path, show=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ga_history["evals"], ga_history["best"], lw=2, label="Genetic Algorithm")
    ax.plot(rs_result["evals"], rs_result["history"], lw=2, label="Random search")
    ax.set_xlabel("Number of plans evaluated")
    ax.set_ylabel("Best fitness so far")
    ax.set_title("GA vs random search (same budget)")
    ax.legend()
    ax.grid(alpha=0.3)
    _finish(fig, path, show)


def plot_sweep(name, labels, curves, finals, path, show=False):
    """curves: list (per setting) of 2D arrays [seeds x generations].
    finals: list (per setting) of final best scores per seed."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for lab, arr in zip(labels, curves):
        mean = arr.mean(axis=0)
        a1.plot(mean, lw=2, label=f"{name} = {lab}")
    a1.set_xlabel("Generation")
    a1.set_ylabel("Best fitness (average of seeds)")
    a1.set_title(f"Convergence for different {name}")
    a1.legend()
    a1.grid(alpha=0.3)

    means = [np.mean(f) for f in finals]
    stds = [np.std(f) for f in finals]
    a2.bar([str(l) for l in labels], means, yerr=stds, capsize=6, color="tab:blue", alpha=0.8)
    a2.set_ylabel("Final best fitness (mean ± std)")
    a2.set_title(f"Final result for different {name}")
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    a2.set_ylim(lo - 0.1 * abs(hi - lo) - 0.01, hi + 0.1 * abs(hi - lo) + 0.01)
    a2.grid(alpha=0.3, axis="y")
    _finish(fig, path, show)


def plot_boxplot_vs_random(ga_scores, rs_scores, path, show=False):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.boxplot([rs_scores, ga_scores])
    ax.set_xticklabels(["Random search", "Genetic Algorithm"])
    ax.set_ylabel("Best fitness found")
    ax.set_title("GA vs random search over several seeds")
    ax.grid(alpha=0.3, axis="y")
    _finish(fig, path, show)
