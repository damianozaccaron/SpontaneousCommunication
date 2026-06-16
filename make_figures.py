"""Figures + statistics from a results directory (default: data/).

Adapts to whatever conditions are present.  Produces, when the relevant runs
exist:
  learning_<controller>.pdf : group yield vs generation, regimes overlaid,
                              channel-on solid / ablated dashed.
  emergence_<controller>.pdf: I(signal;food-state) vs generation + shuffle null.
  value_bar.pdf             : final yield, channel-on vs ablated, per cond.
  alpha_sweep.pdf           : hybrid -- MI and channel value vs alpha.
  neat_complexity.pdf       : NEAT mean #connections and #species vs generation.
  signal_map.pdf            : an exemplar champion's signalling behaviour.
and stats.json with medians, Mann-Whitney tests, and the channel-value table.

Usage: python make_figures.py [results_dir]
"""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

results_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
fig_dir = "figures"; os.makedirs(fig_dir, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150})
regime_color = {"colony": "#c0392b", "individual": "#2c3e50", "hybrid": "#16a085"}


def load_runs(results_dir):
    """Load every result .npz from a directory, skipping legacy v1 files."""
    runs = []
    for path in glob.glob(os.path.join(results_dir, "*.npz")):
        data = dict(np.load(path, allow_pickle=True))
        data = {k: (v.item() if getattr(v, "ndim", 1) == 0 else v) for k, v in data.items()}
        if "controller" not in data:      # skip legacy v1 result files
            continue
        runs.append(data)
    return runs


def select_runs(runs, **filters):
    """Return runs matching every key=value filter, sorted by seed."""
    matched = []
    for run in runs:
        if all(run.get(k) == v for k, v in filters.items()):
            matched.append(run)
    return sorted(matched, key=lambda run: run["seed"])


def stack_curves(rows, key):
    """Stack a per-run curve into a 2-D array, truncating to the shortest run."""
    if not rows:
        return np.empty((0, 0))
    shortest = min(len(row[key]) for row in rows)
    return np.array([row[key][:shortest] for row in rows])


def median_with_ci(matrix, n_boot=2000, rng=None):
    """Median curve across runs plus a bootstrap 95% confidence band.

    ``matrix`` is (n_runs, n_points). Returns ``(median, lo, hi)`` or None when
    there are no runs.
    """
    rng = rng or np.random.default_rng(0)
    if matrix.shape[0] == 0:
        return None
    median = np.median(matrix, 0)
    resample_idx = rng.integers(0, matrix.shape[0], size=(n_boot, matrix.shape[0]))
    boot_medians = np.median(matrix[resample_idx], 1)
    return median, np.percentile(boot_medians, 2.5, 0), np.percentile(boot_medians, 97.5, 0)


def save_figure(fig, name):
    """Save a figure as both PDF and PNG into the figures directory, then close it."""
    fig.savefig(f"{fig_dir}/{name}.pdf"); fig.savefig(f"{fig_dir}/{name}.png"); plt.close(fig)
    print("saved", name)


runs = load_runs(results_dir)
if not runs:
    print("no runs found in", results_dir); sys.exit(0)
controllers = sorted({run["controller"] for run in runs})
regimes = sorted({run["regime"] for run in runs})
print(f"loaded {len(runs)} runs | controllers={controllers} regimes={regimes}")


# ---- learning + emergence, one figure per controller ---------------------
for controller in controllers:
    # learning curve
    fig, ax = plt.subplots(figsize=(4, 3))
    for regime in regimes:
        signal_curves = stack_curves(select_runs(runs, controller=controller, regime=regime, ablate=False), "yield_curve")
        ci = median_with_ci(signal_curves)
        if ci is not None:
            gens = np.arange(ci[0].shape[0])
            ax.plot(gens, ci[0], color=regime_color[regime], lw=1.6, label=f"{regime}")
            ax.fill_between(gens, ci[1], ci[2], color=regime_color[regime], alpha=.15, lw=0)
        ablated_curves = stack_curves(select_runs(runs, controller=controller, regime=regime, ablate=True), "yield_curve")
        if ablated_curves.shape[0]:
            ax.plot(np.arange(ablated_curves.shape[1]), np.median(ablated_curves, 0),
                    color=regime_color[regime], lw=1.1, ls="--", alpha=.8)
    ax.set_xlabel("generation"); ax.set_ylabel("group yield")
    ax.set_title(f"Foraging - {controller} (dashed = ablated)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); save_figure(fig, f"learning_{controller}")

    # emergence
    fig, ax = plt.subplots(figsize=(4, 3)); null_bands = []
    for regime in regimes:
        rows = select_runs(runs, controller=controller, regime=regime, ablate=False)
        mi_curves = stack_curves(rows, "mi"); ci = median_with_ci(mi_curves)
        if ci is not None:
            gens = rows[0]["mi_gen"][:ci[0].shape[0]]
            ax.plot(gens, ci[0], color=regime_color[regime], lw=1.6, label=regime)
            ax.fill_between(gens, ci[1], ci[2], color=regime_color[regime], alpha=.15, lw=0)
            null_bands.append(stack_curves(rows, "mi_null"))
    if null_bands:
        all_nulls = np.concatenate(null_bands, 0); gens = np.arange(all_nulls.shape[1])
        ax.fill_between(gens, 0, np.percentile(all_nulls, 95, 0), color="gray", alpha=.3, lw=0,
                        label="shuffle null 95%")
    ax.set_xlabel("generation"); ax.set_ylabel("I(signal; food-state) [nats]")
    ax.set_title(f"Emergence - {controller}"); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); save_figure(fig, f"emergence_{controller}")


# ---- value bar (channel on vs ablated) ------------------------------------
def final_yields(rows):
    """Pull the scalar final-yield value out of each run."""
    return np.array([float(row["final_yield"]) for row in rows])


conditions = [(controller, regime) for controller in controllers for regime in regimes
              if select_runs(runs, controller=controller, regime=regime, ablate=True)]
if conditions:
    fig, ax = plt.subplots(figsize=(max(4, 1.1 * len(conditions)), 3)); bar_width = .38
    for i, ablate in enumerate((False, True)):
        bar_values, bar_errors = [], []
        for controller, regime in conditions:
            yields = final_yields(select_runs(runs, controller=controller, regime=regime, ablate=ablate))
            ci = median_with_ci(yields[:, None]); bar_values.append(ci[0][0])
            bar_errors.append([ci[0][0]-ci[1][0], ci[2][0]-ci[0][0]])
        ax.bar(np.arange(len(conditions))+i*bar_width, bar_values, bar_width,
               yerr=np.array(bar_errors).T, capsize=3,
               label="ablated" if ablate else "channel on", color=["#27ae60", "#95a5a6"][i])
    ax.set_xticks(np.arange(len(conditions))+bar_width/2)
    ax.set_xticklabels([f"{controller}\n{regime}" for controller, regime in conditions], fontsize=7)
    ax.set_ylabel("final group yield"); ax.set_title("Functional value of the channel")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); save_figure(fig, "value_bar")


# ---- hybrid alpha sweep ---------------------------------------------------
alphas = sorted({float(run["alpha"]) for run in runs if run["regime"] == "hybrid"})
if len(alphas) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for controller in controllers:
        late_mis, channel_values = [], []
        for alpha in alphas:
            signal_runs = select_runs(runs, controller=controller, regime="hybrid", alpha=alpha, ablate=False)
            if not signal_runs:
                late_mis.append(np.nan); channel_values.append(np.nan); continue
            per_run_late_mi = np.array([np.mean(r["mi"][int(.7*len(r["mi"])):]) for r in signal_runs])
            late_mis.append(np.median(per_run_late_mi))
            ablated_runs = select_runs(runs, controller=controller, regime="hybrid", alpha=alpha, ablate=True)
            channel_values.append(np.median(final_yields(signal_runs)) - np.median(final_yields(ablated_runs))
                                  if ablated_runs else np.nan)
        axes[0].plot(alphas, late_mis, "-o", ms=3, label=controller)
        axes[1].plot(alphas, channel_values, "-o", ms=3, label=controller)
    axes[0].set_xlabel("alpha (1=individual, 0=group)"); axes[0].set_ylabel("late MI")
    axes[0].set_title("(a) informativeness vs selection level")
    axes[1].axhline(0, color="k", lw=.8); axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("channel value (yield on - ablated)")
    axes[1].set_title("(b) functional value vs selection level")
    axes[0].legend(frameon=False, fontsize=8); fig.tight_layout(); save_figure(fig, "alpha_sweep")


# ---- NEAT complexity ------------------------------------------------------
neat_rows = select_runs(runs, controller="neat", regime="colony", ablate=False)
if neat_rows and "mean_conns" in neat_rows[0]:
    fig, ax = plt.subplots(figsize=(4, 3))
    mean_conns = stack_curves(neat_rows, "mean_conns")
    ax.plot(np.arange(mean_conns.shape[1]), np.median(mean_conns, 0), color="#8e44ad", label="mean #connections")
    if "n_species" in neat_rows[0]:
        n_species = stack_curves(neat_rows, "n_species")
        ax2 = ax.twinx(); ax2.plot(np.arange(n_species.shape[1]), np.median(n_species, 0),
                                   color="#e67e22", lw=1, label="#species")
        ax2.set_ylabel("#species", color="#e67e22")
    ax.set_xlabel("generation"); ax.set_ylabel("mean #connections", color="#8e44ad")
    ax.set_title("NEAT topology growth"); fig.tight_layout(); save_figure(fig, "neat_complexity")


# ---- signal map exemplar (highest-MI run with stored arrays) --------------
# Only channel-on runs that actually stored the full map arrays are eligible:
# ablated runs save sm_signal but not sm_dist/sm_pos, so an ablated champion
# (high "free" MI) would crash the proximity/position plots below.
exemplars = [run for run in runs
             if all(k in run for k in ("sm_signal", "sm_dist", "sm_pos")) and not run["ablate"]]
if exemplars:
    champion = max(exemplars, key=lambda run: float(run["final_mi"]))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    signal = np.asarray(champion["sm_signal"]).ravel(); dist = np.asarray(champion["sm_dist"]).ravel()
    finite = np.isfinite(dist); signal, dist = signal[finite], dist[finite]
    dist_bins = np.linspace(0, 0.4, 17); bin_of = np.digitize(dist, dist_bins); bin_centers, mean_signals = [], []
    for b in range(1, len(dist_bins)):
        in_bin = bin_of == b
        if in_bin.sum() > 50:
            bin_centers.append(.5*(dist_bins[b-1]+dist_bins[b])); mean_signals.append(signal[in_bin].mean())
    axes[0].plot(bin_centers, mean_signals, "-o", ms=3, color="#8e44ad")
    axes[0].set_xlabel("distance to nearest food"); axes[0].set_ylabel("mean signal")
    axes[0].set_title(f"(a) {champion['controller']}/{champion['regime']} signal vs proximity")
    positions = np.asarray(champion["sm_pos"]); signal_field = np.asarray(champion["sm_signal"])
    focal_xy = positions[:, 0, :, :].reshape(-1, 2); focal_signal = signal_field[:, 0, :].reshape(-1)
    hexbin = axes[1].hexbin(focal_xy[:, 0], focal_xy[:, 1], C=focal_signal, gridsize=20,
                            cmap="viridis", reduce_C_function=np.mean)
    fig.colorbar(hexbin, ax=axes[1], shrink=.85, label="mean signal")
    axes[1].set_title("(b) where signals fire"); fig.tight_layout(); save_figure(fig, "signal_map")


# ---- statistics -----------------------------------------------------------
def late_mi(rows):
    """Mean MI over each run's last 30% of probes (the converged value)."""
    return np.array([np.mean(row["mi"][int(.7*len(row["mi"])):]) for row in rows])


stats = {"controllers": controllers, "regimes": regimes}
for controller in controllers:
    colony_runs = select_runs(runs, controller=controller, regime="colony", ablate=False)
    individual_runs = select_runs(runs, controller=controller, regime="individual", ablate=False)
    if colony_runs and individual_runs:
        colony_mi, individual_mi = late_mi(colony_runs), late_mi(individual_runs)
        stats[f"{controller}_MI_colony"] = float(np.median(colony_mi))
        stats[f"{controller}_MI_individual"] = float(np.median(individual_mi))
        if len(colony_mi) > 1 and len(individual_mi) > 1:
            stats[f"{controller}_MI_p_colony_gt_ind"] = float(
                mannwhitneyu(colony_mi, individual_mi, alternative="greater").pvalue)
    for regime in regimes:
        signal_runs = select_runs(runs, controller=controller, regime=regime, ablate=False)
        ablated_runs = select_runs(runs, controller=controller, regime=regime, ablate=True)
        if signal_runs and ablated_runs:
            signal_yield, ablated_yield = final_yields(signal_runs), final_yields(ablated_runs)
            stats[f"{controller}_{regime}_value_pct"] = float(
                100*(np.median(signal_yield)/np.median(ablated_yield)-1))
            if len(signal_yield) > 1 and len(ablated_yield) > 1:
                stats[f"{controller}_{regime}_p_on_gt_abl"] = float(
                    mannwhitneyu(signal_yield, ablated_yield, alternative="greater").pvalue)
json.dump(stats, open(os.path.join(results_dir, "stats.json"), "w"), indent=2)
print(json.dumps(stats, indent=2))
# end
