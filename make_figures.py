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
    """Load every result .npz from a directory, skipping legacy/unreadable files."""
    runs = []
    for path in glob.glob(os.path.join(results_dir, "*.npz")):
        try:
            data = dict(np.load(path, allow_pickle=True))
        except Exception:                 # skip unreadable / partially-written .npz
            continue
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
            ax.fill_between(gens, ci[1], ci[2], color=regime_color[regime], alpha=.4, lw=0)
            null_bands.append(stack_curves(rows, "mi_null"))
    if null_bands:
        all_nulls = np.concatenate(null_bands, 0); gens = np.arange(all_nulls.shape[1])
        ax.fill_between(gens, 0, np.percentile(all_nulls, 95, 0), color="gray", alpha=.4, lw=0,
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


# ---- signal map exemplar: the feedforward-hybrid champion -----------------
# Only channel-on runs store the map arrays. We feature the FEEDFORWARD HYBRID
# champion (the most communicative condition), falling back to the highest-MI
# eligible run if it is absent. A spatial map is NOT shown: food patches are
# randomly placed per group (and not logged), so an absolute-coordinate map
# aggregated over groups carries no structure. Instead: (a) signal vs distance
# to food -- the conspicuous but "free" food cue; (b) signal vs local crowding
# -- the small, listener-dependent component the referent analysis flags as the
# actual communication.
NEIGHBOUR_RADIUS, WORLD = 0.10, 1.0
eligible = [run for run in runs
            if all(k in run for k in ("sm_signal", "sm_dist", "sm_pos")) and not run["ablate"]]
preferred = [run for run in eligible
             if run["controller"] == "feedforward" and run["regime"] == "hybrid"]
pool = preferred or eligible
if pool:
    champion = max(pool, key=lambda run: float(run["final_mi"]))
    signal = np.asarray(champion["sm_signal"]); positions = np.asarray(champion["sm_pos"])
    signal_flat = signal.ravel(); dist = np.asarray(champion["sm_dist"]).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # (a) signal vs distance to nearest food
    finite = np.isfinite(dist); sig_d, dist_d = signal_flat[finite], dist[finite]
    dist_bins = np.linspace(0, 0.4, 17); bin_of = np.digitize(dist_d, dist_bins)
    centres, mean_sig = [], []
    for b in range(1, len(dist_bins)):
        in_bin = bin_of == b
        if in_bin.sum() > 50:
            centres.append(.5*(dist_bins[b-1]+dist_bins[b])); mean_sig.append(sig_d[in_bin].mean())
    axes[0].plot(centres, mean_sig, "-o", ms=3, color="#8e44ad")
    axes[0].set_xlabel("distance to nearest food"); axes[0].set_ylabel("mean signal")
    axes[0].set_title("(a) signal vs. food distance")

    # (b) signal vs number of groupmates within NEIGHBOUR_RADIUS (toroidal)
    offset = positions[:, :, :, None, :] - positions[:, :, None, :, :]
    offset = (offset + WORLD / 2) % WORLD - WORLD / 2
    pair_dist = np.sqrt((offset ** 2).sum(-1))
    not_self = ~np.eye(positions.shape[2], dtype=bool)[None, None]
    n_neigh = ((pair_dist < NEIGHBOUR_RADIUS) & not_self).sum(-1).ravel()
    counts, mean_sig_n, sem_n = [], [], []
    for c in np.unique(n_neigh):
        in_c = n_neigh == c
        if in_c.sum() > 50:
            counts.append(int(c)); mean_sig_n.append(signal_flat[in_c].mean())
            sem_n.append(signal_flat[in_c].std() / np.sqrt(in_c.sum()))
    axes[1].errorbar(counts, mean_sig_n, yerr=sem_n, fmt="-o", ms=3, color="#16a085", capsize=2)
    axes[1].set_xlabel(f"groupmates within {NEIGHBOUR_RADIUS}"); axes[1].set_ylabel("mean signal")
    axes[1].set_title("(b) signal vs. local crowding")
    fig.tight_layout(); save_figure(fig, "signal_map")


# ---- combined learning panel (both controllers) ---------------------------
# FF and RNN side by side on a SHARED y-scale so the cross-controller yield gap
# is directly comparable, but with y-tick labels on BOTH panels for readability.
# Display curves are smoothed (moving average) so the channel-on vs ablated
# distinction is carried by an explicit line-style legend rather than being lost
# in per-generation noise. The two panels are independent on the x-axis, so the
# controllers may be evolved for a different number of generations (e.g. a longer
# recurrent run that needs more generations to plateau).
def _smooth(curve, window=21):
    """Length-preserving centered moving average, for display only."""
    curve = np.asarray(curve, float)
    if window <= 1 or curve.size < window:
        return curve
    pad = window // 2
    kernel = np.ones(window) / window
    return np.convolve(np.pad(curve, pad, mode="edge"), kernel, mode="valid")[:curve.size]


SMOOTH_W = 21
ctrl_order = [c for c in ["feedforward", "recurrent"] if c in controllers]
regime_order = [r for r in ["colony", "individual", "hybrid"] if r in regimes]
if ctrl_order:
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(ctrl_order), figsize=(7, 3), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, controller in zip(axes, ctrl_order):
        for regime in regime_order:
            on_curves = stack_curves(select_runs(runs, controller=controller, regime=regime, ablate=False), "yield_curve")
            ci = median_with_ci(on_curves)
            if ci is not None:
                gens = np.arange(ci[0].shape[0])
                ax.plot(gens, _smooth(ci[0]), color=regime_color[regime], lw=2.0)
                ax.fill_between(gens, _smooth(ci[1]), _smooth(ci[2]),
                                color=regime_color[regime], alpha=.12, lw=0)
            abl_curves = stack_curves(select_runs(runs, controller=controller, regime=regime, ablate=True), "yield_curve")
            if abl_curves.shape[0]:
                med = np.median(abl_curves, 0)
                ax.plot(np.arange(med.shape[0]), _smooth(med),
                        color=regime_color[regime], lw=1.7, ls=(0, (5, 2)), alpha=.9)
        ax.set_xlabel("generation"); ax.set_title(controller)
        ax.set_ylabel("group yield"); ax.tick_params(labelleft=True)
    # two separate keys: colour = regime, line style = channel condition
    regime_handles = [Line2D([0], [0], color=regime_color[r], lw=2.4) for r in regime_order]
    style_handles = [Line2D([0], [0], color="0.35", lw=2.0, ls="-"),
                     Line2D([0], [0], color="0.35", lw=1.7, ls=(0, (5, 2)))]
    leg_regime = axes[0].legend(regime_handles, regime_order, frameon=False,
                                fontsize=7.5, loc="lower right")
    axes[0].add_artist(leg_regime)
    axes[-1].legend(style_handles, ["channel on", "ablated (deaf)"], frameon=False,
                    fontsize=7.5, loc="lower right")
    fig.tight_layout(); save_figure(fig, "learning_combined")


# ---- multi-referent signal informativeness --------------------------------
# Medians copied from the signal_referents.py console table (KSG estimator,
# raw continuous signal, 10 seeds/condition).  NOT recomputed here: that script
# re-runs 120 logged probe episodes and needs scikit-learn, whereas this file
# only needs what is already saved in data/.  Each value is
# (MI_channel_on, MI_deaf/ablated, p of one-sided MWU test on>deaf).
referent_mi = {
    "on_food": {
        ("feedforward", "colony"): (0.464, 0.495, 0.828),
        ("feedforward", "individual"): (0.453, 0.474, 0.786),
        ("feedforward", "hybrid"): (0.314, 0.478, 0.996),
        ("recurrent", "colony"): (0.218, 0.294, 0.939),
        ("recurrent", "individual"): (0.107, 0.298, 0.995),
        ("recurrent", "hybrid"): (0.053, 0.205, 0.999),
    },
    "neighbors": {
        ("feedforward", "colony"): (0.048, 0.010, 0.000),
        ("feedforward", "individual"): (0.022, 0.009, 0.019),
        ("feedforward", "hybrid"): (0.143, 0.007, 0.011),
        ("recurrent", "colony"): (0.056, 0.089, 0.991),
        ("recurrent", "individual"): (0.027, 0.057, 0.987),
        ("recurrent", "hybrid"): (0.025, 0.052, 0.993),
    },
}
mi_cells = [(c, r) for c in ["feedforward", "recurrent"] for r in ["colony", "individual", "hybrid"]]
mi_cell_labels = [f"{'FF' if c == 'feedforward' else 'RNN'}\n{r[:3]}" for c, r in mi_cells]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
for ax, referent in zip(axes, ["on_food", "neighbors"]):
    x = np.arange(len(mi_cells)); bar_w = 0.4
    mi_on = [referent_mi[referent][cell][0] for cell in mi_cells]
    mi_deaf = [referent_mi[referent][cell][1] for cell in mi_cells]
    ax.bar(x - bar_w / 2, mi_on, bar_w, label="channel on", color="#27ae60")
    ax.bar(x + bar_w / 2, mi_deaf, bar_w, label="deaf (ablated)", color="#95a5a6")
    headroom = 0.04 * max(mi_on + mi_deaf)
    for i, cell in enumerate(mi_cells):
        if referent_mi[referent][cell][2] < 0.05:          # on significantly > deaf
            ax.text(x[i], max(mi_on[i], mi_deaf[i]) + headroom, "*", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(mi_cell_labels, fontsize=7)
    ax.set_ylim(top=max(mi_on + mi_deaf) * 1.20)       # headroom so the * does not hit the title
    ax.set_title(f"I(signal; {referent})"); ax.set_ylabel("MI [nats]")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Signal informativeness by referent  (* : channel-on > deaf, p<0.05)", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95]); save_figure(fig, "referent_mi")


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
