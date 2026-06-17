"""Post-hoc analysis: *what is the evolved signal about?*

For every saved champion we re-run one logged clonal episode and measure the
mutual information between the agent's emitted signal and three candidate
referents -- the things the signal might be "talking about":

    on_food        : is the emitter currently standing on food?      (binary)
    neighbors      : how many groupmates are within NEIGHBOR_RADIUS?  (count)
    recent_intake  : did the emitter eat in the last RECENT_WINDOW steps? (binary)

Three deliberate measurement choices (one line each to justify in the report)
-----------------------------------------------------------------------------
1. Estimator.  The signal is a continuous, near-degenerate sigmoid output.
   Equal-width binning collapses it into a single occupied bin and reads ~0,
   so we hand the raw continuous signal (the feature) and the discrete referent
   (the target) to sklearn's `mutual_info_classif(discrete_features=False)` --
   the Kraskov/Ross k-nearest-neighbour estimator, which needs no bin edges.
2. Deaf control.  A high MI can be "free": the signal neuron shares network
   inputs with the food sensors, so it correlates with food even with nobody
   listening.  We therefore report channel-ON minus the DEAF (ablated) baseline;
   only the excess counts as communication that exists *because* of listeners.
3. Null floor.  k-NN MI is positively biased, so each estimate is paired with a
   label-shuffle null computed by the same estimator: the bias we must clear.
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import mannwhitneyu
from evolved_comm.config import Config
from evolved_comm import env, controllers

# --- measurement parameters ------------------------------------------------- #
NEIGHBOR_RADIUS = 0.10   # "in range" = within this torus distance.  This is a
                         # *local crowding* radius (~2x r_sense / patch_radius);
                         # at earshot range (r_signal=0.55) almost the whole
                         # 8-agent group is always in range, so the count barely
                         # varies and would carry no information by construction.
RECENT_WINDOW   = 8      # a step counts as "recently fed" if the agent ate
                         # within this many steps.
PROBE_GROUPS    = 32     # clonal groups in the probe episode (matches the scale
                         # `evolution.probe_mi` uses during training).
NULL_SHUFFLES   = 20     # label permutations used to estimate the bias floor.
SEEDS           = 10     # seeds per condition in the saved sweep.

# World / GA settings shared by every run in the sweep; only controller, regime,
# alpha and the ablation flag differ between cells, so we overlay those per run.
SHARED_CONFIG = {key: value for key, value in json.load(open("data/config.json")).items()
                 if key in Config.__dataclass_fields__}


def build_run_config(controller: str, regime: str, alpha: float, ablate_signal: bool) -> Config:
    """
    Reconstruct the Config of one saved run.
    """
    return Config(**{**SHARED_CONFIG, "controller": controller, "regime": regime,
                     "alpha": alpha, "ablate_signal": bool(ablate_signal)})


def rebuild_policy(config: Config, champion_genome, n_agents: int):
    """
    Turn a saved champion genome back into a callable policy. A single genome
    cloned across all `n_agents` slots.
    """
    controller = controllers.get_vector_controller(config)
    population = np.asarray(champion_genome, dtype=np.float64)[None]
    clone_ids = np.zeros(n_agents, dtype=int)        # every agent uses genome 0

    return controller.make_policy(population, clone_ids)


def count_neighbors_in_range(positions: np.ndarray, world_size: float, radius: float) -> np.ndarray:
    """Per-agent count of groupmates within `radius` on the torus.

    `positions` has shape (steps, n_groups, group_size, 2).  Returns an integer
    array (steps, n_groups, group_size): for each agent at each step, how many
    *other* members of its group lie within `radius`, using wrapped (toroidal)
    distance so agents near opposite edges are correctly counted as close.
    """
    pairwise_offset = positions[:, :, :, None, :] - positions[:, :, None, :, :]
    pairwise_offset = (pairwise_offset + world_size / 2) % world_size - world_size / 2
    pairwise_distance = np.sqrt((pairwise_offset ** 2).sum(-1))
    group_size = positions.shape[2]
    not_self = ~np.eye(group_size, dtype=bool)[None, None]
    
    return ((pairwise_distance < radius) & not_self).sum(-1)


def mutual_information_with_null(signal_values: np.ndarray, referent_labels: np.ndarray,
                                 rng: np.random.Generator):
    """KSG estimate of I(signal; referent), plus its shuffle-null floor.

    `signal_values` is the continuous emitted signal (the feature); each entry
    of the discrete `referent_labels` is what we ask the signal to be "about".
    The null floor is the median MI over `NULL_SHUFFLES` label permutations --
    which destroy any real dependence while preserving the estimator's positive
    bias -- so a real effect must sit clearly above it.  Returns (mi, null_floor)
    in nats.
    """
    signal_feature = np.asarray(signal_values, float).reshape(-1, 1)
    referent_labels = np.asarray(referent_labels).ravel()
    observed_mi = float(mutual_info_classif(signal_feature, referent_labels,
                                            discrete_features=False, n_neighbors=3,
                                            random_state=0)[0])
    shuffled_mi = [float(mutual_info_classif(signal_feature, rng.permutation(referent_labels),
                                             discrete_features=False, n_neighbors=3,
                                             random_state=0)[0])
                   for _ in range(NULL_SHUFFLES)]
    return observed_mi, float(np.median(shuffled_mi))


def probe_champion(npz_path: str) -> dict:
    """Re-run one champion and return MI(+null) for each referent.

    Loads the saved genome, rebuilds its policy, runs a single logged clonal
    episode under the run's own condition, then measures the signal against each
    referent.  `recent_intake` is only available if `env.simulate` was extended
    to log per-step intake (`rollout.intake_step`); it is skipped otherwise.
    Returns {referent_name: (mi, null_floor)}.
    """
    saved = dict(np.load(npz_path, allow_pickle=True))
    saved = {key: (value.item() if getattr(value, "ndim", 1) == 0 else value)
             for key, value in saved.items()}
    config = build_run_config(saved["controller"], saved["regime"],
                              float(saved["alpha"]), bool(saved["ablate"]))
    rng = np.random.default_rng(2024 + int(saved["seed"]))
    policy = rebuild_policy(config, saved["best_genome"], PROBE_GROUPS * config.group_size)
    rollout = env.simulate(config, PROBE_GROUPS, policy, rng, log=True)

    signal_values = rollout.signal.ravel().astype(float)
    referent_mi = {}
    referent_mi["on_food"] = mutual_information_with_null(
        signal_values, rollout.food_state.astype(int).ravel(), rng)
    referent_mi["neighbors"] = mutual_information_with_null(
        signal_values,
        count_neighbors_in_range(rollout.pos, config.world_size, NEIGHBOR_RADIUS).ravel(),
        rng)

    intake_per_step = getattr(rollout, "intake_step", None)   # (steps, n_groups, group_size)
    if intake_per_step is not None:
        recently_fed = np.zeros_like(intake_per_step)
        for step in range(intake_per_step.shape[0]):
            window_start = max(0, step - RECENT_WINDOW + 1)
            recently_fed[step] = intake_per_step[window_start:step + 1].sum(0)
        referent_mi["recent_intake"] = mutual_information_with_null(
            signal_values, (recently_fed.ravel() > 0).astype(int), rng)
    return referent_mi


def main():
    """Print, per controller x regime, channel-on vs deaf MI for each referent."""
    controllers_to_test = ["feedforward", "recurrent"]
    regimes = ["colony", "individual", "hybrid-a0.5"]

    # results[(controller, regime, ablated)] = list of per-seed referent_mi dicts
    results = {}
    for controller in controllers_to_test:
        for regime in regimes:
            for ablated in (False, True):
                channel_tag = "ablate" if ablated else "signal"
                results[(controller, regime, ablated)] = [
                    probe_champion(f"data/{controller}_{regime}_{channel_tag}_s{seed}.npz")
                    for seed in range(SEEDS)]

    a_sample = results[(controllers_to_test[0], regimes[0], False)][0]
    referents = ["on_food", "neighbors"] + (["recent_intake"] if "recent_intake" in a_sample else [])

    print("KSG MI(signal; referent): channel-ON vs deaf baseline; null in (); "
          "p = Mann-Whitney one-sided (on > deaf)\n")
    for referent in referents:
        print(f"=== referent: {referent} ===")
        print(f"{'controller':<12}{'regime':<11}{'MI_on':>8}{'null':>8}"
              f"{'MI_deaf':>9}{'on-deaf':>9}{'p':>8}")
        for controller in controllers_to_test:
            for regime in regimes:
                channel_on_mi = np.array([seed[referent][0]
                                          for seed in results[(controller, regime, False)]])
                null_floor = np.median([seed[referent][1]
                                        for seed in results[(controller, regime, False)]])
                deaf_mi = np.array([seed[referent][0]
                                    for seed in results[(controller, regime, True)]])
                p_value = mannwhitneyu(channel_on_mi, deaf_mi, alternative="greater").pvalue
                print(f"{controller:<12}{regime.replace('-a0.5', ''):<11}"
                      f"{np.median(channel_on_mi):>8.3f}{null_floor:>8.4f}"
                      f"{np.median(deaf_mi):>9.3f}"
                      f"{np.median(channel_on_mi) - np.median(deaf_mi):>9.3f}{p_value:>8.3f}")
        print()


if __name__ == "__main__":
    main()