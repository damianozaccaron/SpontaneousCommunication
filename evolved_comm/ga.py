"""A small hand-rolled genetic algorithm plus the two regime-specific fitness
evaluations that are the experiment's central manipulation.

Colony / kin regime : every group is a clone army of one genome and that
                      genome's fitness is the whole group's yield.
Individual regime   : each genome is the *focal* member of an otherwise random
                      group and is scored on its own personal intake only.

Both regimes simulate exactly pop_size groups of group_size agents, so they cost
the same and differ *only* in group composition + fitness attribution.
"""
from __future__ import annotations
import numpy as np
from .config import Config
from . import env, mlp, metrics
from .policies import mlp_policy


# --------------------------------------------------------------------------- #
#  Fitness evaluation                                                         #
# --------------------------------------------------------------------------- #
def evaluate(pop: np.ndarray, cfg: Config, rng: np.random.Generator):
    """Return per-genome fitness and the group yield, under cfg.regime."""
    pop_size, group_size = cfg.pop_size, cfg.group_size
    if cfg.regime == "colony":
        agent_ids = np.repeat(np.arange(pop_size)[:, None], group_size, axis=1)   # clones
    elif cfg.regime == "individual":
        agent_ids = rng.integers(0, pop_size, size=(pop_size, group_size))        # random members
        agent_ids[:, 0] = np.arange(pop_size)                                     # focal = genome b
    else:
        raise ValueError(cfg.regime)

    genomes_per_agent = pop[agent_ids.ravel()]
    policy = mlp_policy(genomes_per_agent, cfg)
    rollout = env.simulate(cfg, pop_size, policy, rng)
    intake = rollout.intake                                            # (pop_size, group_size)
    group_yield = intake.sum(1)

    if cfg.regime == "colony":
        fitness = group_yield.copy()
    else:
        fitness = intake[:, 0].copy()                                  # focal only
    return fitness, group_yield


# --------------------------------------------------------------------------- #
#  GA operators (~60 lines of actual algorithm)                               #
# --------------------------------------------------------------------------- #
def _tournament(fitness, k, rng):
    """Tournament selection: for each slot pick the fittest of ``k`` random genomes."""
    contenders = rng.integers(0, len(fitness), size=(len(fitness), k))
    return contenders[np.arange(len(fitness)), fitness[contenders].argmax(1)]


def _next_generation(pop, fitness, cfg, rng):
    """Produce the next population: elitism + tournament + crossover + mutation."""
    pop_size = cfg.pop_size
    ranked = np.argsort(fitness)[::-1]
    next_pop = np.empty_like(pop)
    next_pop[:cfg.elitism] = pop[ranked[:cfg.elitism]]                 # elitism

    parents_a = pop[_tournament(fitness, cfg.tournament_k, rng)]
    parents_b = pop[_tournament(fitness, cfg.tournament_k, rng)]
    for i in range(cfg.elitism, pop_size):
        child = parents_a[i].copy()
        if rng.random() < cfg.crossover_rate:                          # uniform crossover
            from_b = rng.random(child.shape) < 0.5
            child[from_b] = parents_b[i][from_b]
        mutate_at = rng.random(child.shape) < cfg.mut_rate             # Gaussian mutation
        child[mutate_at] += rng.normal(0, cfg.mut_sigma, size=mutate_at.sum())
        next_pop[i] = child
    return next_pop


# --------------------------------------------------------------------------- #
#  Probe for the emergence metric                                             #
# --------------------------------------------------------------------------- #
def probe_mi(genome, cfg, rng, n_groups=64, want_map=False):
    """Run a clonal probe episode of one genome and measure I(signal;food_state)."""
    genomes_per_agent = np.repeat(genome[None], n_groups * cfg.group_size, axis=0)
    rollout = env.simulate(cfg, n_groups, mlp_policy(genomes_per_agent, cfg), rng, log=True)
    mi, null = metrics.signal_food_mi(rollout.signal, rollout.food_state, cfg, rng)
    if want_map:
        return mi, null, rollout
    return mi, null


# --------------------------------------------------------------------------- #
#  Main evolutionary loop                                                     #
# --------------------------------------------------------------------------- #
def evolve(cfg: Config, mi_every: int = 2, probe_groups: int = 32, verbose=False):
    """Run the GA. Returns (history dict, final-probe dict).

    MI is probed on the current champion every ``mi_every`` generations with the
    (cheap) null in cfg.mi_null_shuffles; a final high-resolution probe with a
    50-shuffle null produces the signal-map data.
    """
    from dataclasses import replace
    rng = np.random.default_rng(cfg.seed)
    pop = mlp.random_population(cfg, rng)
    hist = {"yield": [], "best_fit": [], "mi_gen": [], "mi": [], "mi_null": []}

    for gen in range(cfg.generations):
        fitness, group_yield = evaluate(pop, cfg, rng)
        hist["yield"].append(float(group_yield.mean()))
        hist["best_fit"].append(float(fitness.max()))
        if gen % mi_every == 0 or gen == cfg.generations - 1:
            mi, null = probe_mi(pop[fitness.argmax()], cfg, rng, n_groups=probe_groups)
            hist["mi_gen"].append(gen); hist["mi"].append(mi); hist["mi_null"].append(null)
            if verbose and gen % 25 == 0:
                print(f"  {cfg.regime:10s} gen {gen:3d} yield={group_yield.mean():5.1f} "
                      f"MI={mi:.4f} (null {null:.4f})", flush=True)
        pop = _next_generation(pop, fitness, cfg, rng)

    fitness, _ = evaluate(pop, cfg, rng)
    best = pop[fitness.argmax()]
    cfg_full = replace(cfg, mi_null_shuffles=50)
    mi, null, rollout = probe_mi(best, cfg_full, rng, n_groups=96, want_map=True)
    final = {"mi": mi, "mi_null": null, "signal": rollout.signal,
             "food_state": rollout.food_state, "dist_food": rollout.dist_food,
             "pos": rollout.pos, "best_genome": best}
    for key in hist:
        hist[key] = np.array(hist[key])
    return hist, final
