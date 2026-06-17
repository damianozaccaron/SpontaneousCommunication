from __future__ import annotations
import os, pickle
import numpy as np
from dataclasses import replace
from .config import Config
from . import env, metrics, controllers


def get_controller(cfg: Config):
    """Return the controller object named by ``cfg.controller``."""
    return controllers.get_vector_controller(cfg)


def build_agent_ids(regime, pop_size, group_size, rng):
    """Map each agent slot to a genome index -- the central group-assembly step.

    Returns a (pop_size, group_size) int array. Under ``colony`` every member of
    a group is the same genome (clonal groups). Otherwise groups are mixed: each
    group's slot 0 is the focal genome ``b`` and the rest are drawn at random.
    """
    if regime == "colony":
        return np.repeat(np.arange(pop_size)[:, None], group_size, axis=1)   # clones
    agent_ids = rng.integers(0, pop_size, size=(pop_size, group_size))       # random members
    agent_ids[:, 0] = np.arange(pop_size)                                    # focal = genome b
    return agent_ids


def evaluate_population(controller, population, cfg: Config, rng):
    """Score every genome under ``cfg.regime``.

    Returns ``(fitness[pop_size], group_yield[pop_size])``. Colony fitness is
    the whole group's yield; individual fitness is the focal agent's own intake;
    hybrid blends the two. Mixed regimes average over
    ``eval_repeats`` independent group draws to reduce variance.
    """
    pop_size, group_size = cfg.pop_size, cfg.group_size

    if cfg.regime == "colony":
        agent_ids = build_agent_ids("colony", pop_size, group_size, rng)
        rollout = env.simulate(cfg, pop_size, controller.make_policy(population, agent_ids.ravel()), rng)
        group_yield = rollout.intake.sum(1)

        return group_yield.copy(), group_yield

    focal_intake = np.zeros(pop_size); group_total = np.zeros(pop_size)

    for _ in range(max(1, cfg.eval_repeats)):
        agent_ids = build_agent_ids(cfg.regime, pop_size, group_size, rng)
        rollout = env.simulate(cfg, pop_size, controller.make_policy(population, agent_ids.ravel()), rng)
        focal_intake += rollout.intake[:, 0]
        group_total += rollout.intake.sum(1)
    focal_intake /= cfg.eval_repeats; group_total /= cfg.eval_repeats

    if cfg.regime == "individual":
        fitness = focal_intake
    elif cfg.regime == "hybrid":
        fitness = cfg.alpha * focal_intake + (1.0 - cfg.alpha) * group_total
    else:
        raise ValueError(cfg.regime)
    
    return fitness, group_total


def probe_mi(controller, genome, cfg: Config, rng, n_groups=32, want_map=False):
    """
    Clonal probe of one genome -> I(signal; food-state) with shuffle null.

    Fills ``n_groups`` clonal groups with the single ``genome`` and runs a
    logged episode, then measures signal/food-state mutual information against
    its shuffle null. With ``want_map=True`` the full rollout is also returned
    (for the signal-map figures).
    """
    n_agents = n_groups * cfg.group_size
    population = genome[None]
    agent_ids = np.zeros(n_agents, dtype=int)
    rollout = env.simulate(cfg, n_groups, controller.make_policy(population, agent_ids), rng, log=True)
    mi, null = metrics.signal_food_mi(rollout.signal, rollout.food_state, cfg, rng)

    return (mi, null, rollout) if want_map else (mi, null)


#  Fixed-topology GA operators                                                
def _tournament(fitness, k, rng):
    """Tournament selection: for each slot pick the fittest of ``k`` random genomes."""
    contenders = rng.integers(0, len(fitness), size=(len(fitness), k))
    return contenders[np.arange(len(fitness)), fitness[contenders].argmax(1)]


def _next_generation(pop, fitness, cfg, rng):
    """Produce the next population: elitism + tournament + crossover + mutation."""
    pop_size = cfg.pop_size
    ranked = np.argsort(fitness)[::-1]
    next_pop = np.empty_like(pop)
    
    next_pop[:cfg.elitism] = pop[ranked[:cfg.elitism]]
    parents_a = pop[_tournament(fitness, cfg.tournament_k, rng)]
    parents_b = pop[_tournament(fitness, cfg.tournament_k, rng)]

    for i in range(cfg.elitism, pop_size):
        child = parents_a[i].copy()

        if rng.random() < cfg.crossover_rate:
            # if there is crossover, with probability 50% overwrite the gene with the one from parent B
            from_b = rng.random(child.shape) < 0.5
            child[from_b] = parents_b[i][from_b]

        mutate_at = rng.random(child.shape) < cfg.mut_rate
        child[mutate_at] += rng.normal(0, cfg.mut_sigma, size=mutate_at.sum())
        next_pop[i] = child

    return next_pop


#  Checkpointing 
def _save_ckpt(path, state):
    if path:
        with open(path + ".tmp", "wb") as f:
            pickle.dump(state, f)
        os.replace(path + ".tmp", path)


def _load_ckpt(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


#  Main loops 
def evolve(cfg: Config, mi_every=5, probe_groups=32, checkpoint_path=None, log_fn=None):
    """Run the evolution loop. Returns (history dict, final-probe dict)."""
    return _evolve_vector(cfg, mi_every, probe_groups, checkpoint_path, log_fn)


def _record(hist, gen, group_yield, fitness):
    """Append this generation's mean group yield and best fitness to history."""
    hist["yield"].append(float(group_yield.mean()))
    hist["best_fit"].append(float(fitness.max()))


def _evolve_vector(cfg, mi_every, probe_groups, checkpoint_path, log_fn):
    """Evolution loop for the fixed-topology (feedforward/recurrent) controllers."""

    controller = controllers.get_vector_controller(cfg)
    checkpoint = _load_ckpt(checkpoint_path)

    if checkpoint:
        pop, hist = checkpoint["pop"], checkpoint["hist"]
        start_gen, rng = checkpoint["gen"] + 1, checkpoint["rng"]
    else:
        rng = np.random.default_rng(cfg.seed)
        pop = controller.random_population(cfg.pop_size, rng)
        hist = {"yield": [], "best_fit": [], "mi_gen": [], "mi": [], "mi_null": []}
        start_gen = 0

    for gen in range(start_gen, cfg.generations):
        fitness, group_yield = evaluate_population(controller, pop, cfg, rng)
        _record(hist, gen, group_yield, fitness)

        if gen % mi_every == 0 or gen == cfg.generations - 1:
            mi, null = probe_mi(controller, pop[fitness.argmax()], cfg, rng, probe_groups)
            hist["mi_gen"].append(gen); hist["mi"].append(mi); hist["mi_null"].append(null)
            if log_fn:
                log_fn(gen, dict(yield_=float(group_yield.mean()), mi=mi, null=null))

        pop = _next_generation(pop, fitness, cfg, rng)

        if cfg.checkpoint_every and gen % cfg.checkpoint_every == 0:
            _save_ckpt(checkpoint_path, dict(pop=pop, hist=hist, gen=gen, rng=rng))

    fitness, _ = evaluate_population(controller, pop, cfg, rng)

    best = pop[fitness.argmax()]
    mi, null, rollout = probe_mi(controller, best, replace(cfg, mi_null_shuffles=50), rng, 96, True)

    final = dict(mi=mi, mi_null=null, signal=rollout.signal, food_state=rollout.food_state,
                 dist_food=rollout.dist_food, pos=rollout.pos, best_genome=best)
    
    for key in hist:
        hist[key] = np.array(hist[key])

    return hist, final
