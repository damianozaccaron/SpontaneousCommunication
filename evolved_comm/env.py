from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .config import Config


def _torus_delta(point_a, point_b, box_size):
    """Shortest displacement from ``point_b`` to ``point_a`` on a periodic box.

    On a torus the world wraps around, so the true separation between two
    points is the raw difference folded into the half-open range
    (-box_size/2, box_size/2]. Works elementwise on broadcastable arrays whose
    last axis is the (x, y) coordinate.
    """
    delta = point_a - point_b
    delta -= np.round(delta / box_size) * box_size
    return delta


def init_world(cfg: Config, n_groups: int, rng):
    """
    Lay out one fresh world per group at the start of an episode.
    Scatters group_size agents at random and clusters food items tightly
    around n_patches random patch centres. 

    n_groups is the equivalent of pop_size when called normally, otherwise
    it is the number of groups used for probing.

    Returns:
      agent_positions  (n_groups, group_size, 2)
      food_positions   (n_groups, n_food, 2)
      food_alive       (n_groups, n_food) bool -- is each item currently present
      patch_centers    (n_groups, n_patches, 2) -- where eaten food respawns
      food_patch_id    (n_food,) -- which patch each food item belongs to
    """
    world_size, group_size = cfg.world_size, cfg.group_size

    agent_positions = rng.uniform(0, world_size, size=(n_groups, group_size, 2))
    patch_centers = rng.uniform(0, world_size, size=(n_groups, cfg.n_patches, 2))
    patch_offsets = rng.normal(0, cfg.patch_radius, size=(n_groups, cfg.n_patches, cfg.patch_capacity, 2))
    food_positions = (patch_centers[:, :, None, :] + patch_offsets).reshape(n_groups, -1, 2) % world_size
    food_alive = np.ones(food_positions.shape[:2], dtype=bool)
    food_patch_id = np.repeat(np.arange(cfg.n_patches), cfg.patch_capacity)   # (n_food,)

    return agent_positions, food_positions, food_alive, patch_centers, food_patch_id


def observe(cfg: Config, agent_positions, food_positions, food_alive, signal_emitted):
    """
    Build the 9-D observation vector each agent receives this step.

    The observation has three parts: four distance-weighted directional food
    sensors (N/E/S/W) that only fire for food within r_sense, a single
    on-food flag, and four directional signal sensors that pick up groupmates'
    emitted signal within the much longer r_signal (an agent never hears
    itself). 

    Returns (observations, on_food, nearest_food_dist) where on_food is
    a per-agent flag and nearest_food_dist is the distance to the closest
    live food item (both kept for metrics/logging).
    """
    world_size, eps = cfg.world_size, 1e-9

    # food sensing
    agent_to_food = _torus_delta(food_positions[:, None, :, :],
                                 agent_positions[:, :, None, :], world_size)
    food_dist = np.sqrt((agent_to_food ** 2).sum(-1)) + eps
    food_in_range = (food_dist < cfg.r_sense) & food_alive[:, None, :]
    food_weight = food_in_range * (1.0 - food_dist / cfg.r_sense)
    food_dir_x, food_dir_y = agent_to_food[..., 0] / food_dist, agent_to_food[..., 1] / food_dist

    food_E = (food_weight * np.clip(food_dir_x, 0, None)).sum(-1)
    food_W = (food_weight * np.clip(-food_dir_x, 0, None)).sum(-1)
    food_N = (food_weight * np.clip(food_dir_y, 0, None)).sum(-1)
    food_S = (food_weight * np.clip(-food_dir_y, 0, None)).sum(-1)

    on_food = food_in_range.any(-1)
    nearest_food_dist = np.where(food_alive[:, None, :], food_dist, np.inf).min(-1)

    # signal sensing
    agent_to_agent = _torus_delta(agent_positions[:, None, :, :],
                                  agent_positions[:, :, None, :], world_size)
    agent_dist = np.sqrt((agent_to_agent ** 2).sum(-1)) + eps
    group_size = cfg.group_size
    not_self = ~np.eye(group_size, dtype=bool)[None]
    within_earshot = (agent_dist < cfg.r_signal) & not_self
    signal_weight = within_earshot * (1.0 - agent_dist / cfg.r_signal) * signal_emitted[:, None, :]
    sig_dir_x, sig_dir_y = agent_to_agent[..., 0] / agent_dist, agent_to_agent[..., 1] / agent_dist

    sig_E = (signal_weight * np.clip(sig_dir_x, 0, None)).sum(-1)
    sig_W = (signal_weight * np.clip(-sig_dir_x, 0, None)).sum(-1)
    sig_N = (signal_weight * np.clip(sig_dir_y, 0, None)).sum(-1)
    sig_S = (signal_weight * np.clip(-sig_dir_y, 0, None)).sum(-1)

    observations = np.stack([food_N, food_E, food_S, food_W, on_food.astype(float),
                             sig_N, sig_E, sig_S, sig_W], axis=-1)
    
    return observations, on_food, nearest_food_dist


@dataclass
class Rollout:
    """
    Outcome of one simulated episode.

    intake (per-agent items eaten) is always present; the remaining
    per-step trajectories are only filled in when simulate is called with
    log=True (used for metrics and the signal-map figures).
    """
    intake: np.ndarray
    food_state: np.ndarray = None     # per step: was each agent on food
    signal: np.ndarray = None         # per step: each agent's emitted signal
    dist_food: np.ndarray = None      # per step: each agent's distance to nearest food
    pos: np.ndarray = None            # per step: every agent position
    intake_step: np.ndarray = None    # per step: did each agent eat an item this step (bool)


def simulate(cfg: Config, n_groups: int, policy, rng, log: bool = False) -> Rollout:
    """
    Run one foraging episode for n_groups (see above, inside init_world) independent groups in parallel.

    Every step: agents observe the world, the policy returns a move and a
    signal, positions are integrated, eating is resolved (one item per agent
    per step; a contested item goes to the nearest agent), and eaten food
    counts down before respawning at its patch. The policy is the only link
    to the controller, so the physics never knows which brain it is running.

    Returns a Rollout; with log=True it also records the full per-step
    signal/food-state/distance/position trajectories.
    """
    group_size = cfg.group_size
    n_agents = n_groups * group_size
    group_idx = np.arange(n_groups)
    (agent_positions, food_positions, food_alive,
     patch_centers, food_patch_id) = init_world(cfg, n_groups, rng)
    
    n_food = food_positions.shape[1]
    respawn_timer = np.zeros((n_groups, n_food), int)
    intake = np.zeros((n_groups, group_size))
    signal_emitted = np.zeros((n_groups, group_size))

    if hasattr(policy, 'reset'):
        policy.reset(n_agents)     # clear recurrent memory for the episode
    recorded = {k: [] for k in ("food_state", "signal", "dist_food", "pos", "intake_step")}

    for step in range(cfg.episode_steps):
        observations, food_state, nearest_food_dist = observe(
            cfg, agent_positions, food_positions, food_alive, signal_emitted)
        if cfg.ablate_signal:
            observations[..., 5:9] = 0.0
        move, signal = policy(observations.reshape(n_agents, -1), rng)
        agent_positions = (agent_positions
                           + cfg.move_speed * move.reshape(n_groups, group_size, 2)) % cfg.world_size
        signal_emitted = signal.reshape(n_groups, group_size)

        # eating: one item per agent per step, item -> nearest targeting agent
        agent_to_food = _torus_delta(food_positions[:, None, :, :],
                                     agent_positions[:, :, None, :], cfg.world_size)
        food_dist = np.sqrt((agent_to_food ** 2).sum(-1))
        in_eat_range = (food_dist < cfg.r_eat) & food_alive[:, None, :]
        candidate_dist = np.where(in_eat_range, food_dist, np.inf)
        targeted_food = candidate_dist.argmin(2)
        dist_to_target = np.take_along_axis(candidate_dist, targeted_food[:, :, None], 2)[:, :, 0]
        has_target = np.isfinite(dist_to_target)
        nearest_per_food = np.full((n_groups, n_food), np.inf)

        np.minimum.at(nearest_per_food, (group_idx[:, None], targeted_food),
                      np.where(has_target, dist_to_target, np.inf))
        wins_food = has_target & (dist_to_target == nearest_per_food[group_idx[:, None], targeted_food])
        intake += wins_food
        win_group, win_member = np.where(wins_food)
        eaten = np.zeros((n_groups, n_food), bool)
        eaten[win_group, targeted_food[win_group, win_member]] = True
        food_alive &= ~eaten
        respawn_timer[eaten] = cfg.respawn_delay

        # regeneration: dead items count down, then respawn at their patch
        is_dead = ~food_alive
        respawn_timer[is_dead] -= 1
        respawning = is_dead & (respawn_timer <= 0)
        if respawning.any():
            reborn_group, reborn_food = np.where(respawning)
            new_position = (patch_centers[reborn_group, food_patch_id[reborn_food]]
                            + rng.normal(0, cfg.patch_radius, size=(reborn_group.size, 2))) % cfg.world_size
            food_positions[reborn_group, reborn_food] = new_position
            food_alive[reborn_group, reborn_food] = True

        if log:
            recorded["food_state"].append(food_state.copy())
            recorded["signal"].append(signal_emitted.copy())
            recorded["dist_food"].append(nearest_food_dist.copy())
            recorded["pos"].append(agent_positions.copy())
            recorded["intake_step"].append(wins_food.copy())

    if log:
        return Rollout(intake, np.array(recorded["food_state"]),
                       np.array(recorded["signal"]), np.array(recorded["dist_food"]),
                       np.array(recorded["pos"]), np.array(recorded["intake_step"]))
    
    return Rollout(intake)
