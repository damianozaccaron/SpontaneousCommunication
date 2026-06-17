from __future__ import annotations
import numpy as np
from .config import Config
from . import mlp


def mlp_policy(genomes_per_agent: np.ndarray, cfg: Config):
    """Build a fixed-weight MLP policy closure for a whole batch of agents.

    Unpacks the genomes once into weight tensors, then returns a callable that
    maps observations (n_agents, 9) to (move (n_agents, 2), signal (n_agents,)).
    ``genomes_per_agent`` is (n_agents, genome_len) in flattened
    (group, member) order.
    """
    w_hidden, b_hidden, w_out, b_out = mlp.unpack(genomes_per_agent, cfg)

    def act(observations, rng):
        return mlp.forward(observations, w_hidden, b_hidden, w_out, b_out)
    
    return act


def oracle_policy(cfg: Config, mute: bool = False):
    """Hand-written 'ideal' recruiter.  Not evolved -- a ground-truth check.

    Behaviour from the same 9-D observation every MLP sees:
      * on food          -> emit 1, hold position (stay on the patch)
      * food sensed near -> move up the food gradient, stay quiet
      * else hears signal -> move up the signal gradient (get recruited)
      * else              -> random walk
    With ``mute=True`` the agent never signals and ignores signals: the
    non-communicating control that isolates the value of the channel.
    """
    def act(observations, rng):
        food_sensors = observations[:, 0:4]        # N,E,S,W food sensors
        on_food = observations[:, 4]
        signal_sensors = observations[:, 5:9]      # N,E,S,W signal sensors
        n_agents = observations.shape[0]

        food_strength = food_sensors.sum(1)
        signal_strength = signal_sensors.sum(1) if not mute else np.zeros(n_agents)

        def sensors_to_direction(sensors):         # [N,E,S,W] -> (+y,+x,-y,-x)
            return np.stack([sensors[:, 1] - sensors[:, 3],
                             sensors[:, 0] - sensors[:, 2]], 1)

        toward_food = sensors_to_direction(food_sensors)
        toward_signal = sensors_to_direction(signal_sensors)
        random_step = rng.normal(0, 1, size=(n_agents, 2))

        follow_food = (food_strength > 1e-6)[:, None]
        follow_signal = ((food_strength <= 1e-6) & (signal_strength > 1e-6))[:, None]
        move = np.where(follow_food, toward_food,
                        np.where(follow_signal, toward_signal, random_step))
        norm = np.linalg.norm(move, axis=1, keepdims=True) + 1e-9
        move = move / norm

        signal = np.zeros(n_agents) if mute else (on_food > 0.5).astype(float)
        return move, signal
    
    return act
