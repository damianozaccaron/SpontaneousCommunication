"""Fixed-topology MLP controller, evaluated batched over all agents at once.

The architecture is held constant across every condition so that the *only*
thing evolution can vary is whether the signal channel gets used.  Weights for
N agents are stored as stacked arrays and the forward pass is a pair of
per-agent batched matrix multiplies (einsum), so a whole population of groups
steps in one vectorised call.
"""
from __future__ import annotations
import numpy as np
from .config import Config


def unpack(genomes: np.ndarray, cfg: Config):
    """Slice a (n_agents, genome_len) array into per-agent weight tensors.

    Returns the input->hidden weights and hidden biases, then the
    hidden->output weights and output biases:
    w_hidden (n_agents, n_in, n_hidden), b_hidden (n_agents, n_hidden),
    w_out (n_agents, n_hidden, n_out), b_out (n_agents, n_out).
    """
    n_in, n_hidden, n_out = cfg.n_in, cfg.n_hidden, cfg.n_out
    offset = 0
    w_hidden = genomes[:, offset:offset + n_in * n_hidden].reshape(-1, n_in, n_hidden); offset += n_in * n_hidden
    b_hidden = genomes[:, offset:offset + n_hidden];                                    offset += n_hidden
    w_out = genomes[:, offset:offset + n_hidden * n_out].reshape(-1, n_hidden, n_out);  offset += n_hidden * n_out
    b_out = genomes[:, offset:offset + n_out]
    return w_hidden, b_hidden, w_out, b_out


def forward(observations: np.ndarray, w_hidden, b_hidden, w_out, b_out):
    """Batched MLP forward pass over all agents.

    Maps observations (n_agents, n_in) to a move (n_agents, 2) in [-1, 1] and a
    signal (n_agents,) in [0, 1].
    """
    hidden = np.tanh(np.einsum("ni,nih->nh", observations, w_hidden) + b_hidden)
    output = np.einsum("nh,nho->no", hidden, w_out) + b_out
    move = np.tanh(output[:, :2])
    signal = 1.0 / (1.0 + np.exp(-output[:, 2]))
    return move, signal


def random_population(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Initialise a population of ``pop_size`` random Gaussian weight vectors."""
    return rng.normal(0.0, cfg.init_scale, size=(cfg.pop_size, cfg.genome_len))
