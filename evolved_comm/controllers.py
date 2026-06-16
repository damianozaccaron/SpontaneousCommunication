"""Fixed-topology controllers behind a common interface.

A *controller* knows the genome layout and how to turn a set of per-agent
genomes into a *policy* -- a callable ``policy(obs, rng) -> (move, sig)`` that
also exposes ``reset(n_agents)`` to clear any per-episode memory.  The world
(`env.simulate`) only ever sees that callable, so adding a new brain type never
touches the physics, the metric, the oracle, or the ablation.

Two backends here (both batched across all agents with einsum):
  * FeedforwardController : memoryless 9->H->3 MLP.
  * RecurrentController   : Elman RNN, h_t = tanh(W_in x + W_rec h_{t-1} + b),
                            giving the receiver the memory it needs to "keep
                            heading toward a call I heard a moment ago".

NEAT lives in neat.py (variable topology) but implements the same
``make_policy`` / ``random_population`` surface so the evolution core is shared.
"""
from __future__ import annotations
import numpy as np
from .config import Config


def _sigmoid(x):
    """Logistic squashing function, mapping any real value into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
#  Feedforward MLP                                                            #
# --------------------------------------------------------------------------- #
class _FFPolicy:
    """Memoryless MLP policy, batched across every agent at once.

    Slices each agent's flat genome into the two weight matrices and bias
    vectors of a 9->hidden->3 network. Calling it maps observations to a move
    and a signal; ``reset`` is a no-op because the network has no memory.
    """
    def __init__(self, genomes_per_agent, cfg):
        n_in, n_hidden, n_out = cfg.n_in, cfg.n_hidden, cfg.n_out
        offset = 0

        self.w_hidden = genomes_per_agent[:, offset:offset + n_in * n_hidden].reshape(-1, n_in, n_hidden); 
        offset += n_in * n_hidden
        self.b_hidden = genomes_per_agent[:, offset:offset + n_hidden];                                    
        offset += n_hidden

        self.w_out = genomes_per_agent[:, offset:offset + n_hidden * n_out].reshape(-1, n_hidden, n_out);  
        offset += n_hidden * n_out
        self.b_out = genomes_per_agent[:, offset:offset + n_out]

    def reset(self, n_agents):           # memoryless: nothing to do
        """No-op reset (a feedforward network keeps no state between steps)."""
        pass

    def __call__(self, observations, rng):
        """Map observations (n_agents, n_in) to move (n_agents, 2) and signal (n_agents,)."""
        hidden = np.tanh(np.einsum("ni,nih->nh", observations, self.w_hidden) + self.b_hidden)
        output = np.einsum("nh,nho->no", hidden, self.w_out) + self.b_out
        
        return np.tanh(output[:, :2]), _sigmoid(output[:, 2])


class FeedforwardController:
    """Factory for memoryless MLP policies; the simplest of the three brains."""
    kind = "feedforward"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property 
    def genome_len(self):
        """Flat genome length this controller expects (from the config)."""
        return self.cfg.genome_len

    def random_population(self, n_genomes, rng):
        """Draw ``n_genomes`` random genomes as Gaussian weight vectors."""
        return rng.normal(0, self.cfg.init_scale, size=(n_genomes, self.genome_len))

    def make_policy(self, population, ids_flat):
        """Build a batched policy from the genomes selected by ``ids_flat``."""
        return _FFPolicy(population[ids_flat], self.cfg)


# --------------------------------------------------------------------------- #
#  Elman recurrent MLP                                                        #
# --------------------------------------------------------------------------- #
class _RNNPolicy:
    """Elman recurrent policy: like ``_FFPolicy`` but with a hidden state.

    The extra hidden->hidden weight block lets the hidden state carry
    information across steps -- the memory a receiver needs to keep heading
    toward a call it heard a moment ago. ``reset`` zeroes that state at the
    start of each episode.
    """
    def __init__(self, genomes_per_agent, cfg):
        n_in, n_hidden, n_out = cfg.n_in, cfg.n_hidden, cfg.n_out
        offset = 0

        self.w_in = genomes_per_agent[:, offset:offset + n_in * n_hidden].reshape(-1, n_in, n_hidden);      
        offset += n_in * n_hidden
        self.w_rec = genomes_per_agent[:, offset:offset + n_hidden * n_hidden].reshape(-1, n_hidden, n_hidden); 
        offset += n_hidden * n_hidden

        self.b_hidden = genomes_per_agent[:, offset:offset + n_hidden];                                     
        offset += n_hidden

        self.w_out = genomes_per_agent[:, offset:offset + n_hidden * n_out].reshape(-1, n_hidden, n_out);   
        offset += n_hidden * n_out
        self.b_out = genomes_per_agent[:, offset:offset + n_out]
        self.hidden_size = n_hidden
        self.hidden_state = None

    def reset(self, n_agents):
        """Zero the recurrent hidden state for a fresh episode of ``n_agents``."""
        self.hidden_state = np.zeros((n_agents, self.hidden_size))

    def __call__(self, observations, rng):
        """Advance the RNN one step, returning move (n_agents, 2) and signal (n_agents,)."""
        if self.hidden_state is None or self.hidden_state.shape[0] != observations.shape[0]:
            self.reset(observations.shape[0])

        hidden = np.tanh(np.einsum("ni,nih->nh", observations, self.w_in)
                         + np.einsum("nh,nhk->nk", self.hidden_state, self.w_rec) + self.b_hidden)
        self.hidden_state = hidden
        output = np.einsum("nh,nho->no", hidden, self.w_out) + self.b_out

        return np.tanh(output[:, :2]), _sigmoid(output[:, 2])


class RecurrentController:
    """Factory for Elman recurrent policies (memory can evolve in the weights)."""
    kind = "recurrent"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def genome_len(self):
        """Flat genome length this controller expects (includes the recurrent block)."""
        return self.cfg.genome_len

    def random_population(self, n_genomes, rng):
        """Draw ``n_genomes`` random genomes as Gaussian weight vectors."""
        return rng.normal(0, self.cfg.init_scale, size=(n_genomes, self.genome_len))

    def make_policy(self, population, ids_flat):
        """Build a batched recurrent policy from the genomes selected by ``ids_flat``."""
        return _RNNPolicy(population[ids_flat], self.cfg)


def get_vector_controller(cfg: Config):
    """Return the fixed-topology controller named by ``cfg.controller``.

    Handles the two array-genome brains (feedforward / recurrent); NEAT is
    built elsewhere. Raises ``ValueError`` for any other name.
    """
    if cfg.controller == "feedforward":
        return FeedforwardController(cfg)
    if cfg.controller == "recurrent":
        return RecurrentController(cfg)
    raise ValueError(f"not a vector controller: {cfg.controller}")
