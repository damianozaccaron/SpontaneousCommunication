"""Central configuration. One Config object fully specifies a run.

New in this version (vs. the laptop prototype):
  * controller:  'feedforward' | 'recurrent' (Elman memory)
  * regime:      'colony' | 'individual' | 'hybrid' (alpha-blended fitness)
  * eval_repeats: average mixed-group fitness over several random group draws
  * checkpoint_every: periodic resume support for long runs
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- World -----------------------------------------------------------
    world_size: float = 1.0
    group_size: int = 8
    n_patches: int = 2
    patch_capacity: int = 12
    patch_radius: float = 0.05
    episode_steps: int = 150
    respawn_delay: int = 6
    r_sense: float = 0.05
    r_eat: float = 0.03
    r_signal: float = 0.55
    move_speed: float = 0.04

    # ---- Controller ------------------------------------------------------
    controller: str = "feedforward"      # 'feedforward' | 'recurrent'
    n_in: int = 9
    n_hidden: int = 16
    n_out: int = 3
    init_scale: float = 0.5 # Starts with weights distributed around a Gaussian with mean 0 and sd 0.5

    # ---- Genetic algorithm (fixed-topology controllers) ------------------
    pop_size: int = 300
    generations: int = 800
    tournament_k: int = 3
    elitism: int = 4
    mut_sigma: float = 0.10
    mut_rate: float = 0.9
    crossover_rate: float = 0.3

    # ---- Selection regime ------------------------------------------------
    # colony     : clonal groups, fitness = group yield
    # individual : mixed groups,  fitness = personal intake
    # hybrid     : mixed groups,  fitness = alpha*personal + (1-alpha)*group
    regime: str = "colony"
    alpha: float = 0.5 # for hybrid 
    eval_repeats: int = 3 # group re-draws for mixed regimes
    ablate_signal: bool = False # for control

    # ---- Measurement -----------------------------------------------------
    signal_bins: int = 8                 # legacy equal-width binning (unused by signal_food_mi)
    signal_threshold: float = 0.5        # signal>thr => "loud"; MI is computed on this binary symbol
    mi_null_shuffles: int = 50

    # ---- Bookkeeping -----------------------------------------------------
    seed: int = 0
    checkpoint_every: int = 100            # 0 disables; else save every N gens

    @property
    def genome_len(self) -> int:
        """Number of weights in one fixed-topology genome.

        Counts every weight and bias the controller needs, flattened into a
        single vector: input->hidden weights and hidden biases, then
        hidden->output weights and output biases. A recurrent controller also
        carries a hidden->hidden block, so that term is added on demand.
        """
        n_hidden, n_in, n_out = self.n_hidden, self.n_in, self.n_out
        weight_count = n_in * n_hidden + n_hidden + n_hidden * n_out + n_out
        if self.controller == "recurrent":
            weight_count += n_hidden * n_hidden   # recurrent hidden->hidden block
        return weight_count

    def to_dict(self) -> dict:
        """Return all configuration fields as a plain dict (for JSON logging)."""
        return asdict(self)
