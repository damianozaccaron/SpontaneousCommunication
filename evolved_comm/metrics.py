from __future__ import annotations
import numpy as np
from sklearn.metrics import mutual_info_score
from .config import Config


def discretise_signal(signal: np.ndarray, cfg: Config) -> np.ndarray:
    """Bin a continuous signal in [0, 1] into ``cfg.signal_bins`` integer labels.

    Mutual information needs discrete symbols, so the real-valued signal is
    quantised into equal-width bins. ``signal`` may have any shape; it is
    flattened and a 1-D array of bin indices is returned.
    """
    bin_edges = np.linspace(0.0, 1.0, cfg.signal_bins + 1)[1:-1]
    return np.digitize(signal.ravel(), bin_edges)


def signal_food_mi(signal: np.ndarray, food_state: np.ndarray, cfg: Config,
                   rng: np.random.Generator | None = None):
    """Mutual information I(signal; food_state) plus a shuffle-based noise floor.

    Measures how much an agent's emitted signal tells you about whether it is
    currently on food. Because MI is positively biased with finite samples, a
    null distribution is built by repeatedly shuffling the food-state labels
    (which destroys any real dependence) and recomputing MI; the 95th
    percentile of that null is the floor a genuine emergence claim must beat.

    ``signal`` and ``food_state`` are matching arrays of any shape (flattened
    internally). The signal is binarised at ``cfg.signal_threshold`` (loud vs
    silent); the null shuffles food-state against that same binary symbol.
    Returns ``(mi, null_p95)`` in nats (sklearn uses natural log).
    """
    signal_symbols = (signal.ravel() > cfg.signal_threshold).astype(int)
    food_labels = food_state.ravel().astype(int)
    observed_mi = mutual_info_score(food_labels, signal_symbols)

    if rng is None:
        rng = np.random.default_rng(0)

    null_mi = np.empty(cfg.mi_null_shuffles)
    for shuffle in range(cfg.mi_null_shuffles):
        null_mi[shuffle] = mutual_info_score(rng.permutation(food_labels), signal_symbols)
        
    return float(observed_mi), float(np.percentile(null_mi, 95))
