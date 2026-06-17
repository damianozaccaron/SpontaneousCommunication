# Evolving Communication from Scratch — v2

A pure-NumPy platform for studying whether **informative signalling emerges by
selection alone** (no reward ever touches the signal), and how the **level of
selection** controls it. v2 adds a second brain type, a hybrid selection regime,
and a parallel, restartable runner for serious compute.

## What's new vs v1
- **Two controllers** behind one interface: `feedforward` MLP and `recurrent`
  Elman RNN (memory).
- **Three regimes**: `colony` (clones, group fitness), `individual` (mixed,
  personal intake), `hybrid` (mixed, `alpha*personal + (1-alpha)*group`).
- **Multi-group averaging** (`eval_repeats`) to de-noise mixed-group fitness.
- **Parallel runner** over CPU cores, **checkpoint/resume**, JSONL logging.

## Run it
```bash
pip install numpy scipy scikit-learn matplotlib

# 0) always validate the environment first
python validate_oracle.py

# 1) the headline sweep (set --workers to ~cores-1)
python run_experiments.py \
    --controllers feedforward,recurrent \
    --regimes colony,individual,hybrid --alphas 0.25,0.5,0.75 \
    --ablate --seeds 15 --gens 800 --pop 300 --workers 12

# 2) figures + stats (reads ./data by default)
python make_figures.py
```
Jobs are cached per `.npz`, so re-running resumes where you left off. Each run
also checkpoints every `--checkpoint_every` generations.

## File-by-file
| File | Role |
|------|------|
| `evolved_comm/config.py` | One `Config` dataclass = one fully-specified run (world, controller, GA, regime, checkpoint). |
| `evolved_comm/env.py` | Vectorised 2D torus world: renewable patches, short-range sensing, long-range signals, rate-limited eating. Controller-agnostic via a `policy(obs)` callable; clears policy memory each episode. |
| `evolved_comm/controllers.py` | Fixed-topology brains: feedforward + Elman recurrent. Each turns per-agent genomes into a stateful `policy` object. |
| `evolved_comm/evolution.py` | Brain-agnostic core: group assembly + colony/individual/hybrid fitness, the fixed-topology GA, MI probe, checkpointing, and the `evolve()` loop. |
| `evolved_comm/metrics.py` | `I(signal; food-state)` with a label-shuffle null band. |
| `evolved_comm/policies.py` | Hand-scripted **oracle** (and legacy MLP policy). Used for environment validation. |
| `run_experiments.py` | Builds the job matrix, runs it across cores, writes `data/*.npz` + JSONL logs, checkpoints, restartable. |
| `make_figures.py` | Adapts to whatever runs exist: learning + emergence per controller, value bar, hybrid alpha-sweep, signal map, `stats.json`. |
| `validate_oracle.py` | Confirms the world rewards communication (oracle vs mute). |

`evolved_comm/ga.py` and `mlp.py` are **legacy v1** modules, superseded by
`evolution.py` and `controllers.py`; kept only so the v1 report scripts still run.

## Reading the results honestly
MI measures *informativeness*, which can be high for free (the signal neuron
shares inputs with the food sensors — see v1's ablated controls). The
**ablation yield-drop** (channel on vs perception severed) is the test of
*functional* communication. Always read them together; the `alpha_sweep` figure
is the cleanest view of how the selection level moves both.
