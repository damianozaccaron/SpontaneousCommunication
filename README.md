# Spontaneous Emergence of Communication in Evolved Foraging Agents

A pure-NumPy platform for studying whether **informative signalling emerges by
selection alone** (no reward ever touches the signal), and how the **level of
selection** controls it. Uses two types of Neural Networks evolved through custom
Genetic Algorithms: a memoryless `feedforward` network and a `recurrent` Elman RNN.
Three regimes can be used for experiments: `colony` (clones, group fitness), `individual` (mixed,
personal intake), `hybrid` (mixed, `alpha*personal + (1-alpha)*group`).

For more details, check the included paper.

## How to run the code
```bash
pip install -r requirements.txt

# 0) environment check
python validate_oracle.py

# 1) run evolution script
python run_experiments.py \
    --controllers feedforward,recurrent \
    --regimes colony,individual,hybrid --alphas 0.5 \
    --ablate --seeds 10 --gens 600 --pop 250 --workers <NCORES-1>

# 2) figures + stats
python make_figures.py

# 3) Mutual Information Computation
python compute_mi.py
```
Jobs are cached per `.npz`, so re-running resumes where you left off. If the data folder 
is already populated, it won't run anything.
