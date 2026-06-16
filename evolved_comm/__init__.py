"""Evolved-communication mini-framework for the Bio-Inspired AI project.

Modules
-------
config    : the Config dataclass (one object fully specifies a run)
mlp       : fixed-topology batched MLP controller
env       : vectorised 2D foraging world (torus, patchy food, signals)
policies  : MLP closure + hand-scripted oracle signaller
metrics   : I(signal; food_state) with label-shuffle null band
ga        : the two selection regimes + a ~60-line genetic algorithm
"""
from .config import Config
