"""A compact, self-contained NEAT (NeuroEvolution of Augmenting Topologies).

Genomes carry node genes and connection genes with global innovation numbers;
mutation can perturb weights, add a connection, or split a connection into a new
hidden node; crossover aligns genes by innovation; the population is divided into
species by a topological-distance metric with fitness sharing.  Recurrent
connections are permitted, so memory can evolve for free.

Networks are evaluated by ``neat_internal_steps`` synchronous propagation passes
per environment step, batched across all agents that share a genome.  The class
exposes the same ``make_policy`` / ``random_population`` surface as the
fixed-topology controllers, so evolution.py drives all three brain types
identically.  Node id convention: inputs 0..n_in-1, bias n_in,
outputs n_in+1..n_in+n_out, hidden ids above that.
"""
from __future__ import annotations
import numpy as np
from copy import deepcopy
from .config import Config


# --------------------------------------------------------------------------- #
#  Genome                                                                     #
# --------------------------------------------------------------------------- #
class Innovations:
    """Per-run registry: stable innovation numbers for connections and nodes.

    Every distinct connection (source, target) and every newly created node is
    assigned a globally unique, stable id the first time it appears. These ids
    let genes from different genomes be aligned during crossover and distance.
    """
    def __init__(self, first_node):
        self.connection_ids = {}          # (source, target) -> innovation id
        self.next_connection_id = 0
        self.next_node_id = first_node

    def conn_id(self, source, target):
        """Return the innovation id for a connection, creating one if new."""
        key = (source, target)
        if key not in self.connection_ids:
            self.connection_ids[key] = self.next_connection_id
            self.next_connection_id += 1
        return self.connection_ids[key]

    def new_node(self):
        """Allocate and return a fresh, never-before-used hidden node id."""
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id


class Genome:
    """One NEAT individual: a set of connection genes plus its hidden nodes.

    ``conns`` maps innovation id -> ``[source, target, weight, enabled]`` and
    ``hidden`` is the set of hidden-node ids. Input/bias/output nodes are
    implicit (their ids follow a fixed convention), so only hidden nodes are
    stored explicitly.
    """
    __slots__ = ("conns", "hidden")

    def __init__(self):
        self.conns = {}               # innov id -> [source, target, weight, enabled]
        self.hidden = set()           # hidden node ids

    def clone(self):
        """Return a deep copy of this genome (independent gene lists)."""
        twin = Genome()
        twin.conns = {k: v[:] for k, v in self.conns.items()}
        twin.hidden = set(self.hidden)
        return twin


def io_nodes(cfg):
    """Return the fixed input ids, bias id, and output ids for this config."""
    n_in, n_out = cfg.n_in, cfg.n_out
    inputs = list(range(n_in))
    bias = n_in
    outputs = list(range(n_in + 1, n_in + 1 + n_out))
    return inputs, bias, outputs


def random_genome(cfg, innov, rng):
    """Minimal start: every input (+bias) connected to every output."""
    inputs, bias, outputs = io_nodes(cfg)
    genome = Genome()
    for source in inputs + [bias]:
        for target in outputs:
            genome.conns[innov.conn_id(source, target)] = [source, target,
                                                           float(rng.normal(0, 1)), True]
    return genome


# --------------------------------------------------------------------------- #
#  Mutation & crossover                                                       #
# --------------------------------------------------------------------------- #
def mutate(genome, cfg, innov, rng):
    """Return a mutated copy: perturb weights, and maybe add a connection/node.

    Each of the three structural events fires independently with its own
    probability from the config: weight perturbation/replacement, adding one
    new connection, and splitting an existing connection with a new node.
    """
    genome = genome.clone()
    # weight mutation
    if rng.random() < cfg.neat_weight_mut:
        for connection in genome.conns.values():
            if rng.random() < cfg.neat_weight_replace:
                connection[2] = float(rng.normal(0, 1))
            else:
                connection[2] += float(rng.normal(0, cfg.neat_weight_sigma))
    # add connection
    if rng.random() < cfg.neat_add_conn:
        _add_connection(genome, cfg, innov, rng)
    # add node
    if rng.random() < cfg.neat_add_node and genome.conns:
        _add_node(genome, cfg, innov, rng)
    return genome


def _add_connection(genome, cfg, innov, rng):
    """Wire up the first random unused source->target pair found, if any."""
    inputs, bias, outputs = io_nodes(cfg)
    sources = inputs + [bias] + list(genome.hidden) + outputs
    targets = list(genome.hidden) + outputs
    rng.shuffle(sources)
    for source in sources:
        for target in targets:
            if source == target:
                continue
            connection_id = innov.conn_id(source, target)
            if connection_id in genome.conns:
                continue
            genome.conns[connection_id] = [source, target, float(rng.normal(0, 1)), True]
            return


def _add_node(genome, cfg, innov, rng):
    """Split a random enabled connection in two with a new hidden node between.

    The old connection is disabled; the source->new edge gets weight 1 and the
    new->target edge inherits the old weight, so the network's function is
    initially unchanged (the standard NEAT add-node trick).
    """
    enabled = [cid for cid, c in genome.conns.items() if c[3]]
    if not enabled:
        return
    connection_id = enabled[rng.integers(len(enabled))]
    source, target, weight, _ = genome.conns[connection_id]
    genome.conns[connection_id][3] = False               # disable old
    new_node = innov.new_node()
    genome.hidden.add(new_node)
    genome.conns[innov.conn_id(source, new_node)] = [source, new_node, 1.0, True]
    genome.conns[innov.conn_id(new_node, target)] = [new_node, target, weight, True]


def crossover(parent1, fitness1, parent2, fitness2, cfg, rng):
    """Inherit matching genes at random; disjoint/excess from the fitter parent."""
    if fitness2 > fitness1:
        parent1, parent2 = parent2, parent1
    child = Genome()
    for connection_id, connection in parent1.conns.items():
        if connection_id in parent2.conns and rng.random() < 0.5:
            child.conns[connection_id] = parent2.conns[connection_id][:]
        else:
            child.conns[connection_id] = connection[:]
    # hidden nodes have ids at/above the first hidden id; rebuild from conns
    first_hidden = cfg.n_in + 1 + cfg.n_out
    child.hidden = {node_id for connection in child.conns.values()
                    for node_id in (connection[0], connection[1]) if node_id >= first_hidden}
    return child


def distance(genome1, genome2, cfg):
    """Topological + weight distance between two genomes (for speciation).

    Combines the count of non-matching (disjoint/excess) genes, normalised by
    genome size, with the mean weight difference of the matching genes, each
    scaled by its config coefficient.
    """
    all_ids = set(genome1.conns) | set(genome2.conns)
    if not all_ids:
        return 0.0
    matching = set(genome1.conns) & set(genome2.conns)
    n_disjoint = len(all_ids) - len(matching)
    if matching:
        mean_weight_diff = np.mean([abs(genome1.conns[cid][2] - genome2.conns[cid][2])
                                    for cid in matching])
    else:
        mean_weight_diff = 0.0
    larger_size = max(len(genome1.conns), len(genome2.conns), 1)
    return (cfg.neat_c2 * n_disjoint) / larger_size + cfg.neat_c3 * mean_weight_diff


# --------------------------------------------------------------------------- #
#  Network (batched evaluation)                                               #
# --------------------------------------------------------------------------- #
class _Network:
    """A genome compiled into a batched, recurrent forward evaluator.

    Node ids are mapped to dense array slots once; each call runs
    ``neat_internal_steps`` synchronous propagation passes (tanh on hidden
    nodes, linear on outputs) over a batch of agents that share this genome.
    Because connections may point anywhere, recurrence is supported and the
    activation state persists between steps until ``reset``.
    """
    def __init__(self, genome, cfg):
        inputs, bias, outputs = io_nodes(cfg)
        # collect node ids actually used, fix canonical slots for io
        node_ids = set(inputs) | {bias} | set(outputs) | set(genome.hidden)
        for connection in genome.conns.values():
            node_ids.add(connection[0]); node_ids.add(connection[1])
        ordered_ids = sorted(node_ids)
        self.slot_of = {node_id: slot for slot, node_id in enumerate(ordered_ids)}
        self.n_nodes = len(ordered_ids)
        self.input_slots = np.array([self.slot_of[i] for i in inputs])
        self.bias_slot = self.slot_of[bias]
        self.output_slots = np.array([self.slot_of[o] for o in outputs])
        # non-input nodes get activations updated
        update_mask = np.ones(self.n_nodes, bool)
        update_mask[self.input_slots] = False
        update_mask[self.bias_slot] = False
        self.update_mask = update_mask
        is_output = np.zeros(self.n_nodes, bool)
        is_output[self.output_slots] = True
        self.hidden_mask = update_mask & ~is_output
        enabled = [c for c in genome.conns.values() if c[3]]
        self.edge_from = np.array([self.slot_of[c[0]] for c in enabled], dtype=int) if enabled else np.zeros(0, int)
        self.edge_to = np.array([self.slot_of[c[1]] for c in enabled], dtype=int) if enabled else np.zeros(0, int)
        self.edge_weight = np.array([c[2] for c in enabled]) if enabled else np.zeros(0)
        self.n_steps = cfg.neat_internal_steps
        self.activation = None

    def reset(self, n_agents):
        """Zero the activation state for a batch of ``n_agents`` sharing this net."""
        self.activation = np.zeros((n_agents, self.n_nodes))

    def step(self, observations):
        """Run the propagation passes and return (move (M,2), signal (M,))."""
        n_agents = observations.shape[0]
        if self.activation is None or self.activation.shape[0] != n_agents:
            self.reset(n_agents)
        activation = self.activation
        activation[:, self.input_slots] = observations
        activation[:, self.bias_slot] = 1.0
        rows = np.arange(n_agents)[:, None]
        for _ in range(self.n_steps):
            incoming = np.zeros_like(activation)
            if self.edge_from.size:
                np.add.at(incoming, (rows, self.edge_to[None, :]),
                          activation[:, self.edge_from] * self.edge_weight[None, :])
            activation = activation.copy()
            activation[:, self.hidden_mask] = np.tanh(incoming[:, self.hidden_mask])
            activation[:, self.output_slots] = incoming[:, self.output_slots]   # linear outputs
        self.activation = activation
        output = activation[:, self.output_slots]
        return np.tanh(output[:, :2]), np.where(
            output[:, 2] >= 0,
            1.0 / (1.0 + np.exp(-output[:, 2])),
            np.exp(output[:, 2]) / (1.0 + np.exp(output[:, 2]))
        )


class _NeatPolicy:
    """Batched policy over a mixed population of NEAT genomes.

    Agents are grouped by which genome they use; each unique genome is compiled
    into a ``_Network`` once and then driven for just the agents that share it.
    """
    def __init__(self, population, ids_flat, cfg):
        # group agents by which genome they use; compile each unique genome once
        self.groups = []             # (agent_index_array, network)
        compiled = {}
        ids = np.asarray(ids_flat)
        for genome_id in np.unique(ids):
            members = np.where(ids == genome_id)[0]
            if genome_id not in compiled:
                compiled[genome_id] = _Network(population[genome_id], cfg)
            self.groups.append((members, compiled[genome_id]))
        self.N = len(ids)

    def reset(self, n_agents):
        """Reset every per-genome network's activation state for a new episode."""
        for members, network in self.groups:
            network.reset(len(members))

    def __call__(self, observations, rng):
        """Evaluate all agents, dispatching each to its genome's network."""
        move = np.zeros((self.N, 2)); signal = np.zeros(self.N)
        for members, network in self.groups:
            member_move, member_signal = network.step(observations[members])
            move[members] = member_move; signal[members] = member_signal
        return move, signal


class NeatController:
    """Factory exposing the shared controller surface for NEAT genomes."""
    kind = "neat"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        inputs, bias, outputs = io_nodes(cfg)
        self.innov = Innovations(first_node=cfg.n_in + 1 + cfg.n_out)

    def random_population(self, n_genomes, rng):
        """Create ``n_genomes`` minimally-connected starting genomes."""
        return [random_genome(self.cfg, self.innov, rng) for _ in range(n_genomes)]

    def make_policy(self, population, ids_flat):
        """Build a batched policy from the genomes selected by ``ids_flat``."""
        return _NeatPolicy(population, ids_flat, self.cfg)


# --------------------------------------------------------------------------- #
#  Speciated reproduction                                                     #
# --------------------------------------------------------------------------- #
def reproduce(population, fitness, cfg, innov, rng):
    """One generation of speciation + fitness-shared reproduction."""
    pop_size = len(population)
    fitness = np.asarray(fitness, float)
    fitness = fitness - fitness.min() + 1e-6     # keep strictly positive

    # --- adaptive compatibility threshold ---------------------------------
    # The static neat_compat_threshold sits on a scale the within-population
    # distance rarely reaches (close-kin genomes => 1 species forever), which
    # makes fitness sharing a no-op.  Instead we self-tune the threshold each
    # generation toward ~neat_target_species; neat_compat_threshold is only the
    # starting point.  Set neat_target_species <= 0 to restore legacy behaviour.
    target_species = getattr(cfg, "neat_target_species", 12)
    adaptive = bool(target_species and target_species > 0)
    if adaptive:
        compat_thr = getattr(cfg, "_compat_thr", None)
        if compat_thr is None:
            compat_thr = min(cfg.neat_compat_threshold, 1.0)
    else:
        compat_thr = cfg.neat_compat_threshold

    # --- speciation (greedy, threshold on distance to a species rep) ---
    species = []                                 # list of dict(rep, members[idx])
    for i in range(pop_size):
        placed = False
        for group in species:
            if distance(population[i], group["rep"], cfg) < compat_thr:
                group["members"].append(i); placed = True; break
        if not placed:
            species.append({"rep": population[i], "members": [i]})

    # --- offspring allocation by mean adjusted fitness ---
    adjusted_fitness = [np.mean([fitness[i] / len(group["members"]) for i in group["members"]])
                        for group in species]
    total_adjusted = sum(adjusted_fitness) or 1.0
    offspring_counts = [max(1, int(round(a / total_adjusted * pop_size))) for a in adjusted_fitness]
    while sum(offspring_counts) > pop_size:
        offspring_counts[int(np.argmax(offspring_counts))] -= 1
    while sum(offspring_counts) < pop_size:
        offspring_counts[int(np.argmax(adjusted_fitness))] += 1

    next_population = []
    for group, n_offspring in zip(species, offspring_counts):
        members = sorted(group["members"], key=lambda i: fitness[i], reverse=True)
        n_survivors = max(1, int(round(cfg.neat_survival * len(members))))
        breeders = members[:n_survivors]
        if len(members) >= 5:                    # elitism for sizeable species
            next_population.append(population[members[0]].clone()); n_offspring -= 1
        for _ in range(max(0, n_offspring)):
            parent_a = breeders[rng.integers(len(breeders))]
            parent_b = breeders[rng.integers(len(breeders))]
            child = crossover(population[parent_a], fitness[parent_a],
                              population[parent_b], fitness[parent_b], cfg, rng)
            child = mutate(child, cfg, innov, rng)
            next_population.append(child)
        group["rep"] = population[members[rng.integers(len(members))]]

    # nudge the threshold toward the target species count for the next gen
    if adaptive:
        if len(species) < target_species:
            compat_thr = max(0.05, compat_thr - 0.1)
        elif len(species) > target_species:
            compat_thr = compat_thr + 0.1
        cfg._compat_thr = compat_thr
    return next_population[:pop_size], len(species)
