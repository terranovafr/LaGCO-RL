#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

import random
from gymnasium import spaces
import numpy as np
import itertools
import networkx as nx
from utils.features_utils import get_number_nodes, get_number_edges, get_average_node_degree, get_graph_density
from utils.model import Node, Graph, Function, Attribute
from utils.feature_extractors import OneHotEncoding, IdentityEncoding
from utils.pooling_functions import MeanPooling, SumPooling, MinPooling, MaxPooling
from utils.encoding_options import ContinuousValue, MultiClassValue
from envs.env import DiscreteEnv
from wrappers.wrapper import ContinuousEnv
from tqdm import tqdm

class MaxCutEnv(DiscreteEnv):
    """
    MaxCut Environment.

    Agent assigns nodes to one of two partitions to maximize the total cut weight.
    """
    def __init__(self,
                 num_nodes=10,
                 max_weight=10,
                 reward_type="relative",
                 max_sweeps=1000,
                 largest_K_edges=None,
                 **kwargs):

        super().pre_init(**kwargs)
        self.num_nodes = num_nodes
        self.max_weight = max_weight
        self.reward_type = reward_type
        self.largest_K_edges = largest_K_edges
        self.max_sweeps = max_sweeps
        self.scenario_size = num_nodes
        # Observation: partition assignments + upper-triangular weights
        obs_len = self.num_nodes + (self.num_nodes * (self.num_nodes - 1)) // 2
        self.observation_space = spaces.Dict({
            "graph": spaces.Box(low=0, high=self.max_weight, shape=(obs_len,), dtype=np.float32)
        })
        self.current_cut = 0
        self._generate_graph()
        self.best_cut, self.worst_cut = self.estimate_bounds(self.max_sweeps)
        self.semantic_action_list = self.build_semantic_action_list()
        self.action_list = self.build_action_list()
        super().post_init()
        self.reset()


    def is_valid_action(self, action):
        return action in self.graph.nodes

    def build_action_list(self):
        return [action for action in self.graph.nodes]

    def build_semantic_action_list(self):
        # return them ordered by degree, so that the agent can learn to assign high degree nodes first
        degree_dict = dict(self.graph.degree())
        return sorted(self.graph.nodes, key=lambda x: degree_dict[x], reverse=True)

    def _generate_graph(self):
        self.graph = nx.Graph()
        # symmetric adjacency
        self.weights_matrix = np.random.randint(1, self.max_weight + 1, size=(self.num_nodes, self.num_nodes))
        self.weights_matrix = np.triu(self.weights_matrix, 1)
        self.weights_matrix += self.weights_matrix.T

        for i in range(self.num_nodes):
            self.graph.add_node(i, x_dict={'partition': 0})  # start all in partition 0

        # keep only the top K edges per node if largest_K_edges is set, otherwise keep all edges
        if self.largest_K_edges is None:
            for i, j in itertools.combinations(range(self.num_nodes), 2):
                self.graph.add_edge(
                    i, j,
                    edge_attr_dict={'weight': self.weights_matrix[i, j]}
                )
        else:
            for i in range(self.num_nodes):
                topk = np.argsort(self.weights_matrix[i])[::-1]
                topk = [j for j in topk if j != i][:self.largest_K_edges]

                for j in topk:
                    if not self.graph.has_edge(i, j):
                        self.graph.add_edge(
                            i, j,
                            edge_attr_dict={'weight': self.weights_matrix[i, j]}
                        )

    def estimate_bounds(self, samples=1000, local_steps=None):
        # estimate best and worst cut values by sampling random partitions and doing local improvements
        if local_steps is None:
            local_steps = self.num_nodes

        best_cut = -np.inf

        for _ in tqdm(range(samples), desc="Estimating bounds"):
            partition = np.random.randint(0, 2, size=self.num_nodes, dtype=np.int8)
            current_value = self.compute_cut(partition)

            # stochastic local improvement
            for _ in range(local_steps):
                improvements = []
                for i in range(self.num_nodes):
                    partition[i] ^= 1
                    new_value = self.compute_cut(partition)
                    gain = new_value - current_value
                    partition[i] ^= 1

                    if gain > 0:
                        improvements.append((i, gain))

                if not improvements:
                    break

                # random among improving moves, biased toward stronger gains
                idxs = [i for i, g in improvements]
                gains = [g for i, g in improvements]
                chosen = random.choices(idxs, weights=gains, k=1)[0]

                partition[chosen] ^= 1
                current_value = self.compute_cut(partition)

            best_cut = max(best_cut, current_value)

        # worst is when all nodes are in the same partition, which can be computed in closed form
        return best_cut, 0

    def compute_cut(self, partition):
        # compute cut value for a given partition assignment
        partition = np.asarray(partition)
        diff = partition[:, None] != partition[None, :]
        return np.sum(self.weights_matrix[np.triu_indices(self.num_nodes, 1)] *
                      diff[np.triu_indices(self.num_nodes, 1)])

    def reset(self, **kwargs):
        super().pre_reset()
        # generate random weighted graph
        for i in range(self.num_nodes):
            self.graph.nodes[i]['x_dict']['partition'] = 0
        self.assigned_nodes = set()
        self.current_cut = 0
        self.prev_cut = 0
        self.obs = self._get_obs()
        self.info = {}
        super().post_reset()
        return self.obs, {}

    def _get_obs(self):
        # Observation is a concatenation of partition assignments and upper-triangular weights
        partitions = np.array([self.graph.nodes[i]['x_dict']['partition'] for i in range(self.num_nodes)], dtype=np.float32)
        upper_tri_indices = np.triu_indices(self.num_nodes, 1)
        weights_flat = self.weights_matrix[upper_tri_indices]
        obs = np.concatenate([partitions, weights_flat])
        return {"graph": obs.astype(np.float32)}

    def step(self, node):
        node = super().pre_step(node)

        self.reward = 0.0
        self.info = {}

        if self.invalid_action or self.no_action:
            self.obs = self._get_obs()
            super().post_step()
            return self.obs, self.reward, self.done, self.truncated, self.info

        # flip the partition of the selected node and compute the change in cut value
        old_part = self.graph.nodes[node]["x_dict"]["partition"]
        new_part = 1 - old_part

        delta = self._compute_flip_gain(node, old_part)

        self.graph.nodes[node]["x_dict"]["partition"] = new_part
        self.assigned_nodes.add(node)

        self.prev_cut = self.current_cut
        self.current_cut += delta

        if self.reward_type == "relative":
            denom = max(self.best_cut - self.worst_cut, 1e-8)
            self.reward = delta / denom
        else:
            self.reward = delta

        if self.terminate_at_approximate_best:
            self.done = (self.current_cut >= self.best_cut)
        else:
            self.done = False

        self.obs = self._get_obs()
        super().post_step()
        return self.obs, self.reward, self.done, self.truncated, self.info

    def _compute_flip_gain(self, node, old_part):
        # easy to compute incrementally: only edges connected to the flipped node can change their crossing status
        delta = 0.0
        for neighbor in range(self.num_nodes):
            if neighbor == node:
                continue
            w = self.weights_matrix[node, neighbor]
            neigh_part = self.graph.nodes[neighbor]["x_dict"]["partition"]

            # before flip:
            # - if same partition => edge does NOT cross
            # - if different      => edge DOES cross
            #
            # after flip:
            # - if same before    => becomes crossing  => +w
            # - if different      => becomes non-cross => -w
            delta += w if neigh_part == old_part else -w

        return delta

    def update_metrics(self):
        normalized_perf = (self.current_cut - self.worst_cut) / (self.best_cut - self.worst_cut + 1e-8)
        self._metrics.update({
            "current_cut": self.current_cut,
            "partition0_size": sum(1 for i in range(self.num_nodes) if self.graph.nodes[i]['x_dict']['partition'] == 0),
            "partition1_size": sum(1 for i in range(self.num_nodes) if self.graph.nodes[i]['x_dict']['partition'] == 1),
            "relative_performance": normalized_perf
        })

    def update_config(self, config):
        super().update_config(config)
        self.reward_type = config.get('reward_type', self.reward_type)
        old_largest_K_edges = self.largest_K_edges
        self.largest_K_edges = config.get('largest_K_edges', self.largest_K_edges)
        if self.largest_K_edges != old_largest_K_edges:
            # need to regenerate the graph with the new sparsity pattern
            self._generate_graph()
            self.best_cut, self.worst_cut = self.estimate_bounds(self.max_sweeps)
        self.reset()

class ExtendedMaxCutEnv(MaxCutEnv, ContinuousEnv):
    def __init__(self, **config):
        super().__init__(**config)
        self.update_spec()

    def update_spec(self):
        self._observation_type = {
            'graph': Graph(poolings=[MeanPooling, SumPooling, MinPooling, MaxPooling], graph_name='graph'),
            'nodes_number': Function(func=get_number_nodes, graph_name='graph'),
            'edges_number': Function(func=get_number_edges, graph_name='graph'),
            'average_node_degree': Function(func=get_average_node_degree, graph_name='graph'),
            'graph_density': Function(func=get_graph_density, graph_name='graph'),
        }
        self._action_type = {
            'node': Node(graph_name='graph')
        }

        self._node_attributes = {
            'partition': Attribute(feature_extractor=OneHotEncoding,
                                   feature_extractor_args={'set': [0, 1]},
                                   encoding=MultiClassValue(num_classes=2),
                                   embedding_size=2,
                                   graph_name='graph')
        }
        self._edge_attributes = {
            'weight': Attribute(feature_extractor=IdentityEncoding,
                                encoding=ContinuousValue(
                                    normalization="min_max",
                                    min_value=np.min(self.weights_matrix),
                                    max_value=np.max(self.weights_matrix)
                                ),
                                embedding_size=1,
                                graph_name='graph')
        }
        if self.no_action_support:
            self.discrete_actions = ["no_action"]
        else:
            self.discrete_actions = []
        self.action_space_reconstruction_each_step = True
        self.action_candidates_reconstruction = True
        self.reconstruct_only_changed_nodes = False

    def sample_valid_action(self):
        # Randomly select unassigned node and partition
        node_list = [i for i in range(self.num_nodes) if i not in self.assigned_nodes]
        if len(node_list) == 0:
            node_list = [i for i in range(self.num_nodes)]
        node = random.choice(node_list)
        return node

    def get_graphs(self):
        return {'graph': self.graph}