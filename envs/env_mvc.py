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
from utils.model import Node, Graph,  Function, Attribute
from utils.feature_extractors import OneHotEncoding
from utils.pooling_functions import MeanPooling, SumPooling, MinPooling, MaxPooling
from utils.encoding_options import MultiClassValue
from wrappers.wrapper import ContinuousEnv
from envs.env import DiscreteEnv

class MVCEnv(DiscreteEnv):
    """
    Minimum Vertex Cover Environment with node/edge feature vectors.

    Agent builds a vertex cover by selecting nodes.
    Reward is based on reduction of uncovered edges.
    """

    def __init__(self,
                 num_nodes=10,
                 edge_prob=0.3,
                 reward_type="relative",
                 max_sweeps=1000,
                 hard_constraints=False,
                 invalid_action_penalty=-1.0,
                 **kwargs):

        super().pre_init(**kwargs)
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.reward_type = reward_type
        self.max_sweeps = max_sweeps
        self.invalid_action_penalty = invalid_action_penalty
        self.hard_constraints = hard_constraints
        self.scenario_size = num_nodes
        # Placeholder for graph features (will compute in reset)
        obs_len = self.num_nodes + (self.num_nodes * (self.num_nodes - 1)) // 2  # 1 feature per node + edge features
        self.observation_space = spaces.Dict({
            "graph": spaces.Box(
                low=0,
                high=1,
                shape=(obs_len,),
                dtype=np.float32
            )
        })
        self._generate_graph()
        self.current_cover_size = 0
        self.length_best, self.length_worst = self.estimate_bounds()
        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_semantic_action_list()
        super().post_init()
        self.reset()

    def build_action_list(self):
        return list(range(self.num_nodes))

    def build_semantic_action_list(self):
        # order them by degree (highest degree first) as a simple heuristic
        degree_dict = dict(self.graph.degree())
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
        return sorted_nodes

    def _generate_graph(self):
        self.graph = nx.erdos_renyi_graph(self.num_nodes, self.edge_prob)
        # Node features: {'selected': 0/1}
        for n in self.graph.nodes:
            self.graph.nodes[n]['x_dict'] = {'selected': 0}

        # Edge features: {'covered': 0/1}
        for u, v in self.graph.edges:
            self.graph.edges[u, v]['edge_attr_dict'] = {'covered': 0}

        self.remaining_edges = {tuple(sorted(e)) for e in self.graph.edges}

    def estimate_bounds(self, max_sweeps=1000, top_k=3):
        # Simple heuristic: repeatedly select the node that covers the most uncovered edges until all edges are covered.
        edges = set(self.graph.edges)
        nodes = list(self.graph.nodes)
        best_found = self.num_nodes

        for _ in range(max_sweeps):
            uncovered = edges.copy()
            selected = set()

            while uncovered:
                # dynamic score = how many uncovered edges this node still covers
                scores = []
                for node in nodes:
                    cover_count = sum(1 for u, v in uncovered if node == u or node == v)
                    if cover_count > 0:
                        scores.append((node, cover_count))

                if not scores:
                    break

                # keep stochasticity: sample among the current top-k
                scores.sort(key=lambda x: x[1], reverse=True)
                k = min(top_k, len(scores))
                chosen_node = random.choice(scores[:k])[0]

                selected.add(chosen_node)
                uncovered = {e for e in uncovered if chosen_node not in e}

            # optional pruning: remove redundant selected nodes
            selected_list = list(selected)
            random.shuffle(selected_list)
            for node in selected_list:
                trial = selected - {node}
                if all((u in trial or v in trial) for u, v in edges):
                    selected.remove(node)

            best_found = min(best_found, len(selected))

        return best_found, self.num_nodes

    def reset(self, **kwargs):
        super().pre_reset()
        for n in self.graph.nodes:
            self.graph.nodes[n]['x_dict'] = {'selected': 0}
        for u, v in self.graph.edges:
            self.graph.edges[u, v]['edge_attr_dict'] = {'covered': 0}
        self.remaining_edges = {tuple(sorted(e)) for e in self.graph.edges}
        self.current_cover_size = 0

        self.prev_worst_case_edges = len(self.graph.edges)
        self.prev_worst_case_nodes = 0
        self.obs = self._get_obs()
        self.info = {}
        super().post_reset()
        return self.obs, self.info

    def update_config(self, config):
        self.reward_type = config.get('reward_type', self.reward_type)
        self.hard_constraints = config.get('hard_constraints', self.hard_constraints)
        # no need to rebuild action space

    def _get_obs(self):
        # Node features
        node_features = np.array([self.graph.nodes[n]['x_dict']['selected'] for n in range(self.num_nodes)], dtype=np.float32)
        # Edge features (flattened upper triangular)
        edge_features = []
        for i, j in itertools.combinations(range(self.num_nodes), 2):
            if self.graph.has_edge(i, j):
                edge_features.append(self.graph.edges[i, j]['edge_attr_dict']['covered'])
            else:
                edge_features.append(0.0)
        obs = np.concatenate([node_features, edge_features])
        return {"graph": obs.astype(np.float32)}

    def is_valid_action(self, action):
        # In MVC, valid actions are selecting any node that hasn't been selected yet.
        if action >= self.num_nodes:
            return False
        node = action
        return self.graph.nodes[node]['x_dict']['selected'] == 0

    def step(self, node):
        node = super().pre_step(node)

        if not self.invalid_action and not self.no_action:
            duplicate_selection = self.graph.nodes[node]['x_dict']['selected'] == 1

            if self.hard_constraints:
                newly_covered = 0
                if duplicate_selection:
                    self.reward = self.invalid_action_penalty
                else:
                    # Select node
                    self.graph.nodes[node]['x_dict']['selected'] = 1
                    self.current_cover_size += 1
                    # Cover edges
                    for neighbor in self.graph.neighbors(node):
                        edge = tuple(sorted((node, neighbor)))
                        if self.graph.edges[edge]['edge_attr_dict']['covered'] == 0:
                            self.graph.edges[edge]['edge_attr_dict']['covered'] = 1
                            self.remaining_edges.discard(edge)
                            newly_covered += 1
                    self.reward = newly_covered - 0.5 #(0.5 * num_nodes involved = 1)
            else:
                newly_covered = 0

                if not duplicate_selection:
                    self.graph.nodes[node]['x_dict']['selected'] = 1
                    self.current_cover_size += 1

                    for neighbor in self.graph.neighbors(node):
                        edge = tuple(sorted((node, neighbor)))
                        if self.graph.edges[edge]['edge_attr_dict']['covered'] == 0:
                            self.graph.edges[edge]['edge_attr_dict']['covered'] = 1
                            self.remaining_edges.discard(edge)
                            newly_covered += 1
                    # ---- Soft reward formulation ----
                    self.reward += (
                            1.0 * newly_covered
                            - 0.5 # small penalty for selecting a node (encourages smaller covers)
                    )
                else:
                    # small penalty for reselection
                    self.reward = self.invalid_action_penalty

        self.done = len(self.remaining_edges) == 0
        self.obs = self._get_obs()
        self.info = {}
        super().post_step()
        return self.obs, self.reward, self.done, self.truncated, self.info

    def update_metrics(self):
        super().update_metrics()
        current_nodes = self.current_cover_size
        remaining_edges = len(self.remaining_edges)
        min_nodes = self.length_best
        max_nodes = self.length_worst
        node_efficiency = (max_nodes - current_nodes) / max(1, max_nodes - min_nodes)
        node_efficiency = max(0.0, min(1.0, node_efficiency))  # Clamp to [0,1]
        edge_efficiency = (len(self.graph.edges) - remaining_edges) / max(1, len(self.graph.edges))
        # If not all edges are covered and hard constraints, performance = 0
        if self.hard_constraints and remaining_edges == 0:
            relative_performance = node_efficiency
        elif self.hard_constraints and remaining_edges > 0:
            relative_performance = 0.0
        else: # not hard constraints
            relative_performance = 0.5 * node_efficiency + 0.5 * edge_efficiency
        uncovered_ratio = remaining_edges / max(1, len(self.graph.edges))
        selected_ratio = current_nodes / max(1, len(self.graph.nodes))
        self._metrics.update({
            "relative_performance": relative_performance,
            "current_cover_size": current_nodes,
            "uncovered_edges_ratio": uncovered_ratio,
            "selected_ratio": selected_ratio,
            "cover_complete": remaining_edges == 0,
            "node_efficiency": node_efficiency,
            "edge_efficiency": edge_efficiency
        })

class ExtendedMVCEnv(MVCEnv, ContinuousEnv):
    def __init__(self, **config):
        super().__init__(**config)
        self.update_spec()

    def update_spec(self):
        # Observation types for graph-based RL
        self._observation_type = {
            'graph': Graph(poolings=[MeanPooling, SumPooling, MinPooling, MaxPooling], graph_name='graph'),
            'nodes_number': Function(func=get_number_nodes, graph_name='graph'),
            'edges_number': Function(func=get_number_edges, graph_name='graph'),
            'average_node_degree': Function(func=get_average_node_degree, graph_name='graph'),
            'graph_density': Function(func=get_graph_density, graph_name='graph'),
        }

        # Action type: selecting a node
        if self.remove_invalid_actions:
            self._action_type = {
                'node': Node(graph_name='graph', spec={'selected': False})
            }
        else:
            self._action_type = {
                'node': Node(graph_name='graph')
            }

        self._node_attributes = {
            'selected': Attribute(
                feature_extractor=OneHotEncoding,
                feature_extractor_args={'set': [0, 1]},
                encoding=MultiClassValue(num_classes=2),
                embedding_size=2,
                graph_name='graph'
            )
        }

        self._edge_attributes = {
            'covered': Attribute(
                feature_extractor=OneHotEncoding,
                feature_extractor_args={'set': [0, 1]},
                encoding=MultiClassValue(num_classes=2),
                embedding_size=2,
                graph_name='graph'
            )
        }

        if self.no_action_support:
            self.discrete_actions = ["no_action"]
        else:
            self.discrete_actions = []

        self.action_candidates_reconstruction = False
        self.reconstruct_only_changed_nodes = False
        if self.remove_invalid_actions:
            self.action_candidates_reconstruction = True
        self.action_space_reconstruction_each_step = True

    def sample_valid_action(self):
        nodes_set = [n for n in range(self.num_nodes) if self.graph.nodes[n]['x_dict']['selected'] == 0]
        return random.choice(nodes_set)

    def get_graphs(self):
        return {'graph': self.graph}