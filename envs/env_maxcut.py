#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.


import random
from gymnasium import spaces
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from utils.features_utils import get_number_nodes, get_number_edges, get_average_node_degree, get_graph_density
from utils.model import Node, Graph, Function, Attribute, Object
from utils.feature_extractors import OneHotEncoding, IdentityEncoding
from utils.pooling_functions import MeanPooling, SumPooling, MinPooling, MaxPooling
from utils.encoding_options import ContinuousValue, MultiClassValue, BinaryValue
from envs.env import DiscreteEnv
from wrappers.wrapper import ContinuousEnv
from tqdm import tqdm

class MaxCutEnv(DiscreteEnv):
    """
    MaxCut Environment.

    Agent assigns nodes to one of two partitions to maximize the total cut weight.
    """
    def __init__(self,
                 num_nodes=None,
                 max_weight=None,
                 graph=None,
                 reward_type="relative",
                 max_sweeps=1000,
                 largest_K_edges=None,
                 revisiting=False,
                 **kwargs):

        super().pre_init(**kwargs)
        # If a NetworkX graph is supplied, infer its size and edge-weight range.
        # Nodes are relabelled to contiguous integers because the environment's
        # observation/action logic indexes nodes from 0 to num_nodes - 1.
        self.source_graph = None
        if graph is not None:
            if not isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
                raise TypeError("graph must be an undirected networkx.Graph")
            if graph.number_of_nodes() == 0:
                raise ValueError("graph must contain at least one node")
            self.source_graph = graph.copy()
            self.original_node_labels = list(graph.nodes)
            relabel = {label: i for i, label in enumerate(self.original_node_labels)}
            graph = nx.relabel_nodes(graph, relabel, copy=True)
            inferred_weights = [float(data.get("weight", data.get("edge_attr_dict", {}).get("weight", 1.0)))
                                for _, _, data in graph.edges(data=True)]
            if any(not np.isfinite(w) or w < 0 for w in inferred_weights):
                raise ValueError("edge weights must be finite and non-negative")
            num_nodes = graph.number_of_nodes()
            max_weight = max(inferred_weights, default=1.0) if max_weight is None else max_weight
            self._provided_graph = graph
        else:
            num_nodes = 10 if num_nodes is None else num_nodes
            max_weight = 10 if max_weight is None else max_weight
            self.original_node_labels = list(range(num_nodes))
            self._provided_graph = None

        self.num_nodes = int(num_nodes)
        self.max_weight = float(max_weight)
        self.reward_type = reward_type
        self.largest_K_edges = largest_K_edges
        self.max_sweeps = max_sweeps
        self.scenario_size = num_nodes
        self.revisiting = revisiting
        # Observation: partition assignments + upper-triangular weights
        obs_len = self.num_nodes + (self.num_nodes * (self.num_nodes - 1)) // 2
        self.observation_space = spaces.Dict({
            "graph": spaces.Box(low=0, high=self.max_weight, shape=(obs_len,), dtype=np.float32)
        })
        self.current_cut = 0
        self.episode_history = []
        self._generate_graph()
        actual_max_weight = float(np.max(self.weights_matrix)) if self.weights_matrix.size else 0.0
        self.max_weight = max(self.max_weight, actual_max_weight)
        self.observation_space = spaces.Dict({
            "graph": spaces.Box(low=0, high=self.max_weight,
                                shape=(self.num_nodes + (self.num_nodes * (self.num_nodes - 1)) // 2,),
                                dtype=np.float32)
        })
        self.best_cut, self.worst_cut = self.estimate_bounds(self.max_sweeps)
        self.semantic_action_list = self.build_semantic_action_list()
        self.action_list = self.build_action_list()
        super().post_init()
        self.reset()

    def is_valid_action(self, action, partition=None):
        if action not in self.graph.nodes:
            return False
        if self.revisiting:
            return True
        return not self.graph.nodes[action]["x_dict"]["visited"]


    def build_action_list(self):
        if not self.revisiting:
            return [(action, partition) for action in self.graph.nodes for partition in [0, 1]]
        else:
            return [action for action in self.graph.nodes]

    def build_semantic_action_list(self):
        # return them ordered by degree, so that the agent can learn to assign high degree nodes first
        degree_dict = dict(self.graph.degree())
        if not self.revisiting:
            return sorted([(action, partition) for action in self.graph.nodes for partition in [0, 1]], key=lambda x: degree_dict[x[0]], reverse=True)
        else:
            return sorted(self.graph.nodes, key=lambda x: degree_dict[x], reverse=True)

    def export_graph(self, path):
        """
        Export the complete weighted graph as a weighted edge list.

        Format
        ------
        # Weighted MaxCut graph
        # Nodes <n>
        # Edges <m>
        u v weight
        """

        with open(path, "w") as f:

            num_edges = self.num_nodes * (self.num_nodes - 1) // 2

            f.write("# Weighted MaxCut graph\n")
            f.write(f"# Nodes {self.num_nodes}\n")
            f.write(f"# Edges {num_edges}\n")

            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    weight = self.weights_matrix[i, j]
                    f.write(f"{i} {j} {weight}\n")

    def _generate_graph(self):
        """Create a random weighted graph, or load the supplied NetworkX graph."""
        self.graph = nx.Graph()
        self.weights_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)

        for i in range(self.num_nodes):
            self.graph.add_node(i, x_dict={'partition': 0, 'visited': False})

        if self._provided_graph is None:
            random_weights = np.random.randint(1, int(self.max_weight) + 1,
                                               size=(self.num_nodes, self.num_nodes))
            self.weights_matrix = np.triu(random_weights, 1).astype(np.float64)
            self.weights_matrix += self.weights_matrix.T
            edge_iter = ((i, j, float(self.weights_matrix[i, j]))
                         for i, j in itertools.combinations(range(self.num_nodes), 2))
        else:
            for u, v, data in self._provided_graph.edges(data=True):
                weight = float(data.get('weight', data.get('edge_attr_dict', {}).get('weight', 1.0)))
                self.weights_matrix[u, v] = weight
                self.weights_matrix[v, u] = weight
            edge_iter = ((u, v, float(self.weights_matrix[u, v]))
                         for u, v in self._provided_graph.edges())

        for i, j, weight in edge_iter:
            self.graph.add_edge(i, j, edge_attr_dict={'weight': weight}, weight=weight)

        # Optional sparsification is applied consistently to both graph and matrix.
        if self.largest_K_edges is not None and self._provided_graph is None:
            keep = set()
            for i in range(self.num_nodes):
                neighbors = sorted((j for j in range(self.num_nodes) if j != i),
                                   key=lambda j: self.weights_matrix[i, j], reverse=True)
                keep.update(tuple(sorted((i, j))) for j in neighbors[:self.largest_K_edges])
            for u, v in list(self.graph.edges):
                if tuple(sorted((u, v))) not in keep:
                    self.graph.remove_edge(u, v)

    def estimate_bounds(self, samples=1000, local_steps=None):
        # estimate best and worst cut values by sampling random partitions and doing local improvements
        if local_steps is None:
            local_steps = self.num_nodes

        best_cut = -np.inf

        for _ in range(samples):
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
        if self.revisiting:
            return self.reset_revisiting(**kwargs)
        else:
            return self.reset_non_revisiting(**kwargs)

    def reset_non_revisiting(self, **kwargs):
        super().pre_reset()

        # Original initial solution:
        # all nodes start in partition 0.
        for i in range(self.num_nodes):
            self.graph.nodes[i]["x_dict"]["partition"] = 0
            self.graph.nodes[i]["x_dict"]["visited"] = False

        self.assigned_nodes = set()

        self.current_cut = 0
        self.prev_cut = 0
        self.episode_history = [{"step": 0, "cut": float(self.current_cut),
                                 "reward": 0.0, "cumulative_reward": 0.0}]

        self.obs = self._get_obs()
        self.info = {}

        super().post_reset()

        return self.obs, {}

    def reset_revisiting(self, **kwargs):
        super().pre_reset()

        for i in range(self.num_nodes):
            self.graph.nodes[i]["x_dict"]["partition"] = 0
            self.graph.nodes[i]["x_dict"]["visited"] = False

        self.assigned_nodes = set()

        # Calculate the cut of the random initial solution
        partition = np.array([
            self.graph.nodes[i]["x_dict"]["partition"]
            for i in range(self.num_nodes)
        ])

        self.current_cut = self.compute_cut(partition)
        self.prev_cut = self.current_cut


        # Best solution found during this episode
        self.best_episode_cut = self.current_cut
        self.best_episode_partition = partition.copy()
        self.episode_history = [{"step": 0, "cut": float(self.current_cut),
                                 "reward": 0.0, "cumulative_reward": 0.0}]

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

    def step(self, node, partition=None):
        if self.revisiting:
            return self.step_revisiting(node)
        else:
            return self.step_non_revisiting(node, partition)

    def step_non_revisiting(self, node, partition):
        node, partition = super().pre_step((node, partition))

        self.reward = 0.0
        self.info = {}

        if self.invalid_action or self.no_action:
            self.obs = self._get_obs()
            super().post_step()
            return (
                self.obs,
                self.reward,
                self.done,
                self.truncated,
                self.info,
            )

        # Original transition logic
        new_part = partition
        old_part = self.graph.nodes[node]["x_dict"]["partition"]
        if new_part == old_part:
            # No change in partition, no reward
            self.assigned_nodes.add(node)
            self.graph.nodes[node]["x_dict"]["visited"] = True
            all_nodes_assigned = len(self.assigned_nodes) == self.num_nodes

            self.done = all_nodes_assigned
            if self.terminate_at_approximate_best:
                self.done |= (
                        self.current_cut >= self.best_cut
                )
            self.obs = self._get_obs()
            super().post_step()
            return (
                self.obs,
                self.reward,
                self.done,
                self.truncated,
                self.info,
            )

        delta = self._compute_flip_gain(node, old_part)

        self.graph.nodes[node]["x_dict"]["partition"] = new_part
        self.graph.nodes[node]["x_dict"]["visited"] = True

        # Non-revisiting: node is now consumed
        self.assigned_nodes.add(node)

        self.prev_cut = self.current_cut
        self.current_cut += delta

        # Original reward logic
        if self.reward_type == "relative":
            denom = max(self.best_cut - self.worst_cut, 1e-8)
            self.reward = delta / denom
        else:
            self.reward = delta

        # Non-revisiting termination:
        # all nodes have been assigned/flipped once.
        self.done = (
                len(self.assigned_nodes) >= self.num_nodes
        )

        if self.terminate_at_approximate_best:
            self.done |= (
                    self.current_cut >= self.best_cut
            )



        self.obs = self._get_obs()
        self._record_episode_step()

        super().post_step()

        return (
            self.obs,
            self.reward,
            self.done,
            self.truncated,
            self.info,
        )

    def step_revisiting(self, node):
        node = super().pre_step(node)

        self.reward = 0.0
        self.info = {}

        if self.invalid_action or self.no_action:
            self.obs = self._get_obs()
            super().post_step()
            return (
                self.obs,
                self.reward,
                self.done,
                self.truncated,
                self.info,
            )

        # Same transition as original MaxCut
        old_part = self.graph.nodes[node]["x_dict"]["partition"]
        new_part = 1 - old_part

        delta = self._compute_flip_gain(node, old_part)

        self.graph.nodes[node]["x_dict"]["partition"] = new_part
        self.graph.nodes[node]["x_dict"]["visited"] = True

        self.prev_cut = self.current_cut
        self.current_cut += delta

        # Same local transition reward
        if self.reward_type == "relative":
            denom = max(self.best_cut - self.worst_cut, 1e-8)
            self.reward = delta / denom
        else:
            self.reward = delta

        # Track best solution separately
        if self.current_cut > self.best_episode_cut:
            self.best_episode_cut = self.current_cut

            self.best_episode_partition = np.array([
                self.graph.nodes[i]["x_dict"]["partition"]
                for i in range(self.num_nodes)
            ])

        # Revisiting termination = fixed search horizon
        if self.terminate_at_approximate_best:
            self.done |= (
                    self.current_cut >= self.best_cut
            )

        self.obs = self._get_obs()
        self._record_episode_step()

        super().post_step()

        return (
            self.obs,
            self.reward,
            self.done,
            self.truncated,
            self.info,
        )

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

    def _record_episode_step(self):
        # Record per-step values for render() and downstream analysis.
        previous_total = self.episode_history[-1]["cumulative_reward"] if self.episode_history else 0.0
        self.episode_history.append({
            "step": len(self.episode_history),
            "cut": float(self.current_cut),
            "reward": float(getattr(self, "reward", 0.0)),
            "cumulative_reward": float(previous_total + getattr(self, "reward", 0.0)),
        })

    def render(self, mode="human", show=True, save_path=None):
        # Visualize the current graph, cut evolution, and reward history
        if mode not in ("human", "rgb_array"):
            raise ValueError("mode must be 'human' or 'rgb_array'")

        # --------------------------------------------------
        # Create one figure with three horizontal panels
        # --------------------------------------------------
        fig, (ax_graph, ax_cut, ax_reward) = plt.subplots(
            1,
            3,
            figsize=(15, 4),
            gridspec_kw={"width_ratios": [1.2, 1, 1]},
        )

        # --------------------------------------------------
        # 1. Current graph and partition
        # --------------------------------------------------
        positions = nx.spring_layout(self.graph, seed=7)

        partitions = {
            node: self.graph.nodes[node].get("x_dict", {}).get("partition", 0)
            for node in self.graph.nodes
        }

        node_colors = [
            "#4C78A8" if partitions[node] == 0 else "#F58518"
            for node in self.graph.nodes
        ]

        cut_edges = [
            (u, v)
            for u, v in self.graph.edges
            if partitions[u] != partitions[v]
        ]

        other_edges = [
            (u, v)
            for u, v in self.graph.edges
            if partitions[u] == partitions[v]
        ]

        nx.draw_networkx_edges(
            self.graph,
            positions,
            edgelist=other_edges,
            edge_color="#CCCCCC",
            width=1,
            ax=ax_graph,
        )

        nx.draw_networkx_edges(
            self.graph,
            positions,
            edgelist=cut_edges,
            edge_color="#D62728",
            width=2,
            ax=ax_graph,
        )

        nx.draw_networkx_nodes(
            self.graph,
            positions,
            node_color=node_colors,
            node_size=350,
            ax=ax_graph,
        )

        nx.draw_networkx_labels(
            self.graph,
            positions,
            font_size=8,
            ax=ax_graph,
        )

        ax_graph.set_title(
            f"Graph | Cut: {getattr(self, 'current_cut', 0):.3g}",
            fontsize=11,
        )
        ax_graph.axis("off")

        # --------------------------------------------------
        # Read episode history safely
        # --------------------------------------------------
        history = getattr(self, "episode_history", [])

        steps = [entry["step"] for entry in history]
        cuts = [entry["cut"] for entry in history]
        rewards = [entry["reward"] for entry in history]

        # Use stored cumulative rewards when available;
        # otherwise calculate them from step rewards.
        if history and all("cumulative_reward" in entry for entry in history):
            cumulative = [entry["cumulative_reward"] for entry in history]
        else:
            cumulative = np.cumsum(rewards).tolist()

        # --------------------------------------------------
        # 2. Cut value over the episode
        # --------------------------------------------------
        if steps:
            ax_cut.plot(
                steps,
                cuts,
                marker="o",
                markersize=3,
                linewidth=1.6,
            )

        ax_cut.set_title("Cut value", fontsize=11)
        ax_cut.set_xlabel("Step")
        ax_cut.set_ylabel("Cut")
        ax_cut.grid(True, alpha=0.25)

        # --------------------------------------------------
        # 3. Step and cumulative reward
        # --------------------------------------------------
        if steps:
            ax_reward.plot(
                steps,
                rewards,
                label="Step reward",
                linewidth=1.2,
                alpha=0.8,
            )

            ax_reward.plot(
                steps,
                cumulative,
                label="Cumulative reward",
                linewidth=1.6,
            )

        ax_reward.set_title("Reward history", fontsize=11)
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        ax_reward.grid(True, alpha=0.25)
        ax_reward.legend(fontsize=8, loc="best")

        # --------------------------------------------------
        # Layout and optional saving
        # --------------------------------------------------
        fig.tight_layout(pad=1.2)

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save all three panels together in one image.
            fig.savefig(
                path,
                dpi=150,
                bbox_inches="tight",
            )

        # --------------------------------------------------
        # Return RGB image or display the figure
        # --------------------------------------------------
        if mode == "rgb_array":
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            plt.close(fig)
            return image

        if show:
            plt.show(block=False)
            plt.pause(0.001)

        return fig

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
        self.revisiting = config.get('revisiting', self.revisiting if hasattr(self, 'revisiting') else False)
        if self.largest_K_edges != old_largest_K_edges:
            # need to regenerate the graph with the new sparsity pattern
            # clear all edges
            self.graph.remove_edges_from(list(self.graph.edges))
            for i in range(self.num_nodes):
                topk = np.argsort(self.weights_matrix[i])[::-1]
                topk = [j for j in topk if j != i][:self.largest_K_edges]

                for j in topk:
                    if not self.graph.has_edge(i, j):
                        self.graph.add_edge(
                            i, j,
                            edge_attr_dict={'weight': self.weights_matrix[i, j]}
                        )
        for node in self.graph.nodes:
            self.graph.nodes[node]["x_dict"]["visited"] = False
        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_semantic_action_list()
        self.reset()

    def sample_valid_action(self):
        if self.revisiting:
            node_list = list(self.graph.nodes)
        else:
            node_list = [
                i for i in self.graph.nodes
                if not self.graph.nodes[i]["x_dict"]["visited"]
            ]

        if not node_list:
            return None

        return random.choice(node_list)

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
        if self.revisiting:
            self._action_type = {
                'node': Node(graph_name='graph')
            }
        else:
            self.partitions = [0, 1]
            self._action_type = {
                'node': Node(graph_name='graph', spec={"visited": False}),
                'partition': Object(name='partition', set=self.partitions,
                                       feature_extractor=OneHotEncoding,
                                       feature_extractor_args={'set': self.partitions},
                                       embedding_size=len(self.partitions))
            }
        self._node_attributes = {
            'partition': Attribute(feature_extractor=OneHotEncoding,
                                   feature_extractor_args={'set': [0, 1]},
                                   encoding=MultiClassValue(num_classes=2),
                                   embedding_size=2,
                                   graph_name='graph'),
            'visited': Attribute(encoding=BinaryValue(),
                                embedding_size=1,
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



    def get_graphs(self):
        return {'graph': self.graph}