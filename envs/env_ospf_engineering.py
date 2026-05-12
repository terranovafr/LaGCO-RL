#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

import copy
from gymnasium import spaces
import itertools
import networkx as nx
from utils.features_utils import get_number_nodes, get_number_edges, get_average_node_degree, get_graph_density
from utils.model import Graph, Edge, Object, Function, Attribute, ActionSpace
from utils.feature_extractors import OneHotEncoding, IdentityEncoding
from utils.pooling_functions import MeanPooling, SumPooling, MinPooling, MaxPooling
from utils.encoding_options import ContinuousValue
from utils.log_utils import graph_nodes_to_text, graph_edges_to_text
from envs.env import DiscreteEnv
from wrappers.wrapper import ContinuousEnv
import random
import numpy as np
from tqdm import tqdm

class OSPFTrafficEngineeringEnv(DiscreteEnv):
    """
    OSPF Engineering Environment with weight adjustments as actions and shortest path routing (with optional ECMP) for reward computation.
    The agent's goal is to adjust link weights to optimize the maximum link utilization while ensuring feasibility of traffic demands.
    """

    def __init__(self,
                 num_nodes=5,
                 min_capacity=10,
                 max_capacity=10,
                 max_traffic=10,
                 max_weight=4,
                 min_weight=1,
                 incremental_weight_by_unit=True,
                 non_zero_traffic_ratio=0.3,
                 graph_edges_distribution='small_world',
                 small_world_p=0.1,
                 small_world_k=4,
                 communication_edge_ratio=0.3,
                 feasibility_coefficient=0.5,
                 link_util_coefficient=0.5,
                 util_aggregation='mean',
                 routing_ecmp=True,
                 no_change_action_sum=-5,
                 reward_type='relative',
                 bounded_weights=False,
                 always_feasible=False,
                 max_sweeps=1000,
                 terminate_at_approximate_best=False,
                 **kwargs):
        super().pre_init(**kwargs)
        self.num_nodes = num_nodes
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.always_feasible = always_feasible
        self.max_traffic = max_traffic
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.routing_ecmp = routing_ecmp
        self.non_zero_traffic_ratio = non_zero_traffic_ratio
        self.incremental_weight_by_unit = incremental_weight_by_unit
        self.communication_edge_ratio = communication_edge_ratio
        self.feasibility_coefficient = feasibility_coefficient
        self.link_util_coefficient = link_util_coefficient
        self.util_aggregation = util_aggregation
        self.small_world_p = small_world_p
        self.small_world_k = small_world_k
        self.graph_edges_distribution = graph_edges_distribution
        self.no_change_action_sum = no_change_action_sum
        self.reward_type = reward_type
        self.bounded_weights = bounded_weights
        self.max_sweeps = max_sweeps
        self.terminate_at_approximate_best = terminate_at_approximate_best

        if self.verbose > 0:
            print("Initializing OSPF Engineering Environment with parameters:")
            print(f" Number of nodes: {self.num_nodes}")
            print(f" Max capacity: {self.max_capacity}, Weight range: [{self.min_weight}, {self.max_weight}]")
            print(f" Incremental weight by unit: {self.incremental_weight_by_unit}")
            print(f" Max steps coefficient: {self.max_steps_coefficient}")

        self.decrease_actions = 0
        self.increase_actions = 0
        self.no_variation_actions = 0
        self.real_increase_actions = 0
        self.real_decrease_actions = 0
        self.real_no_variation_actions = 0

        self.targeted_edges = set()
        self.targeted_positive_variations = set()
        self.targeted_negative_variations = set()

        if self.graph_edges_distribution == 'spanning_tree':
            base_graph = nx.generators.random_tree(self.num_nodes)
        else:
            base_graph = nx.watts_strogatz_graph(
                self.num_nodes,
                self.small_world_k,
                self.small_world_p
            )

        self.communication_graph = nx.DiGraph()
        self.communication_graph.add_nodes_from(base_graph.nodes())

        self.communication_links = {self._canon_link(u, v) for u, v in base_graph.edges()}

        # Optionally randomly add extra edges with probability p
        # Number of undirected edges desired in total
        E_max = self.num_nodes * (self.num_nodes - 1) // 2
        E_target = int(self.communication_edge_ratio * E_max)

        # Undirected edges already present in base_graph
        all_possible_links = set(itertools.combinations(range(self.num_nodes), 2))
        candidate_links = list(all_possible_links - self.communication_links)

        n_extra = max(0, E_target - len(self.communication_links))
        new_links = random.sample(candidate_links, min(n_extra, len(candidate_links)))
        self.communication_links.update(new_links)

        # Add every physical link symmetrically to the directed graph
        for u, v in self.communication_links:
            self._add_symmetric_link(u, v)

        self.communication_links = sorted(self.communication_links)
        self.communication_edges = list(self.communication_graph.edges)
        self.scenario_size = len(self.communication_links)

        # Initialize capacities and traffic requirements
        success = False
        while not success:
            self.traffic_graph = nx.complete_graph(self.num_nodes, create_using=nx.DiGraph)
            self.traffic_edges = list(self.traffic_graph.edges)

            self.traffic_sum = 0
            candidate_edges = list(self.traffic_graph.edges)
            k = int(self.non_zero_traffic_ratio * len(candidate_edges))
            selected_edges = set(random.sample(candidate_edges, k))
            for u, v in list(self.traffic_edges):
                if (u, v) not in selected_edges:
                    if self.traffic_graph.has_edge(u, v):
                        self.traffic_graph.remove_edge(u, v)
                    self.traffic_edges.remove((u, v))
                else:
                    traffic = np.random.randint(1, self.max_traffic + 1)
                    self.traffic_graph[u][v]['edge_attr_dict'] = {
                        'traffic': traffic
                    }
                    self.traffic_sum += traffic

            for u, v in self.communication_links:
                if 'edge_attr_dict' not in self.communication_graph[u][v]:
                    self.communication_graph[u][v]['edge_attr_dict'] = {}
                if 'edge_attr_dict' not in self.communication_graph[v][u]:
                    self.communication_graph[v][u]['edge_attr_dict'] = {}

                if self.always_feasible:
                    # set capacity to at least total traffic / number of edges to ensure feasibility
                    sampled_capacity = np.random.randint(self.traffic_sum, self.traffic_sum+1)
                else:
                    sampled_capacity = np.random.randint(self.min_capacity, self.max_capacity + 1)

                sampled_weight = np.random.randint(self.min_weight, self.max_weight + 1)
                self._set_link_attr(u, v, 'capacity', sampled_capacity)
                self._set_link_attr(u, v, 'used_capacity', 0)
                self._set_link_attr(u, v, 'weight', sampled_weight)

            self.traffic_edges = list(self.traffic_graph.edges)
            success = self._initialize_empirical_worst_allocation(max_sweeps=self.max_sweeps)
            if not success:
                raise ValueError("Failed to find distinct best and worst configurations. Consider adjusting parameters or increasing max_sweeps.")
        for node in self.communication_graph.nodes:
            self.communication_graph.nodes[node]['x_dict'] = {
                "outgoing_traffic": sum(self.traffic_graph[node][nbr]['edge_attr_dict']['traffic'] for nbr in self.traffic_graph.successors(node)),
                "incoming_traffic": sum(self.traffic_graph[nbr][node]['edge_attr_dict']['traffic'] for nbr in self.traffic_graph.predecessors(node)),
            }
        for node in self.traffic_graph.nodes:
            self.traffic_graph.nodes[node]['x_dict'] = {
                "outgoing_traffic": sum(self.traffic_graph[node][nbr]['edge_attr_dict']['traffic'] for nbr in self.traffic_graph.successors(node)),
                "incoming_traffic": sum(self.traffic_graph[nbr][node]['edge_attr_dict']['traffic'] for nbr in self.traffic_graph.predecessors(node)),
            }

        self.initial_communication_graph = copy.deepcopy(self.communication_graph)
        self.initial_traffic_graph = copy.deepcopy(self.traffic_graph)

        self.num_communication_edges = len(self.communication_links)
        self.num_traffic_edges = self.traffic_graph.number_of_edges()
        self.min_num_communication_edges = self.num_nodes - 1  # spanning tree
        self.max_num_communication_edges = self.num_nodes * (self.num_nodes - 1)  # fully connected directed graph
        self.min_num_traffic_edges = 1
        self.max_num_traffic_edges = self.num_nodes * (self.num_nodes - 1)  # fully connected directed graph

        if self.routing_ecmp:
            self.initial_feasible, link_utils, self.initial_feasible_percentage = self._find_feasible_shortest_paths_ECMP()
        else:
            self.initial_feasible, link_utils, _, self.initial_feasible_percentage = self._find_feasible_shortest_paths()
        self.initial_util = max(link_utils.values()) if link_utils else 0.0
        self.current_util = self.initial_util
        self.current_feasible = 1 if self.initial_feasible else 0
        self.current_feasible_percentage = self.initial_feasible_percentage

        # Observation: concatenated edge features [capacity, weight] + traffic matrix
        obs_len = len(self.communication_links) * 3 + len(self.traffic_edges)  # capacity, weight for each communication edge + traffic for each traffic edge
        self.observation_space = spaces.Dict({
            "graph": spaces.Box(
                low=0,
                high=self.max_capacity,
                shape=(obs_len,),
                dtype=np.float32
            )
        })

        # Action: choose edge + variation coefficient
        if self.incremental_weight_by_unit:
            self.variations = [-1, 0, +1]
        else:
            max_negative = -(self.max_weight - self.min_weight)
            max_positive = (self.max_weight - self.min_weight)
            self.variations = list(range(max_negative, max_positive + 1))

        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_semantic_action_list()
        self.post_init()

    def _canon_link(self, u, v):
        return (u, v) if u < v else (v, u)

    def _add_symmetric_link(self, u, v):
        self.communication_graph.add_edge(u, v)
        self.communication_graph.add_edge(v, u)

    def _set_link_attr(self, u, v, key, value):
        self.communication_graph[u][v]['edge_attr_dict'][key] = value
        self.communication_graph[v][u]['edge_attr_dict'][key] = value

    def _get_link_weight(self, u, v):
        return self.communication_graph[u][v]['edge_attr_dict']['weight']

    def build_action_list(self):
        action_list = []
        for edge in self.communication_links:
            for var in self.variations:
                action_list.append((edge, var))
        return action_list

    def _initialize_empirical_worst_allocation(self, sweep_values=None, max_sweeps=1000):
        # Heuristic based search using:
        # - duplicate-configuration avoidance
        # - stochastic exploration
        # - randomized local search around elite configs
        if sweep_values is None:
            sweep_values = list(range(self.min_weight, self.max_weight + 1))

        self.best_util = np.inf
        self.worst_util = -1.0
        initial_config = {}

        # Fixed edge order ONLY for encoding/applying configs
        edge_list = list(self.communication_links)

        seen_configs = set()
        max_unique = len(sweep_values) ** len(edge_list) if len(edge_list) < 20 else np.inf

        sweeps_done = 0
        attempts = 0
        max_attempts = max_sweeps * 10  # safety guard

        # Small elite pools for intensification without becoming deterministic
        elite_best = []
        elite_worst = []
        elite_pool_size = 8

        def encode_config(config):
            return tuple(config[(u, v)] for (u, v) in edge_list)

        def apply_config(config):
            for (u, v), w in config.items():
                self._set_link_attr(u, v, 'weight', w)

        def aggregate_util(link_utils):
            if not link_utils:
                return 0.0
            if self.util_aggregation == 'max':
                return max(link_utils.values())
            elif self.util_aggregation == 'mean':
                return float(np.mean(list(link_utils.values())))
            else:
                raise ValueError(f"Unknown util_aggregation: {self.util_aggregation}")

        def evaluate_current_graph():
            if self.routing_ecmp:
                feasible, link_utils, feasible_fraction = self._find_feasible_shortest_paths_ECMP()
            else:
                feasible, link_utils, _, feasible_fraction = self._find_feasible_shortest_paths()

            util = aggregate_util(link_utils)
            return feasible, feasible_fraction, util

        def evaluate_config(config):
            apply_config(config)
            return evaluate_current_graph()

        def random_config():
            return {
                (u, v): int(np.random.choice(sweep_values))
                for (u, v) in edge_list
            }

        def perturb_config(base_config):
            """
            Randomized perturbation:
            - changes a random subset of edges
            - sometimes small local changes, sometimes full random reset on chosen edges
            - no deterministic edge ordering for search decisions
            """
            cand = base_config.copy()

            if len(edge_list) == 0:
                return cand

            # Mostly small perturbations, occasionally larger jumps
            if random.random() < 0.75:
                num_changes = random.randint(1, min(3, len(edge_list)))
            else:
                num_changes = random.randint(1, min(max(1, len(edge_list) // 3), len(edge_list)))

            chosen_edges = random.sample(edge_list, num_changes)

            for e in chosen_edges:
                current_w = cand[e]
                possible = [w for w in sweep_values if w != current_w]
                if not possible:
                    continue

                # Keep stochasticity; bias toward nearby values sometimes for local search
                if random.random() < 0.7 and len(sweep_values) > 2:
                    sorted_vals = sorted(sweep_values)
                    idx = sorted_vals.index(current_w)
                    neighborhood = []
                    if idx - 1 >= 0:
                        neighborhood.append(sorted_vals[idx - 1])
                    if idx + 1 < len(sorted_vals):
                        neighborhood.append(sorted_vals[idx + 1])

                    if neighborhood and random.random() < 0.8:
                        cand[e] = random.choice(neighborhood)
                    else:
                        cand[e] = random.choice(possible)
                else:
                    cand[e] = random.choice(possible)

            return cand

        def push_elite(pool, config, util, reverse=False):
            """
            reverse=False  -> pool stores lowest util first (best configs)
            reverse=True   -> pool stores highest util first (worst configs)
            """
            pool.append((util, config.copy()))
            pool.sort(key=lambda x: x[0], reverse=reverse)
            if len(pool) > elite_pool_size:
                pool.pop()

        with tqdm(total=max_sweeps, desc="Sweeps") as pbar:
            while sweeps_done < max_sweeps and attempts < max_attempts:
                attempts += 1

                # -------------------------
                # Build starting config
                # -------------------------
                direction = "best" if (sweeps_done % 2 == 0) else "worst"

                use_elite_restart = (
                        random.random() < 0.35 and
                        ((direction == "best" and elite_best) or (direction == "worst" and elite_worst))
                )

                if use_elite_restart:
                    if direction == "best":
                        seed_config = random.choice(elite_best)[1].copy()
                    else:
                        seed_config = random.choice(elite_worst)[1].copy()

                    # Perturb so sweeps do not collapse to identical trajectories
                    candidate_config = perturb_config(seed_config)
                else:
                    candidate_config = random_config()

                candidate_weights = encode_config(candidate_config)

                # Skip exact duplicate starts; try again
                if candidate_weights in seen_configs:
                    continue

                # -------------------------
                # Evaluate initial candidate
                # -------------------------
                seen_configs.add(candidate_weights)
                sweeps_done += 1

                _,_, max_util = evaluate_config(candidate_config)


                if max_util < self.best_util:
                    self.best_util = max_util
                    push_elite(elite_best, candidate_config, max_util, reverse=False)
                else:
                    # still allow near-best configs into elite occasionally
                    if len(elite_best) < elite_pool_size or random.random() < 0.15:
                        push_elite(elite_best, candidate_config, max_util, reverse=False)

                if max_util > self.worst_util:
                    self.worst_util = max_util
                    initial_config = candidate_config.copy()
                    push_elite(elite_worst, candidate_config, max_util, reverse=True)
                else:
                    if len(elite_worst) < elite_pool_size or random.random() < 0.15:
                        push_elite(elite_worst, candidate_config, max_util, reverse=True)

                # -------------------------
                # Randomized local search around this configuration
                # Budget kept small so "same max_sweeps" stays reasonable
                # -------------------------
                local_trials = 3
                current_config = candidate_config

                for _ in range(local_trials):
                    neighbors = []

                    # generate a few stochastic neighbors
                    for _ in range(4):
                        neigh = perturb_config(current_config)
                        neigh_weights = encode_config(neigh)

                        if neigh_weights in seen_configs:
                            continue

                        seen_configs.add(neigh_weights)
                        _, _, neigh_util = evaluate_config(neigh)
                        neighbors.append((neigh, neigh_util))

                        if neigh_util < self.best_util:
                            self.best_util = neigh_util
                            push_elite(elite_best, neigh, neigh_util, reverse=False)

                        if neigh_util > self.worst_util:
                            self.worst_util = neigh_util
                            initial_config = neigh.copy()
                            push_elite(elite_worst, neigh, neigh_util, reverse=True)

                        if len(seen_configs) >= max_unique:
                            break

                    if not neighbors or len(seen_configs) >= max_unique:
                        break

                    # Stochastic next-state choice:
                    # - best sweeps favor lower util neighbors
                    # - worst sweeps favor higher util neighbors
                    utils = np.array([u for _, u in neighbors], dtype=float)

                    if direction == "best":
                        weights = utils.max() - utils + 1e-8
                    else:
                        weights = utils - utils.min() + 1e-8

                    next_idx = random.choices(range(len(neighbors)), weights=weights, k=1)[0]
                    current_config, current_util = neighbors[next_idx]

                # Early stop if all combinations explored
                if len(seen_configs) >= max_unique:
                    pbar.update(1)
                    break

                pbar.update(1)

        # Apply worst-case configuration
        for (u, v), w in initial_config.items():
            self._set_link_attr(u, v, 'weight', w)

        if self.verbose > 0:
            print(f"Unique sweeps evaluated: {len(seen_configs)}")
            print(f"Initialized empirical worst-case allocation (max utilization={self.worst_util:.4f})")
            print(f"Found empirical best-case allocation (min utilization={self.best_util:.4f})")

        return self.best_util != self.worst_util

    def reset(self, **kwargs):
        # Reset weights and data structures, but keep the same graph and traffic pattern for consistency across episodes
        super().pre_reset()
        self.communication_graph = copy.deepcopy(self.initial_communication_graph)
        if self.routing_ecmp:
            feasible, link_utils, self.initial_feasible_percentage = self._find_feasible_shortest_paths_ECMP()
        else:
            feasible, link_utils, _, self.initial_feasible_percentage = self._find_feasible_shortest_paths()

        if self.util_aggregation == 'max':
            self.initial_util = max(link_utils.values()) if link_utils else 0.0
        elif self.util_aggregation == 'mean':
            self.initial_util = np.mean(list(link_utils.values())) if link_utils else 0.0

        self.prev_feasible_percentage = self.initial_feasible_percentage
        self.prev_util = self.initial_util
        self.initial_feasible = 1 if feasible else 0

        self.current_util = self.initial_util
        self.increase_actions = 0
        self.decrease_actions = 0
        self.real_increase_actions = 0
        self.real_decrease_actions = 0
        self.no_variation_actions = 0
        self.real_no_variation_actions = 0
        self.real_no_change_actions = 0
        self.targeted_edges = set()
        self.targeted_positive_variations = set()
        self.targeted_negative_variations = set()
        self.obs = self._get_obs()
        self.info = {}
        super().post_reset()
        return self.obs, self.info

    def _get_obs(self):
        # Get edge features: [capacity, weight, used_capacity] for communication edges + [traffic] for traffic edges
        edge_features = []
        for u, v in self.communication_links:
            edge_features.extend([self.communication_graph[u][v]['edge_attr_dict']['capacity'], self.communication_graph[u][v]['edge_attr_dict']['weight'], self.communication_graph[u][v]['edge_attr_dict']['used_capacity']])
        for u, v in self.traffic_edges:
            edge_features.append(self.traffic_graph[u][v]['edge_attr_dict']['traffic'])
        return {"graph": np.array(edge_features).astype(np.float32)}

    def update_config(self, config):
        self.feasibility_coefficient = config.get('feasibility_coefficient', self.feasibility_coefficient)
        self.link_util_coefficient = config.get('link_util_coefficient', self.link_util_coefficient)
        self.util_aggregation = config.get('util_aggregation', self.util_aggregation)
        self.routing_ecmp = config.get('routing_ecmp', self.routing_ecmp)
        self.incremental_weight_by_unit = config.get('incremental_weight_by_unit', self.incremental_weight_by_unit)
        self.no_change_action_sum = config.get('no_change_action_sum', self.no_change_action_sum)
        self.reward_type = config.get('reward_type', self.reward_type)
        self.bounded_weights = config.get('bounded_weights', self.bounded_weights)
        if self.incremental_weight_by_unit:
            # add variations by unit as -half range max -min, 0, +half range max -min
            self.variations = [-1, 0, 1]
        else:
            max_negative = -(self.max_weight - self.min_weight)
            max_positive = (self.max_weight - self.min_weight)
            self.variations = list(range(max_negative, max_positive + 1))
        # need to rebuild action list if variations changed
        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_semantic_action_list()


    def build_semantic_action_list(self):
        # heuristic: sort edges by current utilization (descending) and variations by absolute value (descending) to prioritize impactful actions
        sorted_edges = sorted(
            self.communication_links,
            key=lambda e: max(
                self.communication_graph[e[0]][e[1]]['edge_attr_dict']['used_capacity'],
                self.communication_graph[e[1]][e[0]]['edge_attr_dict']['used_capacity'],
            ),
            reverse=True
        )
        # Sort variations (descending)
        sorted_variations = sorted(
            self.variations,
            reverse=True
        )
        mapping = []
        # Semantic cartesian product
        for edge in sorted_edges:
            for var in sorted_variations:
                mapping.append((edge, var))
        return mapping

    def save_step_log(self):
        super().save_step_log()
        self.step_logs[-1].update({
            'observation': self.obs,
            'communication_graph_nodes':  graph_nodes_to_text(self.communication_graph),
            'communication_graph_edges': graph_edges_to_text(self.communication_graph),
            'traffic_graph_nodes':  graph_nodes_to_text(self.traffic_graph),
            'traffic_graph_edges': graph_edges_to_text(self.traffic_graph),
            'action': "Edge: " + str(self.action[0]) + ", Variation: " + str(self.action[1]),
            'most_congested_edge': max(
                self.communication_graph.edges,
                key=lambda e: self.communication_graph[e[0]][e[1]]['edge_attr_dict']['used_capacity'] / self.communication_graph[e[0]][e[1]]['edge_attr_dict']['capacity']
            ),
            'max_utilization': self.current_util,
            'edges_utilization': {
                (u, v): self.communication_graph[u][v]['edge_attr_dict']['used_capacity'] / self.communication_graph[u][v]['edge_attr_dict']['capacity']
                for u, v in self.communication_edges
            },
        })

    def step(self, edge, variation=None):
        # Step action: adjust weight of chosen edge by variation, then recompute shortest paths and link utilizations to calculate reward
        action = super().pre_step((edge, variation))
        if not self.no_action and not self.invalid_action:
            edge, variation = action if action is not None else (edge, variation)

            if isinstance(edge, str):
                u, v = int(edge.split("_")[1]), int(edge.split("_")[2])
            else:
                u, v = edge[0], edge[1]

            # canonicalize edge representation for symmetric
            u, v = self._canon_link(u, v)

            if variation > 0:
                self.increase_actions += 1
            elif variation < 0:
                self.decrease_actions += 1
            else:
                 self.no_variation_actions += 1

            if variation < 0:
                self.targeted_negative_variations.add(variation)
            elif variation > 0:
                self.targeted_positive_variations.add(variation)
            if (u,v) not in self.targeted_edges:
                self.targeted_edges.add((u,v))

            old_weight = self.communication_graph[u][v]['edge_attr_dict']['weight']

            # keep weights within bounds if specified, otherwise just ensure they don't go negative
            if self.bounded_weights:
                if old_weight + variation < self.min_weight:
                    variation = self.min_weight - old_weight
                elif old_weight + variation > self.max_weight:
                    variation = self.max_weight - old_weight
            else:
                if old_weight + variation < 0:
                    variation = -old_weight

            if variation > 0:
                self.real_increase_actions += 1
            elif variation < 0:
                self.real_decrease_actions += 1

            self._set_link_attr(u, v, 'weight', old_weight + variation)

            assert self.communication_graph[u][v]['edge_attr_dict']['weight'] > 0, "Edge weight must be positive!"
            # Reward: difference in maximum link utilization
            if variation != 0:
                if self.routing_ecmp:
                    self.current_feasible, link_utils, self.current_feasible_percentage = self._find_feasible_shortest_paths_ECMP()
                else:
                    self.current_feasible, link_utils, _, self.current_feasible_percentage = self._find_feasible_shortest_paths()

                for (link_u, link_v), util in link_utils.items():
                    if not self.communication_graph.has_edge(link_u, link_v):
                        raise KeyError(f"Edge ({link_u}, {link_v}) not found in communication_graph")
                    self._set_link_attr(link_u, link_v, 'used_capacity', util * self.communication_graph[link_u][link_v]['edge_attr_dict']['capacity'])
                    assert self.communication_graph[link_u][link_v]['edge_attr_dict']['used_capacity'] <= self.communication_graph[link_u][link_v]['edge_attr_dict']['capacity'], "Used capacity exceeds capacity!"

                # Maximum link utilization
                if self.util_aggregation == 'max':
                    self.current_util = max(link_utils.values()) if link_utils else 0.0
                elif self.util_aggregation == 'mean':
                    self.current_util = np.mean(list(link_utils.values())) if link_utils else 0.0

                if self.always_feasible and self.current_feasible_percentage < 1.0:
                    assert False, "Feasibility violated in always feasible setting!"
                if self.reward_type == 'relative':
                    delta_feasible = self.current_feasible_percentage - self.prev_feasible_percentage
                    delta_denom = (self.initial_util - self.best_util) if (self.initial_util - self.best_util) !=0 else 1
                    delta_util = (self.prev_util - self.current_util) / delta_denom
                    self.reward = (self.feasibility_coefficient * delta_feasible
                              + self.link_util_coefficient * delta_util)
                    if self.reward == 0: # lost chance
                        self.real_no_change_actions += 1
                        self.reward = self.no_change_action_sum / (
                                    min(self.max_steps_overall, self.max_steps_coefficient * len(self.communication_graph.edges)))
            self.prev_util = self.current_util
            self.prev_feasible_percentage = self.current_feasible_percentage

            if self.reward_type == 'absolute':
                self.reward = (self.feasibility_coefficient * self.current_feasible_percentage
                          - self.link_util_coefficient * self.current_util)

        if self.feasibility_coefficient > 0 and self.link_util_coefficient == 0:
            self.done = self.current_feasible_percentage >= 1.0
        else:
            if self.terminate_at_approximate_best:
                self.done = self.current_util <= self.best_util
            else:
                self.done = False

        if self.reward_type == 'episodic':
            if self.done or self.truncated:
                # compute final reward at episode end
                self.reward = (self.feasibility_coefficient * self.current_feasible_percentage
                          - self.link_util_coefficient * self.current_util)

        self.action = (edge, variation)
        self.obs = self._get_obs()
        self.info = {}
        self.post_step()
        return self.obs, self.reward, self.done, self.truncated, self.info


    def _find_feasible_shortest_paths(self):
        # Find shortest paths for each traffic demand and check if they can carry the demand without exceeding link capacities

        temp_capacity = {
            (u, v): self.communication_graph[u][v]['edge_attr_dict']['capacity']
            for u, v in self.communication_graph.edges
        }
        feasible_routes = {}
        infeasible = False

        for s, d in self.traffic_edges:
            demand = self.traffic_graph[s][d]['edge_attr_dict']['traffic']
            if demand == 0:
                continue
            try:
                path = nx.shortest_path(
                    self.communication_graph, s, d,
                    weight=lambda a, b, d: self.communication_graph[a][b]['edge_attr_dict']['weight']
                )
            except nx.NetworkXNoPath:
                infeasible = True
                continue
            # check if the path can carry the demand
            if any(temp_capacity[(u, v)] < demand for u, v in zip(path[:-1], path[1:])):
                infeasible = True
                continue

            feasible_routes[(s, d)] = path
            # allocate demand
            for u, v in zip(path[:-1], path[1:]):
                temp_capacity[(u, v)] -= demand

        return not infeasible, self._compute_utilizations(temp_capacity), feasible_routes, (len(feasible_routes) / len(self.traffic_edges))

    def _find_feasible_shortest_paths_ECMP(self):
        # Find all ECMP shortest paths for each traffic demand and check if splitting the demand across them can be carried without exceeding link capacities
        capacity = {
            (u, v): self.communication_graph[u][v]['edge_attr_dict']['capacity']
            for u, v in self.communication_graph.edges
        }

        load = {edge: 0.0 for edge in capacity}
        ecmp_paths = {}
        demands = {}

        for s, d in self.traffic_edges:
            demand = self.traffic_graph[s][d]['edge_attr_dict']['traffic']
            if demand == 0:
                continue

            try:
                all_shortest_paths = list(nx.all_shortest_paths(
                    self.communication_graph,
                    source=s,
                    target=d,
                    weight=lambda u, v, attr: attr['edge_attr_dict']['weight']
                ))
            except nx.NetworkXNoPath:
                continue

            split_demand = demand / len(all_shortest_paths)
            ecmp_paths[(s, d)] = all_shortest_paths
            demands[(s, d)] = split_demand

            for path in all_shortest_paths:
                for u, v in zip(path[:-1], path[1:]):
                    load[(u, v)] += split_demand

        feasible_count = 0
        total_count = len([
            1 for (s, d) in self.traffic_edges
            if self.traffic_graph[s][d]['edge_attr_dict']['traffic'] > 0
        ])

        link_utilization = {
            edge: (load[edge] / capacity[edge] if capacity[edge] > 0 else 0.0)
            for edge in capacity
        }

        for (_, _), paths in ecmp_paths.items():
            path_feasible = True
            for path in paths:
                for u, v in zip(path[:-1], path[1:]):
                    if load[(u, v)] > capacity[(u, v)]:
                        path_feasible = False
                        break
                if not path_feasible:
                    break

            if path_feasible:
                feasible_count += 1

        feasibility_fraction = feasible_count / total_count if total_count > 0 else 1.0
        return feasibility_fraction == 1.0, link_utilization, feasibility_fraction

    # Compute link utilization ratio after tentative allocations
    def _compute_utilizations(self, temp_capacity):
        utilizations = {}
        for u, v in self.communication_graph.edges:
            cap = self.communication_graph[u][v]['edge_attr_dict']['capacity']
            used = cap - temp_capacity.get((u, v), 0)
            utilizations[(u, v)] = used / cap if cap > 0 else 0
        return utilizations

    def update_metrics(self):
        self._metrics.update({
            'initial_link_utilization': self.initial_util,
            'current_link_utilization': self.current_util,
            'ideal_reduction_percentage_relative': (self.initial_util - self.current_util) / (self.initial_util - self.best_util) if (self.initial_util - self.best_util) > 0 else 0.0,
            'link_utilization_difference': self.current_util - self.initial_util,
            'feasible_difference': self.current_feasible - self.initial_feasible,
            'feasibility_percentage_difference': self.current_feasible_percentage - self.initial_feasible_percentage,
            'decrease_actions': self.decrease_actions/self.current_step if self.current_step > 0 else 0.0,
            'increase_action': self.increase_actions/self.current_step if self.current_step > 0 else 0.0,
            'no_variation_actions': self.no_variation_actions/self.current_step if self.current_step > 0 else 0.0,
            'real_decrease_actions': self.real_decrease_actions/self.current_step if self.current_step > 0 else 0.0,
            'real_increase_action': self.real_increase_actions/self.current_step if self.current_step > 0 else 0.0,
            'real_no_variation_actions': self.real_no_variation_actions/self.current_step if self.current_step > 0 else 0.0,
            'real_no_change_actions': self.real_no_change_actions/self.current_step if self.current_step > 0 else 0.0,
            'targeted_edges': len(self.targeted_edges)/self.num_communication_edges,
            'targeted_positive_variations': len(self.targeted_positive_variations)/len([v for v in self.variations if v > 0]) if len([v for v in self.variations if v > 0]) > 0 else 0.0,
            'targeted_negative_variations': len(self.targeted_negative_variations)/len([v for v in self.variations if v < 0]) if len([v for v in self.variations if v < 0]) > 0 else 0.0,
        })


    def is_valid_action(self, edge, variation):
        # check that if bounded we don't go out of bounds, and if remove_invalid_actions is true, that we don't return invalid actions
        u, v = self._canon_link(*edge)
        new_weight = self.communication_graph[u][v]['edge_attr_dict']['weight'] + variation
        if self.bounded_weights:
            return self.min_weight <= new_weight <= self.max_weight
        return new_weight >= 1

class ExtendedOSPFTrafficEngineeringEnv(OSPFTrafficEngineeringEnv, ContinuousEnv):
    def __init__(self, **config):
        super().__init__(**config)
        self.update_spec()

    def update_spec(self):
        self._observation_type = {
            'communication_graph': Graph(poolings=[MeanPooling, SumPooling, MinPooling, MaxPooling], graph_name='communication_graph'),
            'traffic_graph': Graph(poolings=[MeanPooling, SumPooling, MinPooling, MaxPooling], graph_name='traffic_graph'),
            'communication_nodes_number': Function(func=get_number_nodes, graph_name='communication_graph'),
            # should all take graph in input
            'communication_edges_number': Function(func=get_number_edges, graph_name='communication_graph'),
            'communication_average_node_degree': Function(func=get_average_node_degree, graph_name='communication_graph'),
            'communication_graph_density': Function(func=get_graph_density, graph_name='communication_graph'),
            'traffic_nodes_number': Function(func=get_number_nodes, graph_name='traffic_graph'),
            'traffic_edges_number': Function(func=get_number_edges, graph_name='traffic_graph'),
            'traffic_average_node_degree': Function(func=get_average_node_degree, graph_name='traffic_graph'),
            'traffic_graph_density': Function(func=get_graph_density, graph_name='traffic_graph'),
            'action_space': ActionSpace(poolings=[MeanPooling, SumPooling, MaxPooling, MinPooling])
        }
        self._action_type = {
            'link': Edge(graph_name='communication_graph', bidirectional=True, poolings=[MeanPooling]),
            'weight_variation': Object(name='weights_variations', set=self.variations,
                                       feature_extractor=OneHotEncoding,
                                       feature_extractor_args={'set': self.variations},
                                       embedding_size=len(self.variations))
        }
        if not hasattr(self, 'traffic_sum'):
            self.traffic_sum = sum(
                self.traffic_graph[u][v]['edge_attr_dict']['traffic']
                for u, v in self.traffic_edges
            )
        if hasattr(self, 'always_feasible') and self.always_feasible:
            min_capacity = self.traffic_sum
            max_capacity = self.traffic_sum
            max_used_capacity = self.traffic_sum
        else:
            min_capacity = self.min_capacity
            max_capacity = self.max_capacity
            max_used_capacity = self.max_capacity

        # better estimation of max_value for incoming or outgoing traffic as a sum
        max_outgoing_traffic_value = max(
            sum(self.traffic_graph[u][v]['edge_attr_dict']['traffic'] for v in self.traffic_graph.successors(u))
            for u in self.traffic_graph.nodes
        )
        max_incoming_traffic_value = max(
            sum(self.traffic_graph[u][v]['edge_attr_dict']['traffic'] for u in self.traffic_graph.predecessors(v))
            for v in self.traffic_graph.nodes
        )

        self._node_attributes = {
            'outgoing_traffic': Attribute(feature_extractor=IdentityEncoding,
                                            encoding=ContinuousValue(
                                                normalization="min_max",
                                                min_value=0,
                                                max_value=max_outgoing_traffic_value,
                                            ),
                                            embedding_size=1,
                                            graph_name=['communication_graph', 'traffic_graph']),
            'incoming_traffic': Attribute(feature_extractor=IdentityEncoding,
                                        encoding=ContinuousValue(
                                            normalization="min_max",
                                            min_value=0,
                                            max_value=max_incoming_traffic_value,
                                        ),
                                        embedding_size=1,
                                        graph_name=['communication_graph', 'traffic_graph']),
        }
        self._edge_attributes = {
            'capacity': Attribute(feature_extractor=IdentityEncoding,
                                      encoding=ContinuousValue(
                                          normalization="min_max",
                                          min_value=min_capacity,
                                          max_value=max_capacity
                                      ),  # both ranking and continuous reconstruction loss
                                    embedding_size=1,
                                      graph_name='communication_graph'),
            'weight': Attribute(feature_extractor=IdentityEncoding,
                                    encoding=ContinuousValue(
                                        normalization="min_max",
                                        min_value=self.min_weight,
                                        max_value=self.max_weight
                                    ),
                                    embedding_size=1,
                                    graph_name='communication_graph'),
            'used_capacity': Attribute(feature_extractor=IdentityEncoding,
                                            encoding=ContinuousValue(
                                                normalization="min_max",
                                                min_value=0,
                                                max_value=max_used_capacity
                                            ),
                                            embedding_size=1,
                                            graph_name='communication_graph'),
            'traffic': Attribute(feature_extractor=IdentityEncoding,
                                     encoding=ContinuousValue(
                                        normalization="min_max",
                                        min_value=0,
                                        max_value=self.max_traffic
                                     ),
                                     embedding_size=1,
                                     graph_name='traffic_graph'),
        }
        if self.no_action_support:
            self.discrete_actions = ["no_action"]
        else:
            self.discrete_actions = []
        self.action_candidates_reconstruction = False
        self.reconstruct_only_changed_nodes = False
        self.action_space_reconstruction_each_step = True

    def sample_valid_action(self):
        # 1. Randomly select an edge
        edge_idx = np.random.randint(0, len(self.communication_edges))
        edge_source, edge_target = self.communication_edges[edge_idx]
        # 2. Compute all valid variation values (multiples of action_distance)
        variation_values = []
        for variation in self.variations:
            new_weight = self.communication_graph[edge_source][edge_target]['edge_attr_dict']['weight'] + variation
            if self.min_weight <= new_weight <= self.max_weight:
                variation_values.append(variation)
        # 3. Randomly select one valid variation value
        variation_value = np.random.choice(variation_values)
        return ((edge_source, edge_target), variation_value)

    def get_graphs(self):
        graphs = {
            'communication_graph': self.communication_graph,
            'traffic_graph': self.traffic_graph
        }
        return graphs