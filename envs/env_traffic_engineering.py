#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

import copy
from tqdm import tqdm
from gymnasium import spaces
import numpy as np
import itertools
import networkx as nx
from utils.features_utils import get_number_nodes, get_number_edges, get_average_node_degree, get_graph_density
from utils.model import Graph, Edge, Path, Function, Attribute, ActionSpace
from utils.feature_extractors import IdentityEncoding
from utils.pooling_functions import ConcatPooling, MeanPooling, SumPooling, MinPooling, MaxPooling
from utils.encoding_options import ContinuousValue
from utils.log_utils import graph_nodes_to_text, graph_edges_to_text
from wrappers.wrapper import ContinuousEnv
import random
from envs.env import DiscreteEnv

class TrafficEngineeringEnv(DiscreteEnv):
    """
    Traffic Engineering Environment for optimizing routing in a communication network by using path-based selections directly without relying on intermediate layers like OSPF
    """

    def __init__(self,
                 num_nodes=5,
                 min_capacity=10,
                 max_capacity=10,
                 max_len_path=4,
                 non_zero_traffic_ratio=0.3,
                 no_change_penalty_sum=-10,
                 graph_edges_distribution='small_world',
                 small_world_p=0.1,
                 small_world_k=4,
                 max_traffic=100,
                 communication_edge_ratio=0.3,
                 feasibility_coefficient=0.5,
                 link_util_coefficient=0.5,
                 util_aggregation='mean',
                 always_feasible=False,
                 max_sweeps=10000,
                 **kwargs):
        super().pre_init(**kwargs)
        self.num_nodes = num_nodes
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.max_traffic = max_traffic
        self.no_change_penalty_sum = no_change_penalty_sum
        self.non_zero_traffic_ratio = non_zero_traffic_ratio
        self.communication_edge_ratio = communication_edge_ratio
        self.feasibility_coefficient = feasibility_coefficient
        self.link_util_coefficient = link_util_coefficient
        self.util_aggregation = util_aggregation
        self.small_world_p = small_world_p
        self.max_len_path = max_len_path
        self.small_world_k = small_world_k
        self.graph_edges_distribution = graph_edges_distribution
        self.max_sweeps = max_sweeps
        self.always_feasible = always_feasible

        self.change_actions = 0
        self.no_change_actions = 0

        self.targeted_edges = set()
        if self.graph_edges_distribution == 'spanning_tree':
            self.communication_graph = nx.generators.random_tree(self.num_nodes)
        else:
            self.communication_graph = nx.watts_strogatz_graph(self.num_nodes, self.small_world_k, self.small_world_p)
        # add extra edges to reach desired communication edge ratio
        E_max = self.num_nodes * (self.num_nodes - 1) // 2
        E_target = int(self.communication_edge_ratio * E_max)
        existing_edges = set(self.communication_graph.edges)
        all_possible_edges = set(
            itertools.combinations(range(self.num_nodes), 2)
        )
        candidate_edges = list(all_possible_edges - existing_edges)
        new_edges = random.sample(candidate_edges, E_target)
        self.communication_graph.add_edges_from(new_edges)
        self.communication_edges = list(self.communication_graph.edges)

        success = False
        max_trials = 10
        i = 0
        while not success and i < max_trials:
            self.traffic_graph = nx.complete_graph(self.num_nodes, create_using=nx.DiGraph)
            self.traffic_edges = list(self.traffic_graph.edges)

            # Initialize capacities and traffic requirements
            for u, v in self.communication_edges:
                self.communication_graph[u][v]['edge_attr_dict'] = {}
                sampled_capacity = np.random.randint(self.min_capacity, self.max_capacity + 1)
                self.communication_graph[u][v]['edge_attr_dict']['capacity'] = sampled_capacity
                self.communication_graph[u][v]['edge_attr_dict']['used_capacity_ratio'] = 0
                self.communication_graph[u][v]['edge_attr_dict']['used_capacity'] = 0

            self.traffic_sum = 0

            candidate_edges = list(self.traffic_graph.edges)
            feasible_edges = [
                (u, v) for (u, v) in candidate_edges
                if v in nx.single_source_shortest_path_length(
                    self.communication_graph, u, cutoff=self.max_len_path
                )
            ]
            k = int(self.non_zero_traffic_ratio * len(candidate_edges))

            selected_edges = random.sample(feasible_edges, k)

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

            if self.always_feasible:
                # update communication graph to ensure feasibility
                for u, v in self.communication_edges:
                    sampled_capacity = np.random.randint(self.traffic_sum, self.traffic_sum+1)
                    self.communication_graph[u][v]['edge_attr_dict']['capacity'] = sampled_capacity

            self.initial_allocations, success = self._initialize_empirical_worst_allocation(max_sweeps=self.max_sweeps)
            i += 1
        if not success:
            raise ValueError("Failed to find initial worst-case allocation after multiple trials.")

        feasible, link_utils, _, self.initial_feasible_percentage = self._compute_allocation_stats(self.initial_allocations)
        if self.util_aggregation == 'max':
            self.initial_util = max(link_utils.values()) if link_utils else 0.0
        elif self.util_aggregation == 'mean':
            self.initial_util = np.mean(list(link_utils.values())) if link_utils else 0.0
        self.current_util = self.initial_util
        self.initial_feasible = 1 if feasible else 0
        self.current_feasible = self.initial_feasible
        self.current_feasible_percentage = self.initial_feasible_percentage

        # Dictionary to store all paths for each traffic edge
        self.traffic_edges = list(self.traffic_graph.edges)
        self.routing_dict = self._find_routing_options()
        # print for every traffic edge how many routing options are ther
        self.action_list = self.build_action_list()
        self.semantic_action_map = self.build_semantic_action_list()

        self.scenario_size = len(self.traffic_edges)

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
        # compute features to determine complexity of the environment
        self.num_communication_edges = self.communication_graph.number_of_edges()
        self.num_traffic_edges = self.traffic_graph.number_of_edges()
        self.min_num_communication_edges = self.num_nodes - 1  # spanning tree
        self.min_num_traffic_edges = 1  # at least one traffic edge
        self.max_num_communication_edges = self.num_nodes * (self.num_nodes - 1)  # fully connected directed graph
        self.max_num_traffic_edges = self.num_nodes * (self.num_nodes - 1)
        self.chosen_paths_per_edge = {}

        # Observation: concatenated edge features [capacity, weight, used_capacity] + traffic matrix
        obs_len = len(self.communication_edges) * 3 + len(self.traffic_edges)  # capacity, weight, used_capacity for each communication edge + traffic for each traffic edge
        self.observation_space =  spaces.Dict({
            "graph": spaces.Box(
                low=0,
                high=self.max_capacity,
                shape=(obs_len,),
                dtype=np.float32
            )
        })
        super().post_init()

    def build_action_list(self):
        self.discrete_routing_options = []
        for edge in self.routing_dict:
            for path in self.routing_dict[edge]:
                self.discrete_routing_options.append((edge, path))
        return self.discrete_routing_options

    def _find_routing_options(self):
        # For each traffic edge, find all simple paths in the communication graph up to max_len_path
        self.routing_dict = {}
        for (src, dst) in self.traffic_edges:
            try:
                # Get all simple paths from src to dst in communication graph
                paths = list(
                    nx.all_simple_paths(self.communication_graph, source=src, target=dst, cutoff=self.max_len_path))
                self.routing_dict[(src, dst)] = paths
            except nx.NetworkXNoPath:
                # No path exists between src and dst
                self.routing_dict[(src, dst)] = []
        return self.routing_dict


    def _sample_neighbor_allocation(self, base_allocation, direction, feasible=False):
        # Heuristic to discover best and worst empirical solutions
        # Sample a stochastic neighboring allocation by rerouting a small number of demands.
        if direction not in ("best", "worst"):
            raise ValueError(f"direction must be 'best' or 'worst', got {direction}")

        allocation = {
            demand: path.copy()
            for demand, path in base_allocation.items()
        }
        edges = list(self.communication_edges)

        if not allocation:
            return allocation

        # ------------------------------------------------------------
        # Build current edge loads induced by base_allocation
        # ------------------------------------------------------------
        used_capacity = {(u, v): 0.0 for (u, v) in edges}

        for (s, t), path in allocation.items():
            demand = self.traffic_graph[s][t]['edge_attr_dict']['traffic']
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in used_capacity:
                    used_capacity[(u, v)] += demand
                elif (v, u) in used_capacity:
                    used_capacity[(v, u)] += demand

        def edge_capacity(e):
            u, v = e
            return self.communication_graph[u][v]['edge_attr_dict']['capacity']

        def edge_util(e, loads=None):
            if loads is None:
                loads = used_capacity
            cap = edge_capacity(e)
            if cap <= 0:
                return 0.0
            return loads[e] / cap

        def path_edges(path):
            out = []
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in used_capacity:
                    out.append((u, v))
                elif (v, u) in used_capacity:
                    out.append((v, u))
            return out

        def global_max_util(loads):
            max_util = 0.0
            for e in edges:
                util = edge_util(e, loads)
                if util > max_util:
                    max_util = util
            return max_util

        # ------------------------------------------------------------
        # Build graph view for path computation
        # ------------------------------------------------------------
        # We use current communication graph topology but ignore current TE allocation;
        # path search is done over the underlying network.
        G = nx.DiGraph()

        for u, v in edges:
            cap = self.communication_graph[u][v]['edge_attr_dict']['capacity']
            if cap > 0:
                G.add_edge(u, v)

        if len(G.edges) == 0:
            return allocation

        # ------------------------------------------------------------
        # Score demands for rerouting
        # best  -> prefer demands crossing highly utilized edges
        # worst -> prefer demands that can be moved and create concentration
        # ------------------------------------------------------------
        demand_items = list(allocation.items())
        demand_scores = []

        for (s, t), path in demand_items:
            pes = path_edges(path)
            if not pes:
                demand_scores.append(1.0)
                continue

            path_utils = [edge_util(e) for e in pes]

            if direction == "best":
                # prioritize demands touching bottlenecks
                score = max(path_utils) + 1e-6
            else:
                # still prioritize impactful flows, but not deterministically
                traffic = self.traffic_graph[s][t]['edge_attr_dict']['traffic']
                score = traffic * (0.5 + max(path_utils)) + 1e-6

            demand_scores.append(score)

        # reroute a small random number of demands
        num_reroutes = random.randint(1, min(3, len(demand_items)))

        chosen_indices = set()
        while len(chosen_indices) < num_reroutes:
            idx = random.choices(
                range(len(demand_items)),
                weights=demand_scores,
                k=1
            )[0]
            chosen_indices.add(idx)

        # ------------------------------------------------------------
        # Candidate-path generator
        # ------------------------------------------------------------
        def sample_candidate_paths(s, t, current_path, max_candidates=12):
            """
            Produce a diverse random set of simple paths from s to t.
            Mixes shortest-path and k-shortest-path style sampling.
            """
            candidates = []
            seen = set()

            def add_path(p):
                key = tuple(p)
                if key not in seen and key != tuple(current_path):
                    seen.add(key)
                    candidates.append(list(p))

            # 1) plain shortest path if available
            try:
                p = nx.shortest_path(G, source=s, target=t)
                add_path(p)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            # 2) randomized edge weights to induce path diversity
            for _ in range(max_candidates * 2):
                H = G.copy()
                for u, v in H.edges():
                    base = 1.0
                    util_bonus = 0.0

                    edge_key = (u, v) if (u, v) in used_capacity else ((v, u) if (v, u) in used_capacity else None)
                    if edge_key is not None:
                        util_bonus = edge_util(edge_key)

                    if direction == "best":
                        # avoid loaded edges more often
                        w = base + 2.0 * util_bonus + random.random()
                    else:
                        # encourage loaded edges more often
                        w = base + 2.0 * (1.0 - min(util_bonus, 1.0)) + random.random()

                    H[u][v]["weight"] = w

                try:
                    p = nx.shortest_path(H, source=s, target=t, weight="weight")
                    add_path(p)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

                if len(candidates) >= max_candidates:
                    break

            # 3) optionally try a few shortest simple paths
            try:
                gen = nx.shortest_simple_paths(G, source=s, target=t)
                for _ in range(max_candidates):
                    try:
                        p = next(gen)
                        add_path(p)
                        if len(candidates) >= max_candidates:
                            break
                    except StopIteration:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            if not candidates:
                return [current_path]

            return candidates[:max_candidates]

        # ------------------------------------------------------------
        # Evaluate a reroute move
        # ------------------------------------------------------------
        def try_reroute(temp_alloc, demand_key, new_path):
            """
            Return (new_loads, new_global_max_util) if reroute valid, else None.
            """
            s, t = demand_key
            traffic = self.traffic_graph[s][t]['edge_attr_dict']['traffic']
            old_path = temp_alloc[demand_key]

            new_loads = used_capacity.copy()

            # remove old path load
            for e in path_edges(old_path):
                new_loads[e] -= traffic

            # add new path load
            new_edges = []
            for u, v in zip(new_path[:-1], new_path[1:]):
                if (u, v) in new_loads:
                    e = (u, v)
                elif (v, u) in new_loads:
                    e = (v, u)
                else:
                    return None
                new_edges.append(e)
                new_loads[e] += traffic

            if feasible:
                for e in new_edges:
                    cap = edge_capacity(e)
                    if cap > 0 and new_loads[e] - cap > 1e-9:
                        return None

            return new_loads, global_max_util(new_loads)

        # ------------------------------------------------------------
        # Apply reroutes one by one, stochastically
        # ------------------------------------------------------------
        for idx in chosen_indices:
            demand_key, current_path = demand_items[idx]
            s, t = demand_key

            candidate_paths = sample_candidate_paths(s, t, current_path)
            move_candidates = []

            for cand_path in candidate_paths:
                trial = try_reroute(allocation, demand_key, cand_path)
                if trial is None:
                    continue

                new_loads, new_global_util = trial

                old_global_util = global_max_util(used_capacity)

                if direction == "best":
                    gain = old_global_util - new_global_util
                else:
                    gain = new_global_util - old_global_util

                # keep all candidates, but bias later
                move_candidates.append((cand_path, new_loads, new_global_util, gain))

            if not move_candidates:
                continue

            gains = np.array([mc[3] for mc in move_candidates], dtype=float)

            # probabilistic choice, not deterministic argmax
            if direction == "best":
                weights = gains - gains.min() + 1e-6
                if np.all(weights <= 0):
                    # fallback: prefer lower resulting max util
                    vals = np.array([mc[2] for mc in move_candidates], dtype=float)
                    weights = vals.max() - vals + 1e-6
            else:
                weights = gains - gains.min() + 1e-6
                if np.all(weights <= 0):
                    # fallback: prefer higher resulting max util
                    vals = np.array([mc[2] for mc in move_candidates], dtype=float)
                    weights = vals - vals.min() + 1e-6

            chosen = random.choices(
                move_candidates,
                weights=weights,
                k=1
            )[0]

            chosen_path, chosen_loads, _, _ = chosen
            allocation[demand_key] = chosen_path
            used_capacity = chosen_loads

        return allocation

    def _initialize_empirical_worst_allocation(
            self,
            max_sweeps=1000,
            feasible=False,
            local_steps=8,
            candidate_moves=6,
    ):
        # Heuristic local search to find the best and worst solution using function above
        self.best_util = np.inf
        self.worst_util = -1.0

        best_allocation = None
        worst_allocation = None

        seen_signatures = set()
        elite_best = []
        elite_worst = []

        edges = list(self.communication_edges)

        def reset_capacities():
            for u, v in edges:
                self.communication_graph[u][v]['edge_attr_dict']['used_capacity'] = 0.0

        def apply_allocation(allocation):
            reset_capacities()
            for (s, t), path in allocation.items():
                demand = self.traffic_graph[s][t]['edge_attr_dict']['traffic']
                for u, v in zip(path[:-1], path[1:]):
                    self.communication_graph[u][v]['edge_attr_dict']['used_capacity'] += demand

        def compute_signature_and_util():
            signature = []
            max_util = 0.0
            for u, v in edges:
                attr = self.communication_graph[u][v]['edge_attr_dict']
                used = attr['used_capacity']
                signature.append(used)
                if attr['capacity'] > 0:
                    util = used / attr['capacity']
                    if util > max_util:
                        max_util = util
            return tuple(signature), max_util

        for sweep in tqdm(range(max_sweeps), desc="TE empirical sweeps"):
            direction = "best" if sweep % 2 == 0 else "worst"

            # -------------------------
            # Initial state
            # -------------------------
            use_elite = (random.random() < 0.35) and (elite_best or elite_worst)
            if use_elite:
                pool = elite_best if direction == "best" else elite_worst
                if not pool:
                    pool = elite_best + elite_worst
                allocation = copy.deepcopy(random.choice(pool))
                apply_allocation(allocation)

                # perturb a little so sweeps do not repeat same path
                for _ in range(random.randint(1, 3)):
                    allocation = self._sample_neighbor_allocation(
                        allocation,
                        direction=direction,
                        feasible=feasible
                    )
                    apply_allocation(allocation)
            else:
                if feasible:
                    allocation = self.perform_initial_random_feasible_allocation()
                else:
                    allocation = self.perform_initial_random_allocation()

            signature, max_util = compute_signature_and_util()

            if signature not in seen_signatures:
                seen_signatures.add(signature)

                if max_util < self.best_util:
                    self.best_util = max_util
                    best_allocation = copy.deepcopy(allocation)

                if max_util > self.worst_util:
                    self.worst_util = max_util
                    worst_allocation = copy.deepcopy(allocation)

            # -------------------------
            # Local neighborhood search
            # -------------------------
            current_alloc = copy.deepcopy(allocation)
            current_util = max_util

            for _ in range(local_steps):
                candidates = []

                for _ in range(candidate_moves):
                    cand_alloc = self._sample_neighbor_allocation(
                        current_alloc,
                        direction=direction,
                        feasible=feasible
                    )
                    apply_allocation(cand_alloc)
                    cand_sig, cand_util = compute_signature_and_util()

                    if cand_sig in seen_signatures:
                        continue

                    seen_signatures.add(cand_sig)
                    candidates.append((copy.deepcopy(cand_alloc), cand_util))

                    if cand_util < self.best_util:
                        self.best_util = cand_util
                        best_allocation = copy.deepcopy(cand_alloc)

                    if cand_util > self.worst_util:
                        self.worst_util = cand_util
                        worst_allocation = copy.deepcopy(cand_alloc)

                if not candidates:
                    break

                vals = np.array([u for _, u in candidates], dtype=float)
                if direction == "best":
                    weights = vals.max() - vals + 1e-8
                else:
                    weights = vals - vals.min() + 1e-8

                idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                current_alloc, current_util = candidates[idx]
                apply_allocation(current_alloc)

            if best_allocation is not None:
                elite_best = (elite_best + [copy.deepcopy(best_allocation)])[-10:]
            if worst_allocation is not None:
                elite_worst = (elite_worst + [copy.deepcopy(worst_allocation)])[-10:]

            reset_capacities()

        # apply worst allocation
        reset_capacities()
        if worst_allocation is not None:
            for (s, t), path in worst_allocation.items():
                demand = self.traffic_graph[s][t]['edge_attr_dict']['traffic']
                for u, v in zip(path[:-1], path[1:]):
                    self.communication_graph[u][v]['edge_attr_dict']['used_capacity'] += demand

        for u, v in edges:
            attr = self.communication_graph[u][v]['edge_attr_dict']
            attr['used_capacity_ratio'] = (
                attr['used_capacity'] / attr['capacity']
                if attr['capacity'] > 0 else 0.0
            )
        return worst_allocation, self.best_util != self.worst_util

    def perform_initial_random_allocation(self):
        # --- INITIAL TRAFFIC ALLOCATION (Random Path First With Capacity Check) ---
        initial_allocations = {}

        for s, t in list(self.traffic_graph.edges()):
            demand = self.traffic_graph[s][t]['edge_attr_dict']['traffic']

            # get all simple paths between s and t
            paths = list(nx.all_simple_paths(
                self.communication_graph, s, t, cutoff=self.max_len_path
            ))
            chosen_path = random.choice(paths)

            # allocate traffic to this feasible path
            for u, v in zip(chosen_path[:-1], chosen_path[1:]):
                self.communication_graph[u][v]['edge_attr_dict']['used_capacity'] += demand

            initial_allocations[(s, t)] = chosen_path

        # update used capacity ratio for all edges
        for u, v in self.communication_edges:
            cap = self.communication_graph[u][v]['edge_attr_dict']['capacity']
            used = self.communication_graph[u][v]['edge_attr_dict']['used_capacity']
            self.communication_graph[u][v]['edge_attr_dict']['used_capacity_ratio'] = used / cap if cap > 0 else 0.0

        return initial_allocations

    def perform_initial_random_feasible_allocation(self):
        # --- INITIAL TRAFFIC ALLOCATION (Random Path First With Capacity Check) ---
        # Additionally ensure feasibility
        initial_allocations = {}

        for s, t in list(self.traffic_graph.edges()):
            demand = self.traffic_graph[s][t]['edge_attr_dict']['traffic']

            # get all simple paths between s and t
            paths = list(nx.all_simple_paths(
                self.communication_graph, s, t, cutoff=self.max_len_path
            ))

            # shuffle paths to ensure random selection
            random.shuffle(paths)

            allocated = False

            for path in paths:
                feasible = True

                # check capacity feasibility
                for u, v in zip(path[:-1], path[1:]):
                    cap = self.communication_graph[u][v]['edge_attr_dict']['capacity']
                    used = self.communication_graph[u][v]['edge_attr_dict']['used_capacity']
                    if used + demand > cap:
                        feasible = False
                        break

                if not feasible:
                    continue

                # allocate traffic to this feasible path
                for u, v in zip(path[:-1], path[1:]):
                    self.communication_graph[u][v]['edge_attr_dict']['used_capacity'] += demand

                initial_allocations[(s, t)] = path
                allocated = True
                break

            # if no feasible path found → choose a random one anyway
            if not allocated:
                random_path = random.choice(paths)
                for u, v in zip(random_path[:-1], random_path[1:]):
                    self.communication_graph[u][v]['edge_attr_dict']['used_capacity'] += demand

                initial_allocations[(s, t)] = random_path
                print(f"WARNING: No feasible path for {s}->{t}, allocated randomly on {random_path}")

        return initial_allocations

    def reset(self, **kwargs):
        super().pre_reset()
        # Reset weights to initial stage, as well as data structures related
        self.communication_graph = copy.deepcopy(self.initial_communication_graph)
        self.allocations = copy.deepcopy(self.initial_allocations)
        if self.verbose > 1:
            print("Resetting environment. Initial communication graph restored.")

        feasible, link_utils, _, self.initial_feasible_percentage = self._compute_allocation_stats(self.allocations)

        if self.util_aggregation == 'max':
            self.initial_util = max(link_utils.values()) if link_utils else 0.0
        elif self.util_aggregation == 'mean':
            self.initial_util = np.mean(list(link_utils.values())) if link_utils else 0.0

        self.prev_feasible_percentage = self.initial_feasible_percentage
        self.prev_util = self.initial_util
        # before formatting chosen paths print final state
        for edge in self.chosen_paths_per_edge:
            # count unique paths
            unique_paths = set(tuple(path) for path in self.chosen_paths_per_edge[edge])
        self.chosen_paths_per_edge = {}
        self.initial_feasible = 1 if feasible else 0

        if self.verbose > 1:
            print("Post-reset initial feasible:", self.initial_feasible, "Feasible %:", self.initial_feasible_percentage)
            print(f"Post-reset initial max link utilization: {self.initial_util:.4f}")

        self.change_actions = 0
        self.no_change_actions = 0
        self.targeted_edges = set()
        self.obs = self._get_obs()
        self.info = {}
        super().post_reset()
        return self.obs, self.info

    def _get_obs(self):
        # Observation: concatenated edge features [capacity, weight, used_capacity] + traffic for each traffic edge
        edge_features = []
        for u, v in self.communication_edges:
            edge_features.extend([self.communication_graph[u][v]['edge_attr_dict']['capacity'], self.communication_graph[u][v]['edge_attr_dict']['used_capacity']])
        for u, v in self.traffic_edges:
            edge_features.append(self.traffic_graph[u][v]['edge_attr_dict']['traffic'])
        return {"graph": np.array(edge_features).astype(np.float32)}

    def is_valid_action(self, traffic_edge, communication_path):
        # Action is valid if the communication path can accommodate the traffic demand of the traffic edge without exceeding capacities
        demand = self.traffic_graph[traffic_edge[0]][traffic_edge[1]]['edge_attr_dict']['traffic']
        for u, v in zip(communication_path[:-1], communication_path[1:]):
            cap = self.communication_graph[u][v]['edge_attr_dict']['capacity']
            used = self.communication_graph[u][v]['edge_attr_dict']['used_capacity']
            if used + demand > cap:
                return False
        return True

    def update_config(self, config):
        self.feasibility_coefficient = config.get('feasibility_coefficient', self.feasibility_coefficient)
        self.link_util_coefficient = config.get('link_util_coefficient', self.link_util_coefficient)
        self.util_aggregation = config.get('util_aggregation', self.util_aggregation)
        self.max_len_path = config.get('max_len_path', self.max_len_path)
        # find routing options again
        self.routing_dict = self._find_routing_options()
        # need to rebuild action list and semantic action map since routing options may have changed
        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_semantic_action_list()
        super().reconstruct_action_space()

    def build_semantic_action_list(self):
        sorted_edges = sorted(
            self.traffic_edges,
            key=lambda e: self.traffic_graph[e[0]][e[1]]['edge_attr_dict']['traffic'],
            reverse=True
        )
        mapping = []

        # Semantic cartesian product
        for edge in sorted_edges:
            paths = self.routing_dict[edge]
            sorted_paths = sorted(
                paths,
                # use len
                key=len,
                reverse=False
            )
            for path in sorted_paths:
                mapping.append((edge, path))
        return mapping

    def step(self, traffic_edge, communication_path=None):
        action = super().pre_step((traffic_edge, communication_path))

        if not self.invalid_action and not self.no_action:
            traffic_edge, communication_path = action

            u, v = traffic_edge
            if not (u, v) in self.chosen_paths_per_edge:
                self.chosen_paths_per_edge[(u, v)] = []
            self.chosen_paths_per_edge[(u, v)].append(communication_path)
            if (u,v) not in self.targeted_edges:
                self.targeted_edges.add((u,v))

            # find previous allocation
            previous_allocation = self.allocations[(u, v)]
            self.previous_communication_path = previous_allocation
            if previous_allocation == communication_path:
                self.no_change_actions += 1
                self.reward = self.no_change_penalty_sum / (
                                self.max_steps_coefficient * len(self.traffic_graph.edges))
            else:
                self.change_actions += 1
                # deallocate previous allocation
                demand = self.traffic_graph[u][v]['edge_attr_dict']['traffic']
                for a, b in zip(previous_allocation[:-1], previous_allocation[1:]):
                    self.communication_graph[a][b]['edge_attr_dict']['used_capacity'] -= demand
                    self.communication_graph[a][b]['edge_attr_dict']['used_capacity_ratio'] = self.communication_graph[a][b]['edge_attr_dict']['used_capacity'] / self.communication_graph[a][b]['edge_attr_dict']['capacity']
                    assert self.communication_graph[a][b]['edge_attr_dict']['used_capacity'] >= 0, f"It has to be positive, but is {self.communication_graph[a][b]['edge_attr_dict']['used_capacity']}"
                    assert self.communication_graph[b][a]['edge_attr_dict']['used_capacity'] >= 0, "It has to be positive"
                    assert self.communication_graph[a][b]['edge_attr_dict']['used_capacity_ratio'] >= 0, "It has to be positive"
                    assert self.communication_graph[b][a]['edge_attr_dict'][
                               'used_capacity_ratio'] >= 0, "It has to be positive"
                # allocate new path
                for a, b in zip(communication_path[:-1], communication_path[1:]):
                    self.communication_graph[a][b]['edge_attr_dict']['used_capacity'] += demand
                    self.communication_graph[a][b]['edge_attr_dict']['used_capacity_ratio'] = \
                    self.communication_graph[a][b]['edge_attr_dict']['used_capacity'] / \
                    self.communication_graph[a][b]['edge_attr_dict']['capacity']
                    assert self.communication_graph[a][b]['edge_attr_dict']['used_capacity'] <= self.communication_graph[a][b]['edge_attr_dict']['capacity'], "It cannot exceed capacity"
                    assert self.communication_graph[b][a]['edge_attr_dict']['used_capacity'] <= self.communication_graph[b][a]['edge_attr_dict']['capacity'], "It cannot exceed capacity"
                    assert self.communication_graph[a][b]['edge_attr_dict']['used_capacity_ratio'] <= 1.0, "It cannot exceed 1.0"
                    assert self.communication_graph[b][a]['edge_attr_dict'][
                                 'used_capacity_ratio'] <= 1.0, "It cannot exceed 1.0"

                self.allocations[(u, v)] = communication_path
                feasible, link_utils, _, self.current_feasible_percentage = self._compute_allocation_stats(
                    self.allocations)
                if self.util_aggregation == 'max':
                    self.current_util = max(link_utils.values()) if link_utils else 0.0
                elif self.util_aggregation == 'mean':
                    self.current_util = np.mean(list(link_utils.values())) if link_utils else 0.0
                if self.prev_feasible_percentage != 0:
                    feasibility_change = (
                                                         self.current_feasible_percentage - self.prev_feasible_percentage) / self.prev_feasible_percentage
                else:
                    feasibility_change = 0.0

                if self.always_feasible:
                    assert self.current_feasible_percentage == 1.0, f"Not all are feasible, {self.current_feasible_percentage}"

                if self.prev_util != 0:
                    util_change = (self.prev_util - self.current_util) / (self.initial_util - self.best_util)
                else:
                    util_change = 0.0
                # Final reward
                self.reward = (self.feasibility_coefficient * feasibility_change +
                              self.link_util_coefficient * util_change)
                self.prev_util = self.current_util
                self.prev_feasible_percentage = self.current_feasible_percentage

        if self.feasibility_coefficient > 0 and self.link_util_coefficient == 0:
            self.done = self.current_feasible_percentage >= 1.0
        else:
            if self.terminate_at_approximate_best:
                self.done = self.current_util <= self.best_util
            else:
                self.done = False

        self.obs = self._get_obs()
        self.action = (traffic_edge, communication_path)
        self.info = {}
        super().post_step()
        return self.obs, self.reward, self.done, self.truncated, self.info

    def save_step_log(self):
        super().save_step_log()
        self.step_logs[-1].update({
            'observation': self.obs,
            'communication_graph_nodes':  graph_nodes_to_text(self.communication_graph),
            'communication_graph_edges': graph_edges_to_text(self.communication_graph),
            'traffic_graph_nodes':  graph_nodes_to_text(self.traffic_graph),
            'traffic_graph_edges': graph_edges_to_text(self.traffic_graph),
            'action': "Traffic Edge: " + str(self.action[0]) + ", Communication Path: " + str(self.action[1]),
            'previous_communication_path': self.previous_communication_path if hasattr(self, 'previous_communication_path') else None,
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

    def _compute_allocation_stats(self, allocations):
        temp_capacity = {
            (u, v): self.communication_graph[u][v]['edge_attr_dict']['capacity']
            for u, v in self.communication_graph.edges
        }
        feasible_routes = {}
        infeasible = False
        total_traffic = len(self.traffic_edges)
        satisfied = 0  # counts satisfied DEMANDS
        for (s, t), path in allocations.items():
            demand = self.traffic_graph[s][t]['edge_attr_dict']['traffic']

            path_feasible = True
            used_edges = []

            # --- Check full path feasibility ---
            for u, v in zip(path[:-1], path[1:]):
                edge = (u, v) if (u, v) in temp_capacity else (v, u)

                if temp_capacity[edge] < demand:
                    path_feasible = False
                    infeasible = True
                    break

                used_edges.append(edge)

            # --- Apply result ---
            if path_feasible:
                satisfied += 1
                feasible_routes[(s, t)] = path

                for edge in used_edges:
                    temp_capacity[edge] -= demand

        utilizations = self._compute_utilizations(temp_capacity)
        satisfaction_ratio = satisfied / total_traffic if total_traffic > 0 else 0.0
        return not infeasible, utilizations, feasible_routes, satisfaction_ratio

    # Compute link utilization ratio after tentative allocations
    def _compute_utilizations(self, temp_capacity):
        utilizations = {}
        for u, v in self.communication_graph.edges:
            cap = self.communication_graph[u][v]['edge_attr_dict']['capacity']
            used = cap - temp_capacity.get((u, v), 0)
            utilizations[(u, v)] = used / cap if cap > 0 else 0
        return utilizations

    def update_metrics(self):
        super().update_metrics()
        self._metrics.update({
            'initial_link_utilization': self.initial_util,
            'current_link_utilization': self.current_util,
            'ideal_reduction_percentage_relative': (self.initial_util - self.current_util) / (self.initial_util - self.best_util) if (self.initial_util - self.best_util) > 0 else 0.0,
            'link_utilization_difference': self.current_util - self.initial_util,
            'feasible_difference': self.current_feasible - self.initial_feasible,
            'feasibility_percentage_difference': self.current_feasible_percentage - self.initial_feasible_percentage,
            'change_actions': self.change_actions / self.current_step if self.current_step > 0 else 0.0,
            'no_change_actions': self.no_change_actions / self.current_step if self.current_step > 0 else 0.0,
            'targeted_edges': len(self.targeted_edges),
        })

def edge_key_from_tuple(x):
    return f"edge_{x[0]}_{x[1]}"

class ExtendedTrafficEngineeringEnv(TrafficEngineeringEnv, ContinuousEnv):
    def __init__(self, path_encoding, **config):
        super().__init__(**config)
        self.path_encoding = path_encoding
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
        # convert routing dict in another dict where keys are called edge_u_v
        self.routing_dict_formatted = {}
        for edge in self.routing_dict:
            key_name = f"edge_{edge[0]}_{edge[1]}"
            self.routing_dict_formatted[key_name] = self.routing_dict[edge]

        if self.path_encoding == 'concat':
            self._action_type = {
                'traffic_link': Edge(graph_name='traffic_graph', poolings=[ConcatPooling]),
                'communication_path': Path(graph_name='communication_graph',
                                       set_dict=self.routing_dict_formatted,
                                       set_dict_key_name='traffic_link',
                                       set_dict_function=edge_key_from_tuple,
                                       max_len=self.max_len_path,
                                       poolings=[ConcatPooling])
            }
        elif self.path_encoding == 'pooling':
            self._action_type = {
                'traffic_link': Edge(graph_name='traffic_graph'),
                'communication_path': Path(graph_name='communication_graph',
                                       set_dict=self.routing_dict_formatted,
                                       set_dict_key_name='traffic_link',
                                       set_dict_function=edge_key_from_tuple,
                                       max_len=self.max_len_path,
                                       poolings=[MeanPooling, SumPooling]),
            }

        # computed to ensure correct normalization
        if not hasattr(self, 'traffic_sum'):
            self.traffic_sum = sum(
                self.traffic_graph[u][v]['edge_attr_dict']['traffic']
                for u, v in self.traffic_edges
            )
        if hasattr(self, 'always_feasible') and self.always_feasible:
            min_capacity = self.traffic_sum
            max_capacity = self.traffic_sum
        else:
            min_capacity = self.min_capacity
            max_capacity = self.max_capacity

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
            'used_capacity_ratio': Attribute(feature_extractor=IdentityEncoding,
                                           encoding=ContinuousValue(), # already between 0 and 1
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
        self.reconstruct_only_changed_nodes = True
        self.action_space_reconstruction_each_step = True

    def sample_valid_action(self):
        # 1. Randomly select an edge
        edge_idx = np.random.randint(0, len(self.traffic_edges))
        # 2. Randomly select a routing option for that edge
        routing_options = self.routing_dict[self.traffic_edges[edge_idx]]
        communication_path = routing_options[np.random.randint(0, len(routing_options))]
        # derive discrete action index
        index = self.discrete_routing_options.index((self.traffic_edges[edge_idx], communication_path))
        return index

    def get_graphs(self):
        graphs = {
            'communication_graph': self.communication_graph,
            'traffic_graph': self.traffic_graph
        }
        return graphs