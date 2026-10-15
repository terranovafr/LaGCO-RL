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
from utils.feature_extractors import IdentityEncoding
from utils.pooling_functions import MeanPooling, SumPooling, MinPooling, MaxPooling
from utils.encoding_options import ContinuousValue, BinaryValue
from wrappers.wrapper import ContinuousEnv
from tqdm import tqdm
from envs.env import DiscreteEnv

class TSPEnv(DiscreteEnv):

    def __init__(self,
                 num_cities=10,
                 max_coord=100,
                 reward_type="relative",
                 max_sweeps=1000,
                 invalid_action_penalty_sum=-10,
                 shortest_K_edges=None,
                 hard_constraints=False,
                 **kwargs):
        super().pre_init(**kwargs)
        self.num_cities = num_cities
        self.max_coord = max_coord
        self.reward_type = reward_type
        self.invalid_action_penalty_sum = invalid_action_penalty_sum
        self.shortest_K_edges = shortest_K_edges
        self.hard_constraints = hard_constraints
        self.max_sweeps = max_sweeps

        # Generate city coordinates
        self._generate_cities()
        self.scenario_size = self.num_cities

        # Precompute distance matrix
        self._compute_distance_matrix()

        obs_len = self.num_cities * 2 + self.num_cities * 2 + (self.num_cities * (self.num_cities - 1)) // 2
        self.observation_space = spaces.Dict({
            "graph": spaces.Box(
                low=0,
                high=self.max_coord,
                shape=(obs_len,),
                dtype=np.float32
            )
        })
        self.current_length = 0.0
        self.current_length_worst_case = 0.0
        self.start_city = None  # first chosen city
        self.current_city = None
        self.tour = []
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.graph = nx.Graph()
        for i in range(self.num_cities):
            self.graph.add_node(i, x_dict={'x': self.city_coords[i][0], 'y': self.city_coords[i][1],
                                           'visited': self.visited[i], 'current': False})
        if self.shortest_K_edges is None:
            for i, j in itertools.combinations(range(self.num_cities), 2):
                self.graph.add_edge(i, j, edge_attr_dict={'weight': self.dist_matrix[i, j]})
        else:
            for i in range(self.num_cities):
                knn = np.argsort(self.dist_matrix[i])[1:self.shortest_K_edges + 1]
                for j in knn:
                    if not self.graph.has_edge(i, j):
                        self.graph.add_edge(
                            i,
                            j,
                            edge_attr_dict={'weight': self.dist_matrix[i, j]}
                        )
        self.length_best, self.length_worst = self.estimate_bounds(self.max_sweeps)
        self.action_list = list(range(self.num_cities))
        self.semantic_action_list = self.build_semantic_action_list()
        super().post_init()
        self.reset()

    def set_optimal_length(self, optimal_length):
        # Set the optimal length of the episode based on the number of VMs and PMs
        self.length_best = optimal_length  # For example, twice the number of VMs

    def build_semantic_action_list(self):
        centroid = np.mean(self.city_coords, axis=0)
        distances = np.linalg.norm(
            self.city_coords - centroid,
            axis=1
        )
        return np.argsort(distances).tolist()

    def _normalized_coords(self):
        coords = self.city_coords.copy()

        # Center at origin
        coords -= coords.mean(axis=0)

        # Scale to unit variance
        std = coords.std(axis=0)
        std[std < 1e-8] = 1.0
        coords /= std

        return coords

    def _tour_length(self, tour):

        length = 0.0

        for i in range(len(tour) - 1):
            u = tour[i]
            v = tour[i + 1]

            d = self.dist_matrix[u, v]

            length += d

        d = self.dist_matrix[tour[-1], tour[0]]

        length += d

        return length

    def _two_opt_swap(self, tour, i, j):
        # Perform 2-opt swap by reversing the segment between indices i and j
        new_tour = tour.copy()
        new_tour[i:j + 1] = new_tour[i:j + 1][::-1]
        return new_tour

    def estimate_bounds(self, samples=1000, local_steps=20, candidate_pairs=20):
        cities = np.arange(self.num_cities)
        best_length = np.inf
        worst_length = -np.inf

        for _ in tqdm(range(samples), desc="Estimating bounds"):

            tour = np.random.permutation(cities)
            base_length = self._tour_length(tour)


            # sanity check
            assert len(np.unique(tour)) == self.num_cities
            assert len(tour) == self.num_cities

            ####################################################################
            # BEST
            ####################################################################

            best_tour = tour.copy()
            best_local = base_length

            for step in range(local_steps):

                candidates = []

                for pair in range(candidate_pairs):

                    i, j = sorted(random.sample(range(1, self.num_cities), 2))

                    new_tour = self._two_opt_swap(best_tour, i, j)
                    new_len = self._tour_length(new_tour)

                    gain = best_local - new_len

                    if gain > 0:
                        candidates.append((new_tour, new_len, gain))

                if not candidates:
                    break

                chosen = random.choices(
                    candidates,
                    weights=[gain for _, _, gain in candidates],
                    k=1,
                )[0]

                best_tour, best_local, gain = chosen



            ####################################################################
            # WORST
            ####################################################################

            worst_tour = tour.copy()
            worst_local = base_length

            for step in range(local_steps):

                candidates = []

                for pair in range(candidate_pairs):

                    i, j = sorted(random.sample(range(1, self.num_cities), 2))

                    new_tour = self._two_opt_swap(worst_tour, i, j)
                    new_len = self._tour_length(new_tour)

                    gain = new_len - worst_local

                    if gain > 0:
                        candidates.append((new_tour, new_len, gain))



                if not candidates:
                    break

                chosen = random.choices(
                    candidates,
                    weights=[gain for _, _, gain in candidates],
                    k=1,
                )[0]

                worst_tour, worst_local, gain = chosen

            best_length = min(best_length, best_local)
            worst_length = max(worst_length, worst_local)


        return best_length, worst_length

    def export_graph(self, path):
        """
        Export the current TSP instance in TSPLIB format.

        The exported instance always contains the complete problem,
        regardless of shortest_K_edges.
        """
        with open(path, "w") as f:
            f.write(f"NAME: tsp_{self.num_cities}\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {self.num_cities}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")

            for i, (x, y) in enumerate(self.city_coords, start=1):
                f.write(f"{i} {x:.10f} {y:.10f}\n")

            f.write("EOF\n")

    def update_config(self, config):
        super().update_config(config)
        self.reward_type = config.get('reward_type', self.reward_type)
        self.hard_constraints = config.get('hard_constraints', self.hard_constraints) # self.hard_constraints)
        self.remove_invalid_actions = config.get('remove_invalid_actions', self.remove_invalid_actions)
        self.shortest_K_edges = config.get('shortest_K_edges', self.shortest_K_edges)
        self.invalid_action_penalty_sum = config.get('invalid_action_penalty_sum', self.invalid_action_penalty_sum)
        # no need to recompute action space as it will remain the same list

    def _generate_cities(self):
        self.city_coords = np.random.uniform(
            0, self.max_coord, size=(self.num_cities, 2)
        )

    def _compute_distance_matrix(self):
        self.dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i, j in itertools.product(range(self.num_cities), repeat=2):
            self.dist_matrix[i, j] = np.linalg.norm(
                self.city_coords[i] - self.city_coords[j]
            )

    def reset(self, **kwargs):
        super().pre_reset()
        self.visited = np.zeros(self.num_cities, dtype=bool)
        # Start from random city
        self.start_city = np.random.randint(self.num_cities)
        self.current_city = self.start_city

        self.tour = [self.start_city]
        self.visited[self.start_city] = True

        self.current_length = 0.0
        self.prev_length = 0.0


        self.current_length_worst_case = self.length_worst
        self.prev_length_worst_case = self.length_worst


        # Rebuild graph with updated visited status
        if not hasattr(self, 'graph'):
            self.graph = nx.Graph()
            for i in range(self.num_cities):
                self.graph.add_node(i, x_dict={'x': self.city_coords[i][0], 'y': self.city_coords[i][1],
                                               'visited': self.visited[i], 'current': False})
            if self.shortest_K_edges is None:
                for i, j in itertools.combinations(range(self.num_cities), 2):
                    self.graph.add_edge(i, j, edge_attr_dict={'weight': self.dist_matrix[i, j]})
            else:
                for i in range(self.num_cities):
                    knn = np.argsort(self.dist_matrix[i])[1:self.shortest_K_edges + 1]
                    for j in knn:
                        if not self.graph.has_edge(i, j):
                            self.graph.add_edge(
                                i,
                                j,
                                edge_attr_dict={'weight': self.dist_matrix[i, j]}
                            )
        for i in range(self.num_cities):
            self.graph.nodes[i]["x_dict"]["visited"] = False
            self.graph.nodes[i]["x_dict"]["current"] = False

        self.graph.nodes[self.start_city]["x_dict"]["visited"] = True
        self.graph.nodes[self.start_city]["x_dict"]["current"] = True

        self.obs = self._get_obs()
        self.info = {}
        self.post_reset()
        return self.obs, self.info

    def _get_obs(self):
        # Observation consists of coordinates, visited status, and flattened upper triangle of distance matrix
        coords_flat = self.city_coords.flatten()
        visited_float = self.visited.astype(np.float32)
        current_city_float = np.zeros(self.num_cities, dtype=np.float32)
        current_city_float[self.current_city] = 1.0
        weights_flat = []
        for i, j in itertools.combinations(range(self.num_cities), 2):
            weights_flat.append(self.dist_matrix[i, j])
        obs = np.concatenate([coords_flat, visited_float, current_city_float, weights_flat])
        return {"graph": obs.astype(np.float32)}

    def is_valid_action(self, node_id):
        # In hard constraints mode, only unvisited cities are valid actions. In normal mode, all cities are valid (but visiting visited cities will be penalized).
        if node_id >= self.num_cities:
            return False
        if self.visited[node_id]:
            return False
        return True

    def step(self, action):
        action = self.pre_step(action)

        if not self.invalid_action and not self.no_action:
            if self.hard_constraints:
                self.done = False
                if action < self.num_cities and self.visited[action]:
                    duplicate_visit = True
                else:
                    duplicate_visit = False
                if duplicate_visit:
                    self.reward = self.invalid_action_penalty_sum / min(
                        self.max_steps_overall,
                        self.max_steps_coefficient * self.num_cities
                    )
                elif not self.invalid_action and not self.no_action:
                    # make city visited and update tour length
                    next_city = action
                    dist = self.dist_matrix[self.current_city, next_city]
                    self.prev_length = self.current_length
                    self.current_length += dist

                    self.visited[next_city] = True
                    self.tour.append(next_city)
                    self.graph.nodes[self.current_city]['x_dict']['current'] = False
                    self.current_city = next_city

                    self.graph.nodes[next_city]['x_dict']['visited'] = True
                    self.graph.nodes[self.current_city]['x_dict']['current'] = True


                    all_visited_at_least_once = all(self.visited)

                    if len(self.tour) >= self.num_cities:
                        self.done = True
                        if all_visited_at_least_once:
                            # Valid solution → evaluate tour quality
                            self.current_length += self.dist_matrix[self.current_city, self.start_city]

                            if self.reward_type == "relative":
                                self.reward = (
                                        (self.length_worst - self.current_length)
                                        / (self.length_worst - self.length_best + 1e-8)
                                )
                            else:
                                self.reward += -self.current_length

                        else:
                            # Did not visit all cities → strong penalty
                            self.reward = self.invalid_action_penalty_sum / min(
                                self.max_steps_overall,
                                self.max_steps_coefficient * self.num_cities
                            )
            else:
                # soft constraints mode: all actions are valid, but visiting already visited cities will be penalized in the reward
                next_city = action

                dist = self.dist_matrix[self.current_city, next_city]
                self.prev_length = self.current_length
                self.current_length += dist

                self.visited[next_city] = True
                self.tour.append(next_city)
                self.graph.nodes[self.current_city]['x_dict']['current'] = False
                self.current_city = next_city

                self.graph.nodes[next_city]['x_dict']['visited'] = True
                self.graph.nodes[next_city]['x_dict']['current'] = True

                remaining_nodes = [i for i in range(self.num_cities) if not self.visited[i]]
                worst_case_remaining = 0.0
                current = self.current_city

                if len(self.tour) == self.num_cities:
                    self.current_length += self.dist_matrix[self.current_city, self.start_city]

                temp_remaining = remaining_nodes.copy()
                while temp_remaining:
                    farthest_node = max(temp_remaining, key=lambda x: self.dist_matrix[current, x])
                    worst_case_remaining += self.dist_matrix[current, farthest_node]
                    current = farthest_node
                    temp_remaining.remove(farthest_node)

                if remaining_nodes:
                    worst_case_remaining += self.dist_matrix[current, self.start_city]

                self.current_length_worst_case = self.current_length + worst_case_remaining

                if self.reward_type == "relative":
                    self.reward = (
                        self.prev_length_worst_case - self.current_length_worst_case
                    ) / (self.length_worst - self.length_best + 1e-8)
                else:
                    self.reward = -self.current_length

                self.prev_length_worst_case = self.current_length_worst_case

        self.done = len(self.tour) == self.num_cities
        self.obs = self._get_obs()
        self.info = {}
        super().post_step()
        return self.obs, self.reward, self.done, self.truncated, self.info

    def update_metrics(self):
        super().update_metrics()
        # relative performance in [0,1], using worst-case projection
        if self.hard_constraints:
            if (not len(self.tour) == self.num_cities) or (not all(self.visited)):
                relative_performance = 0.0
            else:
                relative_performance = (self.length_worst - self.current_length) / (self.length_worst - self.length_best + 1e-8)
        else:
            relative_performance = (self.length_worst - self.current_length_worst_case) / (self.length_worst - self.length_best + 1e-8)

        self._metrics.update({
            "all_visited": all(self.visited),
            "relative_performance": relative_performance,
            "current_tour_length": self.current_length,
            "current_worst_case_length": self.current_length_worst_case,
            "visited_ratio": np.sum(self.visited) / self.num_cities,
            "tour_completed": len(self.tour) == self.num_cities
        })


class ExtendedTSPEnv(TSPEnv, ContinuousEnv):
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
        if self.remove_invalid_actions:
            self._action_type = {
                'city': Node(graph_name='graph', spec={'visited': False})
            }
        else:
            self._action_type = {
                'city': Node(graph_name='graph')
            }

        self._node_attributes = {
            'x': Attribute(feature_extractor=IdentityEncoding,
                            encoding=ContinuousValue(
                                normalization="min_max",
                                min_value=0,
                                max_value=self.max_coord
                            ),
                            embedding_size=1,
                            graph_name='graph'),
            'y': Attribute(feature_extractor=IdentityEncoding,
                            encoding=ContinuousValue(
                                normalization="min_max",
                                min_value=0,
                                max_value=self.max_coord
                            ),
                            embedding_size=1,
                            graph_name='graph'),
            'visited': Attribute(feature_extractor=IdentityEncoding,
                            encoding=BinaryValue(),
                            embedding_size=1,
                            graph_name='graph'),
            'current': Attribute(
                feature_extractor=IdentityEncoding,
                encoding=BinaryValue(),
                embedding_size=1,
                graph_name='graph'
            ),
        }
        # compute min_weight and max_weights
        self.min_weight = np.min(self.dist_matrix)
        self.max_weight = np.max(self.dist_matrix)
        self._edge_attributes = {
            'weight': Attribute(feature_extractor=IdentityEncoding,
                                    encoding=ContinuousValue(
                                        normalization="min_max",
                                        min_value=self.min_weight,
                                        max_value=self.max_weight
                                    ),
                                    embedding_size=1,
                                    graph_name='graph'),
        }
        if self.no_action_support:
            self.discrete_actions = ["no_action"]
        else:
            self.discrete_actions = []
        if self.remove_invalid_actions:
            self.action_candidates_reconstruction = True
        else:
            self.action_candidates_reconstruction = False
        self.action_space_reconstruction_each_step = True
        self.reconstruct_only_changed_nodes = False

    def sample_valid_action(self):
        # Randomly select valid action
        valid_actions = []
        for city in range(self.num_cities):
            if not self.visited[city]:
                valid_actions.append(city)
        if not valid_actions:
            valid_actions = list(range(self.num_cities))  # if all visited, all actions are valid (will be penalized in step)
        return random.choice(valid_actions)

    def get_graphs(self):
        return {'graph': self.graph}