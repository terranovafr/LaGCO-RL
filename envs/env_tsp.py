#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

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

        obs_len = self.num_cities * 2 + self.num_cities + (self.num_cities * (self.num_cities - 1)) // 2
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
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.start_city = random.randint(0, self.num_cities - 1)
        self.visited[self.start_city] = True
        self.tour = [self.start_city]
        self.bounds_per_start_city = {}
        self.length_best, self.length_worst = self.estimate_bounds(self.max_sweeps)
        self.bounds_per_start_city[self.start_city] = (self.length_best, self.length_worst)
        self.action_list = list(range(self.num_cities))
        self.semantic_action_list = self.build_semantic_action_list()
        super().post_init()
        self.reset()

    def build_semantic_action_list(self):
        # order nodes based on distance from current city (closest first)
        cities = np.arange(self.num_cities)
        other_cities = np.delete(cities, self.start_city)  # exclude start city
        distances = self.dist_matrix[self.start_city, other_cities]
        sorted_indices = np.argsort(distances)
        sorted_cities = other_cities[sorted_indices]
        return sorted_cities.tolist()

    def _tour_length(self, tour):
        # Calculate total length of the given tour (including return to start)
        length = 0.0
        for i in range(len(tour) - 1):
            length += self.dist_matrix[tour[i], tour[i + 1]]
        length += self.dist_matrix[tour[-1], tour[0]]
        return length

    def _two_opt_swap(self, tour, i, j):
        # Perform 2-opt swap by reversing the segment between indices i and j
        new_tour = tour.copy()
        new_tour[i:j + 1] = new_tour[i:j + 1][::-1]
        return new_tour

    def estimate_bounds(self, samples=1000, local_steps=20, candidate_pairs=20):
        # Heuristic-driven estimation of best and worst tour lengths using randomized 2-opt local search
        cities = np.arange(self.num_cities)
        other_cities = np.delete(cities, self.start_city)

        best_length = np.inf
        worst_length = -np.inf

        for _ in tqdm(range(samples), desc="Estimating bounds"):
            perm = np.random.permutation(other_cities)
            tour = np.concatenate(([self.start_city], perm))

            base_length = self._tour_length(tour)

            # ----- best: randomized improving 2-opt -----
            best_tour = tour.copy()
            best_local = base_length

            for _ in range(local_steps):
                candidates = []
                for _ in range(candidate_pairs):
                    i, j = sorted(random.sample(range(1, self.num_cities), 2))  # keep start fixed
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
                    k=1
                )[0]
                best_tour, best_local, _ = chosen

            # ----- worst: randomized worsening 2-opt -----
            worst_tour = tour.copy()
            worst_local = base_length

            for _ in range(local_steps):
                candidates = []
                for _ in range(candidate_pairs):
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
                    k=1
                )[0]
                worst_tour, worst_local, _ = chosen

            best_length = min(best_length, best_local)
            worst_length = max(worst_length, worst_local)

        return best_length, worst_length

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
        self.start_city = random.randint(0, self.num_cities - 1)
        self.visited[self.start_city] = True
        self.tour = [self.start_city]
        self.current_city = self.start_city
        # Rebuild graph with updated visited status
        self.graph = nx.Graph()
        for i in range(self.num_cities):
            self.graph.add_node(i, x_dict={'x': self.city_coords[i][0], 'y': self.city_coords[i][1], 'visited': self.visited[i]})
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
        self.tour = [self.start_city]
        self.current_length = 0.0
        self.prev_length = 0.0
        self.prev_length_worst_case = 0.0
        self.current_length_worst_case = self.length_worst
        self.obs = self._get_obs()
        self.info = {}
        self.post_reset()
        return self.obs, self.info

    def _get_obs(self):
        # Observation consists of coordinates, visited status, and flattened upper triangle of distance matrix
        coords_flat = self.city_coords.flatten()
        visited_float = self.visited.astype(np.float32)
        weights_flat = []
        for i, j in itertools.combinations(range(self.num_cities), 2):
            weights_flat.append(self.dist_matrix[i, j])
        obs = np.concatenate([coords_flat, visited_float, weights_flat])
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
                    self.current_city = next_city

                    self.graph.nodes[next_city]['x_dict']['visited'] = True

                    all_visited_at_least_once = all(self.visited)

                    if len(self.tour) >= self.num_cities:
                        self.done = True
                        if all_visited_at_least_once:
                            # Valid solution → evaluate tour quality
                            self.current_length += self.dist_matrix[self.current_city, self.start_city]

                            if self.reward_type == "relative":
                                self.reward += (
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
                self.current_city = next_city

                self.graph.nodes[next_city]['x_dict']['visited'] = True

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