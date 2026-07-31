#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    agent_training_wrapper.py
    Wrapper that encodes the graph using a GNN and defines a continuous action space based on the action types defined in the environment.
'''

from gymnasium import spaces
import torch
import faiss
from typing import Union
from torch_geometric.utils import from_networkx
from itertools import product
from utils.model import Graph, Node, Edge, Path, Object, SubGraph, NonExistingEdge, Function, ActionSpace
from wrappers.wrapper import GraphWrapper
from utils.distance_utils import sample_farthest_k_points
from utils.pooling_functions import ConcatPooling
from utils.heuristics import get_function_from_protoknnconfig
from utils.hash_utils import is_hashable, make_hashable
from utils.plot_utils import plot_umap_actions, compute_space_cmp_stats
from utils.cache_utils import load_action_cache, save_action_cache
from wrappers.wrapper import ContinuousEnv
from envs.env import DiscreteEnv
from gae.model import GAEEncoder, VGAEEncoder
import time
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from itertools import combinations
import networkx as nx
import random
from utils.feature_extractors import OneHotEncoding
from utils.pooling_functions import compute_pooled_bounds
import os
import numpy as np

# global attribute used to do periodic plotting across many class objects
num_iterations = 0
cmp_at_k_collection = []

class AgentTrainingWrapper(GraphWrapper):
    def __init__(self,
                 env: [ContinuousEnv, DiscreteEnv],
                 env_name,
                 gnn_models,
                 logs_folder,
                 epsilon_lower=1,
                 epsilon_upper=1,
                 approximate_distance=False,
                 distance_metric='euclidean',
                 index_type='flat',
                 IVF_nlist=100,
                 norm_action_space=False,
                 distance_temperature=0,
                 knn = 1,
                 protoknn_heuristic: [Union[dict, float], str, None] = None,
                 pca_percentage_target=1,
                 pca_minimum_without_loss=False,
                 remove_non_valid_actions=False,
                 sample_subset_actions=100,
                 sample_subset_actions_strategy="random",  # "random" or "farthest"
                 plot_action_space=False,
                 plot_action_space_interval=1000,
                 gae_model_type='gae',
                 vgae_projection='mean',
                 no_action_factor=0,
                 tau_quartile=100,
                 algorithm_type="projection",
                 GNN_observations=False,
                 concatenate_actions_observation=False,
                 use_feature_vectors=False,
                 distance_computation=False,
                 **config):
        super().__init__(env, **config)
        self.gnn_models = gnn_models
        self.env = env
        self.env_name = env_name
        self.algorithm_type = algorithm_type
        self.GNN_observations = GNN_observations
        self.logs_folder = logs_folder

        self.node_embeddings = {}

        self.gae_model_type = gae_model_type
        self.vgae_projection = vgae_projection
        self.use_feature_vectors = use_feature_vectors
        self.distance_computation = distance_computation

        self.action_lower_bounds = None
        self.action_upper_bounds = None
        self.continuous_action_pca = None
        self.discrete_actions = None

        self._new_action_entries_since_save = 0
        self._current_action_cache_file = None
        self._action_lookup_cache = {}

        self.candidate_action_KNNs = None
        self.last_knn_candidates = None  # np.array of shape [K, proto_dim]
        self.last_selected_knn_index = None  # int
        self.local_num_iterations = 0
        self.remove_non_valid_actions = remove_non_valid_actions
        self.plot_action_space = plot_action_space
        self.plot_action_space_interval = plot_action_space_interval
        self.epsilon_lower = epsilon_lower
        self.epsilon_upper = epsilon_upper
        self.approximate_distance = approximate_distance # whether to use approximate distance (e.g. via FAISS) for nearest neighbor search in the action space, or compute exact distances
        self.distance_metric = distance_metric
        self.index_type = index_type
        self.IVF_nlist = IVF_nlist
        self.norm_action_space = norm_action_space
        self.distance_temperature = distance_temperature # temperature for scaling distances when using them as similarities (e.g. in softmax weighting), only applied if > 0
        self.knn = knn # number of nearest neighbors to retrieve for ProtoKNN
        self.protoknn_heuristic = protoknn_heuristic  # heuristic to use for protoknn
        self.pca_percentage_target = pca_percentage_target  # number of PCA components to use for action space
        self.pca_minimum_without_loss = pca_minimum_without_loss # whether to compute minimum number of PCA components needed to preserve all explained variance
        self.sample_subset_actions = sample_subset_actions  # number of actions to sample from the action set
        self.sample_subset_actions_strategy = sample_subset_actions_strategy
        self.concatenate_actions_observation = concatenate_actions_observation # whether to concatenate action vectors in the observation vector (only if sample_subset_actions > 0, which means number of actions has a maximum)
        self.action_candidates = {}

        self.observation_space_creation_time = 0
        self.action_space_creation_time = 0
        self.action_set_construction_time = 0
        self.attribute_encoding_time = 0
        self.graph_encoding_time = 0
        self.action_mapping_time = 0

        Gs = self.env.get_graphs()
        encoded_Gs = {}
        for G_name in Gs:
            G = Gs[G_name]
            G, _, _ = self.attribute_encoding(G, G_name)
            encoded_Gs[G_name] = G
            self.encode(G, G_name)

        self.no_action_factor = no_action_factor # factor to determine position of where to place no action point if desired
        self.tau_quartile = tau_quartile # quartile of distance distribution to use as tau for no action point if no_action_factor is set

        if self.algorithm_type == "discrete" and self.GNN_observations:
            self.action_space = env.action_space
        elif self.algorithm_type == "projection" or self.algorithm_type == "iterative":
            self.action_set, self.action_lower_bounds, self.action_upper_bounds = self.create_continuous_action_space(encoded_Gs)
            self.action_set_reversed = {tuple(v): k for k, v in self.action_set.items()} # used to retrieve action discrete ID tuple
            self.define_action_space(self.action_lower_bounds, self.action_upper_bounds, Gs, self.epsilon_lower, self.epsilon_upper)
        self.define_observation_space()

        self.update_metrics()



    def set_continuous_bounds(self, action_lower_bounds, action_upper_bounds):
        # update action lower and upper bounds and redefine action space accordingly
        self.define_action_space(action_lower_bounds, action_upper_bounds, self.epsilon_lower,
                                 self.epsilon_upper)
        # redefine observation space as it may have action space summary and has to update dimensions
        self.define_observation_space()

    def set_continuous_pca_object(self, pca):
        self.pca_object = pca
        self.perform_pca_reduction()

    def perform_pca_reduction(self):
        # reduction of action space dimensions via PCA, applied if pca_percentage_target < 1.0 or pca_minimum_without_loss is True
        if not hasattr(self, 'pca_object') or self.pca_object is None:
            return
        if (self.pca_percentage_target is not None and self.pca_percentage_target < 1.0) or \
            getattr(self, 'pca_minimum_without_loss', False):
            all_actions = list(self.action_set.values())
            all_actions_np = np.array(all_actions, dtype=np.float32)
            all_actions_np = self.pca_object.transform(all_actions_np)
            for i, aid in enumerate(self.action_set):
                self.action_set[aid] = all_actions_np[i]
            # assert that all have same dimensions
            first_dim = len(self.action_set[list(self.action_set.keys())[0]])
            for aid in self.action_set:
                assert len(self.action_set[aid]) == first_dim, "PCA transformed actions have inconsistent dimensions"

            lower_bounds = np.min(all_actions_np, axis=0)
            upper_bounds = np.max(all_actions_np, axis=0)

            if self.norm_action_space == "min-max":
                range_bounds = upper_bounds - lower_bounds
                range_bounds[range_bounds == 0] = 1e-8  # prevent divide-by-zero

                for aid in self.action_set:
                    self.action_set[aid] = (self.action_set[aid] - lower_bounds) / range_bounds

                # new action lower and upper bounds
                self.action_lower_bounds = np.array([0] * len(lower_bounds), dtype=np.float32)
                self.action_upper_bounds = np.array([1] * len(upper_bounds), dtype=np.float32)

            elif self.norm_action_space == "z-score":
                mean = np.mean(all_actions_np, axis=0)
                std = np.std(all_actions_np, axis=0)
                std[std == 0] = 1e-8  # prevent divide-by-zero

                for aid in self.action_set:
                    self.action_set[aid] = (self.action_set[aid] - mean) / std

                # compute new action lower and upper bounds as arrays
                self.action_lower_bounds = -3 * np.ones_like(mean)
                self.action_upper_bounds = 3 * np.ones_like(mean)
            self.action_set_reversed = {tuple(v): k for k, v in self.action_set.items()}
            self.action_space = spaces.Box(low=self.action_lower_bounds, high=self.action_upper_bounds, dtype=np.float32)

    def set_discrete_actions(self, discrete_action_list):
        # add discrete actions
        if hasattr(self.env, 'discrete_actions') and (self.env.discrete_actions is not None) and len(self.env.discrete_actions) > 0:
            for discrete_action in self.env.discrete_actions:
                discrete_action_aid = (("global", discrete_action),)
                self.action_set[discrete_action_aid] = discrete_action_list[("global", discrete_action)]
                self.action_set_reversed[tuple(discrete_action_list[("global", discrete_action)])] = discrete_action_aid

    def get_discrete_actions(self):
        return self.env.discrete_actions

    def _brute_force_search(self, query_vec, candidate_vecs, knn):
        # compute distances between query_vec and each vector in candidate_vecs using the specified distance metric
        if self.distance_computation == 'gpu':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.distance_metric == 'euclidean':
                candidates = torch.tensor(candidate_vecs, device=device)
                query = torch.tensor(query_vec, device=device)
                dists = torch.cdist(query.unsqueeze(0), candidates)
                vals, idx = torch.topk(dists, knn, largest=False)
                return vals.cpu().numpy(), idx.cpu().numpy()
            elif self.distance_metric == 'cosine':
                candidates = torch.tensor(candidate_vecs, device=device, dtype=torch.float32)  # (N,d)
                query = torch.tensor(query_vec, device=device, dtype=torch.float32).unsqueeze(0)  # (1,d)
                candidates = torch.nn.functional.normalize(candidates, p=2, dim=1)  # (N,d)
                query = torch.nn.functional.normalize(query, p=2, dim=1)  # (1,d)
                sims = query @ candidates.T  # shape (1,N)
                dists = 1.0 - sims
                # top-k along last dimension
                if knn >= len(candidate_vecs):
                    knn = len(candidate_vecs)
                vals, idx = torch.topk(dists, knn, largest=False, dim=1)  # shape (1,knn)
                # convert to numpy
                vals = vals.cpu().numpy()
                idx = idx.cpu().numpy()
                return vals, idx
        else: # cpu python-based computation
            dists = None
            if self.distance_metric == 'euclidean':
                dists = np.linalg.norm(candidate_vecs - query_vec, axis=1)
            elif self.distance_metric  == 'manhattan':
                dists = np.sum(np.abs(candidate_vecs - query_vec), axis=1)
            elif self.distance_metric  == 'chebyshev':
                dists = np.max(np.abs(candidate_vecs - query_vec), axis=1)
            elif self.distance_metric  == 'cosine':
                qn = np.linalg.norm(query_vec) + 1e-12
                cn = np.linalg.norm(candidate_vecs, axis=1) + 1e-12
                dists = 1.0 - (candidate_vecs @ query_vec) / (cn * qn)

            indices = np.argsort(dists)[:knn]
            dists_sorted = dists[indices]
            return dists_sorted.reshape(1, -1), indices.reshape(1, -1)

    def set_knn(self, knn):
        self.knn = knn

    def define_observation_space(self):
        # define the observation space based on the observation types defined in the environment, including dimensions for GNN-based observations and any additional features or summaries as specified. For GNN-based observations, compute the dimensions based on the embedding sizes of the GNN models and the pooling operations applied.
        # For function-based observations, compute the output dimensions based on the function outputs and whether they are applied to specific graphs or all graphs.
        # For action space summary observations, compute the dimensions based on the pooling operations applied to the action space. Store the resulting observation space as a gym.spaces.Dict in self.observation_space and also keep track of a flat dimension count in self.observation_space_flat_dim for use in models that require a flat input.
        start_time = time.time()
        self.observation_space_dict = {}
        self.observation_space_flat_dim = 0
        for key in self.env._observation_type:
            dimensions = 0
            if isinstance(self.env._observation_type[key], Graph):
                if self.env._observation_type[key].graph_name != None:
                    for graph in self.env._observation_type[key].graph_name:
                         model = self.gnn_models[graph]
                         dimensions += model.embedding_dim

                else:
                    # all of them
                    for G_name in self.gnn_models:
                        model = self.gnn_models[G_name]
                        dimensions += model.embedding_dim
                dimensions *= len(self.env._observation_type[key].poolings)
                self.observation_space_flat_dim += dimensions
                # check if otherwise is a simple method
                self.observation_space_dict[key] = spaces.Box(low=-np.inf, high=np.inf,
                                                              shape=(dimensions,), dtype=np.float32)
            elif isinstance(self.env._observation_type[key], Function):
                Gs = self.env.get_graphs()
                G_sample = Gs[list(Gs.keys())[0]]
                sample_result = self.env._observation_type[key].func(G_sample)
                if self.env._observation_type[key].graph_name != None:
                    if isinstance(sample_result, (list, np.ndarray)):
                        output_dim = len(sample_result)
                    elif isinstance(sample_result, (int, float)):
                        output_dim = 1
                    else:
                        raise ValueError(f"Unsupported observation type for {key}: {type(sample_result)}")
                else:
                    if isinstance(sample_result, (list, np.ndarray)):
                        output_dim = len(sample_result) * len(Gs)
                    elif isinstance(sample_result, (int, float)):
                        output_dim = 1 * len(Gs)
                    else:
                        raise ValueError(f"Unsupported observation type for {key}: {type(sample_result)}")
                self.observation_space_flat_dim += output_dim
                self.observation_space_dict[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(output_dim,), dtype=np.float32)
            elif isinstance(self.env._observation_type[key], ActionSpace) and (self.algorithm_type == "projection" or self.algorithm_type == "iterative"):
                # add as much features as needed for pooling the action space
                dimensions = 0
                if self.concatenate_actions_observation and self.sample_subset_actions > 0:
                    dimensions += self.action_space.shape[0] * self.sample_subset_actions

                bounds_low_parts = []
                bounds_high_parts = []

                # If concatenating sampled actions directly
                if self.concatenate_actions_observation and self.sample_subset_actions > 0:
                    for _ in range(self.sample_subset_actions):
                        bounds_low_parts.append(self.action_lower_bounds)
                        bounds_high_parts.append(self.action_upper_bounds)

                # For each pooling
                for pooling in self.env._observation_type[key].poolings:
                    low_p, high_p = compute_pooled_bounds(pooling, self.action_lower_bounds, self.action_upper_bounds, self.sample_subset_actions)
                    bounds_low_parts.append(low_p)
                    bounds_high_parts.append(high_p)
                    dimensions += len(low_p)

                low = np.concatenate(bounds_low_parts, axis=0)
                high = np.concatenate(bounds_high_parts, axis=0)

                self.observation_space_flat_dim += dimensions
                self.observation_space_dict[key] = spaces.Box(
                    low=low,
                    high=high,
                    shape=(dimensions,),
                    dtype=np.float32
                )
        self.observation_space = spaces.Dict(self.observation_space_dict)
        self.observation_space_creation_time = time.time() - start_time


    def define_action_space(self, lower_bounds, upper_bounds, Gs, epsilon_lower=1, epsilon_upper=1):
        # Define the continuous action space based on the action types defined in the environment and the embedding sizes of the GNN models.
        # For each action type, compute the dimensions of the corresponding part of the action space based on the pooling operations applied to the GNN embeddings and any additional features concatenated.
        # Store the resulting action space as a gym.spaces.Box in self.action_space and also keep track of the action components and their indices for use in mapping between continuous action vectors and discrete actions.
        start_time = time.time()
        dimensions = 0
        self.action_components = []
        current_offset = 0

        if (self.pca_percentage_target and self.pca_percentage_target < 1) or \
            getattr(self, 'pca_minimum_without_loss', False):
            pass
        elif self.use_feature_vectors:
            pass
        else:
            embedding_sizes = {}
            for G_name in self.gnn_models:
                model = self.gnn_models[G_name]
                embedding_sizes[G_name] = model.embedding_dim
            for key in self.env._action_type:
                if isinstance(self.env._action_type[key], Node):
                    emb_dim = (
                        embedding_sizes[self.env._action_type[key].graph_name[0]]
                        if self.env._action_type[key].graph_name
                        else max(embedding_sizes.values())
                    )

                    self.action_components.append({
                        "name": f"{key}_node_embedding",
                        "type": "continuous",
                        "indices": list(range(current_offset, current_offset + emb_dim))
                    })
                    current_offset += emb_dim
                    dimensions += emb_dim

                    if self.env._action_type[key].concat_features is not None:
                        for feature in self.env._action_type[key].concat_features:
                            feat_dim = self.env._node_attributes[feature].embedding_size
                            self.action_components.append({
                                "name": f"{key}_node_feature_{feature}",
                                "type": "continuous",
                                "indices": list(range(current_offset, current_offset + feat_dim))
                            })
                            current_offset += feat_dim
                            dimensions += feat_dim


                elif isinstance(self.env._action_type[key], Edge):
                    edge_embedding_size = 0
                    for pooling in self.env._action_type[key].poolings:
                        if callable(pooling) and pooling == ConcatPooling:
                            edge_embedding_size += (
                                embedding_sizes[self.env._action_type[key].graph_name[0]] * 2
                                if self.env._action_type[key].graph_name
                                else max(embedding_sizes.values()) * 2
                            )
                            break
                        else:
                            edge_embedding_size += (
                                embedding_sizes[self.env._action_type[key].graph_name[0]]
                                if self.env._action_type[key].graph_name
                                else max(embedding_sizes.values())
                            )
                    self.action_components.append({
                        "name": f"{key}_edge_embedding",
                        "type": "continuous",
                        "indices": list(range(current_offset, current_offset + edge_embedding_size))
                    })

                    current_offset += edge_embedding_size
                    dimensions += edge_embedding_size
                    if self.env._action_type[key].concat_node_features is not None:
                        for feature in self.env._action_type[key].concat_node_features:
                            feat_dim = self.env._node_attributes[feature].embedding_size
                            self.action_components.append({
                                "name": f"{key}_edge_node_feature_{feature}",
                                "type": "continuous",
                                "indices": list(range(current_offset, current_offset + feat_dim))
                            })
                            current_offset += feat_dim
                            dimensions += feat_dim
                elif isinstance(self.env._action_type[key], NonExistingEdge):
                    # defined as concatenation of source and target node embeddings
                    edge_embedding_size = 0
                    for pooling in self.env._action_type[key].poolings:

                        if callable(pooling) and pooling == ConcatPooling:
                            edge_embedding_size += embedding_sizes[self.env._action_type[key].graph_name] * 2 if self.env._action_type[key].graph_name else max(embedding_sizes.values()) * 2
                            break
                        else:
                            edge_embedding_size += embedding_sizes[self.env._action_type[key].graph_name] if self.env._action_type[key].graph_name else max(embedding_sizes.values())
                    dimensions += edge_embedding_size
                elif isinstance(self.env._action_type[key], Path):
                    path_embedding_size = 0
                    for pooling in self.env._action_type[key].poolings:
                        if callable(pooling) and pooling == ConcatPooling:
                            path_embedding_size += (
                                    embedding_sizes[self.env._action_type[key].graph_name[0]]
                                    * (self.env._action_type[key].max_len + 1)
                            )
                            break
                        else:
                            path_embedding_size += embedding_sizes[self.env._action_type[key].graph_name[0]]

                    self.action_components.append({
                        "name": f"{key}_path_embedding",
                        "type": "continuous",
                        "indices": list(range(current_offset, current_offset + path_embedding_size))
                    })
                    current_offset += path_embedding_size
                    dimensions += path_embedding_size
                    if self.env._action_type[key].concat_path_features is not None:
                        for feature in self.env._action_type[key].concat_path_features:
                            if feature == "len":
                                self.action_components.append({
                                    "name": f"{key}_path_len",
                                    "type": "continuous",
                                    "indices": [current_offset]
                                })
                                current_offset += 1
                                dimensions += 1

                elif isinstance(self.env._action_type[key], SubGraph):
                    # Assuming subgraph selection is based on node count
                    subgraph_embedding_size = 0
                    for pooling in self.env._action_type[key].poolings:
                        if callable(pooling) and pooling == ConcatPooling:
                            raise ValueError("ConcatPooling not supported for SubGraph actions")
                        else:
                            subgraph_embedding_size += embedding_sizes[self.env._action_type[key].graph_name] if self.env._action_type[key].graph_name else max(embedding_sizes.values())
                    dimensions += subgraph_embedding_size
                elif isinstance(self.env._action_type[key], Object):
                    obj_dim = self.env._action_type[key].embedding_size
                    if self.env._action_type[key].feature_extractor == OneHotEncoding:
                        self.action_components.append({
                            "name": f"{key}_object_embedding",
                            "type": "onehot",
                            "indices": list(range(current_offset, current_offset + obj_dim))
                        })
                    else:
                        self.action_components.append({
                            "name": f"{key}_object_embedding",
                            "type": "continuous",
                            "indices": list(range(current_offset, current_offset + obj_dim))
                        })
                    current_offset += obj_dim
                    dimensions += obj_dim
                else:
                    raise ValueError(f"Unsupported action type for {key}: {self.env._action_type[key]}")
            assert len(lower_bounds) == dimensions, \
                f"Expected {dimensions} dimensions but got lower_bounds shape {lower_bounds.shape}"
            assert current_offset == dimensions, \
                f"Internal error: offset {current_offset} != dimensions {dimensions}"

        # extend lower bounds with epsilon to consider larger case
        lower_bounds = np.maximum(lower_bounds, -np.inf)
        # extend upper bounds with epsilon to consider larger case
        upper_bounds = np.minimum(upper_bounds, np.inf)

        # add epsilons
        if not self.norm_action_space:
            lower_bounds += epsilon_lower
            upper_bounds += epsilon_upper
        self.action_space = spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.float32)
        self.action_space_creation_time = time.time() - start_time

        if self.no_action_factor:
            all_distances = []
            continuous_actions_array = np.array(list(self.action_set.values()), dtype=np.float32)
            for vec in continuous_actions_array:
                dists, _ = self._brute_force_search(vec, continuous_actions_array, knn=2)
                # Exclude self (first neighbor)
                all_distances.append(dists[0, 1])
            all_distances = np.array(all_distances)
            self.tau = np.percentile(all_distances, self.tau_quartile)
        self.action_lower_bounds = lower_bounds
        self.action_upper_bounds = upper_bounds

    def encode(self, G, G_name):
        # encode the graph using the specified GNN model and store the resulting node embeddings in self.node_embeddings
        start_time = time.time()
        # use the GNN model to encode the graph
        gnn_model = self.gnn_models[G_name]
        data = from_networkx(G)
        with torch.no_grad():
            if isinstance(gnn_model, GAEEncoder):
                node_emb = gnn_model(data.x, data.edge_index, data.edge_attr)
            elif isinstance(gnn_model, VGAEEncoder):
                # distinguish if we have to use mean or full distribution
                if self.vgae_projection == 'mean':
                    _, node_emb, _ = gnn_model(data.x, data.edge_index, data.edge_attr)
                elif self.vgae_projection == 'sample':
                    node_emb, _, _ = gnn_model(data.x, data.edge_index, data.edge_attr)

        new_embeddings = {n: node_emb[i].numpy() for i, n in enumerate(G.nodes())}
        self.node_embeddings[G_name] = new_embeddings
        self.graph_encoding_time += time.time() - start_time


    def create_continuous_action_space(self, Gs, nodes_changed=None):
        # create the continuous action space by checking specifications and concatenating valid combinations
        start_time = time.time()
        G = Gs[list(Gs.keys())[0]]
        if not self._action_lookup_cache:
            self._action_lookup_cache, self._current_action_cache_file = load_action_cache(self.cache_dir)

        def filter_by_spec(items, spec_dict, get_featvec):
            if not spec_dict:
                return items
            return [
                item for item in items
                if all(get_featvec(item).get(k) == v for k, v in spec_dict.items())
            ]

        if self.action_candidates_reconstruction or len(self.action_candidates) == 0:
            self.action_candidates = {}
            for key in self.env._action_type:
                action_type = self.env._action_type[key]
                self.action_candidates[key] = []
                if isinstance(action_type, Node):
                    for G_name in Gs:
                        G = Gs[G_name]
                        if action_type.graph_name != None and G_name not in action_type.graph_name:
                            continue
                        else:
                            candidates = list(G.nodes())
                            #  Filter nodes by spec
                            candidates = filter_by_spec(candidates, action_type.spec, lambda n: G.nodes[n]["x_dict"])
                            # inject in every candidate as first element of the tuple the G_name
                            candidates = [(G_name,n) for n in candidates]
                            self.action_candidates[key] = candidates
                elif isinstance(action_type, Edge):
                    for G_name in Gs:
                        if action_type.graph_name != None and G_name not in action_type.graph_name:
                            continue
                        else:
                            G = Gs[G_name]
                            if action_type.bidirectional:
                            # Keep only one canonical representative per undirected pair
                                candidates = sorted({
                                    (u, v) if u <= v else (v, u)
                                    for (u, v) in G.edges()
                                })
                            else:
                                candidates = list(G.edges())

                            # Filter edges by edge spec
                            candidates = filter_by_spec(candidates, action_type.spec, lambda e: G.edges[e]["edge_attr_dict"])
                            # Filter edge nodes by nodes_spec
                            if action_type.nodes_spec:
                                candidates = [
                                    (u, v) for (u, v) in candidates
                                    if all(G.nodes[n].get(k) == v for k, v in action_type.nodes_spec.items() for n in (u, v))
                                ]
                            # inject in every candidate as first element of the tuple the G_name
                            candidates = [(G_name,u,v) for (u,v) in candidates]
                            self.action_candidates[key] = candidates
                elif isinstance(action_type, NonExistingEdge):
                    for G_name in Gs:
                        if action_type.graph_name != None and G_name not in action_type.graph_name:
                            continue
                        else:
                            G = Gs[G_name]
                            candidates = []
                            nodes = list(G.nodes())
                            for u, v in combinations(nodes, 2):
                                if not G.has_edge(u, v):
                                    candidates.append((u, v))
                            # Filter edge nodes by nodes_spec
                            if action_type.nodes_spec:
                                candidates = [
                                    (u, v) for (u, v) in candidates
                                    if all(G.nodes[n].get(k) == v for k, v in action_type.nodes_spec.items() for n in (u, v))
                                ]
                            # inject in every candidate as first element of the tuple the G_name
                            candidates = [(G_name, u, v) for (u, v) in candidates]
                            self.action_candidates[key] = candidates
                elif isinstance(action_type, Path):
                    for G_name in Gs:
                        if action_type.graph_name != None and G_name not in action_type.graph_name:
                            continue
                        else:
                            G = Gs[G_name]
                            candidates = []
                            # compute all possible paths between
                            if action_type.set_dict:
                                if action_type.set_dict_key_name:
                                    for path_list in action_type.set_dict.values():
                                        for p in path_list:
                                            candidates.append(p)
                                else:
                                    for path in action_type.set_dict:
                                        candidates.append(path)
                            else:
                                # find all paths in G of len up to max_len
                                for path in nx.all_simple_paths(G, cutoff=action_type.max_len):
                                    if len(path) >= action_type.min_len:
                                        candidates.append(path)

                            # Filter edge nodes by nodes_spec
                            if action_type.nodes_spec:
                                candidates = [
                                    path for path in candidates
                                    if all(G.nodes[n].get(k) == v for k, v in action_type.nodes_spec.items() for n in path)
                                ]
                            # inject in every candidate as first element of the tuple the G_name

                            candidates = [(G_name, path) for path in candidates]
                            self.action_candidates[key] = candidates
                elif isinstance(action_type, SubGraph):
                    for G_name in Gs:
                        if action_type.graph_name != None and G_name not in action_type.graph_name:
                            continue
                        else:
                            G = Gs[G_name]
                            node_candidates = filter_by_spec(list(G.nodes()), action_type.nodes_spec, lambda n: G.nodes[n]["x_dict"])
                            size_min = action_type.size_min
                            size_max = action_type.size_max or len(node_candidates)
                            subgraphs = []
                            for i in range(size_min, size_max + 1):
                                subgraphs.extend(combinations(node_candidates, i))
                            # inject in every candidate as first element of the tuple the G_name
                            subgraphs = [(G_name,) + sg for sg in subgraphs]
                            self.action_candidates[key] = subgraphs
                elif isinstance(action_type, Object):
                    # Simply include all elements from the set
                    if not action_type.set:
                        raise ValueError(f"Object action type {key} requires a non-empty `set`")
                    for G_name in Gs:
                        if action_type.reference_graph_name != None and G_name not in action_type.reference_graph_name:
                            continue
                        else:
                            if action_type.set:
                                if action_type.set_dict_key_name:
                                    action_list = []
                                    for obj in action_type.set.values():
                                        action_list.append(obj)
                                else:
                                    action_list = list(action_type.set)
                            else:
                                raise ValueError(f"Object action type {key} requires a non-empty `set`")
                            # inject in every candidate as first element of the tuple the G_name
                            action_list = [(G_name, obj) for obj in action_list]
                            self.action_candidates[key] = action_list
                    else:
                        action_list = list(action_type.set)
                        action_list = [("global", obj) for obj in action_list]
                        self.action_candidates[key] = action_list
                else:
                    raise ValueError(f"Unsupported action type for key '{key}': {action_type}")

            def violates_different_than(combination, key, different_than_keys):
                # Returns True if current combination violates any different_than_keys constraint
                if different_than_keys:
                    for diff_key in different_than_keys:
                        if diff_key in combination and combination[key] == combination[diff_key]:
                            return True
                return False

            def valid_object_binding(combination, key, obj, G):
                # Ensure the object is contained in the FV of the reference_key element
                if obj.reference_key != "global":
                    ref_elem = combination.get(obj.reference_key)
                    if ref_elem is None:
                        return False
                    # get feature vector of the referenced element
                    fv = G.nodes[ref_elem[1]]['x_dict']
                    set_items = fv.get(obj.reference_feature_vector_key, [])
                    if obj.key_to_extract is not None:
                        set_items = [item[obj.key_to_extract] for item in set_items if obj.key_to_extract in item]
                    return combination[key][1] in set_items
                return True

            def valid_set_dict_key_name(combination, key, action_obj):
                # Ensure that if path_obj.set_dict_key_name is set, the path chosen in combination[key] only contains nodes/edges corresponding to the value of the reference key in combination.
                if action_obj.set_dict_key_name is None:
                    return True  # no restriction

                ref_key = action_obj.set_dict_key_name
                ref_value = combination.get(ref_key)
                if ref_value is None:
                    # If the reference key isn't yet selected, we can't validate, assume valid for now
                    return True

                combination = combination[key][1]
                # derive key from action_obj.set_dict_function, removing first element and reprsenting the rest as a tuple
                key_list = []
                for index, elem in enumerate(ref_value):
                    if index == 0:
                        continue
                    key_list.append(elem)
                key = action_obj.set_dict_function(key_list)
                if action_obj.set_dict[key] is not None:
                    valid_objs = action_obj.set_dict.get(key)
                    return combination in valid_objs

            # merge raw products per graph
            filtered_actions = {}
            keys = list(self.action_candidates.keys())
            number_keys = len(keys)
            raw_product = product(*(self.action_candidates[k] for k in keys))
            for combo in raw_product:
                assert len(combo) == number_keys, f"Mismatch in number of keys and combo {combo} length"
                combination = dict(zip(keys, combo))

                # remove graph name from combination for easier processing
                valid = True
                for key, action_type in self.env._action_type.items():
                    # Rule: different_than_keys
                    if violates_different_than(combination, key, getattr(action_type, "different_than_keys", None)):
                        valid = False
                        break

                    # Rule: Edge with source/target node keys
                    if isinstance(action_type, Edge) or isinstance(action_type, NonExistingEdge):
                        # Disallow edge (u, v) from containing any node that appears in a different_than_keys element
                        for other_key in action_type.different_than_keys or []:
                            forbidden_nodes = combination.get(other_key)
                            if forbidden_nodes is None:
                                continue
                            if isinstance(forbidden_nodes, (list, tuple)):
                                if any(n in combination[key] for n in forbidden_nodes):
                                    valid = False
                                    break
                            else:
                                if forbidden_nodes in combination[key]:
                                    valid = False
                                    break
                        if not valid:
                            break

                        if action_type.source_node_key and action_type.target_node_key:
                            source = combination.get(action_type.source_node_key)
                            target = combination.get(action_type.target_node_key)
                            if (source, target) != combination[key] and (target, source) != combination[key]:
                                valid = False
                                break
                        if action_type.source_node_key:
                            source = combination.get(action_type.source_node_key)
                            # check should be as first element in edge tuple
                            if source is None or source != combination[key][0]:
                                valid = False
                                break
                        if action_type.target_node_key:
                            target = combination.get(action_type.target_node_key)
                            if target is None or target != combination[key][1]:
                                valid = False
                                break
                        if action_type.source_node_key_different:
                            source = combination.get(action_type.source_node_key_different)
                            if source is None or source in combination[key]:
                                valid = False
                                break
                        if action_type.target_node_key_different:
                            target = combination.get(action_type.target_node_key_different)
                            if target is None or target in combination[key]:
                                valid = False
                                break

                    # Rule: SubGraph with different_than_keys
                    if isinstance(action_type, SubGraph):
                        for other_key in action_type.different_than_keys or []:
                            other_val = combination.get(other_key)

                            # Normalize other_val to a set of nodes in case of edge
                            if other_val is None:
                                continue
                            elif isinstance(other_val, (list, set, tuple)):
                                other_nodes = set(other_val)
                            else:
                                other_nodes = {other_val}

                            # Check for overlap
                            if set(combination[key]) & other_nodes:
                                valid = False
                                break

                    # Rule: Object with reference_key != global
                    if isinstance(action_type, Object):

                        if not valid_object_binding(combination, key, action_type, G):
                            valid = False
                            break

                    if isinstance(action_type, Path) or isinstance(action_type, Object):
                        if not valid_set_dict_key_name(combination, key, action_type):
                            valid = False
                            break

                if valid:
                    key_name = tuple(make_hashable(v) for k, v in combination.items())
                    # now you can safely use key_name as a dict key
                    filtered_actions[key_name] = combination
                    # convert into tuple
                    key_name = tuple(key_name)
                    # check if env has method is_valid_action
                    if hasattr(self.env, 'is_valid_action') and self.remove_invalid_actions:
                        # remove graph name from combination for easier processing
                        filtered_combination = combination.copy()
                        for elem in filtered_combination:
                            if isinstance(filtered_combination[elem], tuple) and (filtered_combination[elem][0] in Gs or filtered_combination[elem][0] == "global"):
                                if len(filtered_combination[elem]) == 2:
                                    filtered_combination[elem] = filtered_combination[elem][1]
                                else:
                                    filtered_combination[elem] = filtered_combination[elem][1:]
                        if self.env.is_valid_action(*filtered_combination.values()):
                            filtered_actions[key_name] = combination
                    else:
                        filtered_actions[key_name] = combination
            self.action_candidates = filtered_actions
        else:
            filtered_actions = self.action_candidates
        if len(filtered_actions) > 0:
            continuous_actions, lower_bounds, upper_bounds = self.map_actions_to_continuous_space(filtered_actions, self.env._action_type, nodes_changed, Gs)
        else:
            continuous_actions = {}
            lower_bounds = np.array([0 for _ in range(self.action_space.shape[0])])
            upper_bounds = np.array([0 for _ in range(self.action_space.shape[0])])

        global num_iterations
        num_iterations += 1
        if not hasattr(self, 'local_num_iterations'):
            self.local_num_iterations = 0
        self.local_num_iterations += 1
        global cmp_at_k_collection

        # first iteration, compute cmp@k for the whole set before sampling, to have a reference point. Then, if plotting is enabled, keep track of cmp@k for each iteration and plot the evolution every plot_action_space_interval iterations. Note that cmp@k can be computed on the sampled subset or on the whole set, but for consistency we will compute it on the whole set at each iteration to track the true evolution of the action space quality, even if only a subset is used for training.
        if self.local_num_iterations == 1:
            cmp_at_k = compute_space_cmp_stats(continuous_actions)
            cmp_at_k_collection.append(cmp_at_k)

        if self.plot_action_space and (num_iterations != 0 and (num_iterations % self.plot_action_space_interval == 0)):
            # periodically plot the action space using UMAP, coloring points by their discrete action type and optionally showing cmp@k statistics in the title or as annotations. If cmp@k statistics are available for multiple iterations, consider plotting the evolution of these statistics over iterations in a separate plot or as part of the same visualization.
            if cmp_at_k_collection:
                # average should be computed knowing that inside each there wil be a dict indexed with k
                cmp_at_k_dict = {}
                for cmp_at_k in cmp_at_k_collection:
                    for k, v in cmp_at_k.items():
                        if k not in cmp_at_k_dict:
                            cmp_at_k_dict[k] = []
                        cmp_at_k_dict[k].append(v)
                avg_cmp_at_k = {}
                for k in cmp_at_k_dict:
                    avg_cmp_at_k[k] = np.mean(cmp_at_k_dict[k])
                plot_umap_actions(continuous_actions, cmp_at_k=avg_cmp_at_k, logs_folder=os.path.join(self.logs_folder, "plots"),
                                    discrete_actions=self.env.discrete_actions, environment_name=self.env_name,
                                    num_iterations=num_iterations)

        if self.sample_subset_actions and self.sample_subset_actions > 0 and len(continuous_actions) > self.sample_subset_actions:
            # Sample a subset of actions if specified
            if self.sample_subset_actions_strategy == "random":
                sampled_keys = random.sample(list(continuous_actions.keys()), self.sample_subset_actions)
                continuous_actions = {k: continuous_actions[k] for k in sampled_keys}
            elif self.sample_subset_actions_strategy == "farthest":
                continuous_actions = sample_farthest_k_points(continuous_actions, self.sample_subset_actions)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sample_subset_actions_strategy}")
            self.original_set_length = None
        else:
            self.original_set_length = len(continuous_actions)

        self.action_set_construction_time += time.time() - start_time

        return continuous_actions, lower_bounds, upper_bounds


    def map_actions_to_continuous_space(self, actions, action_types, nodes_changed, Gs):
        # map the discrete actions to continuous space by encoding the components of each action according to their type and the specifications
        reconstructed_actions = set()
        count_actions = 0
        reconstructed_parts = 0
        count_parts = 0

        if not hasattr(self, 'action_set_components'):
            self.action_set_components = {}

        def action_involves_nodes(action_value, action_type, nodes_changed):
            # Return True if the action involves any node in nodes_changed.
            graph_name = action_value[0]
            nodes_changed = nodes_changed[graph_name] if graph_name in nodes_changed else set()
            if isinstance(action_type, Node):
                node = action_value[1] if isinstance(action_value, tuple) else action_value
                return node in nodes_changed

            elif isinstance(action_type, Edge) or isinstance(action_type, NonExistingEdge):
                u, v = action_value[1], action_value[2]
                return u in nodes_changed or v in nodes_changed

            elif isinstance(action_type, Path):
                nodes_in_path = action_value[1]  # action_value = (graph_name, path)
                return any(n in nodes_changed for n in nodes_in_path)

            elif isinstance(action_type, SubGraph):
                nodes_in_subgraph = action_value[1:]  # action_value = (graph_name, n1, n2, ...)
                return any(n in nodes_changed for n in nodes_in_subgraph)

            elif isinstance(action_type, Object):
                # If object refers to nodes, implement logic based on reference_key if needed
                return False  # safe default

            return False

        continuous_actions = {}
        # ignored if self.use_feature_vector is True
        for action_id, action in actions.items():
            continuous_action = {}
            count_actions += 1
            for key, action_type in action_types.items():
                count_parts += 1
                if (hasattr(self, 'action_set_components') and
                        self.reconstruct_only_changed_nodes and
                        nodes_changed and
                        not action_involves_nodes(action[key], action_type, nodes_changed) and
                        action_id in self.action_set_components and
                        key in self.action_set_components[action_id]):
                    continuous_action[key] = self.action_set_components[action_id][key]

                    continue
                reconstructed_parts += 1
                reconstructed_actions.add(action_id)
                if isinstance(action_type, Node):
                    graph_name = action[key][0]
                    action_value = action[key][1:]
                    # if action id is tuple of one element, unpack it
                    if isinstance(action_value, tuple) and len(action_value) == 1:
                        action_value = action_value[0]

                    if not self.use_feature_vectors:
                        continuous_action[key] = self.node_embeddings[graph_name][action_value]
                        if action_type.concat_features is not None:
                            concat_encoding = []
                            for feature in action_type.concat_features:
                                attribute = self.env._node_attributes[feature]
                                if feature not in self.env.get_graphs()[graph_name].nodes[action_value]['x_dict']:
                                    encoding = [0] * attribute.embedding_size
                                else:
                                    attribute_value = self.env.get_graphs()[graph_name].nodes[action_value]['x_dict'][feature]
                                    if attribute.feature_extractor:
                                        # List of elements -> apply pooling
                                        if isinstance(attribute_value, list):
                                            elements_encodings = [attribute.feature_extractor(el, **attribute.feature_extractor_args) for el in
                                                                  attribute_value]
                                            if not elements_encodings:
                                                encoding = [0] * attribute.embedding_size
                                            else:
                                                encoding = []
                                                for pooling in attribute.poolings:
                                                    encoding.extend(pooling(elements_encodings))
                                        else:  # single element
                                            encoding = attribute.feature_extractor(attribute_value, **attribute.feature_extractor_args)
                                    else:
                                        encoding = attribute_value
                                if not isinstance(encoding, list) and not isinstance(encoding, np.ndarray):
                                    encoding = [encoding]
                                concat_encoding.extend(encoding)
                            continuous_action[key] = np.concatenate(
                                [continuous_action[key], np.array(concat_encoding, dtype=np.float32)])
                    else:
                        continuous_action[key] = Gs[graph_name].nodes[action_value]['x']
                elif isinstance(action_type, Edge):

                    graph_name = action[key][0]
                    source_emb = self.node_embeddings[graph_name][action[key][1]]
                    target_emb = self.node_embeddings[graph_name][action[key][2]]
                    continuous_action[key] = []
                    for pooling in action_type.poolings:
                        continuous_action[key].extend(pooling([source_emb, target_emb]))
                    if action_type.concat_node_features is not None:
                        encodings = []
                        for feature in action_type.concat_node_features:
                            attribute = self.env._node_attributes[feature]

                            for node in [action[key][1], action[key][2]]:
                                if feature not in self.env.get_graphs()[graph_name].nodes[node]['x_dict']:
                                    encoding = [0] * attribute.embedding_size
                                    encodings.append(encoding)
                                else:
                                    attribute_value = self.env.get_graphs()[graph_name].nodes[node]['x_dict'][
                                        feature]
                                    if attribute.feature_extractor:
                                        # List of elements -> apply pooling
                                        if isinstance(attribute_value, list):
                                            elements_encodings = [
                                                attribute.feature_extractor(el, **attribute.feature_extractor_args) for el in
                                                attribute_value]
                                            if not elements_encodings:
                                                encoding = [0] * attribute.embedding_size
                                            else:
                                                encoding = []
                                                for pooling in attribute.poolings:
                                                    encoding.extend(pooling(elements_encodings))
                                            encodings.append(encoding)
                                        else:
                                            encodings.append(attribute.feature_extractor(attribute_value,
                                                                                   **attribute.feature_extractor_args))
                                    else:
                                        encodings.append(attribute_value)
                            # do pooling of encodings
                            encoding = []
                            for pooling in attribute.poolings:
                                encoding.extend(pooling(encodings))
                            continuous_action[key] = np.concatenate(
                                [continuous_action[key], np.array(encoding, dtype=np.float32)])
                    if not self.use_feature_vectors:
                        if action_type.concat_edge_features is not None:
                            for feature in action_type.concat_edge_features:
                                attribute = self.env._edge_attributes[feature]
                                edge_tuple = (action[key][1], action[key][2])
                                attribute_value = self.env.get_graphs()[graph_name].edges[edge_tuple]['edge_attr_dict'][feature]
                                if attribute.feature_extractor:
                                    # List of elements -> apply pooling
                                    if isinstance(attribute_value, list):
                                        elements_encodings = [attribute.feature_extractor(el, **attribute.feature_extractor_args) for el in
                                                              attribute_value]
                                        if not elements_encodings:
                                            encoding = [0] * attribute.embedding_size
                                        else:
                                            encoding = []
                                            for pooling in attribute.poolings:
                                                encoding.extend(pooling(elements_encodings))
                                    else:
                                        encoding = attribute.feature_extractor(attribute_value, **attribute.feature_extractor_args)
                                else:
                                    encoding = attribute_value
                                continuous_action[key] = np.concatenate([continuous_action[key], np.array(encoding, dtype=np.float32)])
                elif isinstance(action_type, NonExistingEdge):
                    graph_name = action[key][0]
                    if not self.use_feature_vectors:
                        source_emb = self.node_embeddings[graph_name][action[key][1]]
                        target_emb = self.node_embeddings[graph_name][action[key][2]]
                    else:
                        source_emb = Gs[graph_name].nodes[action[key][1]]['x']
                        target_emb = Gs[graph_name].nodes[action[key][2]]['x']
                    continuous_action[key] = []
                    for pooling in action_type.poolings:
                        continuous_action[key].extend(pooling([source_emb, target_emb]))
                elif isinstance(action_type, SubGraph):
                    graph_name = action[key][0]
                    if not self.use_feature_vectors:
                        subgraph_embs = [self.node_embeddings[graph_name][n] for n in action[key][1:]]
                    else:
                        subgraph_embs = [Gs[graph_name].nodes[n]['x'] for n in action[key][1:]]
                    continuous_action[key] = []
                    for pooling in action_type.poolings:
                        continuous_action[key].extend(pooling(subgraph_embs))
                elif isinstance(action_type, Path):
                    graph_name = action[key][0]
                    # cases transpose graph and normal graph
                    continuous_action[key] = []
                    if not self.use_feature_vectors:
                        if action[key][1][0] in self.node_embeddings[graph_name]:
                            # normal graph
                            path_node_embs = [self.node_embeddings[graph_name][n] for n in action[key][1]]
                        else:
                            # transposed graph
                            # convert every two nodes into a node like edge_u_v or edge_v_u
                            path_node_embs = []
                            for i in range(len(action[key][1]) - 1):
                                if "edge_" + f"{action[key][1][i]}_{action[key][1][i+1]}" in self.node_embeddings[graph_name]:
                                    path_node_embs.append(self.node_embeddings[graph_name][f"edge_{action[key][1][i]}_{action[key][1][i+1]}"])
                                else:
                                    path_node_embs.append(self.node_embeddings[graph_name][f"edge_{action[key][1][i+1]}_{action[key][1][i]}"])
                    else:
                        path_node_embs = [Gs[graph_name].nodes[n]['x'] for n in action[key][1]]
                    for pooling in action_type.poolings:
                        continuous_action[key].extend(pooling(path_node_embs))
                        if pooling == ConcatPooling:
                            # pad to max lenpath_node_embs
                            max_len = action_type.max_len
                            current_len = len(path_node_embs)
                            if current_len <= max_len + 1:
                                pad_size = (max_len + 1 - current_len) * len(path_node_embs[0])
                                continuous_action[key].extend([0] * pad_size)
                            elif current_len > max_len + 1:
                                # remove key
                                continuous_actions.pop(key, None)
                    if not self.use_feature_vectors:
                        if action_type.concat_node_features is not None:
                            encodings = []
                            for feature in action_type.concat_node_features:
                                attribute = self.env._node_attributes[feature]
                                for node in action[key][1]:
                                    if feature not in self.env.get_graphs()[graph_name].nodes[node]['x_dict']:
                                        encoding = [0] * attribute.embedding_size
                                        encodings.append(encoding)
                                    else:
                                        attribute_value = self.env.get_graphs()[graph_name].nodes[node]['x_dict'][
                                            feature]
                                        if attribute.feature_extractor:
                                            # List of elements -> apply pooling
                                            if isinstance(attribute_value, list):
                                                elements_encodings = [
                                                    attribute.feature_extractor(el, **attribute.feature_extractor_args) for el
                                                    in
                                                    attribute_value]
                                                if not elements_encodings:
                                                    encoding = [0] * attribute.embedding_size
                                                else:
                                                    encoding = []
                                                    for pooling in attribute.poolings:
                                                        encoding.extend(pooling(elements_encodings))
                                                encodings.append(encoding)
                                            else:
                                                encodings.append(attribute.feature_extractor(attribute_value,
                                                                                             **attribute.feature_extractor_args))
                                        else:
                                            encodings.append(attribute_value)
                                # do pooling of encodings
                                encoding = []
                                for pooling in attribute.poolings:
                                    encoding.extend(pooling(encodings))
                                continuous_action[key] = np.concatenate(
                                    [continuous_action[key], np.array(encoding, dtype=np.float32)])
                    if action_type.concat_path_features is not None:
                        for feature in action_type.concat_path_features:
                            if feature == "len":
                                path_len = len(action[key][1])
                                encoding = [path_len]
                            else:
                                # to further implement features extracted from the path
                                encoding = []
                                pass
                            continuous_action[key] = np.concatenate(
                                [continuous_action[key], np.array(encoding, dtype=np.float32)])
                elif isinstance(action_type, Object):
                    # check if caching enabled on object
                    if action_type.caching:
                        action_key_hashable = (key, is_hashable(action[key][1:]))
                        if action_key_hashable in self._action_lookup_cache:
                            continuous_action[key] = self._action_lookup_cache[action_key_hashable]
                        else:
                            # call the feature extractor if available
                            continuous_action[key] = action_type.feature_extractor(action[key][1:], **action_type.feature_extractor_args)
                            self._action_lookup_cache[action_key_hashable] = continuous_action[key]
                            self._new_action_entries_since_save += 1
                            if self._new_action_entries_since_save >= self.cache_save_interval:
                                save_action_cache(self._current_action_cache_file, self._action_lookup_cache)
                                self._new_action_entries_since_save = 0
                    else:
                        continuous_action[key] = action_type.feature_extractor(action[key][1:],
                                                                               **action_type.feature_extractor_args)
                else:
                    raise ValueError(f"Unsupported action type for key '{key}': {action_type}")
                self.action_set_components.setdefault(action_id, {})[key] = continuous_action[key]

            # Flatten the continuous action into a single vector
            continuous_vector = []
            for key in action_types:
                if isinstance(continuous_action[key], np.ndarray):
                    elem = continuous_action[key].tolist()
                    # check if dimensionality is 2 unpack
                    if isinstance(elem[0], list):
                        for subelem in elem:
                            continuous_vector.extend(subelem)
                    else:
                        continuous_vector.extend(elem)
                else:
                    continuous_vector.extend(continuous_action[key])
            continuous_actions[action_id] = np.array(continuous_vector, dtype=np.float32)


        all_actions = list(continuous_actions.values())

        all_actions_np = np.array(all_actions, dtype=np.float32)

        lower_bounds = np.min(all_actions_np, axis=0)
        upper_bounds = np.max(all_actions_np, axis=0)

        if self.norm_action_space == "min-max":
            range_bounds = upper_bounds - lower_bounds
            range_bounds[range_bounds == 0] = 1e-8  # prevent divide-by-zero

            for aid in continuous_actions:
                continuous_actions[aid] = (continuous_actions[aid] - lower_bounds) / range_bounds

            # new action lower and upper bounds
            lower_bounds = np.array([0] * len(lower_bounds), dtype=np.float32)
            upper_bounds = np.array([1] * len(upper_bounds), dtype=np.float32)

        elif self.norm_action_space == "z-score":
            mean = np.mean(all_actions_np, axis=0)
            std = np.std(all_actions_np, axis=0)
            std[std == 0] = 1e-8  # prevent divide-by-zero

            for aid in continuous_actions:
                continuous_actions[aid] = (continuous_actions[aid] - mean) / std

            # compute new action lower and upper bounds as arrays
            lower_bounds = -3 * np.ones_like(mean)
            upper_bounds = 3 * np.ones_like(mean)

        if not self.norm_action_space:
            lower_bounds = [-100] * all_actions_np.shape[1]
            upper_bounds = [+100] * all_actions_np.shape[1]

        return continuous_actions, lower_bounds, upper_bounds


    def reset(self, **kwargs):
        # Call reset function of the environemnt and handle additional logic such as encoding the initial observation and creating the action space
        print("Times:")
        print(f"  Observation space creation: {self.observation_space_creation_time:.4f} seconds")
        print(f"  Action space creation: {self.action_space_creation_time:.4f} seconds")
        print(f"  Action set construction: {self.action_set_construction_time:.4f} seconds")
        print(f"  Attribute encoding: {self.attribute_encoding_time:.4f} seconds")
        print(f"  Graph encoding: {self.graph_encoding_time:.4f} seconds")
        print(f"  Action mapping: {self.action_mapping_time:.4f} seconds")
        self.action_mapping_time = 0  # Reset for next episode
        self.attribute_encoding_time = 0  # Reset for next episode
        self.graph_encoding_time = 0  # Reset for next episode
        self.action_set_construction_time = 0  # Reset for next episode
        self.node_embeddings_changes = []

        _, info = self.env.reset(**kwargs)
        Gs = self.env.get_graphs()
        encoded_Gs = {}
        graphs_changed = {}
        changed_nodes_Gs = {}
        for G_name in Gs:
            G = Gs[G_name]
            G, graph_changed, nodes_changed = self.attribute_encoding(G, G_name)
            encoded_Gs[G_name] = G
            graphs_changed[G_name] = graph_changed
            changed_nodes_Gs[G_name] = nodes_changed
            if graph_changed or nodes_changed:
                self.encode(G, G_name)
        self.observation = self.get_observation(encoded_Gs)
        if self.algorithm_type == "projection" or self.algorithm_type == "iterative":
            self.action_set, new_action_lower_bounds, new_action_upper_bounds = self.create_continuous_action_space(encoded_Gs, changed_nodes_Gs)
            self.action_set_reversed = {tuple(v): k for k, v in self.action_set.items()}
            self.perform_pca_reduction()
            if self.distance_computation == 'faiss' and self.algorithm_type != "iterative":
                self.rebuild_index()
        self.cmp_at_k_saving()
        return self.observation, info

    def cmp_at_k_saving(self):
        # Store stats from global cmp_at_k_collection in current folder
        global cmp_at_k_collection
        if cmp_at_k_collection and not os.path.exists(os.path.join(self.logs_folder, "cmp_at_k_statistics.json")):
            # average should be computed knowing that inside each there wil be a dict indexed with k
            cmp_at_k_stats = {}
            cmp_at_k_dict = {}
            for cmp_at_k in cmp_at_k_collection:
                for k, v in cmp_at_k.items():
                    if k not in cmp_at_k_dict:
                        cmp_at_k_dict[k] = []
                    cmp_at_k_dict[k].append(v)
            for k in cmp_at_k_dict:
                cmp_at_k_stats[k] = {}
                cmp_at_k_stats[k]['mean'] = np.mean(cmp_at_k_dict[k])
                cmp_at_k_stats[k]['std'] = np.std(cmp_at_k_dict[k])
                cmp_at_k_stats[k]['count'] = len(cmp_at_k_dict[k])
                cmp_at_k_stats[k]['median'] = np.median(cmp_at_k_dict[k])
                cmp_at_k_stats[k]['min'] = np.min(cmp_at_k_dict[k])
                cmp_at_k_stats[k]['max'] = np.max(cmp_at_k_dict[k])
                cmp_at_k_stats[k]['iqr'] = cmp_at_k_stats[k]['max'] - cmp_at_k_stats[k]['min']
                cmp_at_k_stats[k]['all_values'] = cmp_at_k_dict[k]
            with open(os.path.join(self.logs_folder, "cmp_at_k_statistics.json"), "w") as f:
                import json
                json.dump(cmp_at_k_stats, f, indent=4)

    def rebuild_index(self):
        # If actions change, we need to rebuild the FAISS index with the new action vectors
        action_ids = list(self.action_set.keys())
        action_vectors = np.array(
            [self.action_set[aid] for aid in action_ids],
            dtype=np.float32
        )
        action_vectors = np.ascontiguousarray(action_vectors)
        dim = action_vectors.shape[1]
        metric_name = getattr(self, "distance_metric", "euclidean")

        if metric_name == "euclidean":
            metric = faiss.METRIC_L2

        elif metric_name == "cosine":
            # cosine similarity = inner product on normalized vectors
            faiss.normalize_L2(action_vectors)
            metric = faiss.METRIC_INNER_PRODUCT

        elif metric_name == "ip":
            metric = faiss.METRIC_INNER_PRODUCT

        else:
            raise ValueError(f"Unsupported distance metric: {metric_name}")

        if self.approximate_distance:
            index_type = "ivf"  # default to IVF for approximate search
        else:
            index_type = "flat"  # default to exact search

        if metric == faiss.METRIC_L2:
            quantizer = faiss.IndexFlatL2(dim)
        else:
            quantizer = faiss.IndexFlatIP(dim)

        if index_type == "flat":
            index = quantizer
        elif index_type == "ivf":
            index = faiss.IndexIVFFlat(quantizer, dim, self.IVF_nlist, metric)
            index.train(action_vectors)
        elif index_type == "ivfpq":
            index = faiss.IndexIVFPQ(quantizer, dim, self.IVF_nlist)
            index.train(action_vectors)
        elif index_type == "hnsw":
            if metric == faiss.METRIC_L2:
                index = faiss.IndexHNSWFlat(dim)
            else:
                index = faiss.IndexHNSWFlat(dim)
                index.metric_type = metric
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Optional GPU Support
        use_gpu = getattr(self, "use_gpu", True)

        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(action_vectors)

        self.index = index
        self.action_ids = action_ids

    def get_observation(self, Gs):
        # Return observation dict based on the observation type specified in the environment, using the encoded graphs Gs and node embeddings.
        observation_dict = {}
        for key in self.env._observation_type:
            if isinstance(self.env._observation_type[key], Graph):
                observation_dict[key] = []
                for G_name in self.env._observation_type[key].graph_name:
                    # do poolings specified in the graph
                    for pooling in self.env._observation_type[key].poolings:
                        observation_dict[key].append(pooling(self.node_embeddings[G_name].values()))
                observation_dict[key] = np.concatenate(observation_dict[key], axis=0).astype(np.float32)
            elif isinstance(self.env._observation_type[key], Function):
                observation_dict[key] = []
                for G_name in Gs:
                    if self.env._observation_type[key].graph_name != None and G_name not in self.env._observation_type[key].graph_name:
                        continue
                    else:
                        G = Gs[G_name]
                        result = self.env._observation_type[key].func(G)
                        if isinstance(result, (list, np.ndarray)):
                            observation_dict[key].extend(result)
                        elif isinstance(result, (int, float)):
                            observation_dict[key].append(result)
            elif isinstance(self.env._observation_type[key], ActionSpace) and (self.algorithm_type == "projection" or self.algorithm_type == "iterative"):
                observation_dict[key] = []
                action_space_embedding = []
                # add sample subset of actions concatenated
                if self.concatenate_actions_observation and self.sample_subset_actions > 0:
                    if self.original_set_length is None:
                        assert len(self.action_set) == self.sample_subset_actions, f"Action set size {len(self.action_set)} does not match expected sample subset size {self.sample_subset_actions} for concatenation into observation."
                    for action_id in self.action_set:
                        action_space_embedding.extend(self.action_set[action_id])
                    if self.original_set_length is not None and len(self.action_set) < self.sample_subset_actions:
                        # pad with zeros to original set length
                        padding_size = (self.sample_subset_actions - len(self.action_set)) * len(next(iter(self.action_set.values())))
                        action_space_embedding.extend([0] * padding_size)
                for pooling in self.env._observation_type[key].poolings:
                    if callable(pooling):
                        action_space_embedding.extend(pooling(self.action_set.values()))
                observation_dict[key] = np.array(action_space_embedding, dtype=np.float32)
        return observation_dict

    def step(self, action_vector, **kwargs):
        # Adapt step function call with correct args based on type of algorithm
        if self.algorithm_type == "projection" or self.algorithm_type == "iterative":
            if not isinstance(action_vector, list) and not isinstance(action_vector, np.ndarray):
                # iterative DQN
                real_action = (action_vector,) + tuple(kwargs[key] for key in sorted(kwargs.keys()))
                map_action_time = 0
            else:
                # check if action vector is in bounds
                assert len(action_vector) == len(self.action_lower_bounds), \
                    f"Action vector dimensionality {len(action_vector)} does not match action space lower bounds dimensionality {len(self.action_lower_bounds)}."
                assert len(action_vector) == len(self.action_upper_bounds), \
                    f"Action vector dimensionality {len(action_vector)} does not match action space upper bounds dimensionality {len(self.action_upper_bounds)}."
                # assert with small tolerance
                if self.algorithm_type == "iterative":
                    selected_action = self.action_set_reversed.get(tuple(action_vector), None)
                    real_action = ()
                    for component in selected_action:
                        values = component[1:]
                        if len(values) == 1:
                            # Single element — append it directly as (value,)
                            real_action += (values[0],)
                        else:
                            # Multiple elements — append the tuple
                            real_action += (values,)
                else:
                    real_action, _ = self.map_action(action_vector)
        else:
            # discrete action made of action vector and kwargs
            real_action = (action_vector, ) + tuple(kwargs[key] for key in sorted(kwargs.keys()))
        obs, reward, done, truncated, info = self.env.step(*real_action)
        Gs = self.env.get_graphs()
        encoded_Gs = {}
        graphs_changed = {}
        changed_nodes_Gs = {}
        for G_name in Gs:
            G = Gs[G_name]
            G, graph_changed, nodes_changed = self.attribute_encoding(G, G_name)
            encoded_Gs[G_name] = G
            graphs_changed[G_name] = graph_changed
            changed_nodes_Gs[G_name] = nodes_changed
            if graph_changed or nodes_changed:
                self.encode(G, G_name)
        self.observation = self.get_observation(encoded_Gs)
        if changed_nodes_Gs:
            first_val = next(iter(changed_nodes_Gs.values()))

            if isinstance(first_val, set):
                graph_changed = sum(len(nodes) for nodes in changed_nodes_Gs.values())
            else:
                graph_changed = sum(changed_nodes_Gs.values())
        else:
            graph_changed = 0

        if graph_changed and self.action_space_reconstruction_each_step and (self.algorithm_type == "projection" or self.algorithm_type == "iterative"):
            self.action_set, _, _ = self.create_continuous_action_space(encoded_Gs, changed_nodes_Gs)
            self.action_set_reversed = {tuple(v): k for k, v in self.action_set.items()}
            self.perform_pca_reduction()
            if len(self.action_set) > 0 and self.distance_computation == 'faiss' and self.algorithm_type != "iterative":
                self.rebuild_index()
        if done or truncated:
            self.update_metrics()
        return self.observation, reward, done, truncated, info

    def update_metrics(self):
        self.env.update_metrics()

    def map_action(self, action_vector):
        # Map the given action vector to the closest action in the action set using KNN search, and return the corresponding discrete action and the vectors of the KNN candidates. If no action is close enough (based on tau), return a special "no_action" token.
        # The mapping can be done using different heuristics specified in protoknn_heuristic, such as selecting the closest neighbor, or using a more complex predefined heuristic
        # You can use FAISS for efficiency or using a brute-force approach for exact results, based on the distance_computation setting.
        start_time = time.time()
        action_ids = list(self.action_set.keys())
        action_vectors = np.array([self.action_set[aid] for aid in action_ids]).astype('float32')
        action_vector = action_vector.astype('float32')
        if self.gae_model_type == 'gae' or (self.gae_model_type == 'vgae' and (self.vgae_projection == 'mean' or self.vgae_projection == 'sample')):
            if self.distance_computation == 'faiss':
                distances, indices = self._search_faiss(action_vector, self.knn)
            else:
                distances, indices = self._brute_force_search(action_vector, action_vectors, self.knn)
        else:
            if self.distance_computation == 'faiss':
                distances, indices = self._search_faiss(action_vector, self.knn)
            else:
                distances, indices = self._brute_force_search(action_vector, action_vectors, self.knn)

        # compare distance first neighbor with tau
        if self.no_action_factor:
            if distances[0][0] > self.no_action_factor*self.tau:
                return ("no_action",), None

        self.last_knn_candidates = action_vectors[indices[0]]  # shape: [K, proto_dim]

        selected_index = self._select_best_index(distances, indices[0], action_ids,
                                                 get_function_from_protoknnconfig(self.protoknn_heuristic))

        self.last_selected_knn_index = np.where(indices[0] == selected_index)[0][0]

        self.action_mapping_time += time.time() - start_time

        selected_action = ()
        for component in action_ids[selected_index]:
            values = component[1:]
            if len(values) == 1:
                # Single element — append it directly as (value,)
                selected_action += (values[0],)
            else:
                # Multiple elements — append the tuple
                selected_action += (values,)

        return selected_action, action_vectors[indices[0]]

    def store_KNNs(self, candidate_action, no_normalization=False):
        # compute KNN candidates for the given candidate action vector and store them for retrieval
        _, action_vectors = self.map_action(candidate_action)
        if not no_normalization:
            action_vectors = np.array([self.convert_norm_action_to_original_space(vec) for vec in action_vectors])
        self.candidate_action_KNNs = action_vectors

    def convert_norm_action_to_original_space(self, action_vector):
        # Convert a normalized action vector back to the original action space
        if self.norm_action_space == "min-max":
            range_bounds = self.action_upper_bounds - self.action_lower_bounds
            range_bounds[range_bounds == 0] = 1e-8
            original_action = action_vector * range_bounds + self.action_lower_bounds
        elif self.norm_action_space == "z-score":
            mean = np.mean(list(self.action_set.values()), axis=0)
            std = np.std(list(self.action_set.values()), axis=0)
            std[std == 0] = 1e-8
            original_action = action_vector * std + mean
        else:
            original_action = action_vector
        return original_action

    def convert_action_to_normalized_space(self, action_vector):
        # Convert an action vector from the original action space to the normalized action space
        if self.norm_action_space == "min-max":
            range_bounds = self.action_upper_bounds - self.action_lower_bounds
            range_bounds[range_bounds == 0] = 1e-8
            norm_action = (action_vector - self.action_lower_bounds) / range_bounds
        elif self.norm_action_space == "z-score":
            mean = np.mean(list(self.action_set.values()), axis=0)
            std = np.std(list(self.action_set.values()), axis=0)
            std[std == 0] = 1e-8
            norm_action = (action_vector - mean) / std
        else:
            norm_action = action_vector
        return norm_action

    def set_terminate_at_approximate_best(self, terminate):
        self.terminate_at_approximate_best = terminate

    def get_candidate_action_KNNs(self):
        # Retrieve the stored KNN candidates for the last mapped action, or return all action vectors in the action set if no candidates are stored.
        if self.candidate_action_KNNs is not None:
            return self.candidate_action_KNNs
        else:
            # return all action vectors in the action set
            return np.array(list(self.action_set.values())).astype('float32')

    def _search_faiss(self, query_vec, knn):
        # Search the FAISS index for the K nearest neighbors of the query vector, and return their distances and indices.
        # The search can be done using different distance metrics specified in distance_metric, such as euclidean, cosine, or inner product.
        query_vec = np.asarray(query_vec, dtype=np.float32)
        query_vec = np.expand_dims(query_vec, axis=0)
        query_vec = np.ascontiguousarray(query_vec)

        metric = getattr(self, "distance_metric", "euclidean")

        if metric == "cosine":
            faiss.normalize_L2(query_vec)

        dists, indices = self.index.search(query_vec, knn)
        if metric == "euclidean":
            # FAISS returns squared L2
            dists = np.sqrt(dists)

        elif metric == "cosine":
            # FAISS returns cosine similarity
            # Convert to cosine distance
            dists = 1.0 - dists

        elif metric == "ip":
            # Inner product similarity → convert to negative distance
            # So smaller = better
            dists = -dists

        return dists, indices

    def _select_best_index(self, dists, indices, action_ids, heuristic):
        # Select the best index among the KNN candidates based on distance temperature and optional heuristic
        if self.distance_temperature != 0:
            scaled_dists = -dists[0] / self.distance_temperature
            probs = np.exp(scaled_dists - np.max(scaled_dists))
            probs /= np.sum(probs)
            chosen = np.random.choice(len(indices), p=probs)
        else:
            chosen = 0

        if heuristic is not None and len(indices) > 1:
            heuristic_scores = [heuristic(self.action_set[action_ids[i]]) for i in indices]
            best_idx = np.argmax(heuristic_scores)
            return indices[best_idx]

        return indices[chosen]