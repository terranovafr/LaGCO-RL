#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    wrapper.py
    General wrapper and ContinuousEnv built to support our methodology.
'''


import os
import time
from collections import defaultdict
import gymnasium as gym
import numpy as np

from utils.encoding_options import (
    BinaryValue,
    MultiClassValue,
    ContinuousValue,
    RankingValue,
    RankingContinuousValue,
)
from utils.cache_utils import load_features_cache_file, save_features_cache_file_atomic
from utils.hash_utils import make_hashable

class ContinuousEnv:
    '''
        Continuous environment class that defines the interface for environments with continuous action spaces.
        The environment attributes should be filled with proper information to enable the automated framework.
    '''
    def __init__(self, **config):
        self._observation_type = {}
        self._action_type = {}
        self._node_attributes = {}
        self._edge_attributes = {}
        self.discrete_actions = []
        self.action_candidates_reconstruction = False
        self.reconstruct_only_changed_nodes = False
        self.action_space_reconstruction_each_step = False
        self.action_space_varying_node_embeddings = False

    def sample_valid_action(self):
        raise NotImplementedError

    def update_metrics(self):
        self._metrics = {}

    def get_graphs(self):
        raise NotImplementedError

    def get_metrics(self):
        return self._metrics


# ============================================================
# Pure utility functions
# ============================================================

ATTRIBUTE_ENCODING_ORDER = (
    BinaryValue,
    MultiClassValue,
    ContinuousValue,
    RankingValue,
    RankingContinuousValue,
)

def normalize_value(value, attribute, attribute_name):
    normalization = attribute.encoding.normalization

    if normalization is None:
        return value

    if normalization == "min_max":
        vmin = attribute.encoding.min_value
        vmax = attribute.encoding.max_value
        assert value >= vmin and value <= vmax, (
            f"Value {value} out of bounds [{vmin}, {vmax}] "
            f"for attribute {attribute_name}"
        )
        return (value - vmin) / (vmax - vmin + 1e-8)

    if normalization == "l2":
        assert getattr(value, "ndim", 0) != 0 and getattr(value, "size", 1) != 1, (
            f"L2 normalization requires vector values for attribute {attribute_name}"
        )
        norm = np.linalg.norm(value)
        if norm == 0:
            return value
        return value / norm

    if normalization == "z_score":
        mean = attribute.encoding.mean
        std = attribute.encoding.std
        return (value - mean) / (std + 1e-8)

    raise ValueError(f"Unknown normalization {normalization}")

def ensure_list_encoding(encoding):
    if isinstance(encoding, np.ndarray):
        return encoding.tolist()
    if isinstance(encoding, list):
        return encoding
    return [encoding]


def should_process_attribute(attribute, graph_name, attr_type):
    if attribute.graph_name is not None and graph_name not in attribute.graph_name:
        return False
    return isinstance(attribute.encoding, attr_type)


def encode_with_feature_extractor(current_value, attribute):
    if isinstance(current_value, list):
        element_encodings = [
            attribute.feature_extractor(el, **attribute.feature_extractor_args)
            for el in current_value
        ]

        if not element_encodings:
            return [0] * attribute.embedding_size

        encoding = []
        for pooling in attribute.poolings:
            encoding.extend(pooling(element_encodings))
        return encoding

    encoding = attribute.feature_extractor(
        current_value, **attribute.feature_extractor_args
    )
    return ensure_list_encoding(encoding)


def encode_without_feature_extractor(current_value):
    if isinstance(current_value, (list, np.ndarray)):
        return list(current_value)
    return [current_value]


def maybe_normalize_raw_value(current_value, attribute, attr_name):
    if not isinstance(
        attribute.encoding,
        (ContinuousValue, RankingValue, RankingContinuousValue),
    ):
        return current_value

    if attribute.feature_extractor:
        return current_value

    if isinstance(current_value, list):
        return [normalize_value(v, attribute, attr_name) for v in current_value]

    return normalize_value(current_value, attribute, attr_name)


def maybe_normalize_encoded_value(encoding, attribute, attr_name):
    if not isinstance(
        attribute.encoding,
        (ContinuousValue, RankingValue, RankingContinuousValue),
    ):
        return encoding

    if attribute.encoding.normalization == "min_max":
        return [normalize_value(v, attribute, attr_name) for v in encoding]

    return normalize_value(np.array(encoding), attribute, attr_name).tolist()


def values_equal(val1, val2):
    """Check equality between attribute values, handling numpy arrays."""
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        return np.array_equal(val1, val2)
    return val1 == val2



class GraphWrapper(gym.Wrapper):
    '''
        Wrapper extended either for agent or GAE training, defining basics utility functions
    '''
    def __init__(self, env, cache_dir="cache", cache_save_interval=1, **kwargs):
        super().__init__(env)

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.cache_save_interval = cache_save_interval
        self._new_entries_since_save = 0
        self._feature_lookup_cache = {}
        self._current_cache_files = {}

    # --------------------------------------------------------
    # Cache handling
    # --------------------------------------------------------
    def _load_cache(self, graph_name):
        cache_file = os.path.join(self.cache_dir, f"{graph_name}_feature_cache.pkl")
        self._current_cache_files[graph_name] = cache_file

        if graph_name not in self._feature_lookup_cache:
            self._feature_lookup_cache[graph_name] = {}

        if not os.path.exists(cache_file):
            print(f"No existing cache for {graph_name}, starting fresh.")
            return

        self._feature_lookup_cache[graph_name] = load_features_cache_file(cache_file)

    def _save_cache(self):
        if not self._current_cache_files:
            return

        for graph_name, cache_file in self._current_cache_files.items():
            save_features_cache_file_atomic(
                cache_file,
                self._feature_lookup_cache.get(graph_name, {})
            )

        self._new_entries_since_save = 0

    def _maybe_save_cache(self):
        if self._new_entries_since_save >= self.cache_save_interval:
            self._save_cache()

    def attribute_encoding(self, G, G_name):
        # Encode node and edge attributes in the order: Binary -> MultiClass -> Continuous -> Ranking -> RankingContinuous
        nodes_changed = set()

        if G_name not in self._feature_lookup_cache:
            try:
                self._load_cache(G_name)
            except Exception:
                self._feature_lookup_cache[G_name] = {}

        graph_changed = False
        timing_stats = defaultdict(lambda: {"total": 0.0, "count": 0})

        self._encode_nodes(G, G_name, nodes_changed, timing_stats)
        edge_changed = self._encode_edges(G, G_name, nodes_changed, timing_stats)

        graph_changed |= edge_changed

        self._maybe_save_cache()
        return G, graph_changed, nodes_changed

    def _encode_nodes(self, G, G_name, nodes_changed, timing_stats):
        # Encode node attributes in the order: Binary -> MultiClass -> Continuous -> Ranking -> RankingContinuous
        graph_changed = False

        for node in G.nodes():
            x_vector = []
            G.nodes[node].setdefault("x_raw_cache", {})

            for attr_type in ATTRIBUTE_ENCODING_ORDER:
                for attr, attribute in self.env._node_attributes.items():
                    if not should_process_attribute(attribute, G_name, attr_type):
                        continue

                    current_value = G.nodes[node]["x_dict"].get(attr, None)

                    # Fix: extract sub-value instead of returning from _encode_nodes
                    if current_value is not None and attribute.key_to_extract is not None:
                        if isinstance(current_value, list):
                            current_value = [
                                item.get(attribute.key_to_extract, None) if isinstance(item, dict) else None
                                for item in current_value
                            ]
                        elif isinstance(current_value, dict):
                            current_value = current_value.get(attribute.key_to_extract, None)

                    if current_value is None:
                        encoding = [0] * attribute.embedding_size
                        changed = False
                    else:
                        t0 = time.perf_counter()
                        encoding, changed = self._encode_attribute(
                            G_name=G_name,
                            current_value=current_value,
                            cached_value=G.nodes[node]["x_raw_cache"].get(attr, None),
                            attribute=attribute,
                            attr_name=attr,
                        )
                        dt = time.perf_counter() - t0

                        timing_stats[attr]["total"] += dt
                        timing_stats[attr]["count"] += 1

                    if changed:
                        nodes_changed.add(node)

                    graph_changed |= changed
                    x_vector.extend(encoding)

                    G.nodes[node]["x_raw_cache"][attr] = {
                        "value": current_value,
                        "encoding": encoding,
                    }

            G.nodes[node]["x"] = np.array(x_vector, dtype=np.float32)

        return graph_changed

    def _encode_edges(self, G, G_name, nodes_changed, timing_stats):
        # Encode edge attributes in the order: Binary -> MultiClass -> Continuous -> Ranking -> RankingContinuous
        graph_changed = False

        for u, v in G.edges():
            edge_vec = []
            G.edges[u, v].setdefault("edge_raw_cache", {})

            for attr_type in ATTRIBUTE_ENCODING_ORDER:
                for attr, attribute in getattr(self.env, "_edge_attributes", {}).items():
                    if not should_process_attribute(attribute, G_name, attr_type):
                        continue

                    current_value = G.edges[u, v]["edge_attr_dict"].get(attr, None)

                    t0 = time.perf_counter()
                    encoding, changed = self._encode_attribute(
                        G_name=G_name,
                        current_value=current_value,
                        cached_value=G.edges[u, v]["edge_raw_cache"].get(attr, None),
                        attribute=attribute,
                        attr_name=attr,
                    )
                    dt = time.perf_counter() - t0

                    timing_stats[f"edge::{attr}"]["total"] += dt
                    timing_stats[f"edge::{attr}"]["count"] += 1

                    if changed:
                        nodes_changed.add(u)
                        nodes_changed.add(v)

                    graph_changed |= changed
                    edge_vec.extend(encoding)

                    G.edges[u, v]["edge_raw_cache"][attr] = {
                        "value": current_value,
                        "encoding": encoding,
                    }

            G.edges[u, v]["edge_attr"] = np.array(edge_vec, dtype=np.float32)

        return graph_changed

    def _encode_attribute(self, G_name, current_value, cached_value, attribute, attr_name):
        # Returns the encoding for the given attribute value, using the feature extractor if available, and applying normalization if specified. Also returns a boolean indicating whether the graph has changed compared to the cached value.
        graph_changed = False

        if current_value is None:
            if attribute.embedding_size == 1:
                current_value =  0.0
            else:
                current_value =np.zeros(attribute.embedding_size)
        current_value = maybe_normalize_raw_value(current_value, attribute, attr_name)

        if cached_value is not None and values_equal(current_value, cached_value["value"]):
            return ensure_list_encoding(cached_value["encoding"]), graph_changed

        graph_changed = True
        lookup_key = (attr_name, make_hashable(current_value))

        if attribute.caching and lookup_key in self._feature_lookup_cache[G_name]:
            return self._feature_lookup_cache[G_name][lookup_key], graph_changed

        if attribute.feature_extractor:
            encoding = encode_with_feature_extractor(current_value, attribute)
            encoding = maybe_normalize_encoded_value(encoding, attribute, attr_name)
        else:
            encoding = encode_without_feature_extractor(current_value)

        if attribute.caching:
            self._new_entries_since_save += 1
            self._feature_lookup_cache[G_name][lookup_key] = encoding

        return encoding, graph_changed