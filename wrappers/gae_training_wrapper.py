#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    gae_training_wrapper.py
    Wrapper that supports the Graph Autoencoder (GAE) training by extracting and encoding node and edge features from the environment's graphs.
'''

from wrappers.wrapper import GraphWrapper
from wrappers.wrapper import ContinuousEnv
from envs.env import DiscreteEnv
import numpy as np
from torch_geometric.utils import from_networkx
from utils.encoding_options import (
    ContinuousValue,
    BinaryValue,
    MultiClassValue,
    RankingContinuousValue,
    RankingValue,
)

class GAETrainingWrapper(GraphWrapper):
    def __init__(self, env: [ContinuousEnv, DiscreteEnv]):
        super().__init__(env)
        self.env = env

        self.binary_indices = {}
        self.continuous_indices = {}
        self.multi_class_info = {}
        self.multi_class_info_order = {}
        self.ranking_indices = {}

        self.edge_binary_indices = {}
        self.edge_continuous_indices = {}
        self.edge_multi_class_info = {}
        self.edge_multi_class_info_order = {}
        self.edge_ranking_indices = {}

        self.node_feature_vector_size = {}
        self.edge_feature_vector_size = {}

        self._initialize_graph_info()

    def _initialize_graph_info(self):
        graphs = self.env.get_graphs()
        for graph_name, graph in graphs.items():
            graph, _, _ = self.attribute_encoding(graph, graph_name)
            self.store_graph_info(graph_name, graph)

    def store_graph_info(self, graph_name, graph):
        # Function to extract and store graph information indices for both nodes and edges.
        # This function should be called during initialization to populate the relevant data structures for each graph in the environment.
        self._initialize_graph_containers(graph_name)
        graph_data = from_networkx(graph)
        self._validate_node_features(graph, graph_data)

        node_info = self._extract_feature_info(
            graph_name=graph_name,
            graph=graph,
            attributes=self.env._node_attributes,
            is_edge=False,
        )

        self.binary_indices[graph_name] = node_info["binary_indices"]
        self.continuous_indices[graph_name] = node_info["continuous_indices"]
        self.multi_class_info[graph_name] = node_info["multi_class_info"]
        self.multi_class_info_order[graph_name] = node_info["multi_class_info_order"]
        self.ranking_indices[graph_name] = node_info["ranking_indices"]
        self.node_feature_vector_size[graph_name] = node_info["feature_vector_size"]

        edge_info = self._extract_edge_feature_info(graph_name, graph, graph_data)
        self.edge_binary_indices[graph_name] = edge_info["binary_indices"]
        self.edge_continuous_indices[graph_name] = edge_info["continuous_indices"]
        self.edge_multi_class_info[graph_name] = edge_info["multi_class_info"]
        self.edge_multi_class_info_order[graph_name] = edge_info["multi_class_info_order"]
        self.edge_ranking_indices[graph_name] = edge_info["ranking_indices"]
        self.edge_feature_vector_size[graph_name] = edge_info["feature_vector_size"]

        self._print_graph_info()

    def _initialize_graph_containers(self, graph_name):
        self.binary_indices.setdefault(graph_name, [])
        self.continuous_indices.setdefault(graph_name, [])
        self.multi_class_info.setdefault(graph_name, {})
        self.multi_class_info_order.setdefault(graph_name, [])
        self.ranking_indices.setdefault(graph_name, [])

        self.edge_binary_indices.setdefault(graph_name, [])
        self.edge_continuous_indices.setdefault(graph_name, [])
        self.edge_multi_class_info.setdefault(graph_name, {})
        self.edge_multi_class_info_order.setdefault(graph_name, [])
        self.edge_ranking_indices.setdefault(graph_name, [])

    def _validate_node_features(self, graph, graph_data):
        # Validate that all nodes in the graph have the same feature size as the first node. This is important to ensure that the node features can be properly processed by the GNN. If any node has a different feature size, a warning will be printed.
        x = graph_data.x
        if isinstance(x, list):
            x = np.array(x)
        elif hasattr(x, "numpy"):
            x = x.numpy()

        for node in graph.nodes():
            if "x" not in graph.nodes[node]:
                print(f"Node {node} does not have 'x' attribute")
                continue

            if len(graph.nodes[node]["x"]) != len(x[0]):
                print(
                    f"Node {node} has different feature size "
                    f"{len(graph.nodes[node]['x'])} vs {len(x[0])}"
                )

    def _extract_edge_feature_info(self, graph_name, graph, graph_data):
        if not hasattr(graph_data, "edge_attr") or graph_data.edge_attr is None:
            return self._empty_feature_info()

        edge_attributes = getattr(self.env, "_edge_attributes", {})
        return self._extract_feature_info(
            graph_name=graph_name,
            graph=graph,
            attributes=edge_attributes,
            is_edge=True,
        )

    def _empty_feature_info(self):
        return {
            "binary_indices": [],
            "continuous_indices": [],
            "multi_class_info": {},
            "multi_class_info_order": [],
            "ranking_indices": [],
            "feature_vector_size": 0,
        }

    def _extract_feature_info(self, graph_name, graph, attributes, is_edge=False):
        # Extract feature information for the given graph and attributes.
        # This function categorizes the attributes into binary, continuous, multi-class, and ranking types based on their encoding.
        # It also calculates the size of the feature vector for each type of attribute and stores the relevant indices and information in the corresponding data structures.
        binary_indices = []
        continuous_indices = []
        multi_class_info = {}
        multi_class_info_order = []
        ranking_indices = []

        feature_idx = 0

        # 1. Binary attributes
        for attr, attribute in attributes.items():
            if not self._attribute_applies_to_graph(attribute, graph_name):
                continue
            if isinstance(attribute.encoding, BinaryValue):
                size = self._get_attribute_size(graph, attr, attribute, is_edge=is_edge)
                if size is None:
                    continue
                binary_indices.extend(range(feature_idx, feature_idx + size))
                feature_idx += size

        # 2. Multi-class attributes
        for attr, attribute in attributes.items():
            if not self._attribute_applies_to_graph(attribute, graph_name):
                continue
            if isinstance(attribute.encoding, MultiClassValue):
                size = self._get_attribute_size(graph, attr, attribute, is_edge=is_edge)
                if size is None:
                    continue
                multi_class_info[attr] = attribute.encoding.num_classes
                multi_class_info_order.append(attr)
                feature_idx += size

        # 3. Continuous / ranking-like attributes
        assigned_ids = {}
        for attr, attribute in attributes.items():
            if not self._attribute_applies_to_graph(attribute, graph_name):
                continue
            if isinstance(
                attribute.encoding,
                (ContinuousValue, RankingContinuousValue, RankingValue),
            ):
                size = self._get_attribute_size(graph, attr, attribute, is_edge=is_edge)
                if size is None:
                    continue
                assigned_ids[attr] = (feature_idx, feature_idx + size)
                feature_idx += size

        # 4. Continuous indices
        for attr, attribute in attributes.items():
            if attr not in assigned_ids:
                continue
            if isinstance(attribute.encoding, (ContinuousValue, RankingContinuousValue)):
                start, end = assigned_ids[attr]
                continuous_indices.extend(range(start, end))

        # 5. Ranking indices
        for attr, attribute in attributes.items():
            if attr not in assigned_ids:
                continue
            if isinstance(attribute.encoding, (RankingValue, RankingContinuousValue)):
                start, end = assigned_ids[attr]
                ranking_indices.extend(range(start, end))

        return {
            "binary_indices": binary_indices,
            "continuous_indices": continuous_indices,
            "multi_class_info": multi_class_info,
            "multi_class_info_order": multi_class_info_order,
            "ranking_indices": ranking_indices,
            "feature_vector_size": feature_idx,
        }

    def _attribute_applies_to_graph(self, attribute, graph_name):
        return attribute.graph_name is None or graph_name in attribute.graph_name

    def _get_attribute_size(self, graph, attr, attribute, is_edge=False):
        # Determine the size of the feature vector for a given attribute. This function checks if the attribute has an encoding defined and then tries to determine the size based on the encoding or by inspecting an example value from the graph. If the attribute has an embedding size defined, it uses that directly. Otherwise, it looks at an example value for that attribute in the graph (either from a node or an edge) and determines the size based on whether it's a scalar, vector, or higher-dimensional feature.
        if not hasattr(attribute, "encoding"):
            raise ValueError(f"Attribute {attr} does not have an encoding defined.")

        if hasattr(attribute, "embedding_size") and attribute.embedding_size is not None:
            return attribute.embedding_size

        example = self._get_example_value(graph, attr, is_edge=is_edge)
        if example is None:
            return None

        if isinstance(example, np.ndarray):
            return example.size

        try:
            return len(example)
        except Exception:
            return 1 if is_edge else 2

    def _get_example_value(self, graph, attr, is_edge=False):
        if is_edge:
            first_edge = next(iter(graph.edges(data=True)), None)
            if first_edge is None:
                return None
            return first_edge[2].get(attr)

        first_node = next(iter(graph.nodes), None)
        if first_node is None:
            return None
        return graph.nodes[first_node].get("x_dict", {}).get(attr)

    def _print_graph_info(self):
        print("Graph info stored:")
        print(f"Node feature vector size: {self.node_feature_vector_size}")
        print(f"Edge feature vector size: {self.edge_feature_vector_size}")
        print(f"Binary indices: {self.binary_indices}")
        print(f"Continuous indices: {self.continuous_indices}")
        print(f"Multi-class info: {self.multi_class_info}")
        print(f"Ranking indices: {self.ranking_indices}")
        print(f"Edge binary indices: {self.edge_binary_indices}")
        print(f"Edge continuous indices: {self.edge_continuous_indices}")
        print(f"Edge multi-class info: {self.edge_multi_class_info}")
        print(f"Edge ranking indices: {self.edge_ranking_indices}")

    def reset(self):
        return self.env.reset()

    def step(self, real_action):
        if isinstance(real_action, int):
            real_action = (real_action,)
        return self.env.step(*real_action)

    def sample_valid_action(self):
        return self.env.sample_valid_action()

    def get_graphs(self):
        graphs = self.env.get_graphs()
        new_graphs = {}
        for graph_name, graph in graphs.items():
            graph, _, _ = self.attribute_encoding(graph, graph_name)
            new_graphs[graph_name] = graph
        return new_graphs