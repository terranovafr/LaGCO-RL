#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    model.py
    This module defines the classes for the different components of the graph-based model, including nodes, edges, functions, paths, subgraphs, and objects.
    These classes are used to specify the structure of the graph and the features to be extracted from it for use in the proposed embedding construction framework.
'''

from utils.pooling_functions import ConcatPooling

class Node:
    def __init__(self, different_than_keys=None, spec=None, graph_name=None, concat_features=None):
        self.different_than_keys = different_than_keys
        self.spec = spec # dict including checks to be done on the feature vector
        if graph_name is not None and not isinstance(graph_name, list):
            self.graph_name = [graph_name]
        else:
            self.graph_name = graph_name
        if concat_features is not None and not isinstance(concat_features, list):
            self.concat_features = [concat_features]  # dict including features to be appended to the embedding from the node feature vector
        else:
            self.concat_features = concat_features

class Function:
    def __init__(self, func=None, graph_name=None):
        self.func = func  # function that takes as input a graph and returns a value
        if graph_name is not None and not isinstance(graph_name, list):
            self.graph_name = [graph_name]
        else:
            self.graph_name = graph_name

class Edge:
    def __init__(self, bidirectional=False, different_than_keys=None, spec=None, graph_name=None, nodes_spec=None, source_node_key=None, target_node_key=None,
                 source_node_key_different=None, target_node_key_different=None, poolings=None,
                 concat_edge_features=None, concat_node_features=None):
        self.bidirectional = bidirectional
        self.different_than_keys = different_than_keys
        self.spec = spec # dict including checks to be done on the feature vector of edge
        if graph_name is not None and not isinstance(graph_name, list):
            self.graph_name = [graph_name]
        else:
            self.graph_name = graph_name
        if concat_edge_features is not None and not isinstance(concat_edge_features, list):
            self.concat_edge_features = [concat_edge_features]
        else:
            self.concat_edge_features = concat_edge_features
        if concat_node_features is not None and not isinstance(concat_node_features, list):
            self.concat_node_features = [concat_node_features]
        else:
            self.concat_node_features = concat_node_features

        self.nodes_spec = nodes_spec # dict including checks to be done on the feature vector of nodes connected by the edge
        self.source_node_key = source_node_key  # key to identify the source node
        self.target_node_key = target_node_key  # key to identify the target node
        self.source_node_key_different = source_node_key_different # key to identify the source node (should be different from these)
        self.target_node_key_different = target_node_key_different # key to identify the target node (should be different from these)
        self.poolings = poolings if poolings else [ConcatPooling]  # list of pooling methods to apply on the edge feature vector

class NonExistingEdge:
    def __init__(self,  different_than_keys=None, nodes_spec=None, graph_name=None, source_node_key=None, target_node_key=None,
                 source_node_key_different=None, target_node_key_different=None, poolings=None):
        self.different_than_keys = different_than_keys
        self.nodes_spec = nodes_spec # dict including checks to be done on the feature vector of nodes connected by the edge
        if graph_name is not None and not isinstance(graph_name, list):
            self.graph_name = [graph_name]
        else:
            self.graph_name = graph_name
        self.source_node_key = source_node_key  # key to identify the source node
        self.target_node_key = target_node_key  # key to identify the target node
        self.source_node_key_different = source_node_key_different  # key to identify the source node (should be different from these)
        self.target_node_key_different = target_node_key_different  # key to identify the target node (should be different from these)
        self.poolings = poolings if poolings else []  # list of pooling methods to apply on the edge feature vector

class Graph:
    def __init__(self, poolings=None, nodes_spec=None, edges_spec=None, graph_name=None):
        self.poolings = poolings if poolings else []
        self.nodes_spec = nodes_spec if nodes_spec else {} # checks to be done on nodes feature vectors
        self.edges_spec = edges_spec if edges_spec else {} # checks to be done on edges feature vectors
        if graph_name is not None and not isinstance(graph_name, list):
            self.graph_name = [graph_name]
        else:
            self.graph_name = graph_name

class Path:
    def __init__(self, different_than_keys=None, set_dict=None, set_dict_key_name=None,
                 set_dict_function=None,
                 nodes_spec=None, min_len=2, max_len=None, poolings=None, graph_name=None,
                concat_path_features=None, concat_node_features=None):
        self.different_than_keys = different_than_keys
        self.nodes_spec = nodes_spec # dict including checks to be done on the feature vector of nodes
        self.set_dict = set_dict # you do not have to take them all but just those here
        self.set_dict_key_name = set_dict_key_name  # name of the key in the dict representing the set
        self.set_dict_function = set_dict_function  # function that takes as input the set dict and returns a subset of it to be used for path extraction
        self.min_len = min_len # if None take all possible cases
        self.max_len = max_len # if None take all possible cases
        self.poolings = poolings if poolings else []
        if graph_name is not None and not isinstance(graph_name, list):
            self.graph_name = [graph_name]
        else:
            self.graph_name = graph_name
        if concat_path_features is not None and not isinstance(concat_path_features, list):
            self.concat_path_features = [concat_path_features]
        else:
            self.concat_path_features = concat_path_features
        if concat_node_features is not None and not isinstance(concat_node_features, list):
            self.concat_node_features = [concat_node_features]
        else:
            self.concat_node_features = concat_node_features

class SubGraph:
    def __init__(self, different_than_keys=None, nodes_spec=None, size_min=None, size_max=None, poolings=None, graph_name=None):
        self.different_than_keys = different_than_keys
        self.nodes_spec = nodes_spec # dict including checks to be done on the feature vector of nodes
        self.size_min = size_min # if None take all possible cases
        self.size_max = size_max # if None take all possible cases
        self.poolings = poolings if poolings else []
        if graph_name is not None and not isinstance(graph_name, list):
            self.graph_name = [graph_name]
        else:
            self.graph_name = graph_name

class Object:
    def __init__(self, name="unknown", set=None, set_dict=None,
                 language_mapping=None,
                 reference_key="global", reference_graph_name=None,
                 reference_feature_vector_key=None, different_than_keys=None, set_dict_key_name=None,
                 feature_extractor=None, feature_extractor_args=None, key_to_extract=None,
                 caching=False, embedding_size=1):
        self.name = name
        self.set = set  # set of objects to be used
        self.set_dict = set_dict
        self.language_mapping = language_mapping  # optional dict to map the object to a different value before feature extraction based on their textual description provided here
        self.set_dict_key_name = set_dict_key_name  # name of the key in the set dict representing the set
        self.reference_key = reference_key # global all from the set, or key of the element that should have the object
        self.reference_graph_name = reference_graph_name  # name of the graph where to find the objects subset
        self.reference_feature_vector_key = reference_feature_vector_key  # key of the feature vector where to find the objects subset
        self.different_than_keys = different_than_keys
        self.feature_extractor = feature_extractor
        self.feature_extractor_args = feature_extractor_args if feature_extractor_args else {} # args to be passed to the feature extractor function, if any
        self.embedding_size = embedding_size  # size of the embedding vector for the object, if None it will be inferred from the feature extractor
        self.key_to_extract = key_to_extract # if the object pointed is structured as a dict you can specify the key with value that should represent the object
        self.caching = caching # whether to enable caching of the embedding for each object (useful if the feature extractor is time-intensive and the same object appears multiple times)

class Attribute:
    def __init__(self,
                 encoding=None,
                 feature_extractor=None,
                 feature_extractor_args=None,
                 poolings=None,
                 embedding_size=None,
                 key_to_extract=None,
                 caching=False,
                 graph_name=None):
        self.encoding = encoding  # e.g., ContinuousValue
        self.feature_extractor = feature_extractor
        self.feature_extractor_args = feature_extractor_args if feature_extractor_args else {}
        self.poolings = poolings if poolings else []
        self.embedding_size = embedding_size
        self.key_to_extract = key_to_extract
        self.caching = caching # enable caching of embeddings if feature extractor procedure is time-intensive
        if graph_name is not None and not isinstance(graph_name, list):
            self.graph_name = [graph_name]
        else:
            self.graph_name = graph_name

# To be used when poolings of continuous action space should be included in the observation space
# which means, action is not a node only
class ActionSpace:
    def __init__(self, poolings=None):
        assert ConcatPooling not in poolings, "ConcatPooling not allowed in ActionSpace poolings"
        self.poolings = poolings if poolings else []