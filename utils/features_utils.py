#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    features_utils.py
    This module provides utility functions for extracting features from a graph using the NetworkX library.
'''

import networkx as nx

def get_number_nodes(graph):
    return graph.number_of_nodes()

def get_number_edges(graph):
    return graph.number_of_edges()

def get_average_node_degree(graph):
    if graph.number_of_nodes() == 0:
        return 0.0
    return sum(dict(graph.degree()).values()) / graph.number_of_nodes()

def get_graph_density(graph):
    if graph.number_of_nodes() <= 1:
        return 0.0
    return nx.density(graph)

def find_node_largest_degree(graph):
    # take node with largest degree
    if not graph.nodes:
        return None
    degrees = dict(graph.degree())
    max_degree_node = max(degrees, key=degrees.get)
    # take its feature vector
    node_features = graph.nodes[max_degree_node].get('feature_vector', None)
    return node_features