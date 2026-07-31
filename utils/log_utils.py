#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    log_utils.py
    Utility functions for logging and graph representation.
'''
import os
import logging


def graph_nodes_to_text(graph):
    # Convert a NetworkX graph to a text representation
    lines = []

    for node, data in graph.nodes(data=True):
        line = f"Node {node}: "
        attributes = ", ".join(f"{key}={value}" for key, value in data.items())
        line += attributes
        lines.append(line)
    return "\n".join(lines)

def graph_edges_to_text(graph):
    # Convert a NetworkX graph edges to a text representation
    lines = []

    for u, v, data in graph.edges(data=True):
        line = f"Edge from {u} to {v}: "
        attributes = ", ".join(f"{key}={value}" for key, value in data.items())
        line += attributes
        lines.append(line)
    return "\n".join(lines)


def setup_logging(logs_folder, log_to_file=True, log_filename='app.log', log_level=logging.INFO):
    # Set up logging configuration
    os.makedirs(logs_folder, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers if any, to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a console handler for logging to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create a file handler for logging to a file
    if log_to_file:
        file_handler = logging.FileHandler(os.path.join(logs_folder, log_filename), mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

