#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    experiments_utils.py
    This module contains utility functions for supporting the experiments, including resolving environment folders based on a config.yaml file and mapping algorithm types to their corresponding folder names and argument overrides.
'''

import os
from pathlib import Path
from .file_utils import load_yaml

def resolve_env_folder(env_name, script_dir):
    # Given an environment name and the script directory, resolve the path to the environment folder based on the config.yaml file. The config.yaml should have a structure where each environment name maps to a dictionary that contains a "default_envs" key specifying the folder name for that environment.
    cfg = load_yaml(os.path.join(script_dir, "..", "config.yaml"))
    if env_name not in cfg:
        raise ValueError(f"Environment '{env_name}' not found in config.yaml")
    env_folder = cfg[env_name]["default_envs"]
    return Path(os.path.join(script_dir, "..", "data", "env_samples")) / env_folder


def map_algorithm_type_to_original_and_override_args(algorithm_type, args):
    # Given an algorithm_type string and an args namespace, return a tuple of (algorithm_folder, copy_args) where algorithm_folder is the folder name corresponding to the base algorithm type (e.g., "projection", "discrete", "iterative") and copy_args is a dictionary of arguments that may have some values overridden based on the presence of certain substrings in the algorithm_type. The function should check for substrings like "projection", "discrete", "iterative", "GO_", "semantic", "valid", "approximate", "pca", and "sample" in the algorithm_type and set corresponding keys in copy_args accordingly. The algorithm_folder should be constructed based on the base algorithm type and any relevant modifiers (e.g., "_semantic", "_M", "_approx", "_pca", "_sample").
    copy_args = vars(args).copy()  # create a copy of the args dictionary
    algorithm_folder = algorithm_type
    if "projection" in algorithm_type:
        real_algorithm_type = "projection"
    elif "discrete" in algorithm_type:
        real_algorithm_type = "discrete"
        algorithm_folder = "DO_discrete"
    elif "iterative" in algorithm_type:
        real_algorithm_type = "iterative"
        algorithm_folder = "iterative"
    else:
        raise ValueError(f"Unknown algorithm_type '{algorithm_type}'")
    copy_args["algorithm_type"] = real_algorithm_type
    if "GO_" in algorithm_type:
        copy_args["GNN_observations"] = True
        algorithm_folder = "GO_discrete"
    else:
        copy_args["GNN_observations"] = False
    if "semantic" in algorithm_type:
        copy_args["semantic_ordering"] = True
        algorithm_folder += "_semantic"
    else:
        copy_args["semantic_ordering"] = False
    if "valid" in algorithm_type:
        copy_args["algorithm"] = "maskable_ppo"
        algorithm_folder += "_M"
    if "approximate" in algorithm_type:
        copy_args["approximate_distance"] = True
        algorithm_folder += "_approx"
    else:
        copy_args["approximate_distance"] = False
    if "pca" in algorithm_type:
        copy_args["pca_minimum_without_loss"] = True
        algorithm_folder += "_pca"
    else:
        copy_args["pca_minimum_without_loss"] = False
    if not "sample" in algorithm_type:
        copy_args["sample_subset_actions"] = False
    else:
        algorithm_folder += "_sample"
    return algorithm_folder, copy_args
