#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    heuristics_utils.py
    This module provides utility functions for handling heuristic functions in the action-selection framework across the top-k candidates.
    It includes functionality to map string identifiers to predefined heuristic functions, as well as loading custom heuristic functions from external scripts.
'''

import numpy as np
import importlib.util
import sys

def map_protoknn_string_to_function(protoknn_string):
    if protoknn_string == 'random':
        return lambda x: np.random.uniform(0, 1)

def load_function_from_script(script_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)

def get_function_from_protoknnconfig(protoknn_config):
    if isinstance(protoknn_config, str):
        return map_protoknn_string_to_function(protoknn_config)
    elif isinstance(protoknn_config, dict):
        if protoknn_config['type'] == 'script':
            script_path = protoknn_config['path']
            function_name = protoknn_config['function_name']
            return load_function_from_script(script_path, function_name)
        else:
            raise ValueError(f"Unknown function type: {protoknn_config['type']}")
    elif protoknn_config == None:
        return None
    else:
        raise ValueError(f"Invalid protoknn_config: {protoknn_config}. Expected a string or a dict.")
