#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    hash_utils.py
    This module provides utility functions for creating hashable representations of complex nested structures, including NumPy arrays, lists, tuples, and dictionaries. It ensures that the resulting representation is deterministic and can be used as a key in caching mechanisms.
'''

import numpy as np

def is_hashable(value):
    """
    Convert value to a hashable and consistent representation.
    For complex nested structures, use JSON serialization with sorted keys.
    """
    if isinstance(value, (tuple, list)):
        # Recursively convert to tuple
        return tuple(is_hashable(v) for v in value)
    elif isinstance(value, dict):
        # Sort keys to ensure deterministic order
        return tuple((k, is_hashable(v)) for k, v in sorted(value.items()))
    else:
        return value


def make_hashable(value):
    """
    Convert nested structures into a hashable deterministic representation.
    """
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, tuple):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, make_hashable(v)) for k, v in sorted(value.items()))
    return value