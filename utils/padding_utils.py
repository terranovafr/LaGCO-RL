#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    padding_utils.py
    This module provides utility functions for padding Gym observations and spaces to ensure compatibility across different environments and agents. It includes functions to pad observations to match a target observation space, as well as functions to pad Gym spaces themselves. The utilities support various Gym space types, including Box, Discrete, MultiDiscrete, Tuple, and Dict.
'''

import gymnasium as gym
import numpy as np
from typing import Any
import itertools


def pad_observation(obs, observation_space, space=None):
    # Recursively pad observation to match the target observation_space.
    if space is None:
        space = observation_space

    # Box observation
    if isinstance(space, gym.spaces.Box):
        target_shape = space.shape
        if isinstance(obs, np.ndarray) and obs.shape != target_shape:
            padded_obs = np.zeros(target_shape, dtype=obs.dtype)
            slices = tuple(slice(0, min(obs_dim, pad_dim)) for obs_dim, pad_dim in zip(obs.shape, target_shape))
            padded_obs[slices] = obs[slices]
            return padded_obs
        return obs

    # Dict observation
    elif isinstance(space, gym.spaces.Dict):
        padded_dict = {}
        for k, subspace in space.spaces.items():
            padded_dict[k] = pad_observation(obs[k], subspace)
        return padded_dict

    # Tuple observation
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(pad_observation(o, s) for o, s in zip(obs, space.spaces))

    # Discrete or other types: return as-is
    else:
        return obs

def pad_space(space, max_shape=None):
    # Pad a Gym space to match max_shape (Box, Discrete, Tuple, Dict)

    if isinstance(space, gym.spaces.Box):
        if max_shape is None:
            max_shape = space.shape
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=max_shape,
            dtype=np.float32  # Always explicitly provide dtype
        )
    elif isinstance(space, gym.spaces.MultiDiscrete):
        if max_shape is None:
            nvec = np.array(space.nvec, dtype=int)
        else:
            nvec = np.array(max_shape, dtype=int)
        return gym.spaces.MultiDiscrete(nvec)
    elif isinstance(space, gym.spaces.Discrete):
        if max_shape is None:
            max_shape = space.n
        return gym.spaces.Discrete(max_shape)
    elif isinstance(space, gym.spaces.Tuple):
        return gym.spaces.Tuple([pad_space(s, max_shape[i] if max_shape else None)
                                 for i, s in enumerate(space.spaces)])
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: pad_space(v, max_shape.get(k) if max_shape else None)
                                for k, v in space.spaces.items()})
    else:
        return space  # Unknown/custom space

def get_space_shape(space: gym.Space) -> Any:
    """Recursively compute 'shape' for any Gym space.
    Returns:
      - Box -> tuple of ints
      - Discrete -> int
      - MultiDiscrete -> tuple of ints (nvec)
      - MultiBinary -> int
      - Tuple -> tuple of sub-shapes
      - Dict -> dict of sub-shapes
      - unknown -> None
    """
    if isinstance(space, gym.spaces.Box):
        return tuple(int(x) for x in space.shape)
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gym.spaces.MultiDiscrete):
        # represent as tuple of ints
        return tuple(int(x) for x in np.array(space.nvec).flatten())
    if isinstance(space, gym.spaces.MultiBinary):
        # n may be int or tuple; convert to int if possible
        return int(space.n) if np.isscalar(space.n) else tuple(int(x) for x in np.array(space.n).flatten())
    if isinstance(space, gym.spaces.Tuple):
        return tuple(get_space_shape(s) for s in space.spaces)
    if isinstance(space, gym.spaces.Dict):
        return {k: get_space_shape(v) for k, v in space.spaces.items()}
    return None


def _tuple_elementwise_max(t1, t2):
    # allow different lengths: result length = max(len1, len2), missing values treated as 0
    out = []
    for a, b in itertools.zip_longest(t1, t2, fillvalue=0):
        out.append(max(int(a), int(b)))
    return tuple(out)


def max_shapes(shape1, shape2):
    # Recursively compute elementwise maximum of two shapes
    if shape1 is None:
        return shape2
    if shape2 is None:
        return shape1

    # dict vs dict
    if isinstance(shape1, dict) and isinstance(shape2, dict):
        keys = set(shape1.keys()).union(shape2.keys())
        return {k: max_shapes(shape1.get(k), shape2.get(k)) for k in keys}

    # tuple vs tuple (and MultiDiscrete-style tuples)
    if isinstance(shape1, tuple) and isinstance(shape2, tuple):
        return _tuple_elementwise_max(shape1, shape2)

    # tuple vs int: broadcast int across tuple
    if isinstance(shape1, tuple) and isinstance(shape2, (int, np.integer)):
        return _tuple_elementwise_max(shape1, (int(shape2),) * len(shape1))
    if isinstance(shape2, tuple) and isinstance(shape1, (int, np.integer)):
        return _tuple_elementwise_max((int(shape1),) * len(shape2), shape2)

    # int vs int
    try:
        return max(int(shape1), int(shape2))
    except Exception:
        # fallback to shape2 if comparision fails
        return shape2
