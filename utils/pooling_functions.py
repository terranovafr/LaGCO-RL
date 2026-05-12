#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    pooling_functions.py
    This module defines various pooling functions (Concat, Mean, Max, Min, Sum) and a utility function to compute bounds for these pooling operations.
'''

import numpy as np


def compute_pooled_bounds(pooling, low, high, N):
    # Utility functions for pooling operations and computing bounds
    if pooling == ConcatPooling:
        # handled outside (just repeat)
        raise RuntimeError("Concat handled separately")

    if pooling in (SumPooling,):
        return N * low, N * high

    if pooling in (MeanPooling,):
        return low, high

    if pooling in (MaxPooling, MinPooling):
        return low, high

    # Fallback: be conservative
    return -np.inf * np.ones_like(low), np.inf * np.ones_like(high)

def ConcatPooling(elements):
    # Concatenates a list of elements into a single vector.
    if not elements:
        return np.array([])

    # Ensure all elements are numpy arrays
    elements = [np.array(el) for el in elements]

    # Concatenate along the last axis
    return np.concatenate(elements, axis=0)

def MeanPooling(elements):
    # Averages a list of elements into a single vector.
    if not elements:
        return np.array([])

    # Ensure all elements are numpy arrays
    elements = [np.array(el) for el in elements]

    # Stack and compute the mean along the first axis
    return np.mean(np.stack(elements), axis=0)

def MaxPooling(elements):
    # Takes the maximum of a list of elements into a single vector.
    if not elements:
        return np.array([])

    # Ensure all elements are numpy arrays
    elements = [np.array(el) for el in elements]

    # Stack and compute the max along the first axis
    return np.max(np.stack(elements), axis=0)

def MinPooling(elements):
    # Takes the minimum of a list of elements into a single vector.
    if not elements:
        return np.array([])

    # Ensure all elements are numpy arrays
    elements = [np.array(el) for el in elements]

    # Stack and compute the min along the first axis
    return np.min(np.stack(elements), axis=0)

def SumPooling(elements):
    # Sums a list of elements into a single vector.
    if not elements:
        return np.array([])

    # Ensure all elements are numpy arrays
    elements = [np.array(el) for el in elements]

    # Stack and compute the sum along the first axis
    return np.sum(np.stack(elements), axis=0)

