#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    distance_utils.py
    This module contains utility functions for distance components.
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

def sample_farthest_k_points(action_dict, k):
    '''
        Function to sample k points from action_dict that are as far apart as possible using a greedy approach
    '''
    keys = list(action_dict.keys())
    vectors = np.array([np.asarray(action_dict[k]) for k in keys])

    n = len(keys)
    if k >= n:
        return action_dict

    # Start from a random point
    first_idx = random.randrange(n)
    selected = [first_idx]

    # Precompute distance matrix (optional but faster for small n)
    dists = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=-1)

    while len(selected) < k:
        # For each point, compute distance to nearest selected point
        min_dist_to_selected = np.min(dists[:, selected], axis=1)

        # Don't re-pick already selected ones
        min_dist_to_selected[selected] = -np.inf

        next_idx = int(np.argmax(min_dist_to_selected))
        selected.append(next_idx)

    sampled_keys = [keys[i] for i in selected]
    return {k: action_dict[k] for k in sampled_keys}


def calculate_most_distant_point(action_points, bounds, margin=0, n_samples=2500):
    '''
    Calculate the point that is furthest from the nearest action point within the given bounds.
    '''
    # Step 1: Fit a KDE model to the action points
    min_bound, max_bound = bounds
    num_dims = len(action_points[0])
    # Step 1: Sample random points within the bounds
    samples = np.random.uniform(min_bound + margin, max_bound - margin, (n_samples, num_dims))

    # Step 2: Fit a nearest neighbor model to the action points
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(action_points)

    # Step 3: Find the nearest neighbors for each sampled point
    distances, indices = nbrs.kneighbors(samples)

    # Step 4: For each sampled point, find the one with the furthest nearest neighbor
    max_distance_idx = np.argmax(distances[:, 1])  # We take the second closest distance (nearest neighbor)
    most_distant_point = samples[max_distance_idx]

    return most_distant_point