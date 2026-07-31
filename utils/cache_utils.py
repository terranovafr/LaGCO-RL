#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    cache_utils.py
    This module contains utility functions for loading and saving caches, including action lookup caches and features caches.
'''


import os
import pickle

def load_action_cache(cache_dir, cache_name="action_cache"):
    # Load cache for a given graph name
    cache_file = os.path.join(cache_dir, f"{cache_name}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            action_lookup_cache = pickle.load(f)
    else:
        action_lookup_cache = {}
    return action_lookup_cache, cache_file


def save_action_cache(cache_file, cache):
    # Save current cache to disk
    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)

def load_features_cache_file(cache_file):
    # Load cache safely. If corrupted, quarantine it and return empty dict.
    if not os.path.exists(cache_file):
        return {}

    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        corrupted = cache_file + ".corrupted"
        os.rename(cache_file, corrupted)
        return {}

def save_features_cache_file_atomic(cache_file, cache_data):
    # Atomically save cache to disk.
    tmp_file = cache_file + ".tmp"
    try:
        with open(tmp_file, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, cache_file)
    except Exception:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
