#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    generation_utils.py
    Utility functions for environment generation, including complexity computation, environment creation, and saving.
'''

import pickle
import yaml
from pathlib import Path
script_dir = Path(__file__).parent


def compute_complexity(config: dict, env, sampling_spec) -> float:
    score = 0.0
    for param, spec in sampling_spec.items():
        direction = spec.get("complexity", None)
        if direction is None:
            continue
        if param in config:
            value = config[param]
        elif hasattr(env, param):
            value = getattr(env, param)
            config[param] = value
        else:
            raise ValueError(f"Parameter '{param}' not found in config or environment attributes.")

        if spec.get("env_attribute", False):
            min_val = getattr(env, "min_" + param, None)
            max_val = getattr(env, "max_" + param, None)
            config["min_" + param] = min_val
            config["max_" + param] = max_val
        else:
            min_val, max_val = spec["min"], spec["max"]

        norm_val = 0.0 if max_val == min_val else (value - min_val) / (max_val - min_val)

        if direction == "max":
            score += norm_val
        elif direction == "min":
            score += 1.0 - norm_val
        else:
            raise ValueError(f"Unknown complexity direction: {direction}")

    config["complexity_score"] = score
    return score, config

def create_env(env_class, config: dict, env_config: dict, verbose: int):
    return env_class(**config, **env_config, verbose=verbose)

def save_env(BASE_DIR, env, env_id, env_name: str, config: dict):
    env_dir = BASE_DIR / env_name / f"{env_id}"
    env_dir.mkdir(parents=True, exist_ok=True)
    with open(env_dir / "env.pkl", "wb") as f:
        pickle.dump(env, f)
    with open(env_dir / "env_info.yaml", "w") as f:
        yaml.dump(config, f)
    return env

def get_important_specs(sampling_spec):
    return {p: spec for p, spec in sampling_spec.items() if spec.get("important", False)}

def get_size_replacement_specs(sampling_spec):
    # parameters that should be replaced with size-based sampling (e.g., number of nodes, edges, etc.) rather than random sampling
    return {p: spec for p, spec in sampling_spec.items() if spec.get("size_replacement", False)}

def is_feature_extreme(x, spec, threshold=0.9):
    direction = spec.get("complexity", None)
    if direction == "max":
        return x >= threshold
    elif direction == "min":
        return x <= (1.0 - threshold)
    else:
        return x <= (1.0 - threshold) or x >= threshold

def feature_extremeness_map(features, sampling_spec, threshold=0.9):
    info = {}
    for p, spec in get_important_specs(sampling_spec).items():
        x = features[p]
        info[p] = {
            "normalized": x,
            "extreme": is_feature_extreme(x, spec, threshold),
            "direction": spec.get("complexity", "both"),
        }
    return info



def compute_normalized_features(config, env, sampling_spec):
    # Compute normalized features for the environment based on the sampling_spec. For each parameter in the sampling_spec, if it is present in the config or as an attribute of the environment, compute its normalized value based on the min and max specified in the sampling_spec (or from the environment attributes if "env_attribute" is True). The normalized value should be between 0 and 1, where 0 corresponds to the minimum value and 1 corresponds to the maximum value. If the max and min values are equal, set the normalized value to 0.5.
    features = {}
    for param, spec in sampling_spec.items():
        if param in config:
            value = config[param]
        elif hasattr(env, param):
            value = getattr(env, param)
        else:
            continue

        if spec.get("env_attribute", False):
            min_val = getattr(env, "min_" + param)
            max_val = getattr(env, "max_" + param)
        else:
            min_val, max_val = spec["min"], spec["max"]

        if max_val == min_val:
            features[param] = 0.5
        else:
            features[param] = (value - min_val) / (max_val - min_val)

    return features

def compute_env_score_based_on_important_params(env_config, important_params):
    # Compute a score based on important parameters
    score = 1
    for param_name, param_info in important_params.items():
        if param_name not in env_config:
            continue
        value = env_config[param_name]
        score *= value
    return score