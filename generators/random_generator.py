#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

"""
    random_generator.py
    Script to generate random environments based on sampling specifications and split them based on complexity or other criteria
"""

import os
import random
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import glob
import sys
import importlib
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))
from utils.file_utils import load_yaml, save_yaml, load_json
import numpy as np
from utils.math_utils import set_seeds
from utils.generation_utils import compute_complexity, create_env, save_env, get_important_specs, get_size_replacement_specs, compute_normalized_features, feature_extremeness_map
# Base directory for generated environments
BASE_DIR = Path(os.path.join(script_dir, "..", "data", "env_samples"))

def sample_config(sampling_spec, n_samples=10, mode="extrapolation", half_range=None, debug=False):
    # Sample a single configuration dictionary based on the mode and the sampling_spec which defines the parameters, their types, ranges, and complexity directions.
    config = {}

    for param, spec in sampling_spec.items():
        if spec.get("env_attribute", False):
            continue

        p_type = spec.get("type")
        direction = spec.get("complexity", None)
        lo, hi = spec.get("min"), spec.get("max")
        if mode == "half_range" and half_range is not None:
            lo, hi = spec.get("min"), spec.get("max")

            # Determine if we have a restricted half for size split
            if param in half_range:
                if half_range[param] == "lower":
                    hi = lo + (hi - lo) / 2
                else:  # "upper"
                    lo = lo + (hi - lo) / 2

        if mode == "half_range" and half_range is not None:
            # Sampling
            if p_type == "int":
                value = random.randint(int(lo), int(hi))
            elif p_type == "float":
                value = random.uniform(lo, hi)
            elif p_type == "choice":
                value = random.choice(spec.get("values"))
            else:
                raise ValueError(f"Unsupported type '{p_type}' for {param}")

        # --- Random mode: fully random sampling ---
        elif mode == "random" or mode == "largest_train" or mode == "smallest_train":
            if p_type == "int":
                value = random.randint(lo, hi)
            elif p_type == "float":
                value = random.uniform(lo, hi)
            elif p_type == "choice":
                value = random.choice(spec.get("values"))
            else:
                raise ValueError(f"Unsupported type '{p_type}' for {param}")

        # --- Complexity legacy mode ---
        elif mode == "complexity":
            extreme_side = random.choice(["low", "high"])
            if p_type in ["int", "float"]:
                if direction == "max":
                    value = hi
                elif direction == "min":
                    value = lo
                else:
                    value = hi if extreme_side == "high" else lo
                if p_type == "int":
                    value = int(round(value))
            elif p_type == "choice":
                value = random.choice(spec.get("values"))
            else:
                raise ValueError(f"Unsupported type '{p_type}' for {param}")

        # --- New uniform linspace modes ---
        elif mode in ["extrapolation", "interpolation"]:
            if direction is not None and p_type in ["int", "float"]:
                n_points = min(n_samples, 10)
                if mode == "extrapolation":
                    if direction == "max":
                        values = np.linspace(lo, hi, n_points)[-n_points:]
                    elif direction == "min":
                        values = np.linspace(lo, hi, n_points)[:n_points]
                    else:
                        values = np.linspace(lo, hi, n_points)
                else: #if mode == "interpolation":
                    margin = 0.1 * (hi - lo)
                    values = np.linspace(lo + margin, hi - margin, n_points)
                values = [int(round(v)) for v in values] if p_type == "int" else values.tolist()
                value = random.choice(values)
            else:
                # Non-important parameters: random sampling
                if p_type == "int":
                    value = random.randint(lo, hi)
                elif p_type == "float":
                    value = random.uniform(lo, hi)
                elif p_type == "choice":
                    value = random.choice(spec.get("values"))
                else:
                    raise ValueError(f"Unsupported type '{p_type}' for {param}")

        else:
            raise ValueError(f"Unknown mode '{mode}'")

        config[param] = value
        if debug:
            kind = "important" if direction else "non-important"
            print(f"[SAMPLER] {param} ({kind}): {value}")

    return config

# ---------------------------
# Splitting
# ---------------------------
def assign_splits(env_infos, train_pct, val_pct, test_pct, split_method="complexity", sampling_spec=None):
    n = len(env_infos)
    if n == 1:
        return {
            "params": {"train_pct": 1.0, "val_pct": 0.0, "test_pct": 0.0},
            "training_set": env_infos,
            "validation_set": [],
            "test_set": []
        }

    train_f = train_pct * n
    val_f = val_pct * n
    test_f = test_pct * n
    # Step 2: floor counts
    n_train = int(np.floor(train_f))
    n_val = int(np.floor(val_f))
    n_test = int(np.floor(test_f))

    # Step 3: distribute remaining environments to the largest fractional parts
    remaining = n - (n_train + n_val + n_test)
    fractions = [(train_f - n_train, 'train'), (val_f - n_val, 'val'), (test_f - n_test, 'test')]
    fractions.sort(reverse=True)  # largest fractional part first
    for frac, set_name in fractions:
        if remaining <= 0:
            break
        if set_name == 'train':
            n_train += 1
        elif set_name == 'val':
            n_val += 1
        elif set_name == 'test':
            n_test += 1
        remaining -= 1
    # Step 4: ensure at least 1 if fraction > 0
    if train_pct > 0 and n_train == 0:
        n_train = 1
        n_test = max(0, n_test - 1)
    if val_pct > 0 and n_val == 0:
        n_val = 1
        n_test = max(0, n_test - 1)
    if test_pct > 0 and n_test == 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    if split_method in ["random", "complexity"]:
        if split_method == "random":
            env_infos_sorted = random.sample(env_infos, k=n)
        else:
            env_infos_sorted = sorted(env_infos, key=lambda x: x["score"])
        return {
            "params": {"train_pct": train_pct, "val_pct": val_pct, "test_pct": test_pct,
                       "train_num": n_train, "val_num": n_val, "test_num": n_test},
            "training_set": env_infos_sorted[:n_train],
            "validation_set": env_infos_sorted[n_train:n_train+n_val],
            "test_set": env_infos_sorted[n_train+n_val:]
        }

    elif split_method == "largest_train":
        env_infos_sorted = sorted(env_infos, key=lambda x: x["score"], reverse=True)
        return {
            "params": {"split": split_method, "train_pct": train_pct, "val_pct": val_pct, "test_pct": test_pct,
                       "train_num": n_train, "val_num": n_val, "test_num": n_test},
            "training_set": env_infos_sorted[:n_train],
            "validation_set": env_infos_sorted[n_train:n_train+n_val],
            "test_set": env_infos_sorted[n_train+n_val:]
        }
    elif split_method == "smallest_train":
        env_infos_sorted = sorted(env_infos, key=lambda x: x["score"])
        return {
            "params": {"split": split_method, "train_pct": train_pct, "val_pct": val_pct, "test_pct": test_pct,
                       "train_num": n_train, "val_num": n_val, "test_num": n_test},
            "training_set": env_infos_sorted[:n_train],
            "validation_set": env_infos_sorted[n_train:n_train+n_val],
            "test_set": env_infos_sorted[n_train+n_val:]
        }

    elif split_method in ["extrapolation", "interpolation"]:
        assert sampling_spec is not None
        important_specs = get_important_specs(sampling_spec)
        important_keys = list(important_specs.keys())

        # Main extremeness score based on important features
        def extremeness_score(env):
            score = 0.0
            for k in important_keys:
                v = env["features"][k]
                spec = important_specs[k]
                dir = spec.get("complexity", None)
                if dir == "max":
                    score += v
                elif dir == "min":
                    score += 1.0 - v
                else:
                    score += max(v, 1.0 - v)
            return score

        def interior_score(env):
            score = 0.0
            for k in important_keys:
                v = env["features"][k]
                dir = important_specs[k].get("complexity", None)
                if dir == "max":
                    score += 1.0 - v  # smaller values are more interior
                elif dir == "min":
                    score += v  # larger values are more interior
                else:
                    score += 1.0 - abs(0.5 - v) * 2  # distance from 0.5 (middle) is more interior
            return score

        # Tie-break score based on all remaining features (with or without complexity)
        other_keys = [k for k in sampling_spec.keys() if k not in important_keys]

        def tie_break(env):
            score = 0.0
            for k in other_keys:
                v = env["features"][k]
                spec = sampling_spec[k]
                dir = spec.get("complexity", None)
                if dir == "max":
                    score += v
                elif dir == "min":
                    score += 1.0 - v
                else:
                    score += max(v, 1.0 - v)  # extremeness even without complexity
            return score

        for env in env_infos:
            env["_ext_score"] = extremeness_score(env)
            env["_tie_score"] = tie_break(env)
            env["_int_score"] = interior_score(env)

        if split_method == "extrapolation":
            # Most extreme first
            env_infos_sorted = sorted(env_infos, key=lambda e: (e["_ext_score"], e["_tie_score"]), reverse=True)
            test_set = env_infos_sorted[:n_test]
            val_set = env_infos_sorted[n_test:n_test+n_val]
            train_set = env_infos_sorted[n_test+n_val:]
        else:  # interpolation
            # Sort by interior score: most interior first
            env_infos_sorted = sorted(env_infos, key=lambda e: (e["_int_score"], e["_tie_score"]), reverse=True)
            # Test should be the MOST interior
            test_set = env_infos_sorted[:n_test]
            val_set = env_infos_sorted[n_test:n_test + n_val]
            # Train on the remaining (more extreme) ones
            train_set = env_infos_sorted[n_test + n_val:]

        # Clean up temp scores
        for env in env_infos:
            env.pop("_ext_score", None)
            env.pop("_tie_score", None)
            env.pop("_int_score", None)

        return {
            "params": {"split": split_method, "strict": False,
                       "train_pct": train_pct, "val_pct": val_pct, "test_pct": test_pct,
                       "train_num": n_train, "val_num": n_val, "test_num": n_test
                       },
            "training_set": train_set,
            "validation_set": val_set,
            "test_set": test_set
        }
    elif split_method == "half_range":
        n_lower = n // 2
        train_set = env_infos[:int(n_lower)]
        val_set = []
        test_set = env_infos[int(n_lower):]
        return {
            "params": {"split": "size",
                       "train_pct": train_pct, "val_pct": val_pct, "test_pct": test_pct,
                       "train_num": n_train, "val_num": n_val, "test_num": n_test
                       },
            "training_set": train_set,
            "validation_set": val_set,
            "test_set": test_set
        }

    else:
        raise ValueError(f"Unknown split method: {split_method}")

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_samples", type=int, default=10)
    parser.add_argument("-e", "--environment", type=str, required=True)
    parser.add_argument("--train_pct", default=None)
    parser.add_argument("--val_pct", default=None)
    parser.add_argument("--test_pct", default=None)
    parser.add_argument("-ms", "--max_sweeps", type=int, default=0)
    parser.add_argument("-s", "--split", type=str, default="smallest_train",
                        choices=["complexity", "random", "largest_train", "smallest_train", "extrapolation", "interpolation", "half_range"])
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="Sizes to use at the place of yaml-specified ranges (for debugging)")
    parser.add_argument("--size_for_all", type=int,default=None,
                        help="Sizes to use at the place of yaml-specified ranges (for debugging)")

    parser.add_argument('-v', '--verbose', default=1, type=int)
    parser.add_argument('--static_seed', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seed', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seed', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument("--force_saving", action="store_true", default=False)
    args = parser.parse_args()

    if args.sizes is not None:
        args.n_samples = len(args.sizes)


    # if none specified, ntrain is 1/n and rest test
    if args.train_pct is None and args.val_pct is None and args.test_pct is None:
        args.train_pct = 1.0 / args.n_samples
        args.val_pct = 0
        args.test_pct = 1.0 - args.train_pct
    else:
        if args.train_pct is None:
            assert False, "train_pct must be specified if any of train_pct, val_pct, test_pct is specified"
        else:
            # convert to float before checking sum
            if args.train_pct is not None:
                args.train_pct = float(args.train_pct)
            if args.val_pct is not None:
                args.val_pct = float(args.val_pct)
            if args.test_pct is not None:
                args.test_pct = float(args.test_pct)
            if args.train_pct is not None and args.val_pct is not None and args.test_pct is not None:
                pass
            elif args.train_pct is not None and args.val_pct is not None:
                args.test_pct = 1.0 - args.train_pct - args.val_pct
            elif args.train_pct is not None and args.test_pct is not None:
                args.val_pct = 1.0 - args.train_pct - args.test_pct


    if abs((args.train_pct + args.val_pct + args.test_pct) - 1.0) > 1e-6:
        raise ValueError("train_pct, val_pct and test_pct must sum to 1.0")

    env_registry = load_yaml(os.path.join(script_dir, "..", "envs", "config", "env_registry.yaml"))
    if args.environment not in env_registry:
        raise ValueError(f"Environment '{args.environment}' not found in env_registry.yaml")

    if args.static_seed:
        seed = [42 for _ in range(args.num_episodes_per_checkpoint)]
    elif args.random_seed:
        seed = [np.random.randint(1000) for _ in range(args.num_episodes_per_checkpoint)]
    else: # args.load_seed, first seed
        with open(os.path.join(script_dir, args.load_seed, 'seeds.yaml'), 'r') as seeds_file:
            seeds_loaded = yaml.safe_load(seeds_file)
        seed = seeds_loaded['seeds'][0]
    set_seeds(seed)

    module_name = env_registry[args.environment]["module"]
    class_name = env_registry[args.environment]["class"]
    module = importlib.import_module(module_name)
    env_class = getattr(module, class_name)

    sampling_spec = load_yaml(os.path.join(script_dir, "config", f"{args.environment}_ranges.yaml"))
    env_config = load_yaml(os.path.join(script_dir, "..", "envs", "config", f"{args.environment}_config.yaml"))

    if args.max_sweeps > 0:
        env_config["max_sweeps"] = args.max_sweeps

    # Load extra configs
    pattern = f"{args.environment}_*.*"
    for file_path in glob.glob(os.path.join(script_dir, "..", "envs", "config", pattern)):
        base_name = os.path.basename(file_path)
        suffix = base_name[len(args.environment) + 1:].split(".")[0]

        if suffix == "config":
            continue
        fmt = base_name.split(".")[-1]
        if fmt in ["yaml", "yml"]:
            env_config[suffix] = load_yaml(file_path)
        elif fmt == "json":
            env_config[suffix] = load_json(file_path)
        else:
            print(f"[WARN] Unsupported file format: {file_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.environment}_{timestamp}"
    out_dir = BASE_DIR / experiment_name

    env_infos = []
    n_lower = args.n_samples // 2
    important_keys = list(get_important_specs(sampling_spec).keys())
    size_replacement_keys = list(get_size_replacement_specs(sampling_spec).keys())
    env = None
    config = None
    print(f"[INFO] Generating {args.n_samples} '{args.environment}' environments in {out_dir}")
    general_config = load_yaml(os.path.join(script_dir, "..", "config.yaml"))
    keys_config = {k: v for k, v in general_config.items() if isinstance(v, dict) and 'key' in v}

    for i in tqdm(range(args.n_samples), desc="Generating environments"):
        print('Generating environment {}/{}...'.format(i, args.n_samples))
        success = False
        while not success:
            try:
                if i < n_lower:
                    half_range = {k: "lower" for k in important_keys}
                else:
                    half_range = {k: "upper" for k in important_keys}

                if args.sizes is not None or args.size_for_all is not None:
                    # change size_replacement keys to have min=max=size for current index
                    if len(size_replacement_keys) > 1:
                        for k in size_replacement_keys:
                            if k in sampling_spec:
                                size = args.sizes[i] if args.sizes is not None else args.size_for_all
                                sampling_spec[k]["min"] = int(size/len(size_replacement_keys))
                                sampling_spec[k]["max"] = int(size/len(size_replacement_keys))
                    else:
                        k = size_replacement_keys[0]
                        if k in sampling_spec:
                            size = args.sizes[i] if args.sizes is not None else args.size_for_all
                            sampling_spec[k]["min"] = size
                            sampling_spec[k]["max"] = size

                config = sample_config(sampling_spec, args.n_samples, half_range=half_range, mode=args.split)
                for key in keys_config:
                    config[key] = keys_config[key]  # add to environment configuration the values of the general configuration that are relevant for the environment (e.g. with same key)

                env = create_env(env_class, config, env_config, args.verbose)
                success = True
            except ValueError:
                env = None
                config = None
                print(f"[WARN] Failed to create environment with config: {config}. Retrying...")

        complexity, config = compute_complexity(config, env, sampling_spec)
        save_env(BASE_DIR, env, i, experiment_name, config)
        features = compute_normalized_features(config, env, sampling_spec)
        feature_info = feature_extremeness_map(features, sampling_spec)
        extremeness = sum(v["extreme"] for v in feature_info.values()) / len(feature_info) if feature_info else 0
        copy_config = {k: v for k, v in config.items() if not (isinstance(v, dict) and 'key' in v)}
        env_infos.append({
            "id": i,
            "config": copy_config,
            "score": complexity,
            "features": features,
            "feature_info": feature_info,
            "extremeness": extremeness
        })

    # Split and save
    split_info = assign_splits(env_infos, args.train_pct, args.val_pct, args.test_pct, args.split, sampling_spec)
    save_yaml(split_info, out_dir, "split.yaml")
    print(f"[INFO] Saved splits and complexity to {out_dir / 'split.yaml'}")

    # === Ask user if this should be default environment ===
    if args.force_saving:
        config_path = os.path.join(script_dir, "../config.yaml")
        main_config = load_yaml(config_path)

        # Ensure default_envs exists
        if args.environment not in main_config:
            main_config[args.environment] = {}

        # Update default_envs
        if args.split == "extrapolation":
            main_config[args.environment]["default_extrapolation_envs"] = experiment_name
        elif args.split == "interpolation":
            main_config[args.environment]["default_interpolation_envs"] = experiment_name
        else:
            main_config[args.environment]["default_envs"] = experiment_name

        # Save updated config.yaml
        save_yaml(main_config, os.path.dirname(config_path), os.path.basename(config_path))

    else:
        user_input = input(
            f"\nDo you want to set '{experiment_name}' as default for '{args.environment}'? (y/n): ").strip().lower()
        if user_input in ["y", "yes"]:
            config_path = os.path.join(script_dir, "../config.yaml")
            if not os.path.exists(config_path):
                print(f"Config file not found at {config_path}. Creating a new one.")
                main_config = {}
            else:
                main_config = load_yaml(config_path)

            # Ensure default_envs exists
            if args.environment not in main_config:
                main_config[args.environment] = {}

            # Update default_envs
            main_config[args.environment]["default_envs"] = experiment_name

            # Save updated config.yaml
            save_yaml(main_config, os.path.dirname(config_path), os.path.basename(config_path))

            print(f"✅ Updated config.yaml: set default_envs.{args.environment} = {experiment_name}")
        else:
            print("Skipped updating default environment.")


if __name__ == "__main__":
    main()
