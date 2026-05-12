#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    random_cartesian_scalability_generator.py
    Script to generate environments for scalability evaluation with Cartesian splits. For each size in the specified range, it generates a fixed number of environments, computes their complexity, and then creates train-test splits based on size. The generated environments and splits are saved in a structured directory for easy access during training and evaluation.
"""

import copy
import os
import random
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import sys
import importlib
import glob
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))
from utils.file_utils import load_yaml, save_yaml, load_json
import numpy as np
from utils.math_utils import set_seeds
from utils.generation_utils import compute_complexity, create_env, save_env

# Base directory for generated environments
BASE_DIR = Path(os.path.join(script_dir, "..", "data", "env_samples"))

nodes_params = {
    "cyberattack": "n_nodes",
    "ospf_engineering": "num_nodes",
    "traffic_engineering": "num_nodes",
    "vmp": ["n_vms", "n_pms"],
    "tsp": "num_cities",
    "mvc": "num_nodes",
    "maxcut": "num_nodes",
}

def sample_other_params(sampling_spec):
    config = {}
    for param, spec in sampling_spec.items():
        if param in [elem for elem in nodes_params.values() if isinstance(elem, str)] or \
              param in [p for p_list in nodes_params.values() if isinstance(p_list, list) for p in p_list]:
            continue
        p_type = spec.get("type")
        if "env_attribute" in spec and spec["env_attribute"]:
            continue
        if p_type == "int":
            config[param] = random.randint(spec["min"], spec["max"])
        elif p_type == "float":
            config[param] = random.uniform(spec["min"], spec["max"])
        elif p_type == "choice":
            config[param] = random.choice(spec["values"])
        else:
            raise ValueError(f"Unsupported type for param '{param}'")
    return config

def generate_cartesian_scenarios(
    env_class,
    sampling_spec,
    env_config,
    env_name,
    min_size,
    max_size,
    interval,
    num_envs_per_size,
    verbose
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{env_name}_cartesian_{min_size}_{max_size}_step{interval}_{timestamp}"
    experiment_dir = BASE_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    sizes = list(range(min_size, max_size + 1, interval))

    print(f"[INFO] Sizes: {sizes}")
    print(f"[INFO] Generating {num_envs_per_size} envs per size")

    # Store all generated environments by size
    all_envs_by_size = {size: [] for size in sizes}

    scenario_global_id = 0

    other_params_dict = {}
    # Sample just once the parameters of size num_envs_per_size, to store the information
    for i in range(num_envs_per_size):
        other_params_dict[i] = sample_other_params(sampling_spec)

    # -------------------------------------------------
    # 1️⃣ Generate ALL environments first
    # -------------------------------------------------
    for size in sizes:
        print(f"[INFO] Generating environments for size {size}")

        for idx in range(num_envs_per_size):
            attempts = 0
            while True:
                if attempts == 0:
                    base_cfg = copy.deepcopy(other_params_dict[idx])
                else:
                    print("[WARN] Sampling new parameters for env creation due to previous failure.")
                    base_cfg = sample_other_params(sampling_spec)


                if isinstance(nodes_params[env_name], str):
                    base_cfg[nodes_params[env_name]] = size
                else:
                    p1, p2 = nodes_params[env_name]
                    v1 = size // 2
                    v2 = size - v1
                    base_cfg[p1] = v1
                    base_cfg[p2] = v2

                attempts += 1
                try:
                    env = create_env(env_class, base_cfg, env_config, verbose)
                    break
                except ValueError:
                    env = None
                    continue

            complexity, base_cfg = compute_complexity(base_cfg, env, sampling_spec)

            env_id = f"{size}_{idx}"
            save_env(BASE_DIR, env, env_id, experiment_dir, base_cfg)

            info = {
                "id": env_id,
                "config": base_cfg,
                "score": complexity,
                "num_nodes": size
            }

            all_envs_by_size[size].append(info)

            scenario_global_id += 1

    # -------------------------------------------------
    # 2️⃣ Create Cartesian splits
    # -------------------------------------------------
    print("[INFO] Creating Cartesian train-test splits")

    for train_size in sizes:
        for test_size in sizes:
            split_data = {
                "params": {
                    "train_size": train_size,
                    "test_size": test_size,
                    "num_envs_per_size": num_envs_per_size
                },
                "training_set": all_envs_by_size[train_size],
                "validation_set": [],
                "test_set": all_envs_by_size[test_size]
            }

            split_filename = f"split_{train_size}_{test_size}.yaml"
            save_yaml(split_data, experiment_dir, split_filename)

    print(f"[INFO] Cartesian scenario generation complete: {experiment_dir}")

    return experiment_name


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str, required=True, help="Environment name")
    parser.add_argument("-s", "--split", type=str, default="random", choices=["complexity", "random"], help="Split method")
    parser.add_argument("-ms", "--max_sweeps", type=int, default=0,
                        help="If different than 0, overwrites the max sweeps defined in the environment")
    parser.add_argument("--min_size", type=int, required=True)
    parser.add_argument("--max_size", type=int, required=True)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--num_envs", type=int, required=True, help="Number of training scenarios per block")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbose output")
    parser.add_argument('--static_seed', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seed', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seed', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument("--force_saving", action="store_true", default=False)
    args = parser.parse_args()

    # Load environment class dynamically
    env_registry = load_yaml(os.path.join(script_dir, "..", "envs", "config", "env_registry.yaml"))
    if args.environment not in env_registry:
        raise ValueError(f"Environment '{args.environment}' not found in env_registry.yaml")

    if args.static_seed:
        seed = [42 for _ in range(args.num_episodes_per_checkpoint)]
    elif args.random_seed:
        seed = [np.random.randint(1000) for _ in range(args.num_episodes_per_checkpoint)]
    else: # args.load_seed, first seed
        with open(os.path.join(args.load_seed, 'seeds.yaml'), 'r') as seeds_file:
            seeds_loaded = yaml.safe_load(seeds_file)
        seed = seeds_loaded['seeds'][0]
    set_seeds(seed)

    module_name = env_registry[args.environment]["module"]
    class_name = env_registry[args.environment]["class"]
    module = importlib.import_module(module_name)
    env_class = getattr(module, class_name)

    # Load sampling spec and static config
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

    experiment_name = generate_cartesian_scenarios(
        env_class=env_class,
        sampling_spec=sampling_spec,
        env_config=env_config,
        env_name=args.environment,
        min_size=args.min_size,
        max_size=args.max_size,
        interval=args.interval,
        num_envs_per_size=args.num_envs,
        verbose=args.verbose,
    )
    if args.force_saving:
        config_path = os.path.join(script_dir, "..", "config.yaml")
        if not os.path.exists(config_path):
            print(f"Config file not found at {config_path}. Creating a new one.")
            main_config = {}
        else:
            main_config = load_yaml(config_path)

        if main_config is None:
            main_config = {}

        # Ensure default_envs exists
        if args.environment not in main_config:
            main_config[args.environment] = {}

        # Update default_envs
        main_config[args.environment]["default_cartesian_envs"] = str(experiment_name)

        # Save updated config.yaml
        save_yaml(main_config, os.path.dirname(config_path), os.path.basename(config_path))

        print(f"✅ Updated config.yaml: set default_cartesian_envs.{args.environment} = {experiment_name}")
    else:
        # === Ask user if this should be default environment ===
        user_input = input(
            f"\nDo you want to set '{experiment_name}' as default for '{args.environment}'? (y/n): ").strip().lower()
        if user_input in ["y", "yes"]:
            config_path = os.path.join(script_dir, "..", "config.yaml")
            if not os.path.exists(config_path):
                print(f"Config file not found at {config_path}. Creating a new one.")
                main_config = {}
            else:
                main_config = load_yaml(config_path)

            if main_config is None:
                main_config = {}

            # Ensure default_envs exists
            if args.environment not in main_config:
                main_config[args.environment] = {}

            # Update default_envs
            main_config[args.environment]["default_cartesian_envs"] = str(experiment_name)

            # Save updated config.yaml
            save_yaml(main_config, os.path.dirname(config_path), os.path.basename(config_path))

            print(f"✅ Updated config.yaml: set default_cartesian_envs.{args.environment} = {experiment_name}")
        else:
            print("Skipped updating default environment.")

    print(f"[INFO] Scenario generation complete.")


if __name__ == "__main__":
    main()


