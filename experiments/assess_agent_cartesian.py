#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
This script runs a full pipeline of experiments to assess the performance of RL agents trained on a single environment and tested on a range of environments of increasing difficulty/size.
The training environment is selected based on a specified sampling strategy (largest, smallest, mean, or random) using important parameters defined in the environment ranges configuration.
'''
import random
import numpy as np
import argparse
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import os
import copy
import glob
from utils.file_utils import load_yaml, save_yaml
from utils.generation_utils import compute_env_score_based_on_important_params
from utils.proc_utils import run, wait_for_slot, wait_all
from utils.experiments_utils import map_algorithm_type_to_original_and_override_args
script_dir = Path(__file__).parent.resolve()

def select_env(envs, sampling_strategy, important_params, already_selected_ids):
    envs = copy.deepcopy(envs)  # Avoid modifying original list
    for env in envs:
        if env['id'] in already_selected_ids:
            envs.remove(env)
    # Compute scores only for available envs
    scored_envs = [
        (env, compute_env_score_based_on_important_params(env["config"], important_params))
        for env in envs
    ]
    if sampling_strategy == "largest":
        scored_envs.sort(key=lambda x: x[1], reverse=True)
    elif sampling_strategy == "smallest":
        scored_envs.sort(key=lambda x: x[1])
    elif sampling_strategy == "mean":
        mean_score = np.mean([s for _, s in scored_envs])
        scored_envs.sort(key=lambda x: abs(x[1] - mean_score))
    elif sampling_strategy == "random":
        random.shuffle(scored_envs)
    else:
        raise ValueError(f"Unknown env_sampling strategy '{sampling_strategy}'")
    return scored_envs[0][0]



def run_cartesian_experiments(env_name,
                              args,
                              common_logs_folder_name=None,
                              seeds_runs=None,
                              ):
    # 1. Load experiment folder from config.yaml
    config_path = os.path.join(script_dir, "..", "config.yaml")
    main_config = load_yaml(config_path)

    if env_name not in main_config or "default_cartesian_envs" not in main_config[env_name]:
        raise ValueError(f"No cartesian experiment found in config.yaml for {env_name}")

    experiment_name = main_config[env_name]["default_cartesian_envs"]
    experiment_dir = Path(script_dir) / ".." / "data" / "env_samples" / experiment_name
    experiment_dir = experiment_dir.resolve()

    print(f"[INFO] Using experiment folder: {experiment_dir}")

    # 2. Detect all possible sizes
    split_files = sorted(experiment_dir.glob("split_*.yaml"))
    sizes = sorted(
        list(set(
            int(Path(f).stem.split("_")[1])
            for f in split_files
        ))
    )

    print(f"[INFO] Detected train sizes: {sizes}")

    if args.option == "action_time":
        # keep only the smallest size for training
        train_sizes = [sizes[0]]
    else:
        train_sizes = sizes

    # 3. Loop per training size
    logs_folders = {}
    for train_size in train_sizes:

        print(f"\n========== TRAIN SIZE {train_size} ==========")

        # load all splits starting with split_{train_size} and store them into an array
        all_splits = sorted(glob.glob(str(experiment_dir / f"split_{train_size}_*.yaml")))
        # exclude those that include run in the name
        all_splits = [s for s in all_splits if "run" not in s]
        # take unique
        all_splits = list(set(all_splits))

        # Select one training environment
        ranges_path = Path(os.path.join(script_dir, "..", "generators", "config")) / f"{env_name}_ranges.yaml"
        ranges_data = load_yaml(ranges_path)
        important_params = {k: v for k, v in ranges_data.items() if v.get("biggest", False)}

        already_selected_ids = set()
        for run_idx in range(args.num_runs):
            biggest_split = experiment_dir / f"split_{train_size}_{sizes[-1]}.yaml"
            biggest_split_data = load_yaml(biggest_split)
            train_envs = biggest_split_data["training_set"]
            train_env = select_env(train_envs, args.env_sampling, important_params, already_selected_ids)

            print(f"[INFO] Selected training env: {train_env['id']}")
            already_selected_ids.add(train_env['id'])

            current_seed = seeds_runs[run_idx] if seeds_runs else None

            new_all_splits = []
            test_envs = []
            for split in all_splits:
                split_data = load_yaml(split)
                test_envs = split_data["test_set"]
                split_data["training_set"] = [train_env]
                split_data["validation_set"] = []  # Clear validation set for now, can be modified if needed
                split_data["params"]["train_pct"] = 1 / (len(test_envs) + 1)
                split_data["params"]["test_pct"] = 1 - split_data["params"]["train_pct"]
                # include run name in split
                split_new = split.replace(".yaml", f"_run{run_idx + 1}.yaml")
                save_yaml(split_data, split_new)
                new_all_splits.append(split_new)

            # save biggest split as the one to load during training to get maximum padding
            save_yaml(biggest_split_data, os.path.join(experiment_dir, "split.yaml"))
            print(f"Run {run_idx + 1}/{args.num_runs}: Training set = {train_env['id']}, Test set = {len(test_envs)}")

            # 2. GAE TRAINING
            if not args.skip_gae:
                cmd = [
                    "python3", os.path.join(script_dir, "..", "gae", "train_gae.py"),
                    "-e", args.environment,
                    "-ti", str(args.num_iter_gae),
                    "--force_saving",
                    "--static_seed", str(current_seed) if current_seed is not None else 0,
                ]
                cmd += ["--load_envs_mode", "cartesian"]

                if args.validation:
                    cmd.append("--validation")

                if args.name:
                    cmd += ["--name", args.name]

                p = run(cmd)
                # add validation optional
                if p.wait() != 0:
                    sys.exit(1)

            # 3. RL AGENT TRAINING (SEQUENTIAL)
            print("\n===== STARTING SEQUENTIAL TRAINING =====\n", flush=True)

            if not args.skip_training:
                for algorithm_type in args.algorithm_type:
                    algorithm_folder, copy_args = map_algorithm_type_to_original_and_override_args(algorithm_type, args)
                    common_logs_folder = os.path.join(script_dir, "..", "agents", "logs", args.environment, algorithm_folder,
                                                      common_logs_folder_name, f"{train_size}_run_" + str(run_idx + 1))
                    cmd = [
                        "python3", os.path.join(script_dir, "..", "agents", "train_agent.py"),
                        "-e", copy_args['environment'],
                        "-at", copy_args['algorithm_type'],
                        "-ti", str(copy_args['num_iter_rl']),
                        "-algo", copy_args['algorithm'],
                    ]
                    logs_folders[algorithm_type] = common_logs_folder
                    os.makedirs(common_logs_folder, exist_ok=True)
                    if copy_args['GNN_observations']:
                        cmd += ["-GO"]
                    if copy_args['approximate_distance']:
                        cmd += ["-approx"]
                    if copy_args['pca_minimum_without_loss']:
                        cmd += ["--pca_minimum_without_loss"]
                    if copy_args['semantic_ordering']:
                        cmd += ["--semantic_ordering"]
                    cmd += ["--logs_folder", common_logs_folder]
                    cmd += ["--static_seed", str(current_seed) if current_seed is not None else 0]
                    cmd += ["--load_envs_mode", "cartesian"]

                    if copy_args['sample_subset_actions']:
                        cmd += ["--sample_subset_actions", str(args.sample_subset_actions)]

                    if args.validation:
                        cmd.append("--validation")

                    if args.name:
                        cmd += ["--name", args.name]

                    p = run(cmd)
                    if p.wait() != 0:
                        sys.exit(1)

            print("\n===== ALL TRAINING FINISHED =====\n", flush=True)

            # 4. TESTING (PARALLEL)
            print("\n===== STARTING PARALLEL TESTING =====\n", flush=True)
            test_procs = []

            args.max_parallel_test = min(args.max_parallel_test, len(args.algorithm_type) * 2)

            for split in new_all_splits:
                for algorithm_type in args.algorithm_type:
                    algorithm_folder, copy_args = map_algorithm_type_to_original_and_override_args(algorithm_type, args)

                    common_logs_folder = os.path.join(script_dir, "..", "agents", "logs", args.environment, algorithm_folder,
                                                      common_logs_folder_name, f"{train_size}_run_" + str(run_idx + 1))
                    save_yaml(load_yaml(split), os.path.join(common_logs_folder, "split.yaml"))
                    split_size = load_yaml(split)["params"]["test_size"]
                    num_envs = load_yaml(split)["params"]["num_envs_per_size"]
                    wait_for_slot(test_procs, args.max_parallel_test)
                    cmd = [
                        "python3", os.path.join(script_dir, "..", "agents", "test_agent.py"),
                        "-algo", copy_args['algorithm'],
                        "-at", copy_args['algorithm_type'],
                        "-e", copy_args['environment'],
                        "--last_checkpoint",
                        "-o", args.option,
                    ]
                    if copy_args['GNN_observations']:
                        cmd += ["-GO"]
                    if copy_args['approximate_distance']:
                        cmd += ["-approx"]
                    if copy_args['semantic_ordering']:
                        cmd += ["--semantic_ordering"]
                    if copy_args['pca_minimum_without_loss']:
                        cmd += ["--pca_minimum_without_loss"]

                    cmd += ["-e", args.environment,
                            "-ne", str(num_envs * args.number_episodes_per_env)]
                    cmd += ["--logs_folder", common_logs_folder]
                    cmd += ["--custom_test_folder_name", f"test_size{split_size}"]
                    cmd.append("--load_default_test_envs")
                    if args.val_checkpoints:
                        cmd.append("--val_checkpoints")

                    test_procs.append(run(cmd))

                wait_all(test_procs)

    print("\n✅ Cartesian experiment complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("-es", "--env_sampling", type=str, choices=["largest", "smallest", "mean", "random"], default=["largest", "smallest", "mean"])
    # ===== Core experiment parameters =====
    parser.add_argument("-e", "--environment", required=True)
    parser.add_argument(
        '-at', '--algorithm_type',
        nargs="+",
        default=[
            "projection",
            "projection_sample",
            "projection_approximate",
            "projection_pca",
            "DO_discrete",
            "DO_discrete_valid",
            "GO_discrete",
            "GO_discrete_valid",
            "iterative"
        ],
    )
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("-ni_gae", "--num_iter_gae", type=int, default=10_000)
    parser.add_argument("-ni_rl", "--num_iter_rl", type=int, default=500_000)
    parser.add_argument("-algo", "--algorithm", required=True)
    parser.add_argument(
        "-ne",
        "--number_episodes_per_env",
        type=int,
        default=20,
    )
    parser.add_argument('-o', '--option', default='agent_performances',
                        choices=['action_time',
                                 'agent_performances'], help='Decide which statistics to plot')
    parser.add_argument("--load_envs_mode", type=str, default=False)
    parser.add_argument('-ssa', '--sample_subset_actions', default=0,
                        help='How many to sample as a subset of actions at each step (only for continuous action spaces)')
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--val_checkpoints", action="store_true")
    parser.add_argument("--skip_gae", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--val_pct", type=float, default=-1)
    parser.add_argument("--test_pct", type=float, default=-1)
    # ===== Resource control =====
    parser.add_argument("--max_parallel_test", type=int, default=1)
    args = parser.parse_args()

    seeds_loaded = load_yaml(os.path.join(script_dir, "..", "agents", "config", 'seeds.yaml'))
    seeds_runs = seeds_loaded['seeds'][0:args.num_runs]
    current_seed = seeds_runs[0] if seeds_runs else None
    # set seed for reproducibility
    if current_seed is not None:
        random.seed(current_seed)
        np.random.seed(current_seed)

    train_config = load_yaml(os.path.join(script_dir, "..", "agents", "config", 'train_config.yaml'))
    previous_train_config = copy.deepcopy(train_config)
    train_config['checkpoints_save_freq'] = args.num_iter_rl
    save_yaml(train_config, os.path.join(script_dir, "..", "agents", "config", 'train_config.yaml'))

    # create a common logs folder where they will all go grouped
    now = time.strftime("%Y%m%d-%H%M%S")
    if not args.skip_training:
        common_logs_folder_name = f"cartesian_runs_{args.num_runs}_{args.env_sampling}_{now}"
    else:
        # load the most recent one
        logs_base = Path(script_dir) / "../agents" / "logs" / args.environment / args.algorithm_type[0]

        existing_folders = sorted(logs_base.glob("cartesian_runs_*"), key=os.path.getmtime, reverse=True)
        if existing_folders:
            common_logs_folder_name = existing_folders[0].name
            print(f"[INFO] Found existing logs folder: {common_logs_folder_name}")
        else:
            raise ValueError("No existing logs folder found. Please run without --skip_training first.")

    run_cartesian_experiments(
        env_name=args.environment,
        args=args,
        common_logs_folder_name=common_logs_folder_name,
        seeds_runs=seeds_runs
    )
    save_yaml(previous_train_config, os.path.join(script_dir, "..", "agents", "config", 'train_config.yaml'))

if __name__ == "__main__":
    main()
