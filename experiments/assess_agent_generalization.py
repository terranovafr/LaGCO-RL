#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

import json
import random
import numpy as np
import argparse
import os
import math
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
import shutil
from pathlib import Path
from utils.file_utils import load_yaml, save_yaml
from utils.generation_utils import compute_env_score_based_on_important_params
from utils.proc_utils import run, wait_for_slot, wait_all
from utils.experiments_utils import resolve_env_folder, map_algorithm_type_to_original_and_override_args
script_dir = Path(__file__).parent.resolve()

def select_envs(
    envs,
    num_train,
    sampling_strategy,
    important_params,
    already_selected_ids,
    random_sampling_pct=None,
    total_envs=None,
):
    # Separate available and forced-test envs
    available_envs = []
    forced_test_envs = []

    for env in envs:
        env_id = env['id']
        if env_id in already_selected_ids:
            forced_test_envs.append(env)
        else:
            available_envs.append(env)

    if len(available_envs) == 0:
        raise ValueError("No available environments to sample from.")

    # Compute scores only when needed
    scored_envs = [
        (env, compute_env_score_based_on_important_params(env["config"], important_params))
        for env in available_envs
    ]

    # ---- NEW STRATEGY: RANDOM PERCENTAGE WITHOUT REPLACEMENT ACROSS RUNS ----
    if sampling_strategy == "random_pct":
        if random_sampling_pct is None or random_sampling_pct <= 0:
            raise ValueError(
                "env_sampling='random_pct' requires --random_sampling_pct > 0"
            )

        if total_envs is None or total_envs <= 0:
            raise ValueError("total_envs must be provided for random_pct sampling.")

        # Compute how many envs to sample this run based on TOTAL env pool
        num_train = math.floor((random_sampling_pct / 100.0) * total_envs)

        if num_train <= 0:
            raise ValueError(
                f"--random_sampling_pct={random_sampling_pct}%% is too small for "
                f"total_envs={total_envs}; it results in 0 sampled environments per run."
            )

        if len(available_envs) < num_train:
            raise ValueError(
                f"Not enough remaining environments for random_pct sampling: "
                f"requested {num_train}, available {len(available_envs)}."
            )

        random.shuffle(scored_envs)

    # ---- NEW STRATEGIES ----
    elif sampling_strategy in ["largest_50", "smallest_50"]:
        reverse = sampling_strategy == "largest_50"
        scored_envs.sort(key=lambda x: x[1], reverse=reverse)
        num_train = len(scored_envs) // 2

    # ---- EXISTING STRATEGIES ----
    elif sampling_strategy == "largest":
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

    # Safety check for strategies using externally computed num_train
    if sampling_strategy not in ["largest_50", "smallest_50", "random_pct"]:
        if len(scored_envs) < num_train:
            raise ValueError("Not enough environments to sample from.")

    # Split
    new_train_envs = [env for env, _ in scored_envs[:num_train]]
    remaining_envs = [env for env, _ in scored_envs[num_train:]]

    test_envs = forced_test_envs + remaining_envs

    return new_train_envs, test_envs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--train_pct", type=float, default=-1)
    parser.add_argument("--env_sampling", type=str, nargs="+",
                        choices=["largest", "smallest", "mean", "random", "random_pct", "largest_50", "smallest_50"], default=["largest", "smallest", "largest_50", "smallest_50"])
    parser.add_argument(
        "-rsp",
        "--random_sampling_pct",
        type=float,
        default=-1,
        help="Percentage of total environments to sample at each run for env_sampling=random_pct. "
             "Example: 20 means 20%%, so with 5 runs max coverage is 100%%."
    )
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
    parser.add_argument("-ms", "--max_sweeps", type=int, default=1000)
    parser.add_argument("-n", "--number_environments", type=int, default=50)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("-ni_gae", "--num_iter_gae", type=int, default=10_000)
    parser.add_argument("-ni_rl", "--num_iter_rl", type=int, default=500_000)
    parser.add_argument("-algo", "--algorithm", required=True)
    parser.add_argument(
        "-ne_train",
        "--number_episodes_per_env_train",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-ne_test",
        "--number_episodes_per_env_test",
        type=int,
        default=20,
    )
    parser.add_argument("--load_envs_mode", type=str, default=False)
    parser.add_argument('-ssa', '--sample_subset_actions', default=0,
                        help='How many to sample as a subset of actions at each step (only for continuous action spaces)')
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--val_checkpoints", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_gae", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_testing", action="store_true")
    parser.add_argument("--val_pct", type=float, default=-1)
    parser.add_argument("--test_pct", type=float, default=-1)
    parser.add_argument("--test_solutions", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument('--no_cuda', action='store_false', dest='cuda',
                        default=True, help='Disable use of cuda even if available')
    parser.add_argument(
        "--runs_ids",
        nargs="+",
        type=int,
        default=None,
        help="If set, only execute these run indices (1-based). "
             "Example: --runs_ids 1 3 5"
    )
    # ===== Resource control =====
    parser.add_argument("--max_parallel_test", type=int, default=1)

    args = parser.parse_args()
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    time_output_path = output_dir / f"time_{args.environment}_{'+'.join(args.algorithm_type)}_{time_now}.json"

    timings = {
        "created_at": time_now,
        "environment": args.environment,
        "algorithm": args.algorithm,
        "num_runs": args.num_runs,
        "env_sampling": {},
    }

    runs_ids = None
    if args.runs_ids is not None:
        runs_ids = set(args.runs_ids)

    def save_timings():
        with open(time_output_path, "w") as f:
            json.dump(timings, f, indent=4)

    # ============================================================
    # 1. RANDOM GENERATION
    # ============================================================
    if args.train_pct == -1:
        # make only 1 train env and the other on test, computing proper percentages
        if not args.skip_generation:
            train_percent = 100.0 / (args.number_environments + 1)
        else:
            env_folder = resolve_env_folder(args.environment, script_dir)
            split_path = env_folder / "split.yaml"
            if not split_path.exists():
                print(f"Error: split.yaml not found in {env_folder}. Please run without --skip_generation first to generate environments and split.")
                sys.exit(1)
            split_data = load_yaml(split_path)
            total_envs = len(split_data["training_set"]) + len(split_data["test_set"]) + len(split_data.get("validation_set", []))
            train_percent = 100.0 / total_envs
        # round to closest 0.1 to avoid floating point issues
        train_percent = round(train_percent, 1)
        # represent it as 0.X
        args.train_pct = train_percent / 100.0
        args.test_pct = 1 - args.train_pct
        args.val_pct = 0

    if not args.skip_generation:
        cmd = [
            "python3", os.path.join(script_dir, "..", "generators", "random_generator.py"),
            "-e", args.environment,
            "-n", str(args.number_environments),
            "-s", "random",
            "--force_saving",
        ]
        cmd += ["--train_pct", str(args.train_pct)]
        cmd += ["--test_pct", str(args.test_pct)]
        if args.val_pct != -1:
            cmd += ["--val_pct", str(args.val_pct)]
        if args.max_sweeps:
            cmd += ["-ms", str(args.max_sweeps)]
        p = run(cmd)
        if p.wait() != 0:
            sys.exit(1)


    env_folder = resolve_env_folder(args.environment, script_dir)
    split_path = env_folder / "split.yaml"
    original_split_path = env_folder / "original_split.yaml"

    # Backup original split.yaml if not done yet
    if not original_split_path.exists():
        shutil.copy(split_path, original_split_path)

    split_data = load_yaml(original_split_path)

    envs = split_data["training_set"] + split_data["test_set"] + split_data.get("validation_set", [])
    total_envs = len(envs)
    num_train = max(1, int(args.train_pct * total_envs))

    # Validate random_pct configuration
    if "random_pct" in args.env_sampling:
        if args.random_sampling_pct <= 0:
            raise ValueError(
                "You selected env_sampling='random_pct' but did not provide a valid "
                "--random_sampling_pct (> 0)."
            )

        max_total_pct = args.num_runs * args.random_sampling_pct
        if max_total_pct > 100:
            raise ValueError(
                f"Incompatible configuration: num_runs={args.num_runs} and "
                f"random_sampling_pct={args.random_sampling_pct}% would require "
                f"{max_total_pct}% of the pool, which is > 100%. "
                f"Example: 20% is compatible with at most 5 runs."
            )

        random_pct_num_train = int((args.random_sampling_pct / 100.0) * total_envs)
        if random_pct_num_train <= 0:
            raise ValueError(
                f"--random_sampling_pct={args.random_sampling_pct}% is too small for "
                f"total_envs={total_envs}; it results in 0 environments per run."
            )

    # Load important parameters from environment_ranges.yaml
    ranges_path = Path(os.path.join(script_dir, "..", "generators", "config")) / f"{args.environment}_ranges.yaml"
    ranges_data = load_yaml(ranges_path)
    important_params = {k: v for k, v in ranges_data.items() if v.get("important", False)}

    seeds_loaded = load_yaml(os.path.join(script_dir, "..", "agents", "config", 'seeds.yaml'))
    seeds_runs = seeds_loaded['seeds'][0:args.num_runs]
    current_seed = seeds_runs[0] if seeds_runs else None
    # set seed for reproducibility
    if current_seed is not None:
        random.seed(current_seed)
        np.random.seed(current_seed)

    # create a common logs folder where they will all go grouped
    for env_sampling in args.env_sampling:
        already_selected_ids = set()
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        common_logs_folder_name = f"runs_{args.num_runs}_{env_sampling}_{now}"
        timings["env_sampling"][env_sampling] = {
            "common_logs_folder_name": common_logs_folder_name,
            "runs": {}
        }

        for run_idx in range(args.num_runs):
            run_number = run_idx + 1
            run_enabled = (runs_ids is None) or (run_number in runs_ids)
            train_envs, test_envs = select_envs(
                envs=envs,
                num_train=num_train,
                sampling_strategy=env_sampling,
                important_params=important_params,
                already_selected_ids=already_selected_ids,
                random_sampling_pct=args.random_sampling_pct,
                total_envs=total_envs,
            )

            current_seed = seeds_runs[run_idx] if seeds_runs else None
            run_key = f"run_{run_idx + 1}"
            timings["env_sampling"][env_sampling]["runs"][run_key] = {
                "seed": current_seed,
                "n_train_envs": len(train_envs),
                "n_test_envs": len(test_envs),
                "gae_training_seconds": None,
                "agent_training_seconds": {},
                "agent_testing_seconds": {},
                "total_run_seconds": None,
            }

            run_global_start = time.perf_counter()

            # Update selected set
            for env in train_envs:
                already_selected_ids.add(env['id'])

            # Update split.yaml
            split_data["training_set"] = train_envs
            split_data["test_set"] = test_envs
            split_data["validation_set"] = []  # Clear validation set for now, can be modified if needed
            split_data["params"]["train_pct"] = args.train_pct
            split_data["params"]["test_pct"] = args.test_pct
            split_data["params"]["val_pct"] = args.val_pct

            save_yaml(split_data, split_path)
            extra_info = ""
            if env_sampling == "random_pct":
                extra_info = f" | random_sampling_pct={args.random_sampling_pct}%"
            print(
                f"Run {run_idx + 1}/{args.num_runs}: "
                f"Training set = {len(train_envs)}, Test set = {len(test_envs)}{extra_info}"
            )
            # ============================================================
            # 2. GAE TRAINING
            # ============================================================
            if run_enabled and not args.skip_gae:
                cmd = [
                    "python3", os.path.join(script_dir, "..", "gae", "train_gae.py"),
                    "-e", args.environment,
                    "-ti", str(args.num_iter_gae),
                    "--force_saving",
                    "--static_seed", str(current_seed) if current_seed is not None else 0,
                    "--name", f"{args.environment}_{run_key}"
                ]

                if args.validation:
                    cmd.append("--validation")

                if args.load_envs_mode:
                    cmd += ["--load_envs_mode", args.load_envs_mode]

                if args.name:
                    cmd += ["--name", args.name]

                gae_start = time.perf_counter()
                p = run(cmd)
                # add validation optional
                if p.wait() != 0:
                    sys.exit(1)
                gae_end = time.perf_counter()
                timings["env_sampling"][env_sampling]["runs"][run_key]["gae_training_seconds"] = gae_end - gae_start
                save_timings()

            if args.skip_gae:
                timings["env_sampling"][env_sampling]["runs"][run_key]["gae_training_seconds"] = None

            # ============================================================
            # 3. RL AGENT TRAINING (SEQUENTIAL)
            # ============================================================
            logs_folders = {}
            print("\n===== STARTING SEQUENTIAL TRAINING =====\n", flush=True)
            if run_enabled:
                if not args.skip_training:
                    for algorithm_type in args.algorithm_type:
                        algo_folder, copy_args = map_algorithm_type_to_original_and_override_args(algorithm_type, args)

                        cmd = [
                            "python3", os.path.join(script_dir, "..", "agents", "train_agent.py"),
                            "-e", copy_args['environment'],
                            "-at", copy_args['algorithm_type'],
                            "-ti", str(copy_args['num_iter_rl']),
                            "-algo", copy_args['algorithm'],
                        ]

                        common_logs_folder = os.path.join(script_dir, "..", "agents", "logs", args.environment, algo_folder, common_logs_folder_name, "run_" + str(run_idx + 1))
                        logs_folders[algo_folder] = common_logs_folder
                        os.makedirs(common_logs_folder, exist_ok=True)
                        cmd += ["--logs_folder", common_logs_folder]
                        cmd += ["--static_seed", str(current_seed) if current_seed is not None else 0]

                        if not copy_args['cuda']:
                            cmd.append("--no_cuda")
                        if copy_args['GNN_observations']:
                            cmd += ["-GO"]
                        if copy_args['approximate_distance']:
                            cmd += ["-approx"]
                        if copy_args['pca_minimum_without_loss']:
                            cmd += ["--pca_minimum_without_loss"]
                        if copy_args['semantic_ordering']:
                            cmd += ["--semantic_ordering"]
                        if args.load_envs_mode:
                            cmd += ["--load_envs_mode", args.load_envs_mode]

                        if copy_args['sample_subset_actions']:
                            cmd += ["--sample_subset_actions", str(args.sample_subset_actions)]

                        if args.validation:
                            cmd.append("--validation")

                        if args.name:
                            cmd += ["--name", args.name]

                        train_start = time.perf_counter()
                        p = run(cmd)
                        if p.wait() != 0:
                            sys.exit(1)
                        train_end = time.perf_counter()

                        timings["env_sampling"][env_sampling]["runs"][run_key]["agent_training_seconds"][algorithm_type] = {
                            "algo_folder": algo_folder,
                            "seconds": train_end - train_start
                        }
                        save_timings()
                elif not args.skip_testing:
                    def extract_timestamp(folder_name):
                        import re
                        m = re.search(r"\d{8}_\d{6}", folder_name)
                        if m:
                            return time.strptime(m.group(), "%Y-%m-%d_%H-%M-%S")
                        return time.gmtime(0)
                    for algorithm_type in args.algorithm_type:
                        algo_folder, _ = map_algorithm_type_to_original_and_override_args(
                            algorithm_type, args)

                        # find common_logs_folder_name as the one that already exists in logs for this algorithm_type,with latest timestep and that should have 'envs' folder inside
                        candidates = []
                        for folder in os.listdir(os.path.join(script_dir, "..", "agents", "logs", args.environment, algo_folder)):
                            if folder.startswith("runs_") and os.path.exists(os.path.join(script_dir, "..", "agents", "logs", args.environment, algo_folder, folder, "run_1", "envs")):
                                candidates.append(folder)
                        # order them by runs_1_largest_2026-03-24_08-09-40 date
                        candidates.sort(key=lambda x: extract_timestamp(x), reverse=True)

                        common_logs_folder = os.path.join(script_dir, "..", "agents", "logs", args.environment, algo_folder,
                                                          candidates[0], "run_" + str(run_idx + 1))
                        logs_folders[algo_folder] = common_logs_folder
                print("\n===== ALL TRAINING FINISHED =====\n", flush=True)

            # ============================================================
            # 4. TESTING (PARALLEL)
            # ============================================================
            if run_enabled:
                print("\n===== STARTING PARALLEL TESTING =====\n", flush=True)
                n_train = len(train_envs)
                n_test = len(test_envs)

                test_procs = []
                test_proc_metadata = []

                args.max_parallel_test = min(args.max_parallel_test, len(args.algorithm_type) * 2)

                for solution in args.test_solutions:
                    reference_number = n_train if solution == "train" else n_test
                    multiplier = args.number_episodes_per_env_test if solution == "test" else args.number_episodes_per_env_train
                    if multiplier == 0:
                        continue
                    for algorithm_type in args.algorithm_type:
                        wait_for_slot(test_procs, args.max_parallel_test)
                        algo_folder, copy_args = map_algorithm_type_to_original_and_override_args(algorithm_type, args)
                        cmd = [
                            "python3", os.path.join(script_dir, "..", "agents", "test_agent.py"),
                            "-algo", copy_args['algorithm'],
                            "-at", copy_args['algorithm_type'],
                            "-e", copy_args['environment'],
                            "-ne", str(multiplier * reference_number),
                            "--last_checkpoint",
                            "-o", "agent_performances",
                        ]
                        cmd += ["--logs_folder", logs_folders[algo_folder]]
                        # no static seed during testing, it should load one different per episode
                        if copy_args['GNN_observations']:
                            cmd += ["-GO"]
                        if not copy_args['cuda']:
                            cmd.append("--no_cuda")
                        if copy_args['approximate_distance']:
                            cmd += ["-approx"]
                        if copy_args['semantic_ordering']:
                            cmd += ["--semantic_ordering"]
                        if copy_args['pca_minimum_without_loss']:
                            cmd += ["--pca_minimum_without_loss"]

                        if solution == "test":
                            cmd.append("--load_default_test_envs")
                        elif solution == "train":
                            cmd.append("--load_default_train_envs")

                        if args.val_checkpoints:
                            cmd.append("--val_checkpoints")

                        test_start = time.perf_counter()
                        proc = run(cmd)
                        test_procs.append(proc)
                        test_proc_metadata.append({
                            "proc": proc,
                            "algorithm_type": algorithm_type,
                            "solution": solution,
                            "algo_folder": algo_folder,
                            "start_time": test_start,
                        })

                wait_all(test_procs)

                for meta in test_proc_metadata:
                    test_end = time.perf_counter()
                    elapsed = test_end - meta["start_time"]

                    algo_name = meta["algorithm_type"]
                    solution_name = meta["solution"]

                    if algo_name not in timings["env_sampling"][env_sampling]["runs"][run_key]["agent_testing_seconds"]:
                        timings["env_sampling"][env_sampling]["runs"][run_key]["agent_testing_seconds"][algo_name] = {}

                    timings["env_sampling"][env_sampling]["runs"][run_key]["agent_testing_seconds"][algo_name][
                        solution_name] = {
                        "algo_folder": meta["algo_folder"],
                        "seconds": elapsed
                    }

            timings["env_sampling"][env_sampling]["runs"][run_key][
                    "total_run_seconds"] = time.perf_counter() - run_global_start
            save_timings()

            print("\n✅ Pipeline completed successfully")

if __name__ == "__main__":
    main()
