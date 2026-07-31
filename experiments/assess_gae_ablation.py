#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

import argparse
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import load_yaml, save_yaml
from utils.proc_utils import run, wait_for_slot, wait_all
script_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser("Automatic ablation study launcher")

    # ===== Core experiment parameters =====
    parser.add_argument("-e", "--environment", type=str, nargs="+", required=True,
                        default=['cyberattack', 'ospf_engineering', 'traffic_engineering', 'vmp'])
    parser.add_argument("-ms", "--max_sweeps", type=int, default=1000)
    parser.add_argument("-n", "--number_environments", type=int, default=50)
    parser.add_argument("-ti", "--train_iterations", type=int, default=10_000)
    parser.add_argument("-s", "--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("-i", "--num_random_inits", type=int, default=5)
    parser.add_argument("-ne", "--number_episodes", type=int, default=50)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--max_parallel_test", type=int, default=4)
    parser.add_argument("--skip_all_maskings", action="store_true")
    args = parser.parse_args()

    # ============================================================
    # 1. RANDOM GENERATION
    # ============================================================
    if not args.skip_generation:
        for env in args.environment:
            print(f"\n===== GENERATING ENVIRONMENTS FOR {env} =====\n", flush=True)
            p = run([
                "python3", os.path.join(script_dir, "../generators/random_generator.py"),
                "-e", env,
                "-n", str(args.number_environments),
                "-s", "random",
                "--force_saving",
                "-ms", str(args.max_sweeps),
            ])
            if p.wait() != 0:
                sys.exit(1)

    # ============================================================
    # 3. LOAD LOSS WEIGHTS (train_config.yaml)
    # ============================================================
    train_config_path = os.path.join(script_dir, "..", "gae", "config", "train_config.yaml")
    train_config = load_yaml(train_config_path)
    loss_weights = train_config.get("weights", {}).get("default", {})
    if not loss_weights:
        print("No weights found in train_config.yaml")
        sys.exit(1)

    # ============================================================
    # 4. BASELINE RUN (NO ABLATION)
    # ============================================================
    print("\n===== BASELINE: FULL MODEL (NO ABLATION) =====\n", flush=True)

    # Ensure config is untouched
    save_yaml(train_config, os.path.join(script_dir, "..", "gae", "config"), "train_config.yaml")

    # Train
    if not args.skip_training:
        for env in args.environment:
            print(f"\n===== TRAINING BASELINE GAE FOR {env} =====\n", flush=True)
            cmd = [
                "python3", os.path.join(script_dir, "..", "gae", "train_gae.py"),
                "-e", env,
                "-ti", str(args.train_iterations),
                "--name", f"{env}_ablation_None",
                "--force_saving",
            ]
            if args.validation:
                cmd.append("--validation")

            p = run(cmd)
            if p.wait() != 0:
                sys.exit(1)

    # Test baseline
    print("\n===== TESTING BASELINE =====\n", flush=True)
    test_procs = []
    max_parallel = min(args.max_parallel_test, len(args.environment))

    for env in args.environment:
        wait_for_slot(test_procs, max_parallel)

        cmd = [
            "python3", os.path.join(script_dir, "..", "gae", "test_gae.py"),
            "-e", env,
            "-ne", str(args.number_episodes),
            "-s", args.split,
            "--num_random_inits", str(args.num_random_inits),
        ]
        if args.validation:
            cmd.append("--validation")

        test_procs.append(run(cmd))

    wait_all(test_procs)

    # ============================================================
    # 5. ABLATION LOOP: set each weight to 0, train/test, restore
    # ============================================================
    if not args.skip_all_maskings:
        for weight_name, original_value in loss_weights.items():
            print(f"\n===== ABLATION: SETTING {weight_name} TO 0 FROM {original_value} =====\n", flush=True)
            original_value_copy = copy.deepcopy(original_value)
            train_config['weights']['default'][weight_name] = 0
            save_yaml(train_config, os.path.join(script_dir, "..", "gae", "config"), "train_config.yaml")

            # Train
            if not args.skip_training:
                for env in args.environment:
                    print(f"\n===== TRAINING GAE FOR {env} =====\n", flush=True)
                    cmd = [
                        "python3", os.path.join(script_dir, "..", "gae", "train_gae.py"),
                        "-e", env,
                        "-ti", str(args.train_iterations),
                        "--name", f"{env}_ablation_{weight_name}",
                        "--force_saving",
                    ]
                    if args.validation:
                        cmd.append("--validation")
                    p = run(cmd)
                    if p.wait() != 0:
                        sys.exit(1)

            # Test (parallel)
            print("\n===== STARTING PARALLEL TESTING =====\n", flush=True)
            test_procs = []
            max_parallel = min(args.max_parallel_test, len(args.environment))
            for env in args.environment:
                wait_for_slot(test_procs, max_parallel)
                cmd = [
                    "python3", os.path.join(script_dir, "..", "gae", "test_gae.py"),
                    "-e", env,
                    "-ne", str(args.number_episodes),
                    "-s", args.split,
                    "--num_random_inits", str(args.num_random_inits),
                ]
                if args.validation:
                    cmd.append("--validation")
                test_procs.append(run(cmd))
            wait_all(test_procs)

            # Restore weight
            print(f"\n===== RESTORING {weight_name} TO {original_value_copy} =====\n", flush=True)
            train_config['weights']['default'][weight_name] = original_value_copy
            save_yaml(train_config, os.path.join("..", "gae", "config"), "train_config.yaml")

    print("\n✅ Ablation study completed successfully")

if __name__ == "__main__":
    main()
