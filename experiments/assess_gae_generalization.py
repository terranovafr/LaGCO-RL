#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import load_yaml, save_yaml
from utils.proc_utils import run, wait_for_slot, wait_all
script_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser("Automatic experiment launcher")

    # ===== Core experiment parameters =====
    parser.add_argument("-e", "--environment", type=str, nargs="+", required=True,
                        default=['cyberattack', 'ospf_engineering', 'traffic_engineering', 'vmp'])
    parser.add_argument("-ms", "--max_sweeps", type=int, default=1000)
    parser.add_argument("-n", "--number_environments", type=int, default=50)
    parser.add_argument("-ti", "--train_iterations", type=int, default=10_000)
    parser.add_argument("-a", "--gnn_algos", nargs='+', type=str, default=["SAGEConv", "GATConv", "GCNConv", "GINConv"],
                        choices=["SAGEConv", "GATConv", "GCNConv", "GINConv"],
                        help="GNN architectures to use")
    parser.add_argument("-s", "--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate")
    parser.add_argument("-i", "--num_random_inits", type=int, default=5,
                        help="Number of random baseline initializations")
    parser.add_argument(
        "-ne",
        "--number_episodes",
        type=int,
        default=50,
    )

    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_training", action="store_true")

    # ===== Resource control =====
    parser.add_argument("--max_parallel_test", type=int, default=4)

    args = parser.parse_args()

    # ============================================================
    # 1. RANDOM GENERATION
    # ============================================================
    if not args.skip_generation:
        for env in args.environment:
            print(f"\n===== GENERATING ENVIRONMENTS FOR {env} =====\n", flush=True)
            p = run([
                "python3", os.path.join(script_dir,"../generators/random_generator.py"),
                "-e", env,
                "-n", str(args.number_environments),
                "-s", "random",
                "--force_saving",
                "-ms", str(args.max_sweeps),
            ])
            if p.wait() != 0:
                sys.exit(1)

    # ============================================================
    # 2. GAE TRAINING
    # ============================================================


    for gnn_algo in args.gnn_algos:
            # subscribe the algo_config yaml putting all layers of this type
            print(f"\n===== SETTING GNN ALGO TO {gnn_algo} =====\n", flush=True)
            algo_config_path = os.path.join(script_dir, "..", "gae", "config","algo_config.yaml")
            algo_config = load_yaml(algo_config_path)
            algo_config['model_config']['layers'] = algo_config['example_blocks'][gnn_algo]
            save_yaml(algo_config, os.path.join(script_dir, "..", "gae", "config"), "algo_config.yaml")
            if not args.skip_training:
                for env in args.environment:
                    print(f"\n===== TRAINING GAE FOR {env} =====\n", flush=True)
                    cmd = [
                        "python3", os.path.join(script_dir, "..", "gae", "train_gae.py"),
                        "-e", env,
                        "-ti", str(args.train_iterations),
                        "--name", f"{env}_{gnn_algo}",
                        "--force_saving",
                    ]
                    if args.validation:
                        cmd.append("--validation")
                    p = run(cmd)
                    # add validation optional
                    if p.wait() != 0:
                        sys.exit(1)

            # reoverwrite algo_config.yaml to default
            print(f"\n===== RESETTING GNN ALGO TO DEFAULT =====\n", flush=True)
            algo_config_path = os.path.join(script_dir, "..", "gae", "config", "algo_config.yaml")
            algo_config = load_yaml(algo_config_path)
            algo_config['model_config'] = algo_config['default_model_config']
            # ============================================================
            # 3. TESTING (PARALLEL)
            # ============================================================
            print("\n===== STARTING PARALLEL TESTING =====\n", flush=True)

            test_procs = []

            args.max_parallel_test = min(args.max_parallel_test, len(args.environment))

            for env in args.environment:
                wait_for_slot(test_procs, args.max_parallel_test)

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

            print("\n✅ Pipeline completed successfully")


if __name__ == "__main__":
    main()
