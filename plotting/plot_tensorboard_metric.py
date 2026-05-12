#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    plot_tensorboard_metric.py
    Script to plot training curves from TensorBoard logs for multiple solutions and cases
"""

import argparse
import os
import re
from datetime import datetime
import fnmatch
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import sanitize_filename
from utils.math_utils import compute_band, interpolate_runs

script_dir = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(script_dir, "..", "agents")


DEFAULT_METRICS = {
    "ospf_engineering": "train/Mean ideal_reduction_percentage_relative",
    "traffic_engineering": "train/Mean ideal_reduction_percentage_relative",
    "maxcut": "train/Mean relative_performance",
    "mvc": "train/Mean relative_performance",
    "tsp": "train/Mean relative_performance",
    "cyberattack": "train/Mean compromised_nodes", # "cyberattack": "train/Mean DOS_nodes_percentage",
    "vmp": "train/Mean weighted_sum_difference"
}

# Fixed colors per solution (consistent across all plots)
SOLUTION_COLOR_MAP = {
    "DO_discrete": "#1f77b4",        # blue
    "DO_discrete_M": "#ff7f0e",      # orange
    "GO_discrete": "#2ca02c",        # green
    "GO_discrete_M": "#d62728",      # red
    "projection": "#9467bd",         # purple
    "iterative": "#8c564b",          # brown
}

ALGO_PATTERNS = ["PPO_*", "IDQN_*", "TRPO_*", "SAC_*", "TD3_*", "DDPG_*", "A2C_*", "DQN_*", "MaskablePPO_*"]

def load_scalar(log_dir, tag):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise ValueError(f"Metric '{tag}' not found in {log_dir}")

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values


def find_best_case_folder(solution_dir, case):
    candidates = []
    pattern = re.compile(
        rf"runs_(\d+)_{case}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})"
    )

    for name in os.listdir(solution_dir):
        full = os.path.join(solution_dir, name)
        if not os.path.isdir(full):
            continue

        m = pattern.fullmatch(name)
        if not m:
            continue

        n_runs = int(m.group(1))
        dt = datetime.strptime(m.group(2), "%Y-%m-%d_%H-%M-%S")
        candidates.append((n_runs, dt, full))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]

def find_algo_dir(run_dir):
    matches = []

    for d in os.listdir(run_dir):
        full = os.path.join(run_dir, d)
        if not os.path.isdir(full):
            continue

        if any(fnmatch.fnmatch(d, p) for p in ALGO_PATTERNS):
            matches.append(d)

    if not matches:
        return None

    def extract_num(name):
        parts = name.split("_")
        return int(parts[-1]) if parts[-1].isdigit() else -1

    matches.sort(key=extract_num, reverse=True)
    return os.path.join(run_dir, matches[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--logs_folder", required=True, help="Root logs folder")
    parser.add_argument("-e", "--environment", required=True, help="Environment folder")
    parser.add_argument(
        "-s",
        "--solutions",
        nargs="+",
        default=[
            "DO_discrete", "DO_discrete_semantic", "projection", "iterative",
            "GO_discrete", "DO_discrete_M", "GO_discrete_M",
            "projection_approximate", "projection_sample", "projection_pca"
        ],
        help="Solutions to compare",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["largest", "smallest"],
        choices=["largest", "smallest", "mean", "random_pct"],
    )
    parser.add_argument(
        "-m",
        "--metric",
        default=None,
        help="TensorBoard scalar tag",
    )
    parser.add_argument(
        "--spread",
        choices=["std", "ci", "bci"],
        default="ci",
        help="Error band across runs",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    plt.figure(figsize=(8, 6))


    if not args.metric:
        args.metric = DEFAULT_METRICS[args.environment]
    # colors = solutions
    # styles = cases
    linestyle_map = {
        "largest": "-",
        "smallest": "--",
        "mean": ":",
        "random_pct": "-.",
    }

    for sol in args.solutions:
        solution_dir = os.path.join(BASE_DIR, args.logs_folder, args.environment, sol)

        if not os.path.isdir(solution_dir):
            print(f"Skipping missing solution: {solution_dir}")
            continue

        for case in args.cases:
            case_dir = find_best_case_folder(solution_dir, case)

            if case_dir is None:
                print(f"No folder for {sol}/{case}")
                continue

            run_series = []
            for name in sorted(os.listdir(case_dir)):
                run_dir = os.path.join(case_dir, name)
                if not (os.path.isdir(run_dir) and name.startswith("run_")):
                    continue

                algo_dir = find_algo_dir(run_dir)
                if algo_dir is None:
                    print(f"No algo dir in {run_dir}")
                    continue

                try:
                    steps, values = load_scalar(algo_dir, args.metric)
                    run_series.append((steps, values))
                except Exception as e:
                    print(f"Warning in {algo_dir}: {e}")

            if not run_series:
                print(f"No valid runs for {sol}/{case}")
                continue

            x, ys = interpolate_runs(run_series)
            mean, low, high = compute_band(ys, args.spread)

            color = SOLUTION_COLOR_MAP.get(sol, "#7f7f7f")  # fallback = gray
            linestyle = linestyle_map.get(case, "-")
            label = f"{sol} ({case}, n={len(run_series)})"

            plt.plot(
                x,
                mean,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
            plt.fill_between(
                x,
                low,
                high,
                color=color,
                alpha=0.15,
            )

            final_mean = mean[-1]
            final_std = np.std(ys[:, -1], ddof=1) if ys.shape[0] > 1 else 0.0
            print(f"{sol} | {case} | final mean={final_mean:.4f} | final std={final_std:.4f}")

    plt.xlabel("Training step", fontsize=30)
    plt.ylabel("Relative score", fontsize=30)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=25)
    plt.grid(True, linestyle="--", alpha=0.4)
    # Put legend BELOW the plot
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.65),  # push below
        ncol=1,  # adjust columns if needed
        frameon=True,
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space for legend
    plt.tight_layout()
    out_dir = os.path.join(script_dir, "plots", "tensorboard")
    os.makedirs(out_dir, exist_ok=True)

    safe_solutions = sanitize_filename("_".join(args.solutions[:3]))
    if len(args.solutions) > 3:
        safe_solutions += "_etc"

    safe_cases = sanitize_filename("_".join(args.cases))

    out_file = f"{args.environment}__{safe_solutions}__{safe_cases}.pdf"
    out_path = os.path.join(out_dir, out_file)

    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()