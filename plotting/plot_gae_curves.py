#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    plot_gae_curves.py
    Script to plot GAE training curves from TensorBoard logs
"""

import argparse
import os
import re
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.math_utils import compute_band, interpolate_runs

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "gae", "logs")

TARGET_SUFFIXES = [
    "total_loss",
    "adj_loss",
    "edge_feature_vector_loss",
    "node_feature_vector_loss",
]

CLEAN_LOSS_NAMES = {
    "total_loss": "Total Loss",
    "adj_loss": "Adjacency Loss",
    "edge_feature_vector_loss": "Edge Feature Loss",
    "node_feature_vector_loss": "Node Feature Loss"
}


# Find ALL matching experiment folders
def find_all_env_folders(base_dir, env_name):
    env_name = env_name.lower()
    folders = []

    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue

        if not name.startswith(env_name):
            continue

        match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name)
        if match:
            dt = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
        else:
            dt = datetime.fromtimestamp(os.path.getmtime(full))

        folders.append((dt, full))

    if not folders:
        return []

    folders.sort(reverse=True)
    return [f for _, f in folders]

# Load TensorBoard scalars
def load_all_matching_scalars(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()

    scalar_tags = ea.Tags()["scalars"]
    matched = {suffix: [] for suffix in TARGET_SUFFIXES}

    for tag in scalar_tags:
        for suffix in TARGET_SUFFIXES:
            if tag.endswith(suffix):
                events = ea.Scalars(tag)
                steps = np.array([e.step for e in events], dtype=float)
                values = np.array([e.value for e in events], dtype=float)
                matched[suffix].append((steps, values))

    return matched




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", required=True)
    parser.add_argument("--spread", choices=["std", "ci", "bci"], default="ci")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # Find all environment folders
    env_folders = find_all_env_folders(BASE_DIR, args.environment)

    if not env_folders:
        print("No matching folders found.")
        return

    print(f"Found {len(env_folders)} experiment folders")

    # Collect all runs
    global_runs = {suffix: [] for suffix in TARGET_SUFFIXES}

    for folder in env_folders:
        matched = load_all_matching_scalars(folder)

        for suffix, series_list in matched.items():
            if not series_list:
                continue
            # each (steps, values) is one independent run
            for steps, values in series_list:
                global_runs[suffix].append((steps, values))

    # Final aggregation and band computation
    results = {}

    for suffix, run_list in global_runs.items():
        if not run_list:
            continue

        x, ys = interpolate_runs(run_list)
        mean, low, high = compute_band(ys, spread=args.spread)

        results[suffix] = (x, ys, mean, low, high)

        print(f"{suffix}: num runs = {ys.shape[0]}")

    if not results:
        print("No matching metrics found.")
        return

    out_dir = os.path.join(SCRIPT_DIR, "plots", "gae")
    os.makedirs(out_dir, exist_ok=True)

    # INDIVIDUAL PLOTS
    for suffix, (x, ys, mean, low, high) in results.items():
        plt.figure(figsize=(7, 5))

        plt.plot(x, mean, linewidth=2)
        plt.fill_between(x, low, high, alpha=0.2)

        plt.title(CLEAN_LOSS_NAMES[suffix], fontsize=30)
        plt.xlabel("Training Step", fontsize=30)
        plt.ylabel("Loss", fontsize=30)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True, linestyle="--", alpha=0.4)

        out_path = os.path.join(out_dir, f"{args.environment}_{suffix}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")

        if args.show:
            plt.show()

        plt.close()

    # COMBINED PLOT
    plt.figure(figsize=(8, 6))

    for suffix, (x, ys, mean, low, high) in results.items():
        plt.plot(x, mean, label=CLEAN_LOSS_NAMES[suffix], linewidth=2)
        plt.fill_between(x, low, high, alpha=0.15)

    plt.xlabel("Training Step", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(out_dir, f"{args.environment}_combined.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()