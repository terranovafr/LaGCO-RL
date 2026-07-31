#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

"""
    plot_gae_generalization.py
    Script to plot GAE generalization results from YAML summaries
"""

import argparse
import yaml
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import load_latest_yaml_from_split_folder

def extract_gnn_layer(folder_name: str):
    """Extract GNN layer from folder name: vmp_GATConv_gae_2026-..."""
    match = re.search(r"_([A-Za-z0-9]+Conv)_gae", folder_name)
    return match.group(1) if match else "Unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, required=True,
                        help="Environment prefix, e.g. vmp")
    parser.add_argument("-s", "--split", type=str, default="training",
                        choices=["training", "test"])
    parser.add_argument("--plot_ci", action="store_true",
                        help="Plot 95% confidence intervals")
    args = parser.parse_args()

    logs_dir = Path(os.path.join(script_dir, "..", "gae", "logs"))
    out_dir = Path(os.path.join(script_dir, "plots", "gae_generalization", args.environment))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect only latest folder per layer type
    layer_latest_run = {}
    for run in logs_dir.iterdir():
        if not run.is_dir():
            continue
        if not run.name.startswith(args.environment + "_"):
            continue

        layer = extract_gnn_layer(run.name)
        # keep the latest timestamp per layer
        current_time = run.name.split("_")[-1]
        if layer not in layer_latest_run or current_time > layer_latest_run[layer].name.split("_")[-1]:
            layer_latest_run[layer] = run

    # layer -> graph -> loss -> stats
    data = defaultdict(lambda: defaultdict(dict))

    for layer, run in layer_latest_run.items():
        yaml_file = load_latest_yaml_from_split_folder(run, args.split)
        if yaml_file is None:
            continue

        with open(yaml_file) as f:
            content = yaml.safe_load(f)

        for graph_name, metrics in content.items():
            for loss_name, stats in metrics.items():
                data[graph_name][layer][loss_name] = stats

    # For each graph type, create separate tables
    for graph_name, graph_data in data.items():
        rows = []
        for layer, losses in graph_data.items():
            for loss_name, stats in losses.items():
                row = {
                    "layer": layer,
                    "loss": loss_name,
                    "mean": stats.get("mean")
                }
                if "ci95_lower" in stats:
                    row["ci95_lower"] = stats["ci95_lower"]
                    row["ci95_upper"] = stats["ci95_upper"]
                rows.append(row)

        df = pd.DataFrame(rows).sort_values(["loss", "layer"])
        # Save CSV and TXT
        csv_path = out_dir / f"{args.environment}_{graph_name}_gae_summary.csv"
        txt_path = out_dir / f"{args.environment}_{graph_name}_gae_summary.txt"
        df.to_csv(csv_path, index=False)
        with open(txt_path, "w") as f:
            f.write(df.to_string(index=False))

        print(f"\n=== {graph_name.upper()} SUMMARY ===")
        print(df)

        # Plot
        for loss in df["loss"].unique():
            sub = df[df["loss"] == loss]
            plt.figure(figsize=(8, 5))
            x = np.arange(len(sub))
            y = sub["mean"].values

            plt.bar(x, y, tick_label=sub["layer"].values)

            if args.plot_ci and "ci95_lower" in sub:
                yerr = [
                    y - sub["ci95_lower"].values,
                    sub["ci95_upper"].values - y
                ]
                plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=4)

            plt.ylabel("Mean loss")
            plt.title(f"{args.environment} – {graph_name} – {loss}")
            plt.tight_layout()

            fig_path = out_dir / f"{args.environment}_{graph_name}_{loss}.png"
            plt.savefig(fig_path)
            plt.close()

        print(f"Saved {graph_name} tables and plots in {out_dir}")


if __name__ == "__main__":
    main()
