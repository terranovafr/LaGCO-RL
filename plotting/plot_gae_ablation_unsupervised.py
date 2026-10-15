#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    plot_gae_ablation_unsupervised.py
    This script plots the results of the GAE ablation study, comparing the performance of different loss term configurations across various environments. It generates visualizations to illustrate the impact of each loss term on the unsupervised GAE model's performance.
'''

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
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import load_latest_yaml_from_split_folder

script_dir = os.path.dirname(os.path.abspath(__file__))

weight_to_lossname = {
    "adj_weight": "adj_loss",
    "node_features": "node_features_loss",
    "edge_features": "edge_features_loss",
}

pretty_loss_names = {
    "adj_loss": "Adjacency",
    "node_features_loss": "Node Features",
    "edge_features_loss": "Edge Features",
}

pretty_weight_names = {
    "adj_weight": "Adjacency",
    "node_features": "Node Features",
    "edge_features": "Edge Features",
}

def build_delta_matrix(df):
    pivot = df.pivot(index="ablated_weight", columns="loss", values="mean")

    # -------------------------------
    # 1. Extract baseline row
    # -------------------------------
    if "baseline" not in pivot.index:
        raise ValueError("Baseline (_ablation_None) not found!")

    # -------------------------------
    # 2. Compute delta
    # -------------------------------
    delta = pivot

    # -------------------------------
    # 3. Rename columns + rows
    # -------------------------------
    delta = delta.rename(columns=pretty_loss_names)

    new_index = []
    for w in delta.index:
        if w == "baseline":
            new_index.append("Full Model")
        else:
            new_index.append(pretty_weight_names.get(w, w))
    delta.index = new_index

    # --------------------------
    # 4. Remove invalid rows (except baseline)
    # -------------------------------
    rows_to_keep = []

    for w in delta.index:
        if w == "Full Model":
            rows_to_keep.append(w)
            continue

        # reverse map
        raw_weight = None
        for k, v in pretty_weight_names.items():
            if v == w:
                raw_weight = k
                break

        if raw_weight is None:
            continue

        raw_loss = weight_to_lossname.get(raw_weight)
        pretty_col = pretty_loss_names.get(raw_loss)

        if pretty_col in delta.columns:
            rows_to_keep.append(w)

    delta = delta.loc[rows_to_keep]

    return delta

def to_latex_neurips(df, caption="", label=""):
    # -------------------------------
    # Column name formatting (compact headers)
    # -------------------------------
    header_map = {
        "Adjacency": r"\thead{Adj}",
        "Node Features": r"\thead{Node}",
        "Edge Features": r"\thead{Edge}",
    }

    row_map = {
        "Adjacency": "Adj",
        "Node Features": "Node",
        "Edge Features": "Edge",
    }

    # Apply mapping (fallback = raw name)
    headers = ["Ablated"]
    formatted_cols = []
    for col in df.columns:
        if col == "CR_avg":
            formatted_cols.append("CR$_{avg}$")
        else:
            formatted_cols.append(header_map.get(col, col))

    headers += formatted_cols

    # -------------------------------
    # Build LaTeX
    # -------------------------------
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{4pt}")

    # column format: l | c | c X X X X
    n_extra = len(df.columns) - 1
    latex.append(
        r"\begin{tabularx}{\linewidth}{l|c|c *{" + str(n_extra) + r"}{>{\centering\arraybackslash}X}}"
    )

    latex.append(r"\toprule")
    latex.append(" & ".join(headers) + r" \\")
    latex.append(r"\midrule")

    # -------------------------------
    # Rows
    # -------------------------------
    # reorder to make Full model first
    df = df.reindex(["Full Model"] + [idx for idx in df.index if idx != "Full Model"])
    for idx, row in df.iterrows():
        values = []

        for col_i, v in enumerate(row):
            if pd.isna(v):
                values.append("-")
            else:
                if idx == "Full Model":
                    values.append(f"{v:.3f}")

                else:
                    # Δ formatting (signed)
                    if col_i == 0:  # CR_avg stays normal
                        values.append(f"{v:.3f}")
                    else:
                        values.append(f"{v:.3f}")

        latex.append(f"{row_map.get(idx, idx)} & " + " & ".join(values) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabularx}")

    if caption:
        latex.append(rf"\caption{{{caption}}}")
    if label:
        latex.append(rf"\label{{{label}}}")

    latex.append(r"\end{table}")

    return "\n".join(latex)

def extract_ablated_weight(folder_name: str):
    match = re.search(r"_ablation_([a-zA-Z0-9_]+)_gae", folder_name)
    if match:
        val = match.group(1)
        if val == "None":
            return "baseline"
        return val
    return "Unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str, required=True,
                        help="Environment prefix, e.g. vmp")
    parser.add_argument("-f", "--folder", type=str, default="logs",
                        help="Logs folder prefix, e.g. logs")
    parser.add_argument("-s", "--split", type=str, default="training",
                        choices=["training", "test"])
    parser.add_argument("--plot_ci", action="store_true",
                        help="Plot 95% confidence intervals")
    parser.add_argument("--print_option", type=str, default="all",
                        choices=["all", "latex", "csv", "none"],
                        help="Output format control")
    args = parser.parse_args()

    logs_dir = Path(os.path.join(script_dir, "..", "gae", args.folder))
    out_dir = Path(os.path.join(script_dir, "plots", "gae_ablation", args.environment))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect latest folder per ablated weight
    weight_latest_run = {}
    for run in logs_dir.iterdir():
        if not run.is_dir():
            continue
        if not run.name.startswith(args.environment + "_ablation_"):
            continue

        weight = extract_ablated_weight(run.name)
        # keep the latest timestamp per ablated weight
        current_time = run.name.split("_")[-1]
        if weight not in weight_latest_run or current_time > weight_latest_run[weight].name.split("_")[-1]:
            weight_latest_run[weight] = run

    # weight -> graph -> loss -> stats
    data = defaultdict(lambda: defaultdict(dict))

    for weight, run in weight_latest_run.items():
        print("RUN", run, "SPLIT", args.split)
        yaml_file = load_latest_yaml_from_split_folder(run, args.split)
        if yaml_file is None:
            continue

        with open(yaml_file) as f:
            content = yaml.safe_load(f)
        content = content["compression_statistics"]

        for graph_name, metrics in content.items():

            node_stats = []
            edge_stats = []

            for loss_name, stats in metrics.items():

                # Skip metrics that are not reconstruction losses
                if loss_name == "CR_avg":
                    continue

                if loss_name == "adj_loss":
                    data[graph_name][weight]["adj_loss"] = stats
                    continue

                # Every edge loss starts with "edge_"
                if loss_name.startswith("edge_"):
                    edge_stats.append(stats)
                else:
                    # Everything else is a node reconstruction loss
                    node_stats.append(stats)

            if node_stats:
                data[graph_name][weight]["node_features_loss"] = {
                    "mean": np.mean([s["mean"] for s in node_stats]),
                    "ci95_lower": np.mean([s["ci95_lower"] for s in node_stats]),
                    "ci95_upper": np.mean([s["ci95_upper"] for s in node_stats]),
                }

            if edge_stats:
                data[graph_name][weight]["edge_features_loss"] = {
                    "mean": np.mean([s["mean"] for s in edge_stats]),
                    "ci95_lower": np.mean([s["ci95_lower"] for s in edge_stats]),
                    "ci95_upper": np.mean([s["ci95_upper"] for s in edge_stats]),
                }

    # For each graph type, create separate tables
    for graph_name, graph_data in data.items():
        rows = []
        for weight, losses in graph_data.items():
            for loss_name, stats in losses.items():
                #if stats is None:
                #    continue
                row = {
                    "ablated_weight": weight,
                    "loss": loss_name,
                    "mean": stats.get("mean")
                }
                if "ci95_lower" in stats:
                    row["ci95_lower"] = stats["ci95_lower"]
                    row["ci95_upper"] = stats["ci95_upper"]
                rows.append(row)

        print("\nRows:")
        for r in rows:
            print(r)

        df = pd.DataFrame(rows).sort_values(["loss", "ablated_weight"])
        # Normalize raw loss names (VERY IMPORTANT)
        print(df)
        print(df["loss"].unique())
        delta_matrix = build_delta_matrix(df)

        # Save delta CSV
        delta_csv_path = out_dir / f"{args.environment}_{graph_name}_delta_matrix.csv"
        delta_matrix.to_csv(delta_csv_path)

        if args.print_option == "latex":
            latex_str = to_latex_neurips(
                delta_matrix,
                caption=f"{' '.join(args.environment.capitalize().split('_'))} {' '.join(graph_name.split('_'))} ablation study (compression ratio)",
                label=f"tab:{args.environment}_{graph_name}_ablation"
            )

            latex_path = out_dir / f"{args.environment}_{graph_name}_table.tex"
            with open(latex_path, "w") as f:
                f.write(latex_str)

            print("\n--- LaTeX Table ---\n")
            print(latex_str)

        else:
            # Save CSV and TXT
            csv_path = out_dir / f"{args.environment}_{graph_name}_gae_ablation_summary.csv"
            txt_path = out_dir / f"{args.environment}_{graph_name}_gae_ablation_summary.txt"
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

                plt.bar(x, y, tick_label=sub["ablated_weight"].values)

                if args.plot_ci and "ci95_lower" in sub:
                    yerr = [
                        y - sub["ci95_lower"].values,
                        sub["ci95_upper"].values - y
                    ]
                    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=4)

                plt.ylabel("Mean loss")
                plt.title(f"{args.environment} – {graph_name} – {loss}")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                fig_path = out_dir / f"{args.environment}_{graph_name}_{loss}.png"
                plt.savefig(fig_path)
                plt.close()

            print(f"Saved {graph_name} tables and plots in {out_dir}")


if __name__ == "__main__":
    main()
