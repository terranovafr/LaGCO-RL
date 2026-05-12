#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    plot_gae_ablation.py
    Script to plot GAE ablation results from YAML logs
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
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import load_latest_yaml_from_split_folder

script_dir = os.path.dirname(os.path.abspath(__file__))

weight_to_lossname = {
    "adj_weight": "adj_loss",
    "node_feature_vector_weight": "node_feature_vector_loss",
    "edge_feature_vector_weight": "edge_feature_vector_loss",
    "diversity_weight": "diversity_loss",
    "node_feature_vector_binary_cat_weight": "node_feature_vector_binary_cat_loss",
    "node_feature_vector_multi_cat_weight": "node_feature_vector_multi_cat_loss",
    "node_feature_vector_cont_weight": "node_feature_vector_cont_loss",
    "node_feature_vector_ranking_weight": "node_feature_vector_ranking_loss",
    "edge_feature_vector_binary_cat_weight": "edge_feature_vector_binary_cat_loss",
    "edge_feature_vector_multi_cat_weight": "edge_feature_vector_multi_cat_loss",
    "edge_feature_vector_cont_weight": "edge_feature_vector_cont_loss",
    "edge_feature_vector_ranking_weight": "edge_feature_vector_ranking_loss",
    "kl_weight": "kl_loss",
}

pretty_loss_names = {
    "adj_loss": "Adjacency Loss",
    "node_feature_vector_binary_cat_loss": "Node Binary Features Loss",
    "node_feature_vector_multi_cat_loss": "Node Categorical Features Loss",
    "node_feature_vector_cont_loss": "Node Continuous Features Loss",
    "node_feature_vector_ranking_loss": "Node Ranking Loss",
    "edge_feature_vector_binary_cat_loss": "Edge Binary Features Loss",
    "edge_feature_vector_multi_cat_loss": "Edge Categorical Features Loss",
    "edge_feature_vector_cont_loss": "Edge Continuous Features Loss",
    "edge_feature_vector_ranking_loss": "Edge Ranking Loss",
    "kl_loss": "KL Loss",
}

pretty_weight_names = {
    "adj_weight": "Adjacency Loss",
    "node_feature_vector_binary_cat_weight": "Node Binary Features Loss",
    "node_feature_vector_multi_cat_weight": "Node Categorical Features Loss",
    "node_feature_vector_cont_weight": "Node Continuous Features Loss",
    "node_feature_vector_ranking_weight": "Node Ranking Loss",
    "edge_feature_vector_binary_cat_weight": "Edge Binary Features Loss",
    "edge_feature_vector_multi_cat_weight": "Edge Categorical Features Loss",
    "edge_feature_vector_cont_weight": "Edge Continuous Features Loss",
    "edge_feature_vector_ranking_weight": "Edge Ranking Loss",
    "kl_weight": "KL Loss",
}

def build_delta_matrix(df):
    pivot = df.pivot(index="ablated_weight", columns="loss", values="mean")

    # -------------------------------
    # 1. Extract baseline row
    # -------------------------------
    if "baseline" not in pivot.index:
        raise ValueError("Baseline (_ablation_None) not found!")

    baseline = pivot.loc["baseline"]

    # -------------------------------
    # 2. Compute delta
    # -------------------------------
    delta = pivot #.subtract(baseline, axis=1)

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

    # -------------------------------
    # 4. Mask diagonal ONLY for ablations
    # -------------------------------
    for weight in df["ablated_weight"].unique():
        if weight == "baseline":
            continue  # 🚨 do NOT mask baseline

        raw_loss = weight_to_lossname.get(weight)
        if raw_loss is None:
            continue

        pretty_row = pretty_weight_names.get(weight)
        pretty_col = pretty_loss_names.get(raw_loss)

        if pretty_row in delta.index and pretty_col in delta.columns:
            delta.loc[pretty_row, pretty_col] = np.nan

    # -------------------------------
    # 5. Remove invalid rows (except baseline)
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
        "Adjacency Loss": r"\thead{Adjacency\\Loss}",
        "Edge Continuous Features Loss": r"\thead{Edge\\Continuous}",
        "Node Binary Features Loss": r"\thead{Node\\Binary}",
        "Node Continuous Features Loss": r"\thead{Node\\Continuous}",
        "Node Categorical Features Loss": r"\thead{Node\\Categorical}",
    }

    row_map = {
        "Adjacency Loss": "Adjacency Loss",
        "Edge Continuous Features Loss": "Edge Continuous Features",
        "Node Binary Features Loss": "Node Binary Features",
        "Node Continuous Features Loss": "Node Continuous Features",
        "Node Categorical Features Loss": "Node Categorical Features",
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
    parser.add_argument("--environment", type=str, required=True,
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
        yaml_file = load_latest_yaml_from_split_folder(run, args.split)
        if yaml_file is None:
            continue

        with open(yaml_file) as f:
            content = yaml.safe_load(f)

        for graph_name, metrics in content.items():
            for loss_name, stats in metrics.items():
                # Determine the loss corresponding to the ablated weight
                expected_loss_name = weight_to_lossname.get(weight, weight)
                if loss_name == expected_loss_name:
                    data[graph_name][weight][loss_name] = None
                else:
                    data[graph_name][weight][loss_name] = stats



    # For each graph type, create separate tables
    for graph_name, graph_data in data.items():
        rows = []
        for weight, losses in graph_data.items():
            for loss_name, stats in losses.items():
                if stats is None:
                    continue
                row = {
                    "ablated_weight": weight,
                    "loss": loss_name,
                    "mean": stats.get("mean")
                }
                if "ci95_lower" in stats:
                    row["ci95_lower"] = stats["ci95_lower"]
                    row["ci95_upper"] = stats["ci95_upper"]
                rows.append(row)

        df = pd.DataFrame(rows).sort_values(["loss", "ablated_weight"])
        # Normalize raw loss names (VERY IMPORTANT)
        loss_rename_map = {
            "binary_cat_loss": "node_feature_vector_binary_cat_loss",
            "multi_cat_loss": "node_feature_vector_multi_cat_loss",
            "cont_loss": "node_feature_vector_cont_loss",
            "edge_continuous_loss": "edge_feature_vector_cont_loss",
        }

        df["loss"] = df["loss"].replace(loss_rename_map)

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
