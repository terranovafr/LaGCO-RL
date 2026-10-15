#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.


"""
    plot_action_time_selection.py
    Script to plot action time selection results across different solutions and test sizes. It loads raw action time data from YAML files, summarizes it by test size, fits scaling laws, and generates boxplots and scaling fit plots for comparison.
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import extract_timestamp, load_yaml
from utils.math_utils import remove_outliers_iqr
script_dir = os.path.dirname(os.path.abspath(__file__))

PRETTY_ENV_NAMES = {
    "tsp": "TSP",
    "mvc": "MinVertex",
    "maxcut": "MaxCut",
    "vmp": "Placement",
    "cyberattack": "Cyber-Path",
    "ospf_engineering": "OSPF",
    "traffic_engineering": "Traffic"
}

PRETTY_METHOD_NAMES = {
    "iterative": "Iterative",
    "projection": "Projection (ours)",
}

def find_latest_cartesian_folder(env_solution_path):
    candidates = []

    for d in os.listdir(env_solution_path):
        if d.startswith("cartesian_runs"):
            ts = extract_timestamp(d)
            candidates.append((ts, d))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return os.path.join(env_solution_path, candidates[0][1])


def extract_test_size(folder_name):
    # test_size50_20260227095330 -> 50
    m = re.match(r"test_size(\d+)_", folder_name)
    if m:
        return int(m.group(1))
    return None

def fit_power_law(sizes, times):
    """
    Fit t = a * n^alpha using log-log linear regression.

    Returns:
        alpha: exponent
        a: prefactor
        r2: coefficient of determination in log-space
    """
    sizes = np.asarray(sizes, dtype=float)
    times = np.asarray(times, dtype=float)

    if len(sizes) != len(times):
        raise ValueError("sizes and times must have the same length.")
    if len(sizes) < 2:
        raise ValueError("At least 2 points are required for fitting.")
    if np.any(sizes <= 0):
        raise ValueError("All graph sizes must be > 0.")
    if np.any(times <= 0):
        raise ValueError("All action times must be > 0 for log-log fitting.")

    log_n = np.log(sizes)
    log_t = np.log(times)

    result = linregress(log_n, log_t)

    alpha = result.slope
    intercept = result.intercept
    a = np.exp(intercept)
    r2 = result.rvalue ** 2

    return alpha, a, r2



def summarize_times_by_size(results_by_size, statistic="median", remove_outliers=False):
    """
    Convert:
        size -> list of raw action times
    into:
        sizes, representative_times

    statistic: one of ["median", "mean"]
    """
    sizes = []
    rep_times = []

    for size in sorted(results_by_size.keys()):
        values = np.asarray(results_by_size[size], dtype=float)

        if len(values) == 0:
            continue

        if remove_outliers:
            filtered = remove_outliers_iqr(values)
            if len(filtered) > 0:
                values = filtered

        if statistic == "median":
            rep = np.median(values)
        elif statistic == "mean":
            rep = np.mean(values)
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")

        if rep > 0:
            sizes.append(size)
            rep_times.append(rep)

    return np.array(sizes, dtype=float), np.array(rep_times, dtype=float)

def build_scaling_table_from_loaded_results(
    all_results,
    statistic="median",
    remove_outliers=False
):
    """
    all_results format:
    {
        "tsp": {
            "ppo": {
                size1: [times...],
                size2: [times...],
            },
            "random": {
                ...
            }
        },

        "maxcut": {
            ...
        }
    }
    """

    rows = []

    for env_name, env_results in all_results.items():

        for method_name, results_by_size in env_results.items():

            sizes, times = summarize_times_by_size(
                results_by_size,
                statistic=statistic,
                remove_outliers=remove_outliers
            )

            if len(sizes) < 2:
                print(
                    f"Skipping fit for "
                    f"{env_name}/{method_name}: "
                    f"not enough valid points."
                )
                continue

            alpha, a, r2 = fit_power_law(sizes, times)

            rows.append({
                "Environment": PRETTY_ENV_NAMES.get(env_name, env_name),
                "Method": PRETTY_METHOD_NAMES.get(method_name, method_name),

                "alpha": alpha,
                "R2": r2,
                "a": a,
                "n_min": np.min(sizes),
                "n_max": np.max(sizes),
                "num_points": len(sizes),

                "statistic": statistic,
                "outliers_removed": remove_outliers,
            })

    df = pd.DataFrame(rows)

    # optional nice sorting
    if not df.empty:
        df = df.sort_values(
            by=["Environment", "alpha"],
            ascending=[True, True]
        )

    return df

def dataframe_to_latex(
    df,
    caption="Scaling-law fit of action time versus graph size.",
    label="tab:scaling_law",
    float_format="%.3f"
):
    return df.to_latex(
        float_format=float_format,
        escape=False,
        caption=caption,
        label=label,
        bold_rows=False,
        multicolumn=True,
        multirow=False,
        index=False,
    )


# ------------------------------------------------------------
# Core loader
# ------------------------------------------------------------

def load_action_times_for_env(env_solution_path):
    latest_cartesian = find_latest_cartesian_folder(env_solution_path)

    if latest_cartesian is None:
        raise RuntimeError(f"No cartesian_runs folder in {env_solution_path}")

    run_folders = [
        d for d in os.listdir(latest_cartesian)
        if "_run_" in d
    ]

    if len(run_folders) == 0:
        raise RuntimeError(f"No run folder found in {latest_cartesian}")

    run_folder = os.path.join(latest_cartesian, run_folders[0])

    test_root = os.path.join(
        run_folder,
        "test",
        "default",
        "train",
        "1",
        "test_set"
    )

    if not os.path.exists(test_root):
        return {}

    results = {}  # size -> list of action times

    for test_folder in os.listdir(test_root):
        test_size = extract_test_size(test_folder)
        if test_size is None:
            continue

        full_test_path = os.path.join(test_root, test_folder)

        for f in os.listdir(full_test_path):
            if f.startswith("action_times_") and f.endswith(".yaml") and "summary" not in f:
                yaml_path = os.path.join(full_test_path, f)
                values = load_yaml(yaml_path)

                if test_size not in results:
                    results[test_size] = []

                results[test_size].extend(values)

    return results

from matplotlib.ticker import LogFormatterMathtext


from matplotlib.ticker import LogLocator
from matplotlib.ticker import LogFormatterMathtext


def plot_grouped_boxplots(
    all_results,
    save_path=None,
    log_scale="auto",
    ratio_threshold=100,
    base_figsize=(12, 8)
):

    n_envs = len(all_results)

    fig_width = base_figsize[0] * n_envs
    fig_height = base_figsize[1]

    fig, axes = plt.subplots(
        1,
        n_envs,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True
    )

    axes = axes.flatten()

    # ---------------------------------------------------------
    # global log-scale decision
    # ---------------------------------------------------------
    all_values = []

    for env_data in all_results.values():
        for method_data in env_data.values():
            for v in method_data.values():
                all_values.extend(v)

    all_values = np.asarray(all_values)

    ratio = np.max(all_values) / max(np.min(all_values), 1e-12)

    if log_scale == "auto":
        use_log = ratio > ratio_threshold
    else:
        use_log = log_scale

    # ---------------------------------------------------------
    # consistent colors across subplots
    # ---------------------------------------------------------
    all_methods = sorted({
        method
        for env_data in all_results.values()
        for method in env_data.keys()
    })

    cmap = plt.get_cmap("tab10")

    colors = {
        method: cmap(i)
        for i, method in enumerate(all_methods)
    }

    # ---------------------------------------------------------
    # shared legend handles
    # ---------------------------------------------------------
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[method],
            lw=10,
            label=PRETTY_METHOD_NAMES.get(method, method)
        )
        for method in all_methods
    ]

    # ---------------------------------------------------------
    # one subplot per environment
    # ---------------------------------------------------------
    for ax, (env_name, env_results) in zip(axes, all_results.items()):

        all_sizes = sorted({
            size
            for method_data in env_results.values()
            for size in method_data.keys()
        })

        methods = list(env_results.keys())
        n_methods = len(methods)

        # ---------------------------------------------------------
        # Bigger grouped boxes
        # ---------------------------------------------------------
        group_spacing = 5.0
        box_width = 0.8
        inner_spacing = 1.0
        band_padding = 1.6

        for i, size in enumerate(all_sizes):

            center = i * group_spacing

            ax.axvspan(
                center - band_padding,
                center + band_padding,
                color="lightgray",
                alpha=0.25,
                zorder=0
            )

            offsets = np.linspace(
                -inner_spacing * (n_methods - 1) / 2,
                inner_spacing * (n_methods - 1) / 2,
                n_methods
            )

            for j, method in enumerate(methods):

                if size not in env_results[method]:
                    continue

                pos = center + offsets[j]

                data = np.asarray(
                    env_results[method][size],
                    dtype=float
                )

                filtered = remove_outliers_iqr(data)

                if len(filtered) > 0:
                    data = filtered

                color = colors[method]

                # ---------------------------------------------------------
                # Boxplot
                # ---------------------------------------------------------
                box = ax.boxplot(
                    data,
                    positions=[pos],
                    widths=box_width,
                    patch_artist=True,
                    manage_ticks=False,

                    medianprops=dict(
                        color=color,
                        linewidth=3.0
                    ),

                    whiskerprops=dict(
                        color=color,
                        linewidth=2.5
                    ),

                    capprops=dict(
                        color=color,
                        linewidth=2.5
                    ),

                    boxprops=dict(
                        edgecolor=color,
                        linewidth=2.5
                    ),

                    flierprops=dict(
                        marker='o',
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markersize=6,
                        alpha=0.7
                    )
                )

                for patch in box["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)
                    patch.set_alpha(0.35)

                # ---------------------------------------------------------
                # Median marker
                # ---------------------------------------------------------
                median = np.median(data)

                ax.scatter(
                    pos,
                    median,
                    color=color,
                    edgecolor=color,
                    s=45,
                    zorder=4
                )

        centers = [
            i * group_spacing
            for i in range(len(all_sizes))
        ]

        # ---------------------------------------------------------
        # x-axis
        # ---------------------------------------------------------
        ax.set_xticks(centers)

        ax.set_xticklabels(
            all_sizes,
            rotation=60,
            ha="right",
            fontsize=48
        )

        ax.tick_params(
            axis='x',
            labelsize=48
        )

        # ---------------------------------------------------------
        # log-scale formatting
        # ---------------------------------------------------------
        if use_log:

            ax.set_yscale("log")

            # Major ticks
            ax.yaxis.set_major_locator(
                LogLocator(
                    base=10.0,
                    numticks=10
                )
            )

            # Minor ticks only at 2 and 5
            ax.yaxis.set_minor_locator(
                LogLocator(
                    base=10.0,
                    subs=(2.0, 5.0),
                    numticks=10
                )
            )

            # Formatting
            formatter = LogFormatterMathtext(base=10.0)

            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)

            ax.yaxis.offsetText.set_visible(False)

            # Tick styling
            ax.tick_params(
                axis='y',
                which='major',
                labelsize=48,
                length=8,
                width=1.5
            )

            ax.tick_params(
                axis='y',
                which='minor',
                labelsize=48,
                length=5,
                width=1.0
            )

        else:

            ax.tick_params(
                axis='y',
                labelsize=38
            )

        # ---------------------------------------------------------
        # labels/titles
        # ---------------------------------------------------------
        ax.set_xlabel(
            "Graph Size",
            fontsize=50
        )

        ax.set_title(
            PRETTY_ENV_NAMES.get(
                env_name,
                env_name
            ),
            fontsize=50,
            pad=20
        )

        # ---------------------------------------------------------
        # grid
        # ---------------------------------------------------------
        ax.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

    # ---------------------------------------------------------
    # Shared y-label
    # ---------------------------------------------------------
    axes[0].set_ylabel(
        "Action Time (s)",
        fontsize=50
    )

    # ---------------------------------------------------------
    # Shared legend centered above plots
    # ---------------------------------------------------------
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1, #len(all_methods),
        fontsize=46,
        frameon=True,
        # make frame background strong white, full opacity, with a thick border
        fancybox=False,
        #edgecolor="black",
        # solid opaque legend box
        framealpha=1.0,
        facecolor="white",
    )

    # ---------------------------------------------------------
    # layout
    # ---------------------------------------------------------
    fig.tight_layout()

    fig.subplots_adjust(
        wspace=0.10,
        left=0.08,
        right=0.98,
        bottom=0.18,
        top=0.82
    )

    # ---------------------------------------------------------
    # save/show
    # ---------------------------------------------------------
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )
    else:
        plt.show()

    plt.close()

def plot_scaling_fits(
    all_results,
    statistic="median",
    remove_outliers=False,
    save_path=None,
    base_figsize=(10, 6)
):

    n_envs = len(all_results)

    fig_width = base_figsize[0] * n_envs
    fig_height = base_figsize[1]

    fig, axes = plt.subplots(
        1,
        n_envs,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    axes = axes.flatten()

    # ---------------------------------------------------------
    # consistent colors across all environments
    # ---------------------------------------------------------
    all_methods = sorted({
        method
        for env_results in all_results.values()
        for method in env_results.keys()
    })

    cmap = plt.get_cmap("tab10")

    colors = {
        method: cmap(i)
        for i, method in enumerate(all_methods)
    }

    # ---------------------------------------------------------
    # one subplot per environment
    # ---------------------------------------------------------
    for ax, (env_name, env_results) in zip(axes, all_results.items()):

        for method_name, results_by_size in env_results.items():

            sizes, times = summarize_times_by_size(
                results_by_size,
                statistic=statistic,
                remove_outliers=remove_outliers
            )

            if len(sizes) < 2:
                print(
                    f"Skipping scaling fit for "
                    f"{env_name}/{method_name}"
                )
                continue

            alpha, a, r2 = fit_power_law(sizes, times)

            fitted = a * (sizes ** alpha)

            order = np.argsort(sizes)

            sizes = sizes[order]
            times = times[order]
            fitted = fitted[order]

            pretty_method = PRETTY_METHOD_NAMES.get(
                method_name,
                method_name
            )

            color = colors[method_name]

            # raw points
            ax.plot(
                sizes,
                times,
                marker="o",
                linestyle="",
                color=color,
                markersize=7,
                label=f"{pretty_method} data"
            )

            # fitted curve
            ax.plot(
                sizes,
                fitted,
                linestyle="--",
                linewidth=2,
                color=color,
                label=(
                    f"{pretty_method} "
                    f"(α={alpha:.2f}, R²={r2:.2f})"
                )
            )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.5
        )

        ax.set_xlabel(
            "Instance Size",
            fontsize=22
        )

        ax.set_title(
            PRETTY_ENV_NAMES.get(env_name, env_name),
            fontsize=26,
            pad=15
        )

        ax.tick_params(axis='x', labelsize=25)

        ax.tick_params(
            axis='y',
            which='major',
            labelsize=22
        )

        ax.tick_params(
            axis='y',
            which='minor',
            labelsize=18
        )

        ax.legend(
            fontsize=14,
            frameon=False,
            # make on two lines
        )

    # only left subplot gets y-label
    axes[0].set_ylabel(
        f"Action Time ({statistic})",
        fontsize=24
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Saved scaling plot to {save_path}")
    else:
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--logs_folder", required=True)
    parser.add_argument(
        "-e",
        "--environments",
        nargs="+",
        required=True,
        help="List of environments / benchmarks"
    )
    parser.add_argument(
        "--solutions",
        nargs="+",
        required=True,
        help="List of env_types / solution folders"
    )
    parser.add_argument(
        "--fit_statistic",
        choices=["mean", "median"],
        default="median",
        help="Representative time per size used for scaling fit"
    )
    parser.add_argument(
        "--fit_remove_outliers",
        action="store_true",
        help="Remove outliers before computing representative time for scaling fit"
    )
    parser.add_argument("--log_scale", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--log_ratio_threshold", type=float, default=100)

    args = parser.parse_args()

    all_results = {}

    log_scale = args.log_scale
    if log_scale == "true":
        log_scale = True
    elif log_scale == "false":
        log_scale = False

    for env in args.environments:

        env_results = {}

        for sol in args.solutions:

            solution_path = os.path.join(
                script_dir,
                "..",
                "agents",
                args.logs_folder,
                env,
                sol
            )

            if not os.path.exists(solution_path):
                print(f"Skipping {env}/{sol} (not found)")
                continue

            print(f"Loading {env} | {sol}...")

            results = load_action_times_for_env(solution_path)

            env_results[sol] = results

        if env_results:
            all_results[env] = env_results

    if not all_results:
        raise RuntimeError("No valid solutions were loaded.")

    save_location = os.path.join(
        script_dir,
        "plots",
        "action_time_selection",
        "_vs_".join(args.environments)
    )
    os.makedirs(save_location, exist_ok=True)

    # 1) Boxplot from raw loaded values
    plot_grouped_boxplots(
        all_results,
        os.path.join(save_location, "grouped_boxplots.pdf"),
        log_scale=log_scale,
        ratio_threshold=args.log_ratio_threshold
    )

    # 2) Scaling coefficients computed from loaded values
    scaling_df = build_scaling_table_from_loaded_results(
        all_results,
        statistic=args.fit_statistic,
        remove_outliers=args.fit_remove_outliers
    )

    print("\nScaling-law coefficients:")
    print(scaling_df)

    scaling_df.to_csv(os.path.join(save_location, "scaling_factor.csv"), index=False)
    print(f"Saved scaling CSV to {save_location}")

    # 3) Optional scaling-fit plot
    plot_scaling_fits(
        all_results,
        statistic=args.fit_statistic,
        remove_outliers=args.fit_remove_outliers,
        save_path=os.path.join(save_location, "scaling_fit.pdf")
    )



if __name__ == "__main__":
    main()