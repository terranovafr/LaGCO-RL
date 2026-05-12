#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    plot_cartesian_time.py
    Script to load action times for cartesian runs, fit scaling laws, and produce plots and tables.
"""
import os
import re
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy.stats import linregress
script_dir = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def extract_timestamp(name):
    m = re.search(r"\d{14}", name)
    if m:
        return datetime.strptime(m.group(), "%Y%m%d%H%M%S")
    return datetime.min


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


def load_yaml_list(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return np.array(data, dtype=float)


# ------------------------------------------------------------
# Scaling-law utilities
# ------------------------------------------------------------

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


def remove_outliers_iqr(data):
    data = np.asarray(data, dtype=float)
    if len(data) < 4:
        return data

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return data[(data >= lower_bound) & (data <= upper_bound)]


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
    all_env_results,
    statistic="median",
    remove_outliers=False
):
    """
    all_env_results format:
        {
            "MethodA": {
                size1: [times...],
                size2: [times...],
                ...
            },
            "MethodB": {
                ...
            }
        }
    """
    rows = []

    for method_name, results_by_size in all_env_results.items():
        sizes, times = summarize_times_by_size(
            results_by_size,
            statistic=statistic,
            remove_outliers=remove_outliers
        )

        if len(sizes) < 2:
            print(f"Skipping fit for {method_name}: not enough valid points.")
            continue

        alpha, a, r2 = fit_power_law(sizes, times)

        rows.append({
            "Method": method_name,
            "alpha": alpha,
            "a": a,
            "R2": r2,
            "n_min": np.min(sizes),
            "n_max": np.max(sizes),
            "num_points": len(sizes),
            "statistic": statistic,
            "outliers_removed": remove_outliers,
        })

    return pd.DataFrame(rows)


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
                values = load_yaml_list(yaml_path)

                if test_size not in results:
                    results[test_size] = []

                results[test_size].extend(values.tolist())

    return results


# ------------------------------------------------------------
# Plot: grouped boxplots
# ------------------------------------------------------------
def plot_grouped_boxplots(
    all_env_results,
    save_path=None,
    log_scale="auto",
    ratio_threshold=100,
    figsize=(4.8, 2.4)
):

    # ---- detect scale difference ----
    all_values = []
    for env_data in all_env_results.values():
        for v in env_data.values():
            all_values.extend(v)

    all_values = np.asarray(all_values)

    ratio = np.max(all_values) / max(np.min(all_values), 1e-12)

    if log_scale == "auto":
        use_log = ratio > ratio_threshold
    else:
        use_log = log_scale

    # ---- sizes ----
    all_sizes = sorted({
        size
        for env_data in all_env_results.values()
        for size in env_data.keys()
    })

    env_types = list(all_env_results.keys())
    n_envs = len(env_types)

    fig, ax = plt.subplots(figsize=figsize)

    group_spacing = 4.0
    box_width = 0.5
    inner_spacing = 0.7
    band_padding = 1.2

    cmap = plt.get_cmap("tab10")
    colors = {env: cmap(i) for i, env in enumerate(env_types)}

    legend_handles = []

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
            -inner_spacing * (n_envs - 1) / 2,
            inner_spacing * (n_envs - 1) / 2,
            n_envs
        )

        for j, env in enumerate(env_types):

            if size not in all_env_results[env]:
                continue

            pos = center + offsets[j]
            data = np.asarray(all_env_results[env][size], dtype=float)

            filtered = remove_outliers_iqr(data)
            if len(filtered) > 0:
                data = filtered

            box = ax.boxplot(
                data,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color="black", linewidth=1.3),
                capprops=dict(color="black", linewidth=1.3),
                boxprops=dict(edgecolor="black", linewidth=1.3)
            )

            for patch in box["boxes"]:
                patch.set_facecolor(colors[env])
                patch.set_alpha(0.8)

            # colored median marker (always visible)
            median = np.median(data)
            ax.scatter(
                pos,
                median,
                color=colors[env],
                edgecolor="black",
                s=18,
                zorder=4
            )

            if i == 0:
                legend_handles.append(
                    plt.Line2D([0], [0], color=colors[env], lw=6, label=env)
                )

    centers = [i * group_spacing for i in range(len(all_sizes))]
    ax.set_xticks(centers)
    ax.set_xticklabels(all_sizes)

    if use_log:
        ax.set_yscale("log")

    ax.set_xlabel("Size", fontsize=14)
    ax.set_ylabel("Time (s)", fontsize=14)

    ax.tick_params(axis='both', labelsize=14)

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(handles=legend_handles, fontsize=10, frameon=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()

# ------------------------------------------------------------
# Plot: scaling curves
# ------------------------------------------------------------

def plot_scaling_fits(
    all_env_results,
    statistic="median",
    remove_outliers=False,
    save_path=None
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, results_by_size in all_env_results.items():
        sizes, times = summarize_times_by_size(
            results_by_size,
            statistic=statistic,
            remove_outliers=remove_outliers
        )

        if len(sizes) < 2:
            continue

        alpha, a, r2 = fit_power_law(sizes, times)
        fitted = a * (sizes ** alpha)

        order = np.argsort(sizes)
        sizes = sizes[order]
        times = times[order]
        fitted = fitted[order]

        ax.plot(sizes, times, marker="o", linestyle="", label=f"{method_name} data")
        ax.plot(
            sizes,
            fitted,
            linestyle="--",
            label=f"{method_name} fit: α={alpha:.3f}, R²={r2:.3f}"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Test Size")
    ax.set_ylabel(f"Representative Action Time ({statistic})")
    ax.set_title("Scaling-law fit: action time vs test size")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved scaling plot to {save_path}")
    else:
        plt.show()

    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--logs_folder", required=True)
    parser.add_argument("-e", "--environment", required=True)
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

    all_env_results = {}

    log_scale = args.log_scale
    if log_scale == "true":
        log_scale = True
    elif log_scale == "false":
        log_scale = False

    for sol in args.solutions:
        solution_path = os.path.join(
            script_dir, "..", "agents",
            args.logs_folder,
            args.environment,
            sol
        )

        if not os.path.exists(solution_path):
            print(f"Skipping {sol} (not found)")
            continue

        print(f"Loading {sol}...")
        results = load_action_times_for_env(solution_path)

        for size, values in sorted(results.items()):
            values = np.asarray(values, dtype=float)
            print(
                f"{sol} | size {size} -> "
                f"mean={np.mean(values):.6f}, "
                f"median={np.median(values):.6f}, "
                f"std={np.std(values):.6f}, "
                f"min={np.min(values):.6f}, "
                f"max={np.max(values):.6f}, "
                f"n={len(values)}"
            )

        all_env_results[sol] = results

    if not all_env_results:
        raise RuntimeError("No valid solutions were loaded.")

    save_location = os.path.join(script_dir, "plots", "cartesian_time", args.environment)
    os.makedirs(save_location, exist_ok=True)

    # 1) Boxplot from raw loaded values
    plot_grouped_boxplots(
        all_env_results,
        os.path.join(save_location, "grouped_boxplots.pdf"),
        log_scale=log_scale,
        ratio_threshold=args.log_ratio_threshold
    )

    # 2) Scaling coefficients computed from loaded values
    scaling_df = build_scaling_table_from_loaded_results(
        all_env_results,
        statistic=args.fit_statistic,
        remove_outliers=args.fit_remove_outliers
    )

    print("\nScaling-law coefficients:")
    print(scaling_df)

    scaling_df.to_csv(os.path.join(save_location, "scaling_factor.csv"), index=False)
    print(f"Saved scaling CSV to {save_location}")

    # 3) Optional scaling-fit plot
    plot_scaling_fits(
        all_env_results,
        statistic=args.fit_statistic,
        remove_outliers=args.fit_remove_outliers,
        save_path=os.path.join(save_location, "scaling_fit.pdf")
    )



if __name__ == "__main__":
    main()