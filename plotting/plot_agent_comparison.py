#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    plot_agent_comparison.py
    Script to compare agent performances across solutions and environment sampling cases, with statistical analysis and LaTeX table generation.
"""

import argparse
import pandas as pd
from datetime import datetime
import sys
import os
import re
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import load_yaml
from utils.math_utils import bootstrap_ci, compute_indicator, compute_mean_std_bci
script_dir = os.path.dirname(os.path.abspath(__file__))
from scipy.stats import shapiro

SOLUTION_DISPLAY_NAMES = {
    "DO_discrete": "P-Discrete",
    "DO_discrete_M": "P-Discrete-M",
    "GO_discrete": "G-Discrete",
    "GO_discrete_M": "G-Discrete-M",
    "iterative": "Iterative",
    "projection": "Projection (ours)",
    "DO_discrete_semantic": "DO-Discrete-Semantic",
    "projection_approximate": "Projection Approx",
    "projection_sample": "Projection Sample",
    "projection_pca": "Projection PCA",
}

ENV_DISPLAY_NAMES = {
    "vmp": "Placement",
    "tsp": "Travel",
    "mvc": "MinVertex",
    "maxcut": "MaxCut",
    "cyberattack": "Cyber",
    "ospf_engineering": "OSPF",
    "traffic_engineering": "Traffic",
}

def format_env_name(env):
    return ENV_DISPLAY_NAMES.get(env.lower(), env)


CSV_PREFIX = "average_performances_best_val_"  # assuming this is defined somewhere

def check_normality(all_results):
    # Shapiro-Wilk test for normality on each (env, solution, case) group
    print("\n=== NORMALITY TEST (Shapiro-Wilk) ===")

    for env in all_results:
        for sol in all_results[env]:
            for case in all_results[env][sol]:
                data = np.asarray(all_results[env][sol][case])
                if len(data) < 3:
                    continue

                stat, p = shapiro(data)

                normal = "YES" if p > 0.05 else "NO"

                print(
                    f"{env} | {sol} | {case} -> "
                    f"W={stat:.3f}, p={p:.3e} => Normal: {normal}"
                )



def load_env_indicators_from_multiple_runs(
    solution_root_folder,
    metric_column,
    indicator,
    env_id_col="environment_ID",
    no_training=False,
):
    """
    Detect run_* folders inside solution_root_folder.
    If found:
        - Load metrics from each run
        - Concatenate test env metrics across runs
        - Aggregate training stats across runs
        - Merge per-env scores across runs by averaging the score of the same env_id
    Otherwise:
        - Fallback to single-run loader
    """

    run_folders = [
        os.path.join(solution_root_folder, d)
        for d in os.listdir(solution_root_folder)
        if d.startswith("run_") and os.path.isdir(os.path.join(solution_root_folder, d))
    ]

    # SINGLE RUN
    if not run_folders:
        return load_env_indicators_from_csv(
            solution_root_folder,
            metric_column,
            indicator,
            env_id_col,
            no_training
        )

    all_test_metrics = []
    all_test_env_scores = {}  # env_id -> list of scores across runs
    all_train_metrics = []

    for run_folder in sorted(run_folders):
        try:
            test_metrics, train_metrics, test_env_score_map = load_env_indicators_from_csv(
                run_folder,
                metric_column,
                indicator,
                env_id_col,
                no_training
            )

            if train_metrics is not None:
                all_train_metrics.extend(train_metrics)

            if test_metrics is not None:
                all_test_metrics.extend(test_metrics)

            for env_id, score in test_env_score_map.items():
                all_test_env_scores.setdefault(env_id, []).append(score)

        except Exception as e:
            print(f"Warning in run folder '{run_folder}': {e}")

    if not all_test_metrics:
        raise RuntimeError("No test metrics found across runs")

    # average repeated env_id scores across runs
    merged_test_env_score_map = {
        env_id: float(np.mean(scores))
        for env_id, scores in all_test_env_scores.items()
    }


    return np.asarray(all_test_metrics),  np.asarray(all_train_metrics), merged_test_env_score_map


def find_best_performance_csv(solution_folder):
    """
    Find the CSV file:
    average_performances_best_val_*.csv
    and pick the one with the highest episode count.
    """
    pattern = re.compile(rf"{CSV_PREFIX}(\d+)\.csv")

    best_file = None
    best_episodes = -1

    for f in os.listdir(solution_folder):
        m = pattern.match(f)
        if not m:
            continue

        episodes = int(m.group(1))
        if episodes > best_episodes:
            best_episodes = episodes
            best_file = f

    if best_file is None:
        raise FileNotFoundError(
            f"No CSV matching '{CSV_PREFIX}*.csv' found in {solution_folder}"
        )
    return os.path.join(solution_folder, best_file), best_episodes

def find_latest_env_with_episodes(logs_path):
    pattern = f"{CSV_PREFIX}*.csv"
    episode_regex = re.compile(rf"{CSV_PREFIX}(\d+)\.csv")

    # will store entries as:
    # split -> list of dicts
    # { "file": ..., "root": ..., "episodes": ..., "timestamp": ... }
    matches = {
        "training_set": [],
        "test_set": []
    }

    def extract_timestamp(path):
        parts = path.split(os.sep)
        for p in reversed(parts):
            if p.isdigit() and len(p) >= 14:
                return datetime.strptime(p[:14], "%Y%m%d%H%M%S")
        return datetime.min

    for root, _, files in os.walk(logs_path):
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                print("Match", f, pattern)
                m = episode_regex.match(f)
                if not m:
                    continue

                episodes = int(m.group(1))

                # go two levels up
                parts = root.split(os.sep)
                if len(parts) < 3:
                    continue

                split_folder = parts[-2]

                if split_folder not in matches:
                    continue

                matches[split_folder].append({
                    "file": os.path.join(root, f),
                    "root": root,
                    "episodes": episodes,
                    "timestamp": extract_timestamp(root)
                })
    results = {}
    for split, files in matches.items():
        if not files:
            results[split] = None
            continue
        # 1️⃣ largest NE
        max_episodes = max(e["episodes"] for e in files)
        best_ne = [e for e in files if e["episodes"] == max_episodes]

        # 2️⃣ latest timestep
        best_ne.sort(key=lambda x: x["timestamp"], reverse=True)
        # store parent of the file
        parent_dir = os.path.dirname(best_ne[0]["file"])
        results[split] = parent_dir

        #print(f"[{split}] Max episodes: {max_episodes}")
        #print(f"[{split}] Selected file: {results[split]}")
    return results["training_set"], results["test_set"]


def plot_mean_ci(metrics_dict, save_path=None):
    """
    Plot mean, IQM, and 95% bootstrap CI for multiple solutions.
    Handles negative values correctly.
    """
    solution_names = list(metrics_dict.keys())
    means = []
    iqms = []
    maxs = []
    lower_errors = []
    upper_errors = []

    for sol in solution_names:
        data = np.array(metrics_dict[sol])
        mean = np.mean(data)
        means.append(mean)

        # IQM: mean of middle 50%
        ci25, ci75 = np.percentile(data, [25, 75])
        iqm = data[(data >= ci25) & (data <= ci75)].mean()
        iqms.append(iqm)

        max = np.max(data)
        maxs.append(max)

        # Bootstrap CI of the mean
        ci_lower, ci_upper = bootstrap_ci(data, 10000, 95)
        lower_errors.append(mean - ci_lower)
        upper_errors.append(ci_upper - mean)

    x = np.arange(len(solution_names))
    plt.figure(figsize=(10, 6))

    yerr = np.array([lower_errors, upper_errors])

    plt.bar(x, means, yerr=yerr, capsize=5, color='black', edgecolor='blue', alpha=0.7, label='Mean ± 95% CI')
    plt.scatter(x, iqms, color='black', zorder=5, label='IQM')
    plt.scatter(x, maxs, color='red', zorder=5, label='Max')
    plt.xticks(x, solution_names)
    plt.ylabel("Agent Score")
    plt.title("Mean, IQM, and 95% Bootstrap CI Across Solutions")
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def plot_boxplots_grouped(metrics_dict, indicator, save_path=None):
    cases = sorted({
        case
        for sol in metrics_dict
        for case in metrics_dict[sol].keys()
    })

    solutions = sorted(metrics_dict.keys())

    # Fixed consistent color map
    color_map = {
        "largest": "#4E79A7",
        "smallest": "#E15759",
        "mean": "#59A14F",
        "random": "#F28E2B",
        "random_pct": "#F28E2B",
    }

    n_cases = len(cases)
    n_solutions = len(solutions)

    width = 0.15
    x = np.arange(n_solutions)

    plt.figure(figsize=(12, 6))

    for case_idx, case in enumerate(cases):
        positions = x + (case_idx - (n_cases - 1) / 2) * width

        data = []
        for sol in solutions:
            if case in metrics_dict[sol]:
                data.append(metrics_dict[sol][case])
            else:
                data.append([])

        bp = plt.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=True,
        )

        for box in bp["boxes"]:
            box.set(facecolor=color_map[case], alpha=0.7)

        for median in bp["medians"]:
            median.set(color="black", linewidth=1.5)

    plt.xticks(x, solutions)
    plt.ylabel(f"{indicator.upper()} per environment")
    plt.title("Comparison across environment sampling cases")

    # Legend
    handles = [
        plt.Line2D([0], [0], color=color_map[c], lw=6)
        for c in cases
    ]
    plt.legend(handles, cases, title="Sampling Case")

    plt.axhline(0, color="gray", linewidth=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()



def load_env_indicators_from_csv(folder, metric_column, indicator, env_id_col="environment_ID", no_training=False):
    if indicator == "overall":
        raise ValueError(
            "'overall' is not compatible with per-env best-solution stats because it does not produce one scalar per env_id."
        )

    train_solution_folder, test_solution_folder = find_latest_env_with_episodes(folder)

    results = {}
    per_env_results = {}
    train_stats = None
    splits = [("test", test_solution_folder)]
    if not no_training:
        splits.append(("train", train_solution_folder))

    for split, sol_folder in splits:
        if sol_folder is None:
            continue

        csv_path, available_episodes = find_best_performance_csv(sol_folder)
        df = pd.read_csv(csv_path)

        if metric_column not in df.columns:
            raise KeyError(f"Column '{metric_column}' not found in {csv_path}")

        env_values = []
        env_score_map = {}

        for env_id, group in df.groupby(env_id_col):
            values = group[metric_column].dropna().values
            if len(values) == 0:
                continue

            env_score = compute_indicator(values, indicator)
            env_values.append(env_score)
            env_score_map[env_id] = env_score

            #print(f"Computed {indicator} for env {env_id}: {env_score}")

        env_values = np.asarray(env_values)

        results[split] = env_values
        per_env_results[split] = env_score_map

        print(
            f"[{split}] envs={len(env_values)}, "
            f"episodes_used={available_episodes}"
        )

    if "test" not in results:
        raise RuntimeError("No test metrics found")


    return results["test"], results.get("train", None), per_env_results["test"]



def plot_mean_ci_by_group(
    metrics_dict,
    indicator,
    train_stats_dict=None,
    save_path=None,
    ci=95,
):

    methods = list(metrics_dict.keys())
    groups = sorted({g for m in methods for g in metrics_dict[m].keys()})

    n_methods = len(methods)
    n_groups = len(groups)

    x = np.arange(n_methods)

    bar_width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10.colors

    for i, group in enumerate(groups):

        means = []
        cis = []

        for method in methods:

            if group not in metrics_dict[method]:
                means.append(np.nan)
                cis.append(0)
                continue

            values = np.asarray(metrics_dict[method][group])

            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)

            ci_val = 1.96 * std / np.sqrt(n)

            means.append(mean)
            cis.append(ci_val)

        offsets = x + (i - n_groups / 2) * bar_width + bar_width / 2

        ax.bar(
            offsets,
            means,
            width=bar_width,
            yerr=cis,
            capsize=3,
            label=group,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )

    # training stats overlay
    if train_stats_dict is not None:

        for i, method in enumerate(methods):

            if method not in train_stats_dict:
                continue

            train_mean = train_stats_dict[method]["mean"]

            ax.hlines(
                train_mean,
                i - 0.45,
                i + 0.45,
                linestyles="dashed",
                colors="black",
                linewidth=2,
                label="Training mean" if i == 0 else None,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(indicator.upper())
    ax.set_title(f"{indicator.upper()} Mean ± {ci}% CI")

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(title="Group", frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()




def print_all_stats(metrics_dict, ci=95):
    print("\n=== STATS OVER ALL ENV SCORES ===")
    for sol in metrics_dict:
        for case in metrics_dict[sol]:
            data = metrics_dict[sol][case]
            mean, std, ci_lower, ci_upper = compute_mean_std_bci(data, ci=ci)
            print(
                f"{sol} | {case} | "
                f"Mean: {mean:.3f} | STD: {std:.3f} | " 
                f"{ci}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
            )


def print_best_per_env_stats(metrics_by_env_dict, ci=95):
    """
    For each case and each solution:
      - take the score stored for each env_id as that env's best/max score
      - compute stats across all env_ids for that solution

    Assumes:
        metrics_by_env_dict[solution][case][env_id] = best scalar for that env_id
    """
    print("\n=== STATS OVER PER-ENV MAX SCORES ===")

    all_cases = sorted({
        case
        for sol in metrics_by_env_dict
        for case in metrics_by_env_dict[sol]
    })

    for case in all_cases:
        print(f"\nCase: {case}")

        for sol in sorted(metrics_by_env_dict.keys()):
            if case not in metrics_by_env_dict[sol]:
                continue

            # one scalar per env_id: its max/best score for this solution
            per_env_maxs = np.asarray(list(metrics_by_env_dict[sol][case].values()))

            mean, std, ci_lower, ci_upper = compute_mean_std_bci(per_env_maxs, ci=ci)

            print(
                f"{sol} | {case} | "
                f"n_envs: {len(per_env_maxs)} | "
                f"Mean over env maxs: {mean:.3f} | "
                f"STD: {std:.3f} | "
                f"{ci}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
            )



def latex_escape(text):
    return str(text).replace("_", r"\_")


def format_mean_std_latex(mean, std, precision=2):
    if np.isnan(mean) or np.isnan(std):
        return ""
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def is_normal(values):
    if len(values) < 3:
        return None, None, ""
    w, p = shapiro(values)
    return w, p, f"\\checkmark (p={p:.2f})" if p > 0.05 else f"\\times (p={p:.2f})"



def print_latex_extended_table(
    all_results,
    env_order,
    solution_order,
    ci=95
):

    regime_order = ["smallest", "largest", "mean", "random_pct"]

    regime_letters = {
        "smallest": "S",
        "largest": "L",
        "mean": "M",
        "random_pct": "V",
    }

    print("\n% ===== LATEX EXTENDED TABLE =====")

    col_spec = "ll" + "c" * len(solution_order)

    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\footnotesize")
    print("\\setlength{\\tabcolsep}{4pt}")
    print("\\renewcommand{\\arraystretch}{1.1}")

    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    header = ["Regime", "Metric"] + [
        SOLUTION_DISPLAY_NAMES.get(s, s) for s in solution_order
    ]
    print(" & ".join(header) + r" \\")

    print("\\midrule")

    for env in env_order:
        env_name = format_env_name(env)

        print(f"\\multicolumn{{{len(solution_order)+2}}}{{c}}{{\\textbf{{{env_name}}}}} \\\\")
        print("\\midrule")

        for regime in regime_order:

            # -------------------------
            # normality cache
            # -------------------------
            norm_cache = {}
            for sol in solution_order:
                try:
                    vals = np.asarray(all_results[env][sol][regime])
                    _, p, norm = is_normal(vals)
                    norm_cache[sol] = norm
                except Exception:
                    norm_cache[sol] = ""

            metrics = ["Mean", "Std", "IQM", "BCI", "Normality"]

            for i, ind in enumerate(metrics):

                row = []

                # =========================
                # TRUE MULTIROW (NeurIPS style)
                # =========================
                if i == 0:
                    row.append(f"\\multirow{{5}}{{*}}{{{regime_letters[regime]}}}")
                else:
                    row.append("")

                row.append(r"\textit{" + ("Normal" if ind == "Normality" else ind) + "}")

                for sol in solution_order:

                    # -------------------------
                    # NORMALITY ROW
                    # -------------------------
                    if ind == "Normality":
                        row.append(norm_cache.get(sol, ""))
                        continue

                    # -------------------------
                    # METRICS
                    # -------------------------
                    try:
                        values = np.asarray(all_results[env][sol][regime])

                        if len(values) == 0:
                            row.append("")
                            continue

                        mean, std, iqm, (ci_low, ci_up) = mean_std_iqm_ci(values, ci=ci)

                        if ind == "Mean":
                            row.append(f"\\textbf{{{mean:.2f}}}")

                        elif ind == "Std":
                            s_low, s_high = asymmetric_std(values)
                            row.append(f"-{s_low:.2f}/+{s_high:.2f}")

                        elif ind == "IQM":
                            row.append(f"{iqm:.2f}")

                        elif ind == "BCI":
                            row.append(f"[{ci_low:.2f}, {ci_up:.2f}]")

                    except Exception:
                        row.append("")

                print(" & ".join(row) + r" \\")

            print("\\midrule")

        print("\\bottomrule")

    print("\\end{tabular}")

    print("\\caption{Extended generalization results with confidence intervals and Shapiro-Wilk normality checks.}")
    print("\\label{tab:generalization_extended}")
    print("\\end{table}")

    print("% ===== END LATEX EXTENDED TABLE =====\n")

def asymmetric_std(values, low=16, high=84):
    """
    Robust asymmetric std estimate using percentile spread.
    Returns (lower_std, upper_std) around mean-ish region.
    """
    values = np.asarray(values)

    lower = np.percentile(values, low)
    upper = np.percentile(values, high)

    center = np.mean(values)

    return center - lower, upper - center

def bootstrap_asymmetric_std(values, n_bootstrap=10000):
    values = np.asarray(values)
    means = []

    n = len(values)
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(np.std(sample, ddof=1))

    lower = np.percentile(means, 16)
    upper = np.percentile(means, 84)

    return lower, upper

def mean_std_iqm_ci(values, ci=95):
    values = np.asarray(values)
    mean = np.mean(values)
    iqm = compute_indicator(values, "iqm")
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    ci_low, ci_up = bootstrap_ci(values, 10000, ci=ci)
    return mean, std, iqm, (ci_low, ci_up)

def build_latex_table_data(all_results, env_order, solution_order, case_order=("smallest", "largest")):
    """
    all_results structure:
        all_results[env_name][solution_name][case_name] = np.array([...])

    Returns:
        table_data[solution][env][case] = (mean, std)
    """
    table_data = {}

    for sol in solution_order:
        table_data[sol] = {}
        for env in env_order:
            table_data[sol][env] = {}
            for case in case_order:
                values = (
                    all_results.get(env, {})
                    .get(sol, {})
                    .get(case, None)
                )

                if values is None or len(values) == 0:
                    table_data[sol][env][case] = (np.nan, np.nan)
                else:
                    table_data[sol][env][case] = (compute_indicator(values, "iqm"), np.std(values, ddof=1) if len(values) > 1 else 0.0)

    return table_data

import math

def truncate(x, decimals=2):
    if np.isnan(x):
        return x
    factor = 10 ** decimals
    return math.trunc(x * factor) / factor


NO_CONV_THRESHOLD = 0.05

def print_latex_table(
    all_results,
    all_train_stats,  # NEW
    env_order,
    solution_order,
    case_order=("smallest", "mean", "largest", "random_pct"),
    precision=2,
):
    case_labels = {
        "smallest": "S",
        "largest": "L",
        "mean": "M",
        "random_pct": "V",
    }

    table_data = build_latex_table_data(
        all_results=all_results,
        env_order=env_order,
        solution_order=solution_order,
        case_order=case_order,
    )

    train_data = build_latex_table_data(
        all_results=all_train_stats,
        env_order=env_order,
        solution_order=solution_order,
        case_order=case_order,
    )

    # ---- compute best per env ----
    print("TABLE DATA (mean IQM):", table_data)
    best_per_env = {}
    for env in env_order:
        best_val = -np.inf
        for sol in solution_order:
            for case in case_order:
                val, _ = table_data[sol][env][case]
                if not np.isnan(val) and val > best_val:
                    best_val = val
        best_per_env[env] = best_val

    # Header (ADD +1 column per env for GAP)
    col_spec = "l|" + "|".join(["".join(["c"] * len(case_order)) + ":c" for _ in env_order])
    print("\n% ===== LATEX TABLE =====")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    # Env header
    env_header_parts = []
    for i, env in enumerate(env_order):
        suffix = "|" if i < len(env_order) - 1 else ""
        env_header_parts.append(
            f"& \\multicolumn{{{len(case_order)+1}}}{{c{suffix}}}{{{format_env_name(env)}}}"
        )
    print(" ".join(env_header_parts) + r" \\")

    # Subheader
    subheader = ["Method"]
    for _env in env_order:
        for case in case_order:
            subheader.append(case_labels.get(case, case))
        subheader.append("\(\\Delta\\)")
    print(" & ".join(subheader) + r" \\")
    print("\\midrule")

    # Rows
    for sol in solution_order:
        display_name = SOLUTION_DISPLAY_NAMES.get(sol, sol)
        cells = [display_name]

        for env in env_order:
            gaps = []

            for case in case_order:
                test_val, _ = table_data[sol][env][case]

                if case in all_train_stats.get(env, {}).get(sol, {}):
                    train_vals = np.asarray(all_train_stats[env][sol][case])
                else:
                    train_vals = None
                # if case == "random_pct":
                #     #print("KEYS IN ", all_train_stats[env][sol].keys())
                #     train_vals_reference = np.asarray(all_train_stats[env][sol]["smallest"])
                # else:
                #     train_vals_reference = None

                #print("TRAIN VALS FOR", sol, env, case, ":", train_vals)
                if train_vals is not None and len(train_vals) > 0 and not np.isnan(test_val):
                    gap = test_val - np.nanmean(train_vals)
                    gaps.append(gap)

                if np.isnan(test_val):
                    cell_str = ""
                else:
                    val = truncate(test_val, precision)
                    cell_str = f"{val:.{precision}f}"

                    #print("CASE:", case, "ENV:", env)
                    #print("TRAIN VALS FOR", sol, env, case, ":", train_vals)
                    # TO DO READD
                    #cell_str += format_failure_rate(train_vals, NO_CONV_THRESHOLD, reference=train_vals_reference)

                EPS = 1e-6
                if not np.isnan(test_val) and test_val >= best_per_env[env] - EPS:
                    cell_str = f"\\bestblock{{{cell_str}}}"

                cells.append(cell_str)

            # GAP column
            avg_gap = np.nanmean(gaps)
            if np.isnan(avg_gap):
                gap_str = ""
            else:
                val = truncate(avg_gap, precision)
                gap_str = f"{val:+.{precision}f}"


            cells.append(gap_str)

        print(" & ".join(cells) + r" \\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("% ===== END LATEX TABLE =====\n")


def format_failure_rate(train_values, threshold=0.05, reference=None):
    """
    Returns LaTeX superscript like (k/n) where k runs failed.
    """
    if train_values is None or len(train_values) == 0:
        return ""

    train_values = np.asarray(train_values)
    # iqm of the train_values
    n = 5
    if reference is not None:
        reference_len = len(reference)
        # aggregate values in train_values to match len reference, e.g. if reference len is 5 the first 5 will be aggregated with iqm and same as others
        if len(train_values) > reference_len:
            aggregated = []
            for i in range(reference_len):
                start_idx = i * (len(train_values) // reference_len)
                end_idx = (i + 1) * (len(train_values) // reference_len) if i < reference_len - 1 else len(train_values)
                aggregated.append(compute_indicator(train_values[start_idx:end_idx], "iqm"))
            train_values = np.asarray(aggregated)
    k = np.sum(train_values < threshold)
    #print("RANDOM VALS:", reference)
    #print("TRAIN VALUES:", train_values, "->", k, "/", len(train_values))
    if k == 0:
        return ""

    return f"^{{({int(k)}/{int(n)})}}"

def select_best_case_folders(solution_root, requested_cases):
    case_folders = [
        d for d in os.listdir(solution_root)
        if d.startswith("runs_") and os.path.isdir(os.path.join(solution_root, d))
    ]

    selected = {}

    for case in requested_cases:
        case_folders_for_case = [
            d for d in case_folders
            if f"_{case}_" in d
        ]

        if not case_folders_for_case:
            print(f"No folders found for case '{case}' in '{solution_root}'")
            continue

        best_folder = None
        best_runs = -1
        best_date = datetime.min

        for folder in case_folders_for_case:
            m = re.match(
                rf"runs_(\d+)_{case}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})",
                folder
            )
            if not m:
                print(f"Folder '{folder}' does not match expected pattern")
                continue

            num_runs = int(m.group(1))
            date_str = m.group(2)

            try:
                date = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                print(f"Invalid date '{date_str}' in folder name '{folder}'")
                continue

            if (num_runs > best_runs) or (num_runs == best_runs and date > best_date):
                best_runs = num_runs
                best_date = date
                best_folder = folder

        if best_folder is not None:
            selected[case] = best_folder
            print(
                f"Selected folder for case '{case}': {best_folder} "
                f"(runs={best_runs}, date={best_date.strftime('%Y-%m-%d %H:%M:%S')})"
            )

    return selected

def main():
    parser = argparse.ArgumentParser(
        description="Plot agent comparison from RL experiment CSV summaries"
    )
    parser.add_argument(
        "-f", "--logs_folder", type=str, required=True, help="Root logs folder"
    )
    parser.add_argument(
        "-e",
        "--environment",
        nargs="+",
        required=True,
        help="One or more environment type folders, in the exact order to print in LaTeX",
    )
    parser.add_argument(
        "-i", "--indicator",
        choices=["mean", "max", "min", "iqm", "overall"],
        default="mean",
        help="Indicator computed per environment before aggregation"
    )
    parser.add_argument(
        "-s",
        "--solutions",
        nargs="+",
        default=[
            "DO_discrete",
            #"DO_discrete_semantic",
            "DO_discrete_M",
            "GO_discrete",
            "GO_discrete_M",
            "iterative",
            "projection",


            #
            #
            # "projection_approximate",
            # "projection_sample",
            # "projection_pca",
        ],
        help="List of solution types to compare",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["largest", "smallest", "mean", "random", "random_pct"],
        default=["smallest", "largest"],
        help="Environment sampling cases to include",
    )
    parser.add_argument(
        "--check_normality",
        action="store_true",
        help="Run Shapiro-Wilk normality test on per-env distributions"
    )

    parser.add_argument(
        "--print_format",
        choices=["text", "latex", "latex_extended", "both"],
        default="text",
        help="How to print the summary results"
    )
    parser.add_argument("--no_training", action="store_true", help="Whether to skip loading training metrics")
    parser.add_argument(
        "-p",
        "--plot",
        choices=["boxplot", "mean", "none"],
        default="none",
        help="Type of plot to generate"
    )
    parser.add_argument("--no_show_std", action="store_true", help="Whether to hide std in LaTeX table")
    args = parser.parse_args()

    env_registry = load_yaml(
        os.path.join(script_dir, "..", "envs", "config", "env_registry.yaml")
    )

    env_order = args.environment
    solution_order = args.solutions
    case_order = args.cases

    # all_results[env][solution][case] = np.array(...)
    all_results = {}
    all_metrics_by_env = {}
    all_train_stats = {}

    for env_name in env_order:
        all_results[env_name] = {}
        all_metrics_by_env[env_name] = {}
        all_train_stats[env_name] = {}

        for sol in solution_order:
            solution_root = os.path.join(
                script_dir, "..", "agents", args.logs_folder, env_name, sol
            )

            if not os.path.isdir(solution_root):
                print(f"Skipping missing solution folder: {solution_root}")
                continue

            selected_case_folders = select_best_case_folders(solution_root, case_order)

            if not selected_case_folders:
                print(f"No valid case folders found for env='{env_name}', solution='{sol}'")
                continue

            all_results[env_name][sol] = {}
            all_metrics_by_env[env_name][sol] = {}
            all_train_stats[env_name][sol] = {}

            for case_name, case_folder in selected_case_folders.items():
                full_case_path = os.path.join(solution_root, case_folder)

                try:
                    env_metrics, train_metrics, env_score_map = load_env_indicators_from_multiple_runs(
                        full_case_path,
                        metric_column=env_registry[env_name]["score_key"],
                        indicator=args.indicator,
                        env_id_col="environment_ID",
                        no_training=args.no_training,
                    )

                    all_results[env_name][sol][case_name] = env_metrics
                    all_metrics_by_env[env_name][sol][case_name] = env_score_map
                    if not args.no_training:
                        all_train_stats[env_name][sol][case_name] = train_metrics
                    print(
                        f"Loaded env={env_name} | sol={sol} | case={case_name} | n={len(env_metrics)}"
                    )

                except Exception as e:
                    print(f"Warning for env={env_name} / sol={sol} / case={case_name}: {e}")

    # Optional plotting: only when exactly one environment
    if args.plot != "none":
        if len(env_order) != 1:
            print("Plotting is only supported when a single environment is provided. Skipping plot.")
        else:
            env_name = env_order[0]
            metrics_dict = all_results.get(env_name, {})
            if metrics_dict:
                save_path = os.path.join(
                    script_dir,
                    "plots",
                    "agent_comparison",
                    f"{env_name}_{args.indicator}_{args.plot}.png"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                if args.plot == "boxplot":
                    plot_boxplots_grouped(metrics_dict, args.indicator, save_path)
                elif args.plot == "mean":
                    plot_mean_ci_by_group(metrics_dict, args.indicator, None, save_path)


    if args.print_format in ["text", "both"]:
        for env_name in env_order:
            print(f"\n================ ENVIRONMENT: {env_name} ================\n")
            env_metrics_dict = all_results.get(env_name, {})
            print_all_stats(env_metrics_dict, ci=95)

    if args.print_format in ["latex_extended"]:
        print("\n\n================ TEST STATS TABLE ================\n")
        print_latex_extended_table(
            all_results=all_results,
            env_order=env_order,
            solution_order=solution_order,
            ci=95
        )
        if not args.no_training:
            print("\n\n================ TRAINING STATS TABLE ================\n")
            print_latex_extended_table(
                all_results=all_train_stats,
                env_order=env_order,
                solution_order=solution_order,
                ci=95
            )

    # LaTeX printing
    if args.print_format in ["latex", "both"]:
        print("\n\n================ TEST TABLE ================\n")
        print_latex_table(
            all_results=all_results,
            all_train_stats=all_train_stats,
            env_order=env_order,
            solution_order=solution_order,
            case_order=case_order,
            precision=2,
        )
        # if not args.no_training:
        #      print_latex_table(
        #          all_results=all_train_stats,
        #          env_order=env_order,
        #          solution_order=solution_order,
        #          case_order=("train",),
        #          precision=2,
        #      )

    if args.check_normality:
        check_normality(all_results)

if __name__ == "__main__":
    main()
