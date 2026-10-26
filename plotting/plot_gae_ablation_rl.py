#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# You may redistribute it and/or modify it under the terms of the AGPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    plot_gae_ablation_rl.py
    This script plots the results of the GAE ablation study, comparing the performance of different loss term configurations across various environments. It generates visualizations to illustrate the impact of each loss term on the RL model's performance.
'''

import argparse
from datetime import datetime
import sys
import os
import numpy as np
import pandas as pd
import re
import fnmatch
from tabulate import tabulate
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
def _is_override_format(metrics_dict):
    """
    Detects:
        sol -> case -> override -> values   (NEW)
        sol -> case -> values              (OLD)
    """

    for sol_data in metrics_dict.values():
        if not isinstance(sol_data, dict):
            continue

        for case_data in sol_data.values():
            if not isinstance(case_data, dict):
                continue

            # If values are arrays → this is override format
            # (because keys are override names mapping to arrays)

            for v in case_data.values():
                if isinstance(v, np.ndarray):
                    return True

            return False

    return False

CSV_PREFIX = "average_performances_best_val_"  # assuming this is defined somewhere

def check_normality(all_results):
    # Shapiro-Wilk test for normality on each (env, solution, case, override)

    print("\n=== NORMALITY TEST (Shapiro-Wilk) ===")

    for env in all_results:
        for sol in all_results[env]:
            for case in all_results[env][sol]:

                data = all_results[env][sol][case]

                # -------------------------------------------------
                # CASE 1: override structure
                # -------------------------------------------------
                if isinstance(data, dict):

                    for override, values in data.items():

                        values = np.asarray(values)

                        if len(values) < 3:
                            continue

                        stat, p = shapiro(values)
                        normal = "YES" if p > 0.05 else "NO"

                        print(
                            f"{env} | {sol} | {case} | {override} -> "
                            f"W={stat:.3f}, p={p:.3e} => Normal: {normal}"
                        )

                # -------------------------------------------------
                # CASE 2: old structure (no override)
                # -------------------------------------------------
                else:

                    values = np.asarray(data)

                    if len(values) < 3:
                        continue

                    stat, p = shapiro(values)
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
    run_folders = [
        os.path.join(solution_root_folder, d)
        for d in os.listdir(solution_root_folder)
        if d.startswith("run_") and os.path.isdir(os.path.join(solution_root_folder, d))
    ]

    # SINGLE RUN CASE
    if not run_folders:
        return load_env_indicators_from_csv(
            solution_root_folder,
            metric_column,
            indicator,
            env_id_col,
            no_training
        )

    # structure:
    # override_name -> aggregated results across runs
    override_results = {}

    for run_folder in sorted(run_folders):
        try:
            run_outputs = load_env_indicators_from_csv(
                run_folder,
                metric_column,
                indicator,
                env_id_col,
                no_training
            )

            # run_outputs is now:
            # {override_name: (test, train, per_env)}

            for override_name, (test_metrics, train_metrics, env_map) in run_outputs.items():

                if override_name not in override_results:
                    override_results[override_name] = {
                        "test": [],
                        "train": [],
                        "env": {}
                    }

                if test_metrics is not None:
                    override_results[override_name]["test"].extend(test_metrics)

                if train_metrics is not None:
                    override_results[override_name]["train"].extend(train_metrics)

                for env_id, score in env_map.items():
                    if indicator == "concat":
                        override_results[override_name]["env"].setdefault(env_id, []).append(
                            np.asarray(score)
                        )
                    else:
                        override_results[override_name]["env"].setdefault(env_id, []).append(score)

        except Exception as e:
            print(f"Warning in run folder '{run_folder}': {e}")

    if not override_results:
        raise RuntimeError("No metrics found across runs")

    # finalize aggregation per override
    final_results = {}

    for override_name, data in override_results.items():

        test_metrics = np.asarray(data["test"])
        train_metrics = np.asarray(data["train"]) if data["train"] else None

        if indicator == "concat":
            merged_env_map = {
                env_id: np.concatenate(scores)
                for env_id, scores in data["env"].items()
            }
            print("MERGED ENV MAP (concat):", {k: v for k, v in merged_env_map.items()})
        else:
            merged_env_map = {
                env_id: float(np.mean(scores))
                for env_id, scores in data["env"].items()
            }

        final_results[override_name] = (
            test_metrics,
            train_metrics,
            merged_env_map,
        )

    return final_results

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
    Supports:
      - OLD: sol -> array
      - NEW: sol -> override -> array
    """

    override_mode = _is_override_format(metrics_dict)

    plt.figure(figsize=(10, 6))

    if not override_mode:
        # =========================
        # OLD BEHAVIOR (UNCHANGED)
        # =========================
        solution_names = list(metrics_dict.keys())

        means, iqms, maxs = [], [], []
        lower_errors, upper_errors = [], []

        for sol in solution_names:
            data = np.asarray(metrics_dict[sol])

            mean = np.mean(data)
            means.append(mean)

            ci25, ci75 = np.percentile(data, [25, 75])
            iqm = data[(data >= ci25) & (data <= ci75)].mean()
            iqms.append(iqm)

            maxs.append(np.max(data))

            ci_lower, ci_upper = bootstrap_ci(data, 10000, 95)
            lower_errors.append(mean - ci_lower)
            upper_errors.append(ci_upper - mean)

        x = np.arange(len(solution_names))
        yerr = np.array([lower_errors, upper_errors])

        plt.bar(x, means, yerr=yerr, capsize=5,
                color='black', edgecolor='blue', alpha=0.7,
                label='Mean ± 95% CI')

        plt.scatter(x, iqms, color='black', label='IQM')
        plt.scatter(x, maxs, color='red', label='Max')

        plt.xticks(x, solution_names)
        plt.ylabel("Agent Score")
        plt.title("Mean, IQM, and 95% Bootstrap CI Across Solutions")
        plt.axhline(0, color='gray', linewidth=0.8)
        plt.legend()
        plt.tight_layout()

    else:
        # =========================
        # NEW OVERRIDE BEHAVIOR
        # =========================

        solutions = list(metrics_dict.keys())
        overrides = sorted({
            o
            for sol in solutions
            for o in metrics_dict[sol].keys()
        })

        x = np.arange(len(solutions))
        bar_width = 0.8 / max(len(overrides), 1)

        colors = plt.cm.tab10.colors

        for i, override in enumerate(overrides):
            means = []
            iqms = []
            maxs = []

            lower_errors = []
            upper_errors = []

            for sol in solutions:
                if override not in metrics_dict[sol]:
                    means.append(np.nan)
                    iqms.append(np.nan)
                    maxs.append(np.nan)
                    lower_errors.append(0)
                    upper_errors.append(0)
                    continue

                data = np.asarray(metrics_dict[sol][override])

                mean = np.mean(data)
                means.append(mean)

                ci25, ci75 = np.percentile(data, [25, 75])
                iqm = data[(data >= ci25) & (data <= ci75)].mean()
                iqms.append(iqm)

                maxs.append(np.max(data))

                ci_lower, ci_upper = bootstrap_ci(data, 10000, 95)
                lower_errors.append(mean - ci_lower)
                upper_errors.append(ci_upper - mean)

            offsets = x + (i - len(overrides)/2) * bar_width + bar_width/2

            plt.bar(
                offsets,
                means,
                width=bar_width,
                yerr=[lower_errors, upper_errors],
                capsize=4,
                alpha=0.8,
                label=f"{override}"
            )

            plt.scatter(offsets, iqms, s=20, color='black')
            plt.scatter(offsets, maxs, s=20, color='red')

        plt.xticks(x, solutions, rotation=0)
        plt.ylabel("Agent Score")
        plt.title("Mean / IQM / Max Across Solutions (per override)")
        plt.axhline(0, color='gray', linewidth=0.8)
        plt.legend(title="Override")
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def plot_boxplots_grouped(metrics_dict, indicator, save_path=None):

    override_mode = _is_override_format(metrics_dict)

    solutions = sorted(metrics_dict.keys())

    x = np.arange(len(solutions))
    plt.figure(figsize=(12, 6))

    # =========================================================
    # CASE A: DEFAULT MODE (UNCHANGED)
    # =========================================================
    if not override_mode:

        cases = sorted({
            case
            for sol in metrics_dict
            for case in metrics_dict[sol].keys()
        })

        color_map = {
            "largest": "#4E79A7",
            "smallest": "#E15759",
            "mean": "#59A14F",
            "random": "#F28E2B",
            "random_pct": "#F28E2B",
        }

        width = 0.15

        for case_idx, case in enumerate(cases):
            positions = x + (case_idx - (len(cases) - 1) / 2) * width

            data = [
                metrics_dict[sol].get(case, [])
                for sol in solutions
            ]

            bp = plt.boxplot(
                data,
                positions=positions,
                widths=width,
                patch_artist=True,
                showfliers=True,
            )

            for box in bp["boxes"]:
                box.set(facecolor=color_map.get(case, "gray"), alpha=0.7)

        plt.xticks(x, solutions)
        plt.title("Comparison across cases")

        handles = [
            plt.Line2D([0], [0], color=color_map.get(c, "gray"), lw=6)
            for c in cases
        ]
        plt.legend(handles, cases, title="Case")

    # =========================================================
    # CASE B: OVERRIDE MODE (FIXED STRUCTURE)
    # =========================================================
    else:
        overrides = sorted({
            o
            for sol in metrics_dict
            for case in metrics_dict[sol]
            for o in metrics_dict[sol][case].keys()
        })

        cases = sorted({
            case
            for sol in metrics_dict
            for case in metrics_dict[sol].keys()
        })

        solutions = sorted(metrics_dict.keys())
        x = np.arange(len(solutions))

        # stable colors per override
        cmap = plt.cm.get_cmap("tab10")
        override_color = {o: cmap(i % 10) for i, o in enumerate(overrides)}

        n_overrides = len(overrides)
        n_cases = len(cases)

        # spacing between case blocks
        case_spacing = 1.0

        # width per box
        width = 0.8 / (n_cases * n_overrides)

        for case_idx, case in enumerate(cases):

            # center each case block
            case_center = (case_idx - (n_cases - 1) / 2) * case_spacing

            for o_idx, override in enumerate(overrides):

                # offset inside case
                override_offset = (o_idx - (n_overrides - 1) / 2) * width * 1.3

                positions = x + case_center + override_offset

                data = []

                for sol in solutions:
                    sol_cases = metrics_dict[sol].get(case, {})

                    if not isinstance(sol_cases, dict):
                        data.append([])
                        continue

                    data.append(sol_cases.get(override, []))

                bp = plt.boxplot(
                    data,
                    positions=positions,
                    widths=width,
                    patch_artist=True,
                    showfliers=True,
                )

                for box in bp["boxes"]:
                    box.set_facecolor(override_color[override])
                    box.set_alpha(0.7)

                for median in bp["medians"]:
                    median.set_color("black")

        # axis + legend
        plt.xticks(x, solutions)
        plt.title("Comparison across cases × overrides")

        handles = [
            plt.Line2D([0], [0], color=override_color[o], lw=6, label=o)
            for o in overrides
        ]
        plt.legend(handles=handles, title="Override")

    plt.ylabel(indicator.upper())
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()



def _load_single_experiment(folder, metric_column, indicator,
                            env_id_col="environment_ID",
                            no_training=False):
    """Original implementation for a single experiment folder."""

    train_solution_folder, test_solution_folder = find_latest_env_with_episodes(folder)

    results = {}
    per_env_results = {}

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

            if indicator == "concat":
                # Keep every episode value
                env_values.extend(values)
                env_score_map[env_id] = values
            else:
                env_score = compute_indicator(values, indicator)
                env_values.append(env_score)
                env_score_map[env_id] = env_score

        env_values = np.asarray(env_values)

        results[split] = env_values
        per_env_results[split] = env_score_map

        print(
            f"[{os.path.basename(folder)}][{split}] "
            f"envs={len(env_values)}, episodes_used={available_episodes}"
        )

    if "test" not in results:
        raise RuntimeError(f"No test metrics found in {folder}")

    return (
        results["test"],
        results.get("train", None),
        per_env_results["test"],
    )


def load_env_indicators_from_csv(folder, metric_column, indicator,
                                 env_id_col="environment_ID",
                                 no_training=False):
    if indicator == "overall":
        raise ValueError(
            "'overall' is not compatible with per-env best-solution stats because it does not produce one scalar per env_id."
        )

    # ------------------------------------------------------------------
    # Case 1: folder itself is an experiment (current behaviour)
    # ------------------------------------------------------------------
    if os.path.isdir(os.path.join(folder, "checkpoints")):
        return {
            "default": _load_single_experiment(
                folder,
                metric_column,
                indicator,
                env_id_col,
                no_training,
            )
        }

    # ------------------------------------------------------------------
    # Case 2: folder contains override_* experiments
    # ------------------------------------------------------------------
    override_dirs = sorted(
        d for d in os.listdir(folder)
        if d.startswith("override_")
        and os.path.isdir(os.path.join(folder, d))
    )

    if not override_dirs:
        raise RuntimeError(
            f"Neither a checkpoints directory nor override_* directories were found in {folder}"
        )

    all_results = {}

    for override in override_dirs:
        exp_folder = os.path.join(folder, override)

        if not os.path.isdir(os.path.join(exp_folder, "checkpoints")):
            print(f"Skipping {override}: no checkpoints directory.")
            continue

        all_results[override] = _load_single_experiment(
            exp_folder,
            metric_column,
            indicator,
            env_id_col,
            no_training,
        )

    return all_results

def plot_mean_ci_by_group(
    metrics_dict,
    indicator,
    train_stats_dict=None,
    save_path=None,
    ci=95,
):
    override_mode = _is_override_format(metrics_dict)

    methods = list(metrics_dict.keys())
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    # ---------------------------------------------------------
    # Collect groups depending on structure
    # ---------------------------------------------------------
    if not override_mode:
        groups = sorted({
            g
            for m in methods
            for g in metrics_dict[m].keys()
        })
    else:
        groups = sorted({
            g
            for m in methods
            for o in metrics_dict[m].values()
            for g in o.keys()
        })

    bar_width = 0.8 / len(groups)

    # ---------------------------------------------------------
    # MAIN PLOT
    # ---------------------------------------------------------
    for i, group in enumerate(groups):

        means = []
        cis = []

        for method in methods:

            if not override_mode:
                values = metrics_dict[method].get(group, None)

            else:
                # collect across overrides
                values_list = []
                for override in metrics_dict[method]:
                    if group in metrics_dict[method][override]:
                        values_list.append(
                            np.asarray(metrics_dict[method][override][group])
                        )

                if len(values_list) == 0:
                    values = None
                else:
                    values = np.concatenate(values_list)

            if values is None or len(values) == 0:
                means.append(np.nan)
                cis.append(0)
                continue

            values = np.asarray(values)

            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)

            ci_val = 1.96 * std / np.sqrt(n)

            means.append(mean)
            cis.append(ci_val)

        offsets = x + (i - len(groups) / 2) * bar_width + bar_width / 2

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

    # ---------------------------------------------------------
    # TRAINING OVERLAY (unchanged)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # FORMATTING
    # ---------------------------------------------------------
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


def _get_overrides(all_results):
    first_env = next(iter(all_results))
    first_sol = next(iter(all_results[first_env]))
    first_case = next(iter(all_results[first_env][first_sol]))

    data = all_results[first_env][first_sol][first_case]

    if isinstance(data, dict):
        return sorted(data.keys())  # override names

    return None  # default-only format

def print_all_stats(metrics_dict, ci=95):
    print("\n=== STATS OVER ALL ENV SCORES ===")

    if _is_override_format(metrics_dict):

        for sol in metrics_dict:
            for override in metrics_dict[sol]:
                data = metrics_dict[sol][override]

                mean, std, ci_lower, ci_upper = compute_mean_std_bci(data, ci=ci)

                print(
                    f"{sol} | {override} | "
                    f"Mean: {mean:.3f} | STD: {std:.3f} | "
                    f"{ci}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
                )

    else:
        for sol in metrics_dict:
            data = metrics_dict[sol]

            mean, std, ci_lower, ci_upper = compute_mean_std_bci(data, ci=ci)

            print(
                f"{sol} | Mean: {mean:.3f} | STD: {std:.3f} | "
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

    overrides = _get_overrides(all_results)

    # =========================
    # CASE 1: NO OVERRIDES
    # =========================
    if overrides is None:
        overrides = ["default"]

    for override in overrides:

        print(f"\n% ===== LATEX EXTENDED TABLE ({override}) =====")

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

                norm_cache = {}

                for sol in solution_order:
                    try:
                        vals = all_results[env][sol][regime].get(override)
                        vals = np.asarray(vals)

                        if len(vals) == 0:
                            norm_cache[sol] = ""
                            continue

                        _, p, norm = is_normal(vals)
                        norm_cache[sol] = norm

                    except Exception:
                        norm_cache[sol] = ""

                metrics = ["Mean", "Std", "IQM", "BCI", "Normality"]

                for i, ind in enumerate(metrics):

                    row = []

                    if i == 0:
                        row.append(f"\\multirow{{5}}{{*}}{{{regime_letters[regime]}}}")
                    else:
                        row.append("")

                    row.append(r"\textit{" + ("Normal" if ind == "Normality" else ind) + "}")

                    for sol in solution_order:

                        if ind == "Normality":
                            row.append(norm_cache.get(sol, ""))
                            continue

                        try:
                            vals = all_results[env][sol][regime].get(override)
                            vals = np.asarray(vals)

                            if len(vals) == 0:
                                row.append("")
                                continue

                            mean, std, iqm, (ci_low, ci_up) = mean_std_iqm_ci(vals, ci=ci)

                            if ind == "Mean":
                                row.append(f"\\textbf{{{mean:.2f}}}")

                            elif ind == "Std":
                                s_low, s_high = asymmetric_std(vals)
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

        if override != "default":
            print(f"\\caption{{Extended results (override={override})}}")
        else:
            print("\\caption{Extended results}")
        print("\\label{tab:generalization_extended_" + override + "}")

        print("\\end{table}")
        print("% ===== END TABLE =====\n")

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
def build_latex_table_data(
    all_results,
    env_order,
    solution_order,
    case_order=("smallest", "largest")
):
    """
    Returns:
        table_data[solution][env][case][override] = (mean, std)
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

                # -------------------------
                # NO DATA
                # -------------------------
                if values is None or len(values) == 0:
                    table_data[sol][env][case] = {
                        "default": (np.nan, np.nan)
                    }
                    continue

                # -------------------------
                # OVERRIDE FORMAT
                # -------------------------
                if isinstance(values, dict):

                    table_data[sol][env][case] = {}

                    for override, v in values.items():

                        v = np.asarray(v)

                        if len(v) == 0:
                            continue

                        table_data[sol][env][case][override] = (
                            compute_indicator(v, "iqm"),
                            np.std(v, ddof=1) if len(v) > 1 else 0.0
                        )

                # -------------------------
                # OLD FORMAT
                # -------------------------
                else:
                    v = np.asarray(values)

                    if len(v) == 0:
                        table_data[sol][env][case] = {
                            "default": (np.nan, np.nan)
                        }
                    else:
                        table_data[sol][env][case] = {
                            "default": (
                                compute_indicator(v, "iqm"),
                                np.std(v, ddof=1) if len(v) > 1 else 0.0
                            )
                        }

    return table_data

import math

def truncate(x, decimals=2):
    if np.isnan(x):
        return x
    factor = 10 ** decimals
    return math.trunc(x * factor) / factor


NO_CONV_THRESHOLD = 0.05

def compute_ablation_deltas(all_results):
    """
    Returns

    delta_results[env][solution][case][ablation] = values - baseline
    """

    delta_results = {}

    for env in all_results:

        delta_results[env] = {}

        for sol in all_results[env]:

            delta_results[env][sol] = {}

            for case in all_results[env][sol]:

                if "None" not in all_results[env][sol][case]:
                    raise RuntimeError(
                        f"Missing baseline ablation_None for "
                        f"{env}/{sol}/{case}"
                    )

                baseline = np.asarray(
                    all_results[env][sol][case]["None"]
                )

                delta_results[env][sol][case] = {}

                for ablation, values in all_results[env][sol][case].items():

                    if ablation == "None":
                        continue

                    values = np.asarray(values)

                    delta_results[env][sol][case][ablation] = (
                        values - baseline
                    )

    return delta_results
def print_ablation_stats(metrics):

    rows = []

    order = [
        "None",
        "node_features",
        "edge_features",
        "adj_weight",
    ]

    labels = {
        "None": "Full model",
        "node_features": "- Node features",
        "edge_features": "- Edge features",
        "adj_weight": "- Adjacency weights",
    }

    for ablation in order:

        if ablation not in metrics:
            continue

        values = np.asarray(metrics[ablation])

        mean = np.mean(values)
        iqm = compute_indicator(values, "iqm")
        std = np.std(values, ddof=1)

        low, high = bootstrap_ci(values, 10000)

        rows.append([
            labels[ablation],
            f"{mean:.3f}",
            f"{iqm:.3f}",
            f"{std:.3f}",
            f"[{low:.3f}, {high:.3f}]",
        ])

    print(tabulate(
        rows,
        headers=[
            "Model",
            "Mean",
            "IQM",
            "Std",
            "95% BCI",
        ],
        tablefmt="github",
    ))


def plot_ablation_boxplots(metrics):

    names = list(metrics.keys())

    data = [metrics[k] for k in names]

    plt.figure(figsize=(7,5))

    plt.boxplot(
        data,
        labels=names,
        showmeans=True,
        patch_artist=True,
    )

    plt.axhline(0, color="red", linestyle="--")

    plt.ylabel("Score difference from full model")

    plt.tight_layout()

    plt.show()

def print_latex_table(
    all_results,
    all_train_stats,
    env_order,
    solution_order,
    no_training,
    case_order=("smallest", "mean", "largest", "random_pct"),
    precision=2,

):


    show_delta = not no_training


    case_labels = {
        "smallest": "S",
        "largest": "L",
        "mean": "M",
        "random_pct": "V",
    }

    overrides = _get_overrides(all_results)
    if overrides is None:
        overrides = ["default"]

    for override in overrides:

        if override != "default":
            print(f"\n% ===== LATEX TABLE (override={override}) =====")
        else:
            print(f"\n% ===== LATEX TABLE =====")

        table_data = build_latex_table_data(
            all_results, env_order, solution_order, case_order
        )
        train_data = build_latex_table_data(
            all_train_stats, env_order, solution_order, case_order
        )

        # NOTE: table_data still needs override-aware access
        # so we patch access inline below instead of rewriting everything

        best_per_env = {}

        for env in env_order:
            best_val = -np.inf

            for sol in solution_order:
                for case in case_order:

                    vals = all_results.get(env, {}).get(sol, {}).get(case, {})
                    if isinstance(vals, dict):
                        vals = vals.get(override, None)

                    if vals is None:
                        continue

                    vals = np.asarray(vals)
                    if len(vals) == 0:
                        continue

                    val = compute_indicator(vals, "iqm")

                    if not np.isnan(val) and val > best_val:
                        best_val = val

            best_per_env[env] = best_val

        delta_col = ":c" if show_delta else ""

        col_spec = "l|" + "|".join(
            ["".join(["c"] * len(case_order)) + delta_col for _ in env_order]
        )

        print("\\begin{tabular}{" + col_spec + "}")
        print("\\toprule")

        env_header_parts = []
        for i, env in enumerate(env_order):
            suffix = "|" if i < len(env_order) - 1 else ""
            n_cols = len(case_order) + (1 if show_delta else 0)

            env_header_parts.append(
                f"& \\multicolumn{{{n_cols}}}{{c{suffix}}}{{{format_env_name(env)}}}"
            )
        print(" ".join(env_header_parts) + r" \\")

        subheader = ["Method"]

        for _ in env_order:
            for case in case_order:
                subheader.append(case_labels.get(case, case))

            if show_delta:
                subheader.append("\\(\\Delta\\)")
        print(" & ".join(subheader) + r" \\")

        print("\\midrule")

        for sol in solution_order:

            display_name = SOLUTION_DISPLAY_NAMES.get(sol, sol)
            cells = [display_name]

            for env in env_order:
                gaps = []

                for case in case_order:

                    vals = all_results.get(env, {}).get(sol, {}).get(case, {})
                    if isinstance(vals, dict):
                        vals = vals.get(override, None)

                    if vals is None:
                        cells.append("")
                        continue

                    vals = np.asarray(vals)
                    test_val = compute_indicator(vals, "iqm")

                    train_vals = all_train_stats.get(env, {}).get(sol, {}).get(case, {})
                    if isinstance(train_vals, dict):
                        train_vals = train_vals.get(override, None)

                    if train_vals is not None:
                        train_vals = np.asarray(train_vals)
                    else:
                        train_vals = None

                    if show_delta and train_vals is not None and len(train_vals) > 0:
                        gaps.append(test_val - np.nanmean(train_vals))

                    if np.isnan(test_val):
                        cell_str = ""
                    else:
                        val = truncate(test_val, precision)
                        cell_str = f"{val:.{precision}f}"

                    EPS = 1e-6
                    if test_val >= best_per_env[env] - EPS:
                        cell_str = f"\\bestblock{{{cell_str}}}"

                    cells.append(cell_str)

                # -----------------------------------------------------
                # Δ COLUMN (ONLY IF TRAINING EXISTS)
                # -----------------------------------------------------
                if show_delta:
                    avg_gap = np.nanmean(gaps)
                    gap_str = "" if np.isnan(avg_gap) else f"{avg_gap:+.{precision}f}"
                    cells.append(gap_str)

            print(" & ".join(cells) + r" \\")

        print("\\bottomrule")
        print("\\end{tabular}")

        if override != "default":
            print(f"\\caption{{Comparison table (override={override})}}")
        else:
            print("\\caption{Comparison table}")
        print("\\label{tab:comparison_" + override + "}")

def build_table(
    all_results,
    all_train_stats,
    env_order,
    solution_order,
    no_training,
    case_order=("smallest", "mean", "largest", "random_pct"),
    precision=2,
):
    show_delta = not no_training

    case_labels = {
        "smallest": "S",
        "largest": "L",
        "mean": "M",
        "random_pct": "V",
    }

    overrides = _get_overrides(all_results)
    if overrides is None:
        overrides = ["default"]

    tables = {}

    for override in overrides:

        # --------------------------------------------------
        # Best IQM per environment
        # --------------------------------------------------
        best_per_env = {}

        for env in env_order:
            best_val = -np.inf

            for sol in solution_order:
                for case in case_order:

                    vals = all_results.get(env, {}).get(sol, {}).get(case, {})
                    if isinstance(vals, dict):
                        vals = vals.get(override)

                    if vals is None:
                        continue

                    vals = np.asarray(vals)
                    if len(vals) == 0:
                        continue

                    val = compute_indicator(vals, "iqm")

                    if not np.isnan(val):
                        best_val = max(best_val, val)

            best_per_env[env] = best_val

        # --------------------------------------------------
        # Headers
        # --------------------------------------------------

        headers = ["Method"]

        for env in env_order:
            prefix = format_env_name(env)

            for case in case_order:
                headers.append(f"{prefix}-{case_labels[case]}")

            if show_delta:
                headers.append(f"{prefix}-Δ")

        # --------------------------------------------------
        # Rows
        # --------------------------------------------------

        rows = []

        for sol in solution_order:

            display_name = SOLUTION_DISPLAY_NAMES.get(sol, sol)
            row = [display_name]

            for env in env_order:

                gaps = []

                for case in case_order:

                    vals = all_results.get(env, {}).get(sol, {}).get(case, {})

                    if isinstance(vals, dict):
                        vals = vals.get(override)

                    if vals is None:
                        row.append("")
                        continue

                    vals = np.asarray(vals)

                    if len(vals) == 0:
                        row.append("")
                        continue

                    test_val = compute_indicator(vals, "iqm")

                    train_vals = (
                        all_train_stats
                        .get(env, {})
                        .get(sol, {})
                        .get(case, {})
                    )

                    if isinstance(train_vals, dict):
                        train_vals = train_vals.get(override)

                    if train_vals is not None:
                        train_vals = np.asarray(train_vals)

                    if (
                        show_delta
                        and train_vals is not None
                        and len(train_vals) > 0
                    ):
                        gaps.append(test_val - np.nanmean(train_vals))

                    if np.isnan(test_val):
                        cell = ""
                    else:
                        cell = f"{truncate(test_val, precision):.{precision}f}"

                        if test_val >= best_per_env[env] - 1e-6:
                            cell = f"*{cell}*"

                    row.append(cell)

                if show_delta:
                    avg_gap = np.nanmean(gaps)
                    row.append("" if np.isnan(avg_gap) else f"{avg_gap:+.{precision}f}")

            rows.append(row)

        tables[override] = (headers, rows)

    return tables


def build_extended_table(
    all_results,
    all_train_stats,
    env_order,
    solution_order,
    no_training,
    case_order=("smallest", "mean", "largest", "random_pct"),
    precision=2,
):
    show_delta = not no_training

    case_labels = {
        "smallest": "S",
        "largest": "L",
        "mean": "M",
        "random_pct": "V",
    }

    overrides = _get_overrides(all_results)
    if overrides is None:
        overrides = ["default"]

    tables = {}

    for override in overrides:

        # --------------------------------------------------
        # Best IQM per environment
        # --------------------------------------------------
        best_per_env = {}

        for env in env_order:
            best_val = -np.inf

            for sol in solution_order:
                for case in case_order:

                    vals = all_results.get(env, {}).get(sol, {}).get(case, {})
                    if isinstance(vals, dict):
                        vals = vals.get(override)

                    if vals is None:
                        continue

                    vals = np.asarray(vals)
                    if len(vals) == 0:
                        continue

                    val = compute_indicator(vals, "iqm")

                    if not np.isnan(val):
                        best_val = max(best_val, val)

            best_per_env[env] = best_val

        # --------------------------------------------------
        # Headers
        # --------------------------------------------------

        headers = ["Method"]

        for env in env_order:
            prefix = format_env_name(env)

            for case in case_order:
                headers.append(f"{prefix}-{case_labels[case]}")

            if show_delta:
                headers.append(f"{prefix}-Δ")

        # --------------------------------------------------
        # Rows
        # --------------------------------------------------

        rows = []

        for sol in solution_order:

            display_name = SOLUTION_DISPLAY_NAMES.get(sol, sol)
            row = [display_name]

            for env in env_order:

                gaps = []

                for case in case_order:

                    vals = all_results.get(env, {}).get(sol, {}).get(case, {})

                    if isinstance(vals, dict):
                        vals = vals.get(override)

                    if vals is None:
                        row.append("")
                        continue

                    vals = np.asarray(vals)

                    if len(vals) == 0:
                        row.append("")
                        continue

                    test_val = compute_indicator(vals, "iqm")

                    train_vals = (
                        all_train_stats
                        .get(env, {})
                        .get(sol, {})
                        .get(case, {})
                    )

                    if isinstance(train_vals, dict):
                        train_vals = train_vals.get(override)

                    if train_vals is not None:
                        train_vals = np.asarray(train_vals)

                    if (
                        show_delta
                        and train_vals is not None
                        and len(train_vals) > 0
                    ):
                        gaps.append(test_val - np.nanmean(train_vals))

                    if len(vals) == 0:
                        row.append("")
                        continue

                    cell = summarize_distribution(vals, precision)

                    # still highlight the best IQM
                    iqm = compute_indicator(vals, "iqm")
                    if iqm >= best_per_env[env] - 1e-6:
                        cell = "*\n" + cell

                    row.append(cell)

                if show_delta:
                    avg_gap = np.nanmean(gaps)
                    row.append("" if np.isnan(avg_gap) else f"{avg_gap:+.{precision}f}")

            rows.append(row)

        tables[override] = (headers, rows)

    return tables

def summarize_distribution(values, precision=2):
    values = np.asarray(values)

    if len(values) == 0:
        return ""

    iqm = compute_indicator(values, "iqm")
    mean = np.mean(values)
    p25 = np.percentile(values, 25)
    minimum = np.min(values)

    # mean of worst 10%
    n = max(1, int(np.ceil(0.10 * len(values))))
    worst10 = np.mean(np.sort(values)[:n])

    return "\n".join([
        f"IQM : {iqm:.{precision}f}",
        f"μ   : {mean:.{precision}f}",
        f"P25 : {p25:.{precision}f}",
        f"W10 : {worst10:.{precision}f}",
        f"Min : {minimum:.{precision}f}",
    ])
def select_best_case_folders(solution_root, requested_cases):
    print("SOLUTION ROOT:", solution_root)
    print("REQUESTED CASES:", requested_cases)
    case_folders = [
        d for d in os.listdir(solution_root)
        if "runs_" in d and os.path.isdir(os.path.join(solution_root, d))
    ]
    print("CASE FOLDERS FOUND:", case_folders)

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

            num_runs = m.group(1)
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
def find_case_folders(solution_root):
    cases = {}

    pattern = re.compile(
        r"gae_\d+_runs_\d+_(smallest|largest|mean|random_pct)_"
    )

    for folder in os.listdir(solution_root):

        full = os.path.join(solution_root, folder)

        if not os.path.isdir(full):
            continue

        m = pattern.match(folder)

        if m is None:
            continue

        case = m.group(1)
        cases[case] = full

    return cases
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
        choices=["mean", "max", "min", "iqm", "overall", "concat"],
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
        choices=["text", "latex", "latex_extended", "both", "ascii", "ascii_extended"],
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

    # all_results[env][solution][case] = np.array(...)
    all_results = {}

    for env_name in env_order:
        all_results[env_name] = {}

        for sol in solution_order:
            solution_root = os.path.join(
                script_dir, "..", "agents", args.logs_folder, env_name, sol
            )
            if not os.path.isdir(solution_root):
                print(f"Skipping missing solution folder: {solution_root}")
                continue
            print("Processing solution folder:", solution_root)
            ablation_pattern = re.compile(
                r"ablation_(.+?)_runs_\d+_(smallest|largest|mean|random_pct)_"
            )

            all_results[env_name][sol] = {}

            for folder in os.listdir(solution_root):

                full = os.path.join(solution_root, folder)

                if not os.path.isdir(full):
                    continue

                m = ablation_pattern.match(folder)
                if m is None:
                    continue

                ablation = m.group(1)
                case_name = m.group(2)

                if case_name not in args.cases:
                    continue

                all_results[env_name][sol].setdefault(case_name, {})

                outputs = load_env_indicators_from_multiple_runs(
                    full,
                    metric_column=env_registry[env_name]["score_key"],
                    indicator=args.indicator,
                    env_id_col="environment_ID",
                    no_training=True,
                )

                test_metrics, train_metrics, env_map = outputs["default"]

                all_results[env_name][sol][case_name][ablation] = test_metrics
    delta_results = compute_ablation_deltas(all_results)

    # Optional plotting: only when exactly one environment
    if args.print_format in ["text", "both"]:
        for env_name in env_order:
            print(f"\n================ ENVIRONMENT: {env_name} ================\n")
            metrics = all_results[env_name]
            for sol in metrics:
                for case in metrics[sol]:
                    print(f"Solution: {sol} | Case: {case} | ")
                    print_ablation_stats(
                        all_results[env_name][sol][case]
                    )
    if args.print_format in ["boxplot", "both"]:
        for env_name in env_order:
            metrics = delta_results[env_name]
            for sol in metrics:
                for case in metrics[sol]:
                    print(f"Solution: {sol} | Case: {case} | ")
                    plot_ablation_boxplots(
                        delta_results[env_name][sol][case]
                    )


if __name__ == "__main__":
    main()
