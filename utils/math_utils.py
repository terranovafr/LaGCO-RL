#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    math_utils.py
    Utility functions for mathematical operations, statistics, and loss weighting in machine learning contexts.
'''

import numpy as np
import random
from typing import Callable
import os
import torch
from scipy import stats

def sample_range_dict(rng_dict, key):
    return random.randint(*rng_dict[key])

def min_max_normalize(x, min_x, max_x, eps=1e-8):
    return (x - min_x) / (max_x - min_x + eps)

def iqm(series):
    # Interquartile mean: mean between 25th and 75th percentiles
    try:
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        return series[(series >= q1) & (series <= q3)].mean()
    except:
        return 0

def ci(values, confidence=0.95):
    arr = np.asarray(values)
    low_p = (1 - confidence) / 2 * 100
    high_p = (1 + confidence) / 2 * 100
    return float(np.percentile(arr, low_p)), float(np.percentile(arr, high_p))

def bootstrap_ci(series, num_samples=1000, ci=95):
    # Compute bootstrapped confidence interval
    if len(series) == 0:
        return [None, None]
    boot_samples = np.random.choice(series, (num_samples, len(series)), replace=True)
    means = boot_samples.mean(axis=1)
    lower = np.percentile(means, (100 - ci)/2)
    upper = np.percentile(means, 100 - (100 - ci)/2)
    return [lower, upper]

# Alignment across runs
def interpolate_runs(run_series):
    all_steps = sorted(set(np.concatenate([s for s, _ in run_series])))
    all_steps = np.array(all_steps, dtype=float)

    aligned = []
    for steps, values in run_series:
        y = np.interp(all_steps, steps, values)
        aligned.append(y)

    return all_steps, np.array(aligned)

# Compute a specified indicator (mean, max, min, or interquartile mean) from a list of values. The function takes a list of values and an indicator string as input and returns the computed indicator value. If the indicator is "mean", it returns the average of the values. If the indicator is "max", it returns the maximum value. If the indicator is "min", it returns the minimum value. If the indicator is "iqm", it computes the interquartile mean by first calculating the 25th and 75th percentiles and then averaging the values that fall within this range. If an unknown indicator is provided, it raises a ValueError.
def compute_indicator(values, indicator):
    values = np.asarray(values)

    if indicator == "mean":
        return np.mean(values)
    elif indicator == "max":
        return np.max(values)
    elif indicator == "min":
        return np.min(values)
    elif indicator == "iqm":
        q25, q75 = np.percentile(values, [25, 75])
        middle = values[(values >= q25) & (values <= q75)]
        return np.mean(middle)
    else:
        raise ValueError(f"Unknown indicator: {indicator}")

def compute_mean_std_bci(data, ci=95):
    data = np.asarray(data)
    if len(data) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean = np.mean(data)
    std = np.std(data, ddof=1) if len(data) > 1 else 0.0
    ci_lower, ci_upper = bootstrap_ci(data, 10000, ci=ci)
    return mean, std, ci_lower, ci_upper

def compute_band(values, spread="ci", confidence=0.95, n_boot=1000):
    mean = np.mean(values, axis=0)
    n = values.shape[0]
    if n > 1:
        std = np.std(values, axis=0, ddof=1)
    else:
        std = np.zeros(values.shape[1])
    if spread == "std":
        low = mean - std
        high = mean + std

    elif spread == "ci":
        if n > 1:
            t_val = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
            err = t_val * std / np.sqrt(n)
        else:
            err = np.zeros_like(std)
        low = mean - err
        high = mean + err
    elif spread == "bci":
        if n > 1:
            low, high = bootstrap_ci(values, n_boot, confidence)
        else:
            low = high = mean
    else:
        raise ValueError("Unknown spread type")

    return mean, low, high

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

# Linear schedule for decay of a metric (e.g. learning rate) from an initial value to a final value
def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value + (final_value - initial_value) * (1.0 - progress_remaining)
    return func

def set_seeds(seed):
    # Set seeds for reproducibility across random, numpy, and torch (both CPU and CUDA). Also set environment variables to ensure deterministic behavior in PyTorch.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Calculate the area under the curve using the trapezoidal rule
def calculate_auc(times, values):
    auc = np.trapz(values, times)
    return auc



class HybridMagnitudeDWAWeighting:
    """
    Hybrid loss weighting:
    - Normalize losses by magnitude so that large losses do not dominate
    - Use Dynamic Weight Averaging (DWA) on normalized losses to favor faster-changing ones
    - Zero-valued losses are ignored
    - Can control contribution of magnitude vs DWA using alpha_mag and alpha_dwa
    """
    def __init__(self, loss_names, T=2.0, eps=1e-8, alpha_mag=0.99, alpha_dwa=0.01):
        self.loss_names = loss_names
        self.T = T
        self.eps = eps
        self.alpha_mag = alpha_mag
        self.alpha_dwa = alpha_dwa
        # store last two normalized losses for each loss
        self.loss_history = {name: [1.0, 1.0] for name in loss_names}

    def update_losses(self, current_losses_dict):
        # Update current losses and normalize by max magnitude across losses.
        non_zero_losses = {name: max(float(current_losses_dict.get(name, 0.0)), self.eps)
                           for name in self.loss_names if float(current_losses_dict.get(name, 0.0)) > 0.0}

        if not non_zero_losses:
            for name in self.loss_names:
                self.loss_history[name].append(self.eps)
                if len(self.loss_history[name]) > 2:
                    self.loss_history[name].pop(0)
            return

        max_val = max(non_zero_losses.values())

        for name in self.loss_names:
            val = non_zero_losses.get(name, 0.0)
            norm_val = val / max_val if val > 0 else 0.0
            self.loss_history[name].append(norm_val)
            if len(self.loss_history[name]) > 2:
                self.loss_history[name].pop(0)

    def get_weights(self):
        # Compute final weights combining magnitude normalization + DWA ratio with controllable alpha.
        r = []
        active_names = []

        for name in self.loss_names:
            prev, curr = self.loss_history[name]
            if curr > self.eps:
                # log ratio for DWA
                ratio = torch.log(torch.tensor(prev / (curr + self.eps) + self.eps, dtype=torch.float32))
                r.append(ratio)
                active_names.append(name)

        if not active_names:
            return {name: 0.0 for name in self.loss_names}

        # DWA weights
        r = torch.tensor(r, dtype=torch.float32)
        weights_dwa = torch.softmax(r / self.T, dim=0) * len(active_names)

        # Magnitude normalization weights
        inv_mag = []
        for name in active_names:
            _, curr = self.loss_history[name]
            inv_mag.append(1.0 / max(curr, self.eps))
        inv_mag = torch.tensor(inv_mag, dtype=torch.float32)
        inv_mag = inv_mag / inv_mag.sum() * len(active_names)

        # Hybrid weighting with alpha control
        hybrid_weights = self.alpha_mag * inv_mag + self.alpha_dwa * weights_dwa

        # Normalize so sum = number of active losses
        hybrid_weights = hybrid_weights / hybrid_weights.sum() * len(active_names)

        # Construct full dictionary
        weight_dict = {}
        j = 0
        for name in self.loss_names:
            if name in active_names:
                weight_dict[name] = hybrid_weights[j].item()
                j += 1
            else:
                weight_dict[name] = 0.0

        return weight_dict
