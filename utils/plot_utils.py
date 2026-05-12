#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    plot_utils.py
    Utility functions for plotting UMAP embeddings of action spaces, including metric computation, representative sampling, and compact visualization styling.
'''

import os
import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

def style_small_umap_axes(ax, keep_ticks=True):
    # Clean, minimalist axes for small UMAP plots. Optionally keep a few ticks for scale reference.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    if keep_ticks:
        ax.locator_params(axis="both", nbins=2)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=5,
            length=2,
            width=0.5,
            pad=1,
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            labelbottom=False,
            labelleft=False,
        )


def compute_embedding_metrics(X, ks=(3, 5, 10)):
    """
    Group-free metrics in [0, 1].

    Returns
    -------
    dst : float
        Distinctiveness based on nearest-neighbor separation.

    cmp_at_k : dict
        Local compactness at different neighborhood sizes.
    """
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(X, dtype=np.float64)
    n = len(X)

    if n < 2:
        return np.nan, {k: np.nan for k in ks}

    D = squareform(pdist(X, metric="euclidean"))
    np.fill_diagonal(D, np.inf)

    pairwise = D[np.isfinite(D)]
    mean_pairwise = float(np.mean(pairwise)) if len(pairwise) > 0 else np.nan

    if not np.isfinite(mean_pairwise) or mean_pairwise <= 1e-12:
        return 0.0, {k: 0.0 for k in ks}

    nn_dist = np.min(D, axis=1)
    mean_nn = float(np.mean(nn_dist))
    dst = float(np.clip(mean_nn / mean_pairwise, 0.0, 1.0))

    sorted_dists = np.sort(D, axis=1)
    cmp_at_k = {}

    for k in ks:
        k_eff = min(k, n - 1)
        knn_dists = sorted_dists[:, :k_eff]
        mean_knn = float(np.mean(knn_dists))
        cmp = float(np.clip(1.0 - (mean_knn / mean_pairwise), 0.0, 1.0))
        cmp_at_k[k] = cmp

    return dst, cmp_at_k


def build_metrics_text(cmp_at_k):
    # Build a multi-line string for the metrics box, showing cmp@k values. If a value is NaN, show "n/a".
    lines = []
    for k in sorted(cmp_at_k.keys()):
        val = cmp_at_k[k]
        if np.isnan(val):
            lines.append(f"cmp@{k}=n/a")
        else:
            lines.append(f"cmp@{k}={val:.2f}")
    return "\n".join(lines)


def choose_metrics_box_position(z_2d, box_w=0.26, box_h=0.16):
    # Choose the best position for the metrics box among 4 corners, based on how many points would be covered and their proximity to the box center. The box is defined in normalized axes coordinates (0 to 1), and we check how many points fall within the box area for each candidate position. We also add a proximity penalty to prefer positions that are farther from dense clusters of points.
    candidates = [
        {"xy": (0.03, 0.97), "ha": "left", "va": "top"},
        {"xy": (0.97, 0.97), "ha": "right", "va": "top"},
        {"xy": (0.03, 0.03), "ha": "left", "va": "bottom"},
        {"xy": (0.97, 0.03), "ha": "right", "va": "bottom"},
    ]

    x = z_2d[:, 0]
    y = z_2d[:, 1]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    xr = max(xmax - xmin, 1e-12)
    yr = max(ymax - ymin, 1e-12)

    x_norm = (x - xmin) / xr
    y_norm = (y - ymin) / yr
    pts = np.column_stack([x_norm, y_norm])

    def box_bounds(candidate):
        x0, y0 = candidate["xy"]

        if candidate["ha"] == "left":
            left, right = x0, x0 + box_w
        else:
            left, right = x0 - box_w, x0

        if candidate["va"] == "top":
            bottom, top = y0 - box_h, y0
        else:
            bottom, top = y0, y0 + box_h

        return left, right, bottom, top

    def score_candidate(candidate):
        left, right, bottom, top = box_bounds(candidate)

        inside = (
            (pts[:, 0] >= left) & (pts[:, 0] <= right) &
            (pts[:, 1] >= bottom) & (pts[:, 1] <= top)
        )
        n_inside = np.sum(inside)

        cx = (left + right) / 2.0
        cy = (bottom + top) / 2.0
        d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        proximity_penalty = np.sum(np.exp(-d2 / 0.01))

        return n_inside * 1000 + proximity_penalty

    return min(candidates, key=score_candidate)


def add_metrics_box(ax, z_2d, cmp_at_k):
    # Add a box with metrics text to the plot, choosing the best position to minimize overlap with points. The text is built from cmp_at_k values, and the box has a white background with some transparency.
    metrics_text = build_metrics_text(cmp_at_k)
    best = choose_metrics_box_position(z_2d)

    ax.text(
        best["xy"][0],
        best["xy"][1],
        metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        va=best["va"],
        ha=best["ha"],
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            alpha=0.85,
            edgecolor="0.7",
            linewidth=0.3,
        ),
        zorder=10,
    )

def sanitize_filename(name):
    # Sanitize a string to be safe for filenames by replacing non-alphanumeric characters with underscores and stripping leading/trailing underscores.
    return re.sub(r"[^\w\-\.]+", "_", str(name)).strip("_")


def compute_umap_embedding(X, n_neighbors=None, min_dist=0.1, metric="euclidean"):
    # Compute a 2D UMAP embedding of the data X. If n_neighbors is not provided, choose a default based on dataset size. If there are too few points, return None. After computing UMAP, optionally align it to PCA for better interpretability.
    if n_neighbors is None:
        n_neighbors = max(5, min(30, len(X) - 1))

    if len(X) < n_neighbors:
        return None

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    z_2d = reducer.fit_transform(X)

    if len(X) >= 2:
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(X)
        R, _ = orthogonal_procrustes(z_2d, z_pca)
        z_2d = z_2d.dot(R)

    return z_2d


def build_group_keys(action_ids, discrete_actions=None):
    # Build group keys for coloring the plot. If discrete_actions is None, use the first element of the action_id tuple or the string representation of the action_id as the group key. If discrete_actions is provided, group action_ids based on whether their first element (if tuple) or string representation is in discrete_actions, using "Other" for those that are not.
    if discrete_actions is None:
        return [aid[0] if isinstance(aid, tuple) else str(aid) for aid in action_ids]

    group_keys = []
    for aid in action_ids:
        if isinstance(aid, tuple):
            group_keys.append(aid[0] if aid[0] in discrete_actions else "Other")
        else:
            aid_str = str(aid)
            group_keys.append(aid_str if aid_str in discrete_actions else "Other")
    return group_keys


def build_color_info(group_keys):
    # Build color information for the plot based on group keys. Assign a unique integer to each group key, then create a colormap and normalization for plotting. The colors array maps each point to its group integer, and the colormap can be used to assign distinct colors to each group.
    unique_groups = sorted(set(group_keys))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    colors = np.array([group_to_int[g] for g in group_keys])

    n_colors = len(unique_groups)
    cmap = plt.cm.get_cmap("tab20", n_colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_colors + 0.5, 1), n_colors)

    return colors, cmap, norm


def _allocate_integer_quotas(weights, total):
    # Allocate integer quotas summing exactly to total, proportionally to weights.
    weights = np.asarray(weights, dtype=np.float64)
    if total <= 0 or len(weights) == 0:
        return np.zeros(len(weights), dtype=int)

    weight_sum = weights.sum()
    if weight_sum <= 0:
        quotas = np.zeros(len(weights), dtype=int)
        quotas[:min(total, len(weights))] = 1
        return quotas

    raw = total * (weights / weight_sum)
    base = np.floor(raw).astype(int)
    remainder = total - base.sum()

    if remainder > 0:
        frac = raw - base
        order = np.argsort(-frac)
        base[order[:remainder]] += 1

    return base


def sample_representative_points(
    z_2d,
    group_keys=None,
    max_points=2500,
    grid_size=30,
    min_per_nonempty_stratum=1,
    random_state=42,
):
    """
    Sample points for plotting while preserving the visual distribution.

    Strategy
    --------
    - Build strata in UMAP 2D space using a regular grid.
    - Optionally split each grid cell by action group.
    - Allocate plot budget proportionally to stratum occupancy.
    - Guarantee at least a small number of points per non-empty stratum when possible.

    This keeps dense regions dense, sparse regions visible, and rare groups less likely
    to disappear completely.
    """
    z_2d = np.asarray(z_2d)
    n = len(z_2d)

    if n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(random_state)

    x = z_2d[:, 0]
    y = z_2d[:, 1]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    xr = max(xmax - xmin, 1e-12)
    yr = max(ymax - ymin, 1e-12)

    # Normalize to [0, 1]
    x_norm = (x - xmin) / xr
    y_norm = (y - ymin) / yr

    gx = np.minimum((x_norm * grid_size).astype(int), grid_size - 1)
    gy = np.minimum((y_norm * grid_size).astype(int), grid_size - 1)

    if group_keys is None:
        strata_keys = [(ix, iy) for ix, iy in zip(gx, gy)]
    else:
        strata_keys = [(ix, iy, g) for ix, iy, g in zip(gx, gy, group_keys)]

    strata_to_indices = {}
    for idx, key in enumerate(strata_keys):
        strata_to_indices.setdefault(key, []).append(idx)

    strata = list(strata_to_indices.keys())
    counts = np.array([len(strata_to_indices[s]) for s in strata], dtype=int)
    n_strata = len(strata)

    if n_strata == 0:
        return np.arange(min(n, max_points))

    # If possible, reserve a minimum per stratum
    quotas = np.zeros(n_strata, dtype=int)
    if min_per_nonempty_stratum > 0 and max_points >= n_strata * min_per_nonempty_stratum:
        quotas += min_per_nonempty_stratum
        remaining = max_points - quotas.sum()
    else:
        remaining = max_points

    # Distribute the remaining budget proportionally to leftover capacity / counts
    if remaining > 0:
        extra_weights = counts.astype(np.float64)
        extra_quotas = _allocate_integer_quotas(extra_weights, remaining)
        quotas += extra_quotas

    # Cannot sample more than available in any stratum
    quotas = np.minimum(quotas, counts)

    # Because of clipping above, we may have a few unused slots -> redistribute
    deficit = max_points - quotas.sum()
    if deficit > 0:
        remaining_capacity = counts - quotas
        expandable = np.where(remaining_capacity > 0)[0]
        if len(expandable) > 0:
            extra = _allocate_integer_quotas(remaining_capacity[expandable], deficit)
            quotas[expandable] += np.minimum(extra, remaining_capacity[expandable])

    selected = []
    for i, stratum in enumerate(strata):
        q = quotas[i]
        idxs = np.array(strata_to_indices[stratum], dtype=int)
        if q >= len(idxs):
            selected.extend(idxs.tolist())
        elif q > 0:
            chosen = rng.choice(idxs, size=q, replace=False)
            selected.extend(chosen.tolist())

    selected = np.array(sorted(set(selected)), dtype=int)

    # Safety fallback
    if len(selected) > max_points:
        selected = rng.choice(selected, size=max_points, replace=False)
        selected = np.array(sorted(selected), dtype=int)

    return selected


def save_umap_plot(z_2d, colors, cmap, norm, cmp_at_k, output_path, keep_ticks, keep_legend):
    # Save a UMAP plot with the given 2D coordinates, colors, colormap, and metrics. The plot is styled for compactness and clarity, with an optional metrics box and ticks.
    fig, ax = plt.subplots(figsize=(1.6, 1.6))

    ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=colors,
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        s=15,
        linewidths=0,
        rasterized=False,
    )

    if keep_legend:
        add_metrics_box(ax, z_2d, cmp_at_k)
    style_small_umap_axes(ax, keep_ticks=keep_ticks)

    plt.tight_layout(pad=0.08)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)


def compute_space_cmp_stats(
    continuous_actions,
    n_neighbors=None,
    min_dist=0.1,
    metric="euclidean",
):
    # Compute cmp@k metrics for the full action set, without plotting. This can be used to report metrics in a table or log them separately from the visualization.
    action_ids = list(continuous_actions.keys())
    X = np.array([continuous_actions[aid] for aid in action_ids], dtype=np.float32)

    if len(X) < 2:
        return

    z_2d = compute_umap_embedding(
        X,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    if z_2d is None:
        return

    _, cmp_at_k = compute_embedding_metrics(X)
    return cmp_at_k


def plot_umap_actions(
    continuous_actions,
    logs_folder,
    discrete_actions=None,
    cmp_at_k=None,
    n_neighbors=None,
    min_dist=0.1,
    metric="euclidean",
    environment_name="environment",
    num_iterations=1,
    plot_max_points=300,
    sampling_grid_size=30,
    sampling_min_per_stratum=1,
    sampling_random_state=42,
):
    """
    Plot compact UMAP action embeddings for paper use.

    Saves:
      - {environment_name}_{num_iterations}_with_ticks.pdf
      - {environment_name}_{num_iterations}_no_ticks.pdf
      - {environment_name}_{num_iterations}_no_ticks_no_legend.pdf

    Notes
    -----
    - UMAP is computed on the full action set.
    - Metrics are computed on the full action set.
    - Only the plotted points may be subsampled, in a distribution-aware way.
    """
    os.makedirs(logs_folder, exist_ok=True)

    action_ids = list(continuous_actions.keys())
    X = np.array([continuous_actions[aid] for aid in action_ids], dtype=np.float32)

    if len(X) < 2:
        return

    z_2d = compute_umap_embedding(
        X,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    if z_2d is None:
        return

    if not cmp_at_k:
        _, cmp_at_k = compute_embedding_metrics(X)

    group_keys = build_group_keys(action_ids, discrete_actions=discrete_actions)

    if len(X) > plot_max_points:
        # Representative subsampling only for visualization
        plot_indices = sample_representative_points(
            z_2d=z_2d,
            group_keys=group_keys,
            max_points=plot_max_points,
            grid_size=sampling_grid_size,
            min_per_nonempty_stratum=sampling_min_per_stratum,
            random_state=sampling_random_state,
        )
        z_plot = z_2d[plot_indices]
    else:
        plot_indices = np.arange(len(X))
        z_plot = z_2d
    group_keys_plot = [group_keys[i] for i in plot_indices]

    colors, cmap, norm = build_color_info(group_keys_plot)

    safe_env_name = sanitize_filename(environment_name)

    save_umap_plot(
        z_2d=z_plot,
        colors=colors,
        cmap=cmap,
        norm=norm,
        cmp_at_k=cmp_at_k,
        output_path=os.path.join(
            logs_folder,
            f"{safe_env_name}_{num_iterations}_with_ticks.pdf",
        ),
        keep_ticks=True,
        keep_legend=True,
    )

    save_umap_plot(
        z_2d=z_plot,
        colors=colors,
        cmap=cmap,
        norm=norm,
        cmp_at_k=cmp_at_k,
        output_path=os.path.join(
            logs_folder,
            f"{safe_env_name}_{num_iterations}_no_ticks.pdf",
        ),
        keep_ticks=False,
        keep_legend=True,
    )

    save_umap_plot(
        z_2d=z_plot,
        colors=colors,
        cmap=cmap,
        norm=norm,
        cmp_at_k=cmp_at_k,
        output_path=os.path.join(
            logs_folder,
            f"{safe_env_name}_{num_iterations}_no_ticks_no_legend.pdf",
        ),
        keep_ticks=False,
        keep_legend=False,
    )

    return cmp_at_k