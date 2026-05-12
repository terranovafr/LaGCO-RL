#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    test_gae.py
    Script to evaluate a trained GAE/VGAE and compute compression ratios with confidence intervals.
"""

from datetime import datetime
import argparse
import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from utils.file_utils import load_yaml
from envs.env_switcher import GraphEnvSwitcher
from torch_geometric.utils import from_networkx
from gae_utils import compute_backward_batch_train_loss
from model import GAE, VGAE
from utils.math_utils import ci
script_dir = Path(__file__).parent
torch.set_default_dtype(torch.float32)


def load_trained_gaes(gae_logs_folder, validation=False, random_init=False):
    models = {}
    models_spec = {}
    configs = {}

    for graph_name in os.listdir(gae_logs_folder):
        if not (graph_name.endswith("_graph") or graph_name.endswith("graph")):
            continue
        graph_path = os.path.join(gae_logs_folder, graph_name)
        config = load_yaml(os.path.join(graph_path, "train_config.yaml"))

        if not os.path.isdir(graph_path) or not graph_name.endswith("graph"):
            continue

        model_file = "model_best_val.pth" if validation and "model_best_val.pth" in os.listdir(graph_path) else "model_final.pth"
        model_path = os.path.join(graph_path, model_file)
        spec = load_yaml(os.path.join(graph_path, "model_spec.yaml"))
        # save in config the spec using graph_name as key
        for elem in spec:
            if not elem in config:
                config[elem] = {}
            config[elem][graph_name] = spec[elem]

        if config['model_type'] == "gae":
            model = GAE(
                in_channels=spec['node_feature_vector_size'],
                cfg_layers=config['model_config']['layers'],
                edge_feat_dim=spec['edge_feature_vector_size'],
                binary_indices=spec.get('binary_indices', []),
                multi_class_info=spec.get('multi_class_info', {}),
                multi_class_info_order=spec.get('multi_class_info_order', {}),
                continuous_indices=spec.get('continuous_indices', []),
                ranking_indices=spec.get('ranking_indices', []),
                edge_binary_indices=spec.get('edge_binary_indices', []),
                edge_continuous_indices=spec.get('edge_continuous_indices', []),
                edge_multi_class_info=spec.get('edge_multi_class_info', {}),
                edge_multi_class_info_order=spec.get('edge_multi_class_info_order', {}),
                edge_ranking_indices=spec.get('edge_ranking_indices', [])
            )
        elif config['model_type'] == "vgae":
            model = VGAE(
                in_channels=spec['node_feature_vector_size'],
                cfg_layers=config['model_config']['layers'],
                edge_feat_dim=spec['edge_feature_vector_size'],
                binary_indices=spec.get('binary_indices', []),
                multi_class_info=spec.get('multi_class_info', {}),
                multi_class_info_order=spec.get('multi_class_info_order', {}),
                continuous_indices=spec.get('continuous_indices', []),
                ranking_indices=spec.get('ranking_indices', []),
                edge_binary_indices=spec.get('edge_binary_indices', []),
                edge_continuous_indices=spec.get('edge_continuous_indices', []),
                edge_multi_class_info=spec.get('edge_multi_class_info', {}),
                edge_multi_class_info_order=spec.get('edge_multi_class_info_order', {}),
                edge_ranking_indices=spec.get('edge_ranking_indices', []),
                latent_dim=32
            )
        else:
            raise ValueError(f"Unknown model_type {config['model_type']}")

        if not random_init:
            model.load_state_dict(torch.load(model_path))
        model.eval()

        models[graph_name] = model
        models_spec[graph_name] = spec
        configs[graph_name] = config

    return models, models_spec, configs

def load_envs_from_folder(envs_folder):
    envs_dict = {}
    for f in os.listdir(envs_folder):
        if f.endswith(".pkl"):
            with open(os.path.join(envs_folder, f), "rb") as fp:
                envs_dict[int(f.replace(".pkl", ""))] = pickle.load(fp)
    return envs_dict


def build_env_switcher(envs_folder, split_yaml, split_name, test_config):
    split = load_yaml(split_yaml)
    split_name = split_name + "_set"
    if split_name not in split:
        raise ValueError(f"Split {split_name} not found in {split_yaml}")

    selected_ids = [elem["id"] for elem in split[split_name]]
    if len(selected_ids) == 0:
        raise ValueError(f"No environments found for split {split_name} in {split_yaml}")

    return GraphEnvSwitcher(
        train_ids=selected_ids,
        val_ids=None,
        save_switch_logs=False,
        algorithm_type='gae',
        switch_strategy=test_config['switch_strategy'],
        switch_interval=test_config['switch_interval'],
        envs_folder=envs_folder
    )


def compute_compression_ratios_with_ci(baseline_losses_list, trained_losses):
    compression = {}

    for g in trained_losses:
        compression[g] = {}
        cr_avg_samples = []

        for loss_name in trained_losses[g]:
            cr_samples = []

            for baseline in baseline_losses_list:
                L0 = baseline[g][loss_name]
                Lt = trained_losses[g][loss_name]

                if L0 <= 1e-8:
                    continue

                cr = (L0 - Lt) / L0
                cr = np.clip(cr, 0.0, 1.0)
                cr_samples.append(cr)

            if len(cr_samples) == 0:
                compression[g][loss_name] = {
                    "compression_ratio_mean": 0.0,
                    "compression_ratio_ci95_lower": 0.0,
                    "compression_ratio_ci95_upper": 0.0,
                }
                continue

            lo, hi = ci(cr_samples)

            compression[g][loss_name] = {
                "compression_ratio_mean": float(np.mean(cr_samples)),
                "compression_ratio_ci95_lower": float(lo),
                "compression_ratio_ci95_upper": float(hi),
            }

            cr_avg_samples.append(np.mean(cr_samples))

        # ---- CR_avg over losses ----
        if len(cr_avg_samples) > 0:
            lo, hi = ci(cr_avg_samples)
            compression[g]["CR_avg"] = {
                "compression_ratio_mean": float(np.mean(cr_avg_samples)),
                "compression_ratio_ci95_lower": float(lo),
                "compression_ratio_ci95_upper": float(hi),
            }
        else:
            compression[g]["CR_avg"] = {
                "compression_ratio_mean": 0.0,
                "compression_ratio_ci95_lower": 0.0,
                "compression_ratio_ci95_upper": 0.0,
            }

    return compression




def compute_losses_for_envs(models, envs, configs, episodes=50):
    losses_all = {
        g: {k: [] for k in [
            "adj_loss",
            "binary_cat_loss",
            "multi_cat_loss",
            "cont_loss",
            "ranking_loss",
            "edge_binary_loss",
            "edge_multi_cat_loss",
            "edge_continuous_loss",
            "edge_ranking_loss",
        ]}
        for g in models
    }



    envs.reset()

    for episode in tqdm(range(episodes), desc="Evaluating losses"):
        action = envs.sample_valid_action()
        if not isinstance(action, tuple):
            action = (action,)

        _, _, done, truncated, _ = envs.step(action)
        done = done or truncated

        for g, G in envs.get_graphs().items():
            if len(G.nodes) == 0:
                continue

            data = from_networkx(G)
            if data.edge_attr is None or data.edge_attr.numel() == 0:
                data.edge_attr = torch.zeros((data.edge_index.size(1), 1))

            model = models[g]
            with torch.no_grad():
                out = compute_backward_batch_train_loss(
                    model,
                    g,
                    data,
                    backward=False,
                    loss_weighting_object=None,
                    optimizer=None,
                    **configs[g],
                )

            (
                _,
                adj_loss,
                _,
                _,
                binary_cat_loss,
                multi_cat_loss,
                cont_loss,
                ranking_loss,
                edge_binary_loss,
                edge_multi_cat_loss,
                edge_continuous_loss,
                edge_ranking_loss,
                _,
            ) = out

            losses_all[g]["adj_loss"].append(adj_loss)
            losses_all[g]["binary_cat_loss"].append(binary_cat_loss)
            losses_all[g]["multi_cat_loss"].append(multi_cat_loss)
            losses_all[g]["cont_loss"].append(cont_loss)
            losses_all[g]["ranking_loss"].append(ranking_loss)
            losses_all[g]["edge_binary_loss"].append(edge_binary_loss)
            losses_all[g]["edge_multi_cat_loss"].append(edge_multi_cat_loss)
            losses_all[g]["edge_continuous_loss"].append(edge_continuous_loss)
            losses_all[g]["edge_ranking_loss"].append(edge_ranking_loss)

        if done:
            envs.reset()
            episode += 1

    return {g: {k: float(np.mean(v)) for k, v in losses.items()} for g, losses in losses_all.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained GAE/VGAE and compute compression ratios")
    parser.add_argument("-f", "--logs_folder", type=str, help="Folder with trained GAEs (inside gae/logs/)")
    parser.add_argument("-s", "--split", type=str, default="test", choices=["train","val","test"], help="Which split to evaluate")
    parser.add_argument("-e", "--environment", type=str, help="Environment to look for if logs folder is not specified")
    parser.add_argument("-ne", "--num_episodes", type=int, default=50, help="Number of episodes to run")
    parser.add_argument("-i", "--num_random_inits", type=int, default=5, help="Number of random baseline initializations")
    parser.add_argument("--validation", action="store_true", help="Use best validation encoder if exists")
    parser.add_argument("--test_config_file", type=str, default=os.path.join(script_dir, "..", "gae", "config", "test_config.yaml"), help="Path to the test config file")
    args = parser.parse_args()

    base_logs_dir = os.path.join(script_dir, "..", "gae", "logs")
    if not args.logs_folder:
        run_dirs = [
            os.path.join(base_logs_dir, d)
            for d in os.listdir(base_logs_dir)
            if os.path.isdir(os.path.join(base_logs_dir, d))
        ]

        if not run_dirs:
            raise FileNotFoundError(f"No log folders found in {base_logs_dir}")
        # keep only those that have environment key in train_config.yaml being the environment in args
        remaining_run_dirs = []
        if args.environment:
            # collect all (run_dir, graph_dir) pairs that contain train_config.yaml

            for run_dir in run_dirs:
                candidate_graph_dirs = []

                for sub_d in os.listdir(run_dir):
                    sub_path = os.path.join(run_dir, sub_d)
                    if (
                            os.path.isdir(sub_path)
                            and (sub_d.endswith("_graph") or sub_d.endswith("graph"))
                            and os.path.exists(os.path.join(sub_path, "train_config.yaml"))
                    ):
                        candidate_graph_dirs.append(sub_path)

                if not candidate_graph_dirs:
                    continue

                # pick a random *_graph folder
                random_graph_dir = random.choice(candidate_graph_dirs)

                # load train_config.yaml
                train_config_path = os.path.join(random_graph_dir, "train_config.yaml")
                train_config = load_yaml(train_config_path)

                # optional: filter by environment
                if train_config.get("environment", "") != args.environment:
                    continue
                else:
                    remaining_run_dirs.append(run_dir)
        # Pick the most recently created matching run
        args.logs_folder = max(remaining_run_dirs, key=os.path.getctime)
    else:
        args.logs_folder = os.path.join(base_logs_dir, args.logs_folder)
    # load test config file
    test_config = load_yaml(args.test_config_file)

    if args.split == 'train':
        args.split = 'training'
    models, enc_spec, configs = load_trained_gaes(args.logs_folder, validation=args.validation)

    envs_dict = load_envs_from_folder(os.path.join(args.logs_folder, "envs"))
    envs_switcher = build_env_switcher(os.path.join(args.logs_folder, "envs"), os.path.join(args.logs_folder, "split.yaml"), args.split, test_config)

    baseline_losses_list = []
    for _ in tqdm(range(args.num_random_inits)):
        rand_models, _, _ = load_trained_gaes(args.logs_folder, random_init=True)
        baseline_losses_list.append(
            compute_losses_for_envs(rand_models, envs_switcher, configs, args.num_episodes)
        )

    trained_losses = compute_losses_for_envs(models, envs_switcher, configs, args.num_episodes)

    # -------- Compression distributions --------
    compression = {
        g: {} for g in trained_losses
    }

    for baseline_losses in baseline_losses_list:
        for g in trained_losses:
            for loss_name, Lt in trained_losses[g].items():

                if Lt == 0:
                    continue

                L0 = baseline_losses[g].get(loss_name, 0)

                if L0 == 0:
                    continue

                cr = max(0.0, min(1.0, (L0 - Lt) / L0))

                compression.setdefault(g, {}).setdefault(loss_name, []).append(cr)

    # -------- CI aggregation --------
    compression_with_ci = {}

    for g, losses in compression.items():
        compression_with_ci[g] = {}

        valid_cr_lists = []

        for loss, values in losses.items():
            if not values:
                continue

            mean = float(np.mean(values))
            lo, hi = ci(values)

            compression_with_ci[g][loss] = {
                "mean": mean,
                "ci95_lower": lo,
                "ci95_upper": hi,
            }

            valid_cr_lists.append(values)

        # ---- CR_avg only if we have valid losses ----
        if valid_cr_lists:
            min_len = min(len(v) for v in valid_cr_lists)
            aligned = np.stack([v[:min_len] for v in valid_cr_lists])

            cr_avg_samples = np.mean(aligned, axis=0)
            lo, hi = ci(cr_avg_samples)

            compression_with_ci[g]["CR_avg"] = {
                "mean": float(np.mean(cr_avg_samples)),
                "ci95_lower": lo,
                "ci95_upper": hi,
            }

    # -------- Save --------
    out_dir = os.path.join(args.logs_folder, "test", args.split)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"compression_{datetime.now():%Y%m%d_%H%M%S}.yaml")

    with open(out_file, "w") as f:
        yaml.dump(compression_with_ci, f, sort_keys=False)

    print(f"\n✅ Saved compression + CI to {out_file}")