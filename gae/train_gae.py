#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

"""
    train_gae.py
    Script to train a Graph Autoencoder (GAE) on a set of environments, with support for multiple graph types, dynamic loss weighting, and validation. The script loads environments, initializes models based on the graph structures, and performs training while periodically evaluating on a validation set. Compression ratios are computed against random initialization baselines to assess the effectiveness of the training.
"""

import argparse
import torch
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.log_utils import setup_logging
from torch_geometric.data import DataLoader
from gae_utils import compute_backward_batch_train_loss
from model import GAE, VGAE
from gae_utils import validate
from envs.env_switcher import GraphEnvSwitcher
from utils.math_utils import set_seeds, HybridMagnitudeDWAWeighting
from pathlib import Path
from utils.file_utils import save_yaml, read_split_file, load_yaml
from tqdm import tqdm
from wrappers.gae_training_wrapper import GAETrainingWrapper
from torch_geometric.utils import from_networkx
import numpy as np
script_dir = Path(__file__).parent
torch.set_default_dtype(torch.float32)
import os
import torch
from copy import deepcopy

# Function used to load a folder where the environments are stored and a sample environment to load information about sizes
def load_envs(config, envs_folder):
    general_config = load_yaml(os.path.join(script_dir, "..", "config.yaml"))
    sample_env = None
    if config['load_envs']: # Added with the idea of adding new modes in the future, like random generation live etc.
        # Networks relative to env_samples
        file_path = os.path.join(script_dir, '..', 'data', 'env_samples', config['load_envs'])
        # wrap in the environments
        config['num_environments'] = 0
        keys_config = {k: v for k, v in general_config.items() if isinstance(v, dict) and 'key' in v}
        for index, folder in tqdm(enumerate(os.listdir(file_path)), desc="Loading networks"):
            if os.path.isdir(os.path.join(file_path, folder)) and folder.isdigit():
                config['num_environments'] += 1
                env_folder = os.path.join(file_path, folder, f"env.pkl")
                with open(env_folder, 'rb') as f:
                    environment = pickle.load(f)
                env_config = load_yaml(
                    os.path.join(script_dir, "..", "envs", "config", f"{config['environment']}_config.yaml"))
                for key in keys_config:
                    env_config[key] = keys_config[key]
                env_config['verbose'] = config['verbose']
                environment.update_config(env_config)
                environment.update_spec()
                env = GAETrainingWrapper(env=environment)
                with open(os.path.join(envs_folder, f"{folder}.pkl"), 'wb') as f:
                    pickle.dump(env, f)
                sample_env = env
    return sample_env

def compute_losses_no_backward(models, envs, config):
    losses = {}

    envs.reset()
    done = False

    while not done:
        action = envs.sample_valid_action()
        if not isinstance(action, tuple):
            action = (action,)
        _, _, done, truncated, _ = envs.step(action)
        done = done or truncated

        Gs = envs.get_graphs()
        for graph_name, G in Gs.items():
            if len(G.nodes) == 0:
                continue

            data = from_networkx(G)
            if data.edge_attr is None or data.edge_attr.numel() == 0:
                data.edge_attr = torch.zeros((data.edge_index.size(1), 1))

            model = models[graph_name]
            model.eval()

            with torch.no_grad():
                out = compute_backward_batch_train_loss(
                    model,
                    graph_name,
                    data,
                    loss_weighting_object=None,
                    optimizer=None,
                    backward=False,
                    **config
                )

            (
                total_loss, adj_loss, feature_loss, edge_feature_loss,
                binary_cat_loss, multi_cat_loss, cont_loss, ranking_loss,
                edge_binary_loss, edge_multi_cat_loss,
                edge_continuous_loss, edge_ranking_loss, kl_loss
            ) = out

            if graph_name not in losses:
                losses[graph_name] = {
                    "adj_loss": [],
                    "binary_cat_loss": [],
                    "multi_cat_loss": [],
                    "cont_loss": [],
                    "ranking_loss": [],
                    "edge_binary_loss": [],
                    "edge_multi_cat_loss": [],
                    "edge_continuous_loss": [],
                    "edge_ranking_loss": [],
                }

            losses[graph_name]["adj_loss"].append(adj_loss)
            losses[graph_name]["binary_cat_loss"].append(binary_cat_loss)
            losses[graph_name]["multi_cat_loss"].append(multi_cat_loss)
            losses[graph_name]["cont_loss"].append(cont_loss)
            losses[graph_name]["ranking_loss"].append(ranking_loss)
            losses[graph_name]["edge_binary_loss"].append(edge_binary_loss)
            losses[graph_name]["edge_multi_cat_loss"].append(edge_multi_cat_loss)
            losses[graph_name]["edge_continuous_loss"].append(edge_continuous_loss)
            losses[graph_name]["edge_ranking_loss"].append(edge_ranking_loss)

    # Average
    for g in losses:
        for k in losses[g]:
            losses[g][k] = float(np.mean(losses[g][k]))

    return losses


# Initialize model and optimizer, with model initialized with the graph from the sample environment (determining feature vector attributes)
def init_models(sample_env, config):
    sample_env.reset()
    Gs = sample_env.get_graphs()
    for _ in Gs:
        config['continuous_indices'] = sample_env.continuous_indices
        config['binary_indices'] = sample_env.binary_indices
        config['multi_class_info'] = sample_env.multi_class_info
        config['multi_class_info_order'] = sample_env.multi_class_info_order
        config['ranking_indices'] = sample_env.ranking_indices
        config['node_feature_vector_size']= sample_env.node_feature_vector_size
        config['edge_feature_vector_size']= sample_env.edge_feature_vector_size
        config['edge_binary_indices']= sample_env.edge_binary_indices
        config['edge_continuous_indices'] = sample_env.edge_continuous_indices
        config['edge_multi_class_info'] = sample_env.edge_multi_class_info
        config['edge_multi_class_info_order'] = sample_env.edge_multi_class_info_order
        config['edge_ranking_indices'] = sample_env.edge_ranking_indices
    models = {}
    optimizers = {}
    configs = {}
    for G_name in Gs:
        copy_config = deepcopy(config)
        model = None
        if config['model_type'] == "gae":
            # check that all have G_name in config for the required keys
            if config['edge_feature_vector_size'][G_name] == 0:
                # repeat twice second layer and remove first one
                copy_config['model_config']['layers'] = [config['model_config']['layers'][1]]
                additional_layers = config['model_config']['layers'][1:]
                # check if it is list
                if isinstance(additional_layers, list):
                    copy_config['model_config']['layers'] += additional_layers
                else:
                    copy_config['model_config']['layers'].append(additional_layers)
            for key in ['node_feature_vector_size', 'edge_feature_vector_size', 'binary_indices', 'multi_class_info', 'multi_class_info_order', 'continuous_indices', 'edge_binary_indices', 'edge_continuous_indices', 'edge_multi_class_info', 'edge_multi_class_info_order']:
                if G_name not in config[key]:
                    raise ValueError(f"Graph name {G_name} not found in config for key {key}")
            model = GAE(copy_config['node_feature_vector_size'][G_name], copy_config['model_config']['layers'], copy_config['edge_feature_vector_size'][G_name], copy_config['binary_indices'][G_name], copy_config['multi_class_info'][G_name], copy_config['multi_class_info_order'][G_name], copy_config['continuous_indices'][G_name], copy_config['ranking_indices'][G_name], copy_config['edge_binary_indices'][G_name], copy_config['edge_continuous_indices'][G_name], copy_config['edge_multi_class_info'][G_name], copy_config['edge_multi_class_info_order'][G_name], copy_config['edge_ranking_indices'][G_name]) #config['num_edge_types'][G_name])
        elif config['model_type'] == 'vgae':
            model = VGAE(copy_config['node_feature_vector_size'][G_name], copy_config['model_config']['layers'], copy_config['edge_feature_vector_size'][G_name], copy_config['binary_indices'][G_name], copy_config['multi_class_info'][G_name], copy_config['multi_class_info_order'][G_name], copy_config['continuous_indices'][G_name], copy_config['ranking_indices'][G_name], copy_config['edge_binary_indices'][G_name], copy_config['edge_continuous_indices'][G_name], copy_config['edge_multi_class_info'][G_name], copy_config['edge_multi_class_info_order'][G_name], copy_config['edge_ranking_indices'][G_name], copy_config['latent_dim_vgae'])

        models[G_name] = model
        optimizer = torch.optim.Adam(model.parameters(), lr=config['model_config']['learning_rate'])
        optimizers[G_name] = optimizer
        configs[G_name] = copy_config
    return models, optimizers, configs

# Training and evaluation function for the model
def train_and_eval(config, envs, models, optimizers, writer, logger, logs_folder):
    batch_graph = {}
    done = True
    dwas = {}
    best_val_loss = float('inf')
    average_train_loss = None
    best_model_states = {name: None for name in models}  # store best encoder states

    for model_name in models:
        dwa = HybridMagnitudeDWAWeighting(loss_names=[
            "adj_loss", "binary_cat_loss", "multi_cat_loss", "cont_loss", "ranking_loss", "edge_binary_loss", "edge_multi_cat_loss",
            "edge_continuous_loss", "edge_ranking_loss",
        ], T=config['temperature_dwa'])
        dwas[model_name] = dwa
    batch_counter = 0
    envs.reset()
    episode_id = 0
    average_val_loss = None
    average_val_loss_window = []
    for iteration in tqdm(range(config['train_iterations']), desc="Launching training"): # Number of different graph configurations to which the GAE is exposed
        if done:
            envs.reset()
        # Sample a valid action and use it
        action = envs.sample_valid_action()
        if not isinstance(action, tuple):
            action = (action,)
        if action == (None,):
            done = True
            episode_id += 1
            envs.done = True
            continue
        _, _, done, truncated, _ = envs.step(action)
        done = done or truncated
        average_train_loss = 0

        Gs = envs.get_graphs()
        # check if any of them has 0 edges, in that case redo
        empty_graph = False
        for graph_name in Gs:
            G = Gs[graph_name]
            if len(G.nodes) == 0 or len(G.edges) == 0:
                empty_graph = True
                break
        if empty_graph:
            done = True
            envs.done = True
            continue
        # for each graph print nodes and edges
        for graph_name in Gs:
            G = Gs[graph_name]
            if len(G.nodes) == 0:
                continue

            data = from_networkx(G)
            # check for every node if they have attributes, both in G and data

            if data.edge_attr is None or data.edge_attr.numel() == 0:
                data.edge_attr = torch.zeros((data.edge_index.size(1), 1))
                data.edge_raw_cache = [{} for _ in range(data.edge_index.size(1))]

            # PRINT DATA EDGE_RAW_CACHE
            if not graph_name in batch_graph:
                batch_graph[graph_name] = []
            batch_graph[graph_name].append(data)
            if len(batch_graph[graph_name]) % config['model_config']['batch_size'] == 0:
                batch_loader = DataLoader(batch_graph[graph_name], batch_size=config['model_config']['batch_size'], shuffle=True)
                for batch_data in batch_loader:


                    if config['dynamic_weights']:
                        total_loss, adj_loss, feature_loss, edge_feature_loss, binary_cat_loss, multi_cat_loss, cont_loss, ranking_loss, edge_binary_loss, edge_multi_cat_loss, edge_continuous_loss, edge_ranking_loss, kl_loss = compute_backward_batch_train_loss(
                            models[graph_name], graph_name, batch_data, dwas, optimizers[graph_name], backward=True, **config)
                    else:
                        if not config['environment'] in config['weights']:
                            config['environment'] = 'default'
                        total_loss, adj_loss, feature_loss, edge_feature_loss, binary_cat_loss, multi_cat_loss, cont_loss, ranking_loss, edge_binary_loss, edge_multi_cat_loss, edge_continuous_loss, edge_ranking_loss, kl_loss = compute_backward_batch_train_loss(
                            models[graph_name], graph_name, batch_data,backward=True, loss_weighting_object=None, optimizer=optimizers[graph_name],
                            adj_weight=config['weights'][config['environment']]['adj_weight'],
                            feature_weight=config['weights'][config['environment']]['node_feature_vector_weight'],
                            edge_feature_weight=config['weights'][config['environment']]['edge_feature_vector_weight'],
                            binary_cat_weight=config['weights'][config['environment']][
                                'node_feature_vector_binary_cat_weight'],
                            multi_cat_weight=config['weights'][config['environment']][
                                'node_feature_vector_multi_cat_weight'],
                            cont_weight=config['weights'][config['environment']]['node_feature_vector_cont_weight'],
                            ranking_weight=config['weights'][config['environment']].get('node_feature_vector_ranking_weight', 0.0),
                            edge_binary_weight=config['weights'][config['environment']]['edge_feature_vector_binary_cat_weight'],
                            edge_multi_cat_weight=config['weights'][config['environment']]['edge_feature_vector_multi_cat_weight'],
                            edge_continuous_weight=config['weights'][config['environment']]['edge_feature_vector_cont_weight'],
                            edge_ranking_weight=config['weights'][config['environment']].get('edge_feature_vector_ranking_weight', 0.0),
                            kl_weight=config['weights'][config['environment']].get('kl_weight', 0.0),
                            **config)
                    batch_counter += 1
                    if total_loss is None:
                        done = True # graph blocked
                        envs.done = True
                        break
                    average_train_loss += total_loss
                    if config['verbose']:
                        logger.info(f"Reporting metrics for graph {graph_name}")
                        logger.info(f" / Training iteration {iteration} - Total loss: {total_loss} Adj loss: {adj_loss} Node feature vector loss: {feature_loss} Edge feature vector loss: {edge_feature_loss} Node feature vector binary cat loss: {binary_cat_loss} Node feature vector multi cat loss: {multi_cat_loss} Node feature vector cont loss: {cont_loss} Edge feature vector binary cat loss: {edge_feature_loss} Edge feature vector multi cat loss: {edge_multi_cat_loss} Edge feature vector cont loss: {edge_continuous_loss}")
                    writer.add_scalar(f'train/{graph_name}/total_loss', total_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/adj_loss', adj_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/node_feature_vector_loss', feature_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/edge_feature_vector_loss', edge_feature_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/node_feature_vector/binary_cat_loss', binary_cat_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/node_feature_vector/multi_cat_loss', multi_cat_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/node_feature_vector/cont_loss', cont_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/node_feature_vector/ranking_loss', ranking_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/edge_feature_vector/binary_cat_loss', edge_binary_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/edge_feature_vector/multi_cat_loss', edge_multi_cat_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/edge_feature_vector/cont_loss', edge_continuous_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/edge_feature_vector/ranking_loss', edge_ranking_loss, iteration)
                    writer.add_scalar(f'train/{graph_name}/kl_loss', kl_loss, iteration)
                    batch_graph = {}
                    break  # take only one batch


        if config['validation'] and iteration % config['val_interval'] == 0 and iteration > 0:
            if config['verbose']:
                logger.info(
                    f"Validation phase at iteration {iteration}")
            envs.set_mode('val')  # Switch to validation mode
            average_val_loss = validate(models, envs, dwas, writer, config, iteration, logger)
            average_val_loss_window.append(average_val_loss)
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                for model_name in models:
                    best_model_states[model_name] = deepcopy(models[model_name].encoder.state_dict())
                if config['verbose']:
                    logger.info(f"New best validation loss: {best_val_loss:.6f}, saving best model weights.")

    for model_name in models:
        model_path = os.path.join(logs_folder, model_name)
        os.makedirs(model_path, exist_ok=True)
        # Save final model
        torch.save(models[model_name].encoder.state_dict(), os.path.join(model_path, 'encoder_final.pth'))
        torch.save(models[model_name].state_dict(), os.path.join(model_path, 'model_final.pth'))
        # Save best validation model
        if best_model_states[model_name] is not None:
            torch.save(best_model_states[model_name], os.path.join(model_path, 'encoder_best_val.pth'))
            if config['verbose']:
                logger.info(
                    f"Best validation model for {model_name} saved at {os.path.join(model_path, 'encoder_best_val.pth')}")

    # Decide return score based on validation or training
    score = np.mean(average_val_loss_window[-5:]) if config['validation'] else average_train_loss
    return score

# Execute several runs of model training
def execute_runs(config, original_logs_folder, envs_folder, logger):
    scores = []
    for run in range(config['num_runs']):
        if config['num_runs'] == 1:
            logs_folder = os.path.join(original_logs_folder)
        else:
            logs_folder = os.path.join(original_logs_folder, f"run_{run+1}",)
        if not os.path.exists(envs_folder):
            os.makedirs(envs_folder, exist_ok=True)
        os.makedirs(logs_folder, exist_ok=True)
        writer = SummaryWriter(str(logs_folder))
        set_seeds(config['seeds'][run])
        env = load_envs(config, envs_folder)
        models, optimizers, config_dict = init_models(env,config=config)

        # Saving configuration yaml file with all information related to the training
        for model in models:
            if not os.path.exists(os.path.join(str(logs_folder), model)):
                os.makedirs(os.path.join(str(logs_folder), model))
            config = config_dict[model]
            filename = os.path.join(str(logs_folder), model, f"model_spec.yaml")
            indices_keys = ['continuous_indices', 'binary_indices', 'multi_class_info', 'multi_class_info_order', 'node_feature_vector_size', 'edge_feature_vector_size', 'edge_binary_indices', 'edge_continuous_indices', 'edge_multi_class_info', 'edge_multi_class_info_order', 'edge_ranking_indices', 'ranking_indices']
            indices_config = {key: config[key] for key in indices_keys}
            # keep only those with graph name
            indices_config = {key: indices_config[key][model] for key in indices_config}
            save_yaml(indices_config, logs_folder, filename)
            copy_config = config.copy()
            for key in indices_keys:
                copy_config.pop(key, None)
            for key in ['default_model_config', 'example_blocks']:
                copy_config.pop(key, None)
            folder_name = os.path.join(str(logs_folder), model)
            # remove from copy config all dicts with key as key
            copy_config = {k: v for k, v in copy_config.items() if not (isinstance(v, dict) and 'key' in v)}
            save_yaml(copy_config, folder_name, "train_config.yaml")
        if config['verbose']:
            logger.info("Setting up holdout environments")

        train_ids, val_ids = read_split_file(config['validation'],
                                            os.path.join(script_dir, '..', 'data', 'env_samples', config['load_envs']),
                                            logs_folder)

        envs_switcher = GraphEnvSwitcher(train_ids=train_ids, val_ids=val_ids,
                                         algorithm_type="gae",
                                         switch_strategy=config['switch_strategy'], save_switch_logs=False,
                                         save_logs_transitions=config['save_logs_transitions'],
                                         save_logs_interval_train=config['save_logs_interval_train'],
                                         save_logs_interval_val=config['save_logs_interval_val'],
                                         switch_interval=config['switch_interval'], envs_folder=envs_folder)

        # === Compression baseline (random init) ===

        if config['compare_random_init']:
            baseline_losses = []

            logger.info(f"Computing compression baseline using {config['num_random_initializations']} random initializations")

            for _ in tqdm(range(config['num_random_initializations']), desc="Random inits for compression baseline"):
                # Re-init models
                rand_models, _, _ = init_models(env, config)
                rand_losses = compute_losses_no_backward(rand_models, env, config)
                baseline_losses.append(rand_losses)

            # Average baseline losses
            baseline_mean = {}
            for graph_name in baseline_losses[0]:
                baseline_mean[graph_name] = {}
                for loss_name in baseline_losses[0][graph_name]:
                    baseline_mean[graph_name][loss_name] = float(np.mean(
                        [b[graph_name][loss_name] for b in baseline_losses]
                    ))

            score = train_and_eval(config, envs_switcher, models, optimizers, writer, logger, logs_folder) # give general config since parameters used are the same
            # === Trained losses ===
            trained_losses = compute_losses_no_backward(models, envs_switcher, config)
            compression = {}

            for graph_name in trained_losses:
                compression[graph_name] = {}
                cr_vals = []

                for loss_name in trained_losses[graph_name]:
                    L0 = baseline_mean[graph_name][loss_name]
                    Lt = trained_losses[graph_name][loss_name]

                    cr = max(0.0, min(1.0, (L0 - Lt) / max(L0, 1e-8)))
                    compression[graph_name][loss_name] = cr
                    cr_vals.append(cr)

                compression[graph_name]["CR_avg"] = float(np.mean(cr_vals))
            import yaml

            compression_path = os.path.join(logs_folder, "compression_ratios.yaml")
            with open(compression_path, "w") as f:
                yaml.dump(
                    {
                        "baseline_losses": baseline_mean,
                        "trained_losses": trained_losses,
                        "compression_ratios": compression,
                    },
                    f,
                    sort_keys=False
                )

            logger.info(f"Compression ratios saved to {compression_path}")
        else:
            score = train_and_eval(config, envs_switcher, models, optimizers, writer, logger,
                                   logs_folder)  # give general config since parameters used are the same

        scores.append(score) # average val loss OR train loss if not holdout
    return np.mean(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GNN Autoencoder')
    parser.add_argument('--train_config', type=str, default=os.path.join(script_dir, '..', 'gae', 'config', 'train_config.yaml'), help='Path to the configuration YAML file')
    parser.add_argument('--algo_config', type=str, default=os.path.join(script_dir, '..', 'gae', 'config', 'algo_config.yaml'), help='Path to the algorithm configuration YAML file')
    parser.add_argument('-ti', '--train_iterations', type=int, default=100,
                        help='Number of training iterations overall')
    parser.add_argument('-e', '--environment', type=str, default='sample',
                        help='Type of environment application to be used for training')
    parser.add_argument('-m', '--model_type', type=str, default='gae', choices=['gae', 'vgae'],
                        help='Type of model to use: gae (Graph Autoencoder) or vgae (Variational Graph Autoencoder)')
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--static_seed', default=0, type=int, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--validation', default=False, action="store_true", help='Use validation set of graphs')
    parser.add_argument('--load_envs', default=False, type=str, help='Path to the .pkl file containing the graph')
    parser.add_argument('--load_envs_mode', default=False,
                        help='If present, specifies extrapolation/interpolation mode for loading environments')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('-v', '--verbose', default=2, type=int, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('-i', '--num_random_initializations', default=10, type=int,
                        help='Number of random initializations to compute compression baseline')
    parser.add_argument("--no_save", action="store_false", dest="save", default=True, help="Disable saving environments as default")
    parser.add_argument("--force_saving", action="store_true", default=False,
                        help="Force saving environment as default")
    parser.add_argument("--compare_random_init", action="store_true", default=False,
                        help="Skip computation of random initialization compression baseline")
    args = parser.parse_args()

    # Creating logs folder
    general_config = load_yaml(os.path.join(script_dir, "..", "config.yaml"))
    if not args.load_envs:
        if args.load_envs_mode == "extrapolation":
            args.load_envs = general_config[args.environment]['default_extrapolation_envs']
        elif args.load_envs_mode == "intrapolation":
            args.load_envs = general_config[args.environment]['default_intrapolation_envs']
        elif args.load_envs_mode == "cartesian":
            args.load_envs = general_config[args.environment]['default_cartesian_envs']
        else:
            args.load_envs = general_config[args.environment]['default_envs']

    if args.name:
        logs_folder = os.path.join(script_dir, 'logs/', args.name + "_" + args.model_type + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join(script_dir, 'logs/', args.model_type + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logs_folder, exist_ok=True)  # Ensure logs folder exists

    envs_folder = os.path.join(logs_folder, "envs") # folder where environments will be stored and copied
    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    if args.verbose:
        logger.info("Logs folder: %s", os.path.basename(logs_folder.rstrip('/')))

    # Potential seed setting before the generation of environment
    if args.static_seed:
        if args.verbose:
            logger.info("Setting static seeds for each run of training")
        seeds_runs = [args.static_seed for _ in range(args.num_runs)]
    elif args.random_seeds:
        if args.verbose:
            logger.info("Setting random seeds for training")
        seeds_runs = [np.random.randint(1000) for _ in range(args.num_runs)]
    else: # args.load_seeds:
        if args.verbose:
            logger.info("Loading seeds from seeds file %s", args.load_seeds)
        seeds_loaded = load_yaml(os.path.join(script_dir, args.load_seeds, 'seeds.yaml'))
        seeds_runs = seeds_loaded['seeds'][:args.num_runs]

    config = load_yaml(os.path.join(script_dir,args.train_config))
    algo_config = load_yaml(os.path.join(script_dir, args.algo_config))
    config.update(algo_config)
    # not updated if loaded configuration file from previous experiment
    config.update({"seeds": seeds_runs})
    config.update(vars(args))
    score = execute_runs(config, logs_folder, envs_folder, logger)

    if args.force_saving:
        config_path = os.path.join(script_dir, "../config.yaml")

        main_config = load_yaml(config_path)

        # Logs folder relative to gae/logs
        relative_logs = os.path.basename(logs_folder.rstrip("/"))

        # Ensure environment key exists
        if args.environment not in main_config:
            main_config[args.environment] = {}

        # Set the new gae_path
        main_config[args.environment]["gae_path"] = relative_logs

        # Save updated config.yaml
        save_yaml(main_config, os.path.dirname(config_path), os.path.basename(config_path))
    else:
        if args.save:
            # === Ask user if this should be default embedding model ===
            user_input = input(f"\nDo you want to set this trained GAE as default for '{args.environment}'? (y/n): ").strip().lower()
            if user_input in ["y", "yes"]:
                config_path = os.path.join(script_dir, "../config.yaml")

                main_config = load_yaml(config_path)

                # Logs folder relative to gae/logs
                relative_logs = os.path.basename(logs_folder.rstrip("/"))

                # Ensure environment key exists
                if args.environment not in main_config:
                    main_config[args.environment] = {}

                # Set the new gae_path
                main_config[args.environment]["gae_path"] = relative_logs

                # Save updated config.yaml
                save_yaml(main_config, os.path.dirname(config_path), os.path.basename(config_path))

                print(f"✅ Updated config.yaml: set {args.environment}.gae_path = {relative_logs}")
        else:
            print("Skipped updating default embedding model.")