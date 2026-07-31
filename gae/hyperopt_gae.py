#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

"""
    hyperopt_gae.py
    This file contains the module used to hyper-optimize the Graph Autoencoder (GAE) model.
    The file relies on the train_gae.py script to train the model with the hyperparameters sampled by Optuna.
"""

import argparse
import copy
from datetime import datetime
import os
import numpy as np
import random
import optuna
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gae.train_gae import execute_runs
from utils.file_utils import load_yaml
from utils.log_utils import setup_logging
from pathlib import Path
script_dir = Path(__file__).parent

map_name_to_sampler = {
    'grid': optuna.samplers.GridSampler,
    'random': optuna.samplers.RandomSampler,
    'tpe': optuna.samplers.TPESampler,
    'cmaes': optuna.samplers.CmaEsSampler,
    'GPS': optuna.samplers.GPSampler,
    'partial_fixed': optuna.samplers.PartialFixedSampler,
    'nsga2': optuna.samplers.NSGAIISampler,
    'nsga3': optuna.samplers.NSGAIIISampler,
    'qmc': optuna.samplers.QMCSampler,
    'bruteforce': optuna.samplers.BruteForceSampler
}

# Sample the characteristics of a layer
def sample_layer(trial, layer_template, layer_idx, is_first_layer=False, is_last_layer=False):
    layer = {}

    for k, v in layer_template.items():
        if isinstance(v, dict):
            param_name = f"layer_{layer_idx}_{k}"

            if v['type'] == 'categorical':
                if k == 'type' and is_first_layer:
                    layer[k] = 'NNConv'  # Force first layer to NNConv to include edge features
                elif k == 'activation' and is_last_layer:
                    layer[k] = 'null'  # Force last layer activation to linear/null
                else:
                    layer[k] = trial.suggest_categorical(param_name, v['values'])

            elif v['type'] == 'int':
                layer[k] = trial.suggest_int(param_name, v['min'], v['max'])

            elif v['type'] == 'uniform':
                layer[k] = trial.suggest_float(param_name, v['low'], v['high'])

            else:
                raise ValueError(f"Unsupported hyperparameter type for layer field '{k}': {v['type']}")
        else:
            layer[k] = v  # fixed value from template

    return layer


def sample_hyperparameters(trial, trial_config, hyperparams_ranges):
    # Sample top-level hyperparameters
    for k, v in hyperparams_ranges.items():
        if k in ['model_config', 'num_layers']:
            continue

        if v['type'] == 'categorical':
            trial_config['model_config'][k] = trial.suggest_categorical(k, v['values'])
        elif v['type'] == 'uniform':
            trial_config['model_config'][k] = trial.suggest_float(k, v['low'], v['high'])
        elif v['type'] == 'int':
            trial_config['model_config'][k] = trial.suggest_int(k, v['min'], v['max'])
        else:
            raise ValueError(f"Unsupported hyperparameter type for top-level field '{k}': {v['type']}")

    # Sample number of layers
    num_layers_cfg = hyperparams_ranges.get('num_layers', {'min': 1, 'max': 4})
    num_layers = trial.suggest_int("num_layers", num_layers_cfg['min'], num_layers_cfg['max'])

    # Sample architecture
    layers = []
    layer_templates = hyperparams_ranges['model_config']['layer_template']

    for i in range(num_layers):
        if i == 0:
            # First layer must be NNConv so edge features are used
            layer_template = layer_templates[0]
        else:
            # Let Optuna choose the layer type/template explicitly
            template_choices = layer_templates[1:]
            template_names = [tpl['type'] for tpl in template_choices]
            chosen_type = trial.suggest_categorical(f"layer_{i}_type", template_names)

            layer_template = next(tpl for tpl in template_choices if tpl['type'] == chosen_type)

        is_first_layer = (i == 0)
        is_last_layer = (i == num_layers - 1)

        sampled_layer = sample_layer(
            trial,
            layer_template,
            layer_idx=i,
            is_first_layer=is_first_layer,
            is_last_layer=is_last_layer
        )
        layers.append(sampled_layer)

    trial_config['model_config']['layers'] = layers
    return trial_config

# Objective function for hyperparameter optimization
def objective(trial, logs_folder, envs_folder, config, hyperparams_ranges, logger):
    trial_config = copy.deepcopy(config)
    trial_config = sample_hyperparameters(trial, trial_config, hyperparams_ranges)
    if config['verbose']:
        logger.info("Trial configuration:")
        indices_keys = ['continuous_indices', 'binary_indices', 'multi_class_info', 'node_feature_vector_size',
                        'edge_feature_vector_size', 'train_config', 'hyperparams_ranges_file']
        trial_config_copy = {key: trial_config[key] for key in trial_config if key not in indices_keys}
        logger.info(trial_config_copy)
    # Creating unique run folder
    trial_folder = os.path.join(logs_folder, f"trial_{str(trial.number + 1)}")
    if config['verbose']:
        logger.info(f"Trial folder: {os.path.basename(trial_folder.rstrip('/'))}")
    os.makedirs(trial_folder, exist_ok=True)
    # Save the modified configuration for reproducibility
    return execute_runs(trial_config, trial_folder, envs_folder, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GNN Autoencoder')
    parser.add_argument('--train_config', type=str, default=os.path.join('config', 'train_config.yaml'),
                        help='Path to the configuration YAML file')
    parser.add_argument('--algo_config', type=str, default=os.path.join('config', 'algo_config.yaml'),
                        help='Path to the algorithm configuration YAML file')
    parser.add_argument('-ti', '--train_iterations', type=int, default=100,
                        help='Number of training iterations overall')
    parser.add_argument('-e', '--environment', type=str, default='sample',
                        help='Type of environment application to be used for training')
    parser.add_argument('-m', '--model_type', type=str, default='gae', choices=['gae', 'vgae'],
                        help='Type of model to use: gae (Graph Autoencoder) or vgae (Variational Graph Autoencoder)')
    parser.add_argument("--compare_random_init", action="store_true", default=False,
                        help="Skip computation of random initialization compression baseline")
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--static_seeds', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--validation', default=False, action="store_true", help='Use validation set of graphs')
    parser.add_argument('--load_envs', default=False, type=str, help='Path to the .pkl file containing the graph')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('-v', '--verbose', default=2, type=int,
                        help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    # Hyper-parameters optimization file
    parser.add_argument('--num_trials', type=int, default=25, help='Number of runs to perform')
    parser.add_argument('--hyperparams_ranges_file', type=str,
                        default=os.path.join('config', 'hyperparams_ranges.yaml'),
                        help='Path to the hyperopt configuration YAML file')
    parser.add_argument('--optimization_type', type=str,
                        choices=['grid', 'random', 'tpe', 'cmaes', 'GPS', 'partial_fixed', 'nsga2', 'nsga3', 'qmc',
                                 'bruteforce'], default='tpe',
                        help='Type of hyperparameter optimization to use')
    args = parser.parse_args()

    general_config = load_yaml(os.path.join(script_dir, "..", "config.yaml"))
    if not args.load_envs:
        args.load_envs = general_config[args.environment]['default_envs']

    args.train_config = os.path.join(script_dir, args.train_config)
    args.hyperparams_ranges_file = os.path.join(script_dir, args.hyperparams_ranges_file)

    # Creating logs folder
    if args.name:
        logs_folder = os.path.join(script_dir, 'logs/',
                                   "hyperopt_" + args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join(script_dir, 'logs/', "hyperopt_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logs_folder, exist_ok=True)  # Ensure logs folder exists
    envs_folder = os.path.join(logs_folder, 'envs')

    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    # Potential seed setting before the generation of environment
    if args.static_seeds:
        if args.verbose:
            logger.info("Setting static seeds for each run of training")
        seeds_runs = [42 for _ in range(args.num_runs)]
    elif args.random_seeds:
        if args.verbose:
            logger.info("Setting random seeds for training")
        seeds_runs = [np.random.randint(1000) for _ in range(args.num_runs)]
    else: # load seeds
        if args.verbose:
            logger.info("Loading seeds from seeds file %s", args.load_seeds)
        seeds_loaded = load_yaml(os.path.join(script_dir, args.load_seeds, 'seeds.yaml'))
        seeds_runs = seeds_loaded['seeds'][:args.num_runs]

    config = load_yaml(args.train_config)
    algo_config = load_yaml(os.path.join(script_dir, args.algo_config))
    config.update(algo_config)
    # not updated if loaded configuration file from previous experiment
    config.update({"seeds": seeds_runs})
    config.update(vars(args))

    # Load hyperopt ranges
    hyperparams_ranges = load_yaml(args.hyperparams_ranges_file)

    if args.optimization_type in map_name_to_sampler:
        sampler = map_name_to_sampler[args.optimization_type]()
    else:
        raise ValueError("Unsupported optimization type specified:", args.optimization_type)

    study = optuna.create_study(direction='minimize', sampler=sampler, study_name='gae_hyperopt', storage=f"sqlite:///{os.path.abspath(logs_folder)}/gae_hyperopt.db", load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, logs_folder=logs_folder, envs_folder=envs_folder,
                                hyperparams_ranges=hyperparams_ranges, config=config, logger=logger),
        n_trials=config['num_trials']
    )
    if config['verbose']:
        logger.info(f"Best hyperparameters: {study.best_params}")
