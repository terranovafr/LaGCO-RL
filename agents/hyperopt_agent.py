#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

"""
    hyperopt_agent.py
    Script to hyper-optimize an algorithm on a set of environments with Optuna.
"""

import argparse
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agents.train_agent import setup_train_via_args, train_rl_algorithm
from utils.train_utils import check_args, replace_with_strings
import numpy as np
from datetime import datetime
import optuna
from utils.file_utils import save_yaml, load_yaml
script_dir = os.path.dirname(__file__)

# sampler of hyper-parameters
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

# metric to consider in the case of validation or not
metrics = {
    "validation" : "validation/undiscounted_return",
    "train": "rollout/ep_rew_mean",
}


# suggest hyperparams based on type of range
def suggest_hyperparameters(trial, hyperparam_ranges):
    suggested_params = {}
    if len(hyperparam_ranges) == 0:
        return suggested_params
    for param_name, param_config in hyperparam_ranges.items():
        print("Suggesting param:", param_name, "with config:", param_config)
        if param_config['type'] == 'categorical':
            suggested_params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
        elif param_config['type'] == 'float':
            suggested_params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'],
                                                               log=param_config.get('log', False))
        elif param_config['type'] == 'int':
            suggested_params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
    return suggested_params

def objective(trial, algo_hyperparams_ranges, hyperparams_ranges, args, original_logs_folder, general_config):
    # sample a set of hyperparams
    algo_suggested_params = suggest_hyperparameters(trial, algo_hyperparams_ranges)
    suggested_params = suggest_hyperparameters(trial, hyperparams_ranges)

    print(f"Training {args.algorithm} with suggested algo params:")
    print(algo_suggested_params)
    print("and general params:")
    print(suggested_params)

    # Loop all the goals and merge the results
    metrics_across_envs = []
    for env_name in args.environment:
        envs_folder = os.path.join(original_logs_folder, env_name, "envs")
        logs_folder = os.path.join(original_logs_folder, env_name,  args.algorithm.upper() + "_trial_" + str(trial.number))
        os.makedirs(logs_folder, exist_ok=True)
        final_suggested_params = {}
        for param_name, value in suggested_params.items():
            final_suggested_params[param_name] = value
        for param_name, value in algo_suggested_params.items():
            final_suggested_params[param_name] = value

        save_yaml(final_suggested_params, logs_folder, "hyperparams.yaml")
        args_copy = copy.deepcopy(args)
        args_copy.environment = env_name
        args_copy.load_envs = args_copy.load_envs[env_name]
        logger, logs_folder, _, config, train_ids, val_ids, test_ids = setup_train_via_args(args_copy, general_config, logs_folder, envs_folder=envs_folder, suggested_params=suggested_params)
        for param_name, value in algo_suggested_params.items():
            config['algorithm_hyperparams'][param_name] = value
        for param_name, value in suggested_params.items():
            config[param_name] = value

        copy_config = copy.deepcopy(config)
        copy_config['policy_kwargs'] = replace_with_strings(copy_config['policy_kwargs'])
        save_yaml(copy_config, logs_folder, "config.yaml")
        config['algorithm_hyperparams']['learning_rate_type'] = "constant"
        # Different metric name according to set and environment
        if args.validation:
            metric_set = 'validation'
        else:
            metric_set = 'train'
        env_registry = load_yaml(os.path.join(script_dir, "..", "envs", "config", "env_registry.yaml"))
        if args.validation:
            config['metric_name'] = metric_set + "/" + "_".join(
                env_registry[env_name]['score_key'].split("_")[1:])  # remove agent
        else:
            config['metric_name'] = metric_set+"/Mean "+"_".join(env_registry[env_name]['score_key'].split("_")[1:]) # remove agent
        runs_metric = train_rl_algorithm(args, logs_folder, envs_folder, config, train_ids, val_ids, test_ids, logger=logger, verbose=args.verbose, metric_name=config['metric_name'])
        avg_metric = np.mean(runs_metric)
        metrics_across_envs.append(avg_metric)
    # average AUC of the metric across all runs for the trial
    print("Metric: ", config['metric_name'], ":", metrics_across_envs)
    return np.mean(metrics_across_envs)

def hyperopt_rl(algo_hyperparams_ranges, hyperparams_ranges, args, logs_folder, general_config):
    if args.optimization_type in map_name_to_sampler:
        sampler = map_name_to_sampler[args.optimization_type]()
    else:
        raise ValueError("Unsupported optimization type specified:", args.optimization_type)

    # Create study or overwrite if it exists
    os.makedirs(logs_folder, exist_ok=True)
    # replace logs folder with relative path to script dir not full
    partial_logs_folder = os.path.relpath(logs_folder, script_dir)
    study = optuna.create_study(direction=args.direction, sampler=sampler, storage=os.path.join('sqlite:///', str(os.path.join(partial_logs_folder, args.name +'.db'))), study_name=args.name, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, algo_hyperparams_ranges, hyperparams_ranges, args, logs_folder, general_config), n_trials=args.num_trials)

    print(f"Best parameters found: {study.best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-optimize RL Agent for C-CyberBattleSim environments")
    # same code of train_agent but with hyperopt logic
    parser.add_argument('-algo', '--algorithm', type=str,
                        choices=['ppo', 'a2c', 'rppo', 'trpo', 'ddpg', 'sac', 'td3', 'tqc', 'maskable_ppo', 'idqn'],
                        default='ppo', help='RL algorithm to train')
    parser.add_argument('-at', '--algorithm_type', type=str, choices=['projection', 'iterative', 'discrete'],
                        default='projection',
                        help='Type of approximator to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument('-GO', '--GNN_observations', action='store_true', default=False,
                        help='Whether to use graph-based observations for discrete algorithms (if not, use vectorized observations)')
    parser.add_argument('-s', '--semantic_ordering', action='store_true', default=False,
                        help='Whether to use semantic ordering for the action space')
    parser.add_argument('-e', '--environment', type=str, default=['tsp'], nargs='+',
                        help='Type of environment application to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument("-m", "--mode", type=str, choices=["parallel", "switch"], default="switch",
                        help="Training mode: parallel or switch")
    parser.add_argument('-ti', '--train_iterations', type=int, default=1000,
                        help='Number of training iterations overall')
    parser.add_argument("-pca", "--pca_percentage_target", type=float, default=1,
                        help="Do PCA dimensionality reduction on the action space")
    parser.add_argument("-ssa", "--sample_subset_actions", type=int, default=0,
                        help="Whether to sample a subset of actions at each step for the agent to choose from (only for continuous action spaces, and if > 0)")
    parser.add_argument("-pca_min", "--pca_minimum_without_loss", default=False, action='store_true',
                        help="Compute minimum number of components to not have loss in PCA reconstruction")
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--validation', action='store_true', default=False,
                        help='Periodically evaluate on validation sets')
    parser.add_argument("-approx", "--approximate_distance", default=False,
                        type=bool, help="Approximate distance for the projection approach")
    parser.add_argument('--finetune_model', type=str,
                        help='Path to the model to eventually finetune (relative to the logs folder)')
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Early stopping on the validation environments setting the number of patience runs')
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--load_envs', default=False,
                        help='Path of the envs folder where the networks should be processed and loaded from')
    parser.add_argument('--load_envs_mode', default=False,
                        choices=[False, 'extrapolation', 'interpolation', 'cartesian'],
                        help='If present, specifies mode for loading environments (e.g. other types of splits)')
    parser.add_argument('--static_seed', default=0, type=int, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('-v', '--verbose', default=2, type=int,
                        help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('-f', '--logs_folder', type=str, default=None,
                        help='Path to the logs folder where to save training information')
    parser.add_argument('-gae', '--gae_folder', type=str, default=None,
                        help='Path to the gae folder to load, optional alternative to general config')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('--no_save_switch_logs', action='store_false', dest='save_switch_logs',
                        default=True, help='Disable saving environment switch logs')
    parser.add_argument('--save_embeddings', action='store_true',
                        default=False, help='Save evolution of the observation vector periodically')
    parser.add_argument("--use_feature_vectors", action='store_true', default=False,
                        help="Whether to use feature vectors as action points in the case of continuous action spaces")
    parser.add_argument('--no_cuda', action='store_false', dest='cuda',
                        default=True, help='Disable use of cuda even if available')
    parser.add_argument('--train_config', type=str, default='config/train_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--algo_config', type=str, default='config/algo_config.yaml',
                        help='Path to the configuration YAML file')
    # hyperopt specific arguments
    parser.add_argument('--optimization_type', type=str, choices=['grid', 'random', 'tpe', 'cmaes', 'GPS', 'partial_fixed', 'nsga2', 'nsga3', 'qmc', 'bruteforce'], default='tpe',
                        help='Type of hyperparameter optimization to use')
    parser.add_argument('--algo_hyperparams_ranges_file', type=str,
                        default="algo_hyperparams_ranges.yaml",
                        help='Path to YAML file specifying hyperparameter ranges')
    parser.add_argument('--hyperparams_ranges_file', type=str,
                        default="hyperparams_ranges.yaml",
                        help='Path to YAML file specifying hyperparameter ranges')
    parser.add_argument('--direction', type=str, default="maximize", choices=['maximize', 'minimize'],
                        help='Direction of the optimization (maximize or minimize)')
    parser.add_argument('--num_trials', type=int, default=50, help='Number of trials for hyperparameter optimization')
    args = parser.parse_args()

    general_config = load_yaml(os.path.join(script_dir, "..", "config.yaml"))
    if not args.load_envs:
        args.load_envs = {}
        for env_name in args.environment:
            if not args.load_envs:
                if args.load_envs_mode == "extrapolation":
                    args.load_envs[env_name] = general_config[env_name]['default_extrapolation_envs']
                elif args.load_envs_mode == "interpolation":
                    args.load_envs[env_name] = general_config[env_name]['default_interpolation_envs']
                elif args.load_envs_mode == "cartesian":
                    args.load_envs[env_name] = general_config[env_name]['default_cartesian_envs']
                else:
                    args.load_envs[env_name] = general_config[env_name]['default_envs']

    check_args(args)

    if args.algorithm_type == 'iterative':
        args.algorithm = 'idqn'  # to use the specific DQN with action reconstruction loss for that type of environment

    # read hyperparams ranges
    algo_hyperparams_ranges = load_yaml(os.path.join(script_dir, "config", args.algo_hyperparams_ranges_file))
    hyperparams_ranges = load_yaml(os.path.join(script_dir, "config", args.hyperparams_ranges_file))
    if not hyperparams_ranges:
        hyperparams_ranges = {}

    # read only those of the target algorithms
    algo_hyperparams_ranges = algo_hyperparams_ranges.get(args.algorithm, {})
    # add those specific to the type of algorithm
    if args.algorithm_type in hyperparams_ranges:
        hyperparams_ranges = hyperparams_ranges[args.algorithm_type]
    elif "default" in hyperparams_ranges:
        hyperparams_ranges = hyperparams_ranges["default"]
    else:
        hyperparams_ranges = {}

    if len(args.environment) == 1:
        # add those of the environment (if hyperopt only involves one type)
        if args.environment[0] in hyperparams_ranges:
            hyperparams_ranges = hyperparams_ranges[args.environment[0]]
        elif "default" in hyperparams_ranges:
            hyperparams_ranges = hyperparams_ranges["default"]
    else:
        hyperparams_ranges = hyperparams_ranges.get("default", {})

    if args.name:
        # save in general path as it can use many environments
        logs_folder = os.path.join(script_dir, 'logs',
                                   "hyperopt_" + args.algorithm_type + "_" + args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join(script_dir, 'logs', "hyperopt_" + args.algorithm_type + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    if not args.name:
        args.name = "hyperopt_" + args.algorithm

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    save_yaml(algo_hyperparams_ranges, logs_folder, "algo_hyperparams_ranges.yaml")
    save_yaml(hyperparams_ranges, logs_folder, "hyperparams_ranges.yaml")

    hyperopt_rl(algo_hyperparams_ranges, hyperparams_ranges, args, logs_folder, general_config)
