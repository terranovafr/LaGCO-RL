#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

"""
    train_agent.py
    Script to train a RL algorithm on a graph-based CO task.
    Several options are available to customize the training process
"""

import argparse
import copy
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import yaml
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import pickle
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
import torch
from tqdm import tqdm
from utils.train_utils import check_args, replace_with_classes, derive_env_type_folder_name, replace_with_strings, algorithm_models, recurrent_algorithms
from utils.file_utils import load_yaml, save_yaml, extract_metric_data
from utils.log_utils import setup_logging
from utils.math_utils import set_seeds, calculate_auc, linear_schedule
from sb3_sa_contrib.policies import StateActionDQNPolicy
from sb3_sa_contrib.buffers import DictReplayBufferWithNextActions
from envs.env_switcher import GraphEnvSwitcher
import shutil
from callbacks import TrainingCallback, ValidationCallback
from wrappers.agent_training_wrapper import AgentTrainingWrapper
from gae.model import GAEEncoder, VGAEEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
torch.set_default_dtype(torch.float32)
script_dir = os.path.dirname(__file__)

# Handles the training for different runs of the algorithm and extraction of average metric values
def train_rl_algorithm(args, logs_folder, envs_folder, config, train_ids, val_ids, test_ids, metric_name=None, logger=None, verbose=1):
    metric_values = []
    if verbose:
        logger.info(f"Training algorithm {config['algorithm']} on {config['num_environments']} environments with {config['num_runs']} runs.")
    for run_id in range(config['num_runs']):

        # Varying the seed (if needed) at each run
        seed = config['seeds_runs'][run_id]
        set_seeds(seed)

        if verbose:
           logger.info(f"Run {run_id + 1}/{config['num_runs']} with seed {seed} started.")
        # in parallel case used only for validation
        envs_switcher = GraphEnvSwitcher(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                                         algorithm_type=args.algorithm_type,
                                         GNN_observations=args.GNN_observations,
                                         pca_percentage_target=args.pca_percentage_target,
                                         pca_minimum_without_loss=args.pca_minimum_without_loss,
                                         envs_folder=envs_folder,
                                         switch_strategy=config['switch_strategy'],
                                             switch_interval=config['switch_interval'],
                                             pad_spaces=config['pad_spaces'], save_embeddings=config['save_embeddings'],
                                             save_embeddings_interval_train=config['save_embeddings_interval_train'],
                                                save_embeddings_interval_val=config['save_embeddings_interval_val'],
                                             save_switch_logs=config['save_switch_logs'],
                                             save_logs_transitions=config['save_logs_transitions'],
                                             save_logs_interval_train=config['save_logs_interval_train'],
                                             save_logs_interval_val=config['save_logs_interval_val'],
                                             log_path=logs_folder)
        # if padding is enabled, get the maximum observation and action shapes across environments to use for padding, and save the configuration with string representations of the policy kwargs (to avoid saving the objects) before replacing them back with the actual classes for training
        if config['pad_spaces']:
            max_obs_shape, max_action_shape = envs_switcher.get_padding_config()
            config['padding_config'] = {
                'max_obs_shape': max_obs_shape,
                'max_action_shape': max_action_shape
            }
            config['policy_kwargs'] = replace_with_strings(config['policy_kwargs'])
            if 'envs_vectorized' in config:
                del config['envs_vectorized']
            save_yaml(config, logs_folder, "train_config.yaml")
            config['policy_kwargs'] = replace_with_classes(config['policy_kwargs'])

        train_model(envs_switcher, args, logs_folder, config, run_id+1, logger=logger, verbose=verbose)

        # extract metric data (used by hyperopt)
        if metric_name:
            if config['algorithm'] == "rppo": # handling name difference in case of recurrent PPO
                tensorboard_dir = os.path.join(logs_folder, "RecurrentPPO_" + str(run_id))
            elif config['algorithm'] == "idqn":
                tensorboard_dir = os.path.join(logs_folder, "IDQN_" + str(run_id))
            else:
                tensorboard_dir = os.path.join(logs_folder, config['algorithm'].upper() + "_" + str(run_id))

            times, values = extract_metric_data(tensorboard_dir, metric_name)
            auc = calculate_auc(times, values)
            # AUC normalized to starting value in order to normalize to initial random performances
            if verbose:
                logger.info(f"The AUC of metric {metric_name} for the run {run_id + 1} is {auc}")
            metric_values.append(auc)

    if verbose:
        logger.info("Training finished.")
    return metric_values


def train_model(switch_envs, args, logs_folder, config, run_id, logger=None, verbose=1):
    device = "cuda" if torch.cuda.is_available() and config['cuda'] else "cpu"
    if verbose:
        logger.info(f"Training on device: {device}")

    # Learning rate scheduling (potential)
    algorithm_config = copy.deepcopy(config['algorithm_hyperparams'])
    if len(algorithm_config) != 0:
        if algorithm_config['learning_rate_type'] == "linear":
            learning_rate = linear_schedule(algorithm_config['learning_rate'], algorithm_config['learning_rate_final'])
        else: # constant learning rate
            learning_rate = algorithm_config['learning_rate']

        algorithm_config.pop('learning_rate_type', None)
        algorithm_config.pop('learning_rate', None)
        algorithm_config.pop('learning_rate_final', None)
    else:
        learning_rate = None
    envs = switch_envs
    envs = DummyVecEnv([lambda: Monitor(envs)])

    switch_envs = DummyVecEnv([lambda: Monitor(switch_envs)])

    # potential normalization
    envs = VecNormalize(envs, norm_obs=config['norm_obs'], norm_reward=config['norm_reward'])
    if verbose:
        logger.info(f"Normalization of observations: {config['norm_obs']}")
        logger.info(f"Normalization of rewards: {config['norm_reward']}")

    model_class = algorithm_models[config['algorithm']]

    # different policy according to whether the algorithm is memory-based or not
    if config['algorithm'] in recurrent_algorithms:
        if config['finetune_model'] and os.path.exists(config['finetune_model']):
            model = model_class.load(config['finetune_model'], env=envs, device=device)
            if verbose:
                logger.info(f"Loaded model to finetune from {config['finetune_model']}")
        else:
            if verbose:
                logger.info("Initialized new model from scratch")

            model = model_class("MultiInputLstmPolicy", envs, policy_kwargs=config['policy_kwargs'],
                                learning_rate=learning_rate,
                                tensorboard_log=logs_folder, **algorithm_config, verbose=verbose, device=device)
    else:
        # not memory-based algorithm
        for key in ['lstm_hidden_size', 'n_lstm_layers']:
            if key in config['policy_kwargs']:
                del config['policy_kwargs'][key]
        if 'finetune_model' in config and config['finetune_model'] and os.path.exists(config['finetune_model']):
            model = model_class.load(config['finetune_model'], tensorboard_log=logs_folder, env=envs, verbose=1, learning_rate=learning_rate, device=device)
            if verbose:
                logger.info(f"Loaded model to finetune from {config['finetune_model']}")
        else:
            if verbose:
                logger.info("Initialized new model from scratch")
            if len(algorithm_config) != 0:
                if 'action_noise' in algorithm_config and algorithm_config['action_noise'] is not None:
                    action_noise = algorithm_config['action_noise']
                    del algorithm_config['action_noise']
                    n_actions = envs.action_space.shape[-1]
                    if action_noise == "ornstein_uhlenbeck":
                        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                    elif action_noise == "normal":
                        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                    algorithm_config['action_noise'] = action_noise
                else:
                    algorithm_config.pop('action_noise', None)
            model_class_args = {
                'policy_kwargs': config['policy_kwargs'],
                'tensorboard_log': logs_folder,
                'verbose': verbose,
                'device': device
            }
            if learning_rate:
                model_class_args['learning_rate'] = learning_rate
            if config['algorithm'] == 'idqn':
                policy = StateActionDQNPolicy
                model_class_args['replay_buffer_class'] = DictReplayBufferWithNextActions
            else:
                policy = "MultiInputPolicy"
            try:
                model = model_class(policy, envs,  **algorithm_config, **model_class_args)
            except Exception:
                # if cuda out of memory, try again with cpu
                device = "cpu"
                model_class_args['device'] = device
                model = model_class(policy, envs,  **algorithm_config, **model_class_args)

    # Checkpoint periodic saving
    checkpoint_callback = CheckpointCallback(save_freq=config['checkpoints_save_freq'],
                                             save_path=os.path.join(logs_folder, "checkpoints", str(run_id)),
                                             name_prefix='checkpoint')

    train_callback = TrainingCallback(env=envs)

    callbacks = [checkpoint_callback, train_callback]

    if config['validation']:
        # Logging validation metrics, saves validation checkpoints, and use early stopping if requested
        val_callback = ValidationCallback(
                    val_envs=switch_envs, # even in parallel case, there we switch
                    n_val_episodes=config['n_val_episodes'],
                    val_freq=config['val_freq'],
                    val_switch_interval=config['val_switch_interval'],
                    score_key=config['val_score_key'],
                    early_stopping=config['early_stopping'],
                    patience=config['early_stopping'],
                    gamma=algorithm_config.get('gamma',1),
                    log_dir=os.path.join(logs_folder, args.algorithm.upper() + "_" + str(run_id-1)) if config['algorithm'] != "rppo" else os.path.join(logs_folder, "RecurrentPPO_" + str(run_id-1)),
                    save_dir=os.path.join(logs_folder, "checkpoints", str(run_id)),
        )
        callbacks.append(val_callback)
    if verbose:
        logger.info(f"Training started for run {run_id} lasting {config['train_iterations']} iterations with episode length max being {config['max_steps_overall']}")

    model.learn(total_timesteps=config['train_iterations'], callback=callbacks, reset_num_timesteps=False)

def setup_train_via_args(args, general_config, logs_folder=None, envs_folder=None, suggested_params=None):
    env_type_folder_name = derive_env_type_folder_name(args)

    if not logs_folder:
        # Creating logs folder
        if args.name:
            logs_folder = os.path.join(script_dir, 'logs', args.environment, env_type_folder_name, args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            logs_folder = os.path.join(script_dir, 'logs', args.environment, env_type_folder_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(logs_folder, exist_ok=True)

    if 'finetune_model' in args and args.finetune_model:
        # find name of the largest checkpoint file inside that folder + /checkpoints/
        args.finetune_model = os.path.join(script_dir, 'logs', args.environment, env_type_folder_name, args.finetune_model)
        checkpoint_folder = os.path.join(args.finetune_model, "checkpoints", "1")
        if os.path.exists(checkpoint_folder):
            checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.startswith('checkpoint_') and f.endswith('.zip')]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                args.finetune_model = os.path.join(checkpoint_folder, latest_checkpoint)
            else:
                raise ValueError(f"No checkpoint files found in {checkpoint_folder}")

    # Setting up the logger (terminal output and or file based)
    logger = setup_logging(logs_folder, log_to_file=args.save_log_file)

    config = load_yaml(os.path.join(script_dir, args.train_config))
    algorithm_config = load_yaml(os.path.join(script_dir, args.algo_config))
    env_config = load_yaml(os.path.join(script_dir, f"../envs/config/{args.environment}_config.yaml"))

    if args.algorithm_type in algorithm_config:
        config['algorithm_hyperparams'] = algorithm_config[args.algorithm_type].get(args.algorithm, {})
    else:
        config['algorithm_hyperparams'] = algorithm_config.get(args.algorithm, {})



    config['policy_kwargs'] = algorithm_config['policy_kwargs']

    config.update(vars(args))  # put all arguments in the configuration dict only if yaml not loaded, otherwise overwrite
    config.update(env_config)

    # possible to concatenate actions in the observation vector only if present in fixed number
    if not config['sample_subset_actions']:
        config['concatenate_actions_observation'] = False
        # otherwise choice by config

    if args.environment in config:
        config.update(config[args.environment])
    if args.algorithm_type in config:
        config.update(config[args.algorithm_type])

    if suggested_params:
        config.update(suggested_params)

    if config['static_seed']:
        if config['verbose']:
            logger.info("Setting a static seed for all runs of training")
        seeds_runs = [config['static_seed'] for _ in range(config['num_runs'])]
    elif config['random_seeds']:
        if config['verbose']:
            logger.info("Setting random seeds for training")
        seeds_runs = [np.random.randint(1000) for _ in range(config['num_runs'])]
    else: # load_seeds case
        if config['verbose']:
            logger.info("Loading seeds from seeds file %s", config['load_seeds'])
        seeds_loaded = load_yaml(os.path.join(script_dir, args.load_seeds, 'seeds.yaml'))
        seeds_runs = seeds_loaded['seeds'][0:config['num_runs']]

    # Saving seeds for reproducibility
    save_yaml({"seeds": seeds_runs}, logs_folder, "seeds.yaml")
    config.update({"seeds_runs": seeds_runs})
    set_seeds(seeds_runs[0]) # do it before initial setting steps

    # Load the graph encoder
    if args.algorithm_type == "projection" or args.algorithm_type == "iterative" or ("discrete" in args.algorithm_type and config['GNN_observations']): # only if using graph-based observations, otherwise not needed
        graph_names = []
        for element in os.listdir(os.path.join(script_dir,"..", "gae", "logs", general_config[args.environment]['gae_path'])):
            if os.path.isdir(os.path.join(script_dir,"..", "gae", "logs", general_config[args.environment]['gae_path'], element)) and element.endswith("graph"):
                graph_names.append(element)
        encoders_config = {}
        encoders_spec = {}
        encoders = {}
        for graph_name in graph_names:
            # prefer validation set checkpoint if available and if validation is enabled, otherwise take final checkpoint (best val or final depending on availability)
            if 'encoder_best_val.pth' in os.listdir(os.path.join(script_dir, "..", "gae", "logs", general_config[args.environment]['gae_path'], graph_name)) and args.validation:
                graph_encoder_path = os.path.join(script_dir, "..", "gae", "logs", general_config[args.environment]['gae_path'], graph_name,
                                                  "encoder_best_val.pth")
            else:
                graph_encoder_path = os.path.join(script_dir, "..", "gae", "logs", general_config[args.environment]['gae_path'], graph_name,
                                              "encoder_final.pth")
            graph_encoder_config_path = os.path.join(script_dir, "..", "gae", "logs",
                                                     general_config[args.environment]['gae_path'], graph_name, "train_config.yaml")
            graph_encoder_spec_path = os.path.join(script_dir, "..", "gae", "logs",
                                                   general_config[args.environment]['gae_path'], graph_name, "model_spec.yaml")
            encoder_config = load_yaml(graph_encoder_config_path)
            encoder_spec = load_yaml(graph_encoder_spec_path)
            if encoder_config['model_type'] == 'gae':
                graph_encoder = GAEEncoder(encoder_spec['node_feature_vector_size'],
                                           encoder_config['model_config']['layers'],
                                           encoder_spec['edge_feature_vector_size'])
            elif encoder_config['model_type'] == 'vgae':
                graph_encoder = VGAEEncoder(encoder_spec['node_feature_vector_size'],
                                            encoder_config['model_config']['layers'],
                                            encoder_spec['edge_feature_vector_size'], latent_dim=32)
            else:
                graph_encoder = None
            if graph_encoder is not None:
                graph_encoder.load_state_dict(torch.load(str(graph_encoder_path)))
                graph_encoder.eval()
                encoders_config[graph_name] = encoder_config
                encoders_spec[graph_name] = encoder_spec
                encoders[graph_name] = graph_encoder
                config.update({"gae_model_type": encoder_config['model_type']})
                config.update({f"graph_encoder_path_{graph_name}": graph_encoder_path})
                config.update({f"graph_encoder_config_path_{graph_name}": graph_encoder_config_path})
                config.update({f"graph_encoder_spec_path_{graph_name}": graph_encoder_spec_path})
    else:
        encoders = None
    # Saving configurations to ensure reproducibility
    copy_config = {k: v for k, v in config.items() if not (isinstance(v, dict) and 'key' in v)}
    save_yaml(copy_config, logs_folder, "train_config.yaml")
    if config['load_envs']: # environments to be processed from a folder in samples
        if config['verbose']:
            logger.info("Loading environments from the folder %s", config['load_envs'])

        original_envs_folder = os.path.join(script_dir, '..', 'data', 'env_samples', config['load_envs'])

        if not envs_folder: # destination folder where to have the processed environments
            envs_folder = os.path.join(logs_folder, "envs")
        if not os.path.exists(os.path.join(envs_folder)):
            os.makedirs(os.path.join(envs_folder))

        config['num_environments'] = 0
        # make copy of general_config leaving only the dicts that have "key" as key
        keys_config = {k: v for k, v in general_config.items() if isinstance(v, dict) and 'key' in v}
        for element in tqdm(os.listdir(original_envs_folder), desc="Loading environments from the folder"):
            if os.path.isfile(os.path.join(original_envs_folder, element)):
                with open(os.path.join(envs_folder, element), 'wb') as f: # saving into the destination folder
                    shutil.copyfile(os.path.join(original_envs_folder, element), os.path.join(envs_folder, element))
            if os.path.isdir(os.path.join(original_envs_folder,element)):
                config['num_environments'] += 1
                with open(os.path.join(original_envs_folder, element, "env.pkl"), 'rb') as f:
                    environment = pickle.load(f)
                # if in that folder there is also config.yaml, copy it as well
                if os.path.exists(os.path.join(original_envs_folder, element, "config.yaml")):
                    shutil.copyfile(os.path.join(original_envs_folder, element, "config.yaml"),
                                    os.path.join(envs_folder, "config.yaml"))
                env_config = load_yaml(
                    os.path.join(script_dir, "..", "envs", "config", f"{args.environment}_config.yaml"))
                # some train configuration should condition environment
                env_config['verbose'] = config['verbose']
                env_config['max_steps_overall'] = config['max_steps_overall']
                env_config['max_steps_coefficient'] = config['max_steps_coefficient']
                env_config['max_sweeps'] = config['max_sweeps']
                env_config['save_logs_transitions'] = config['save_logs_transitions']
                env_config['remove_invalid_actions'] = config['remove_invalid_actions']
                env_config['terminate_at_approximate_best'] = config['terminate_at_approximate_best']
                env_config['padding_invalid_action_penalty_sum'] = config['padding_invalid_action_penalty_sum']
                env_config['no_action_penalty_sum'] = config['no_action_penalty_sum']
                env_config['no_action_support'] = config['no_action_support']
                for key in keys_config:
                    env_config[key] = keys_config[key]  # add to environment configuration the values of the general configuration that are relevant for the environment (e.g. with same key)
                environment.update_config(env_config)  # updating environment configuration
                if suggested_params:
                    env_config.update(suggested_params)
                # Use the network graph environment and map it to a C-CyberBattleSim environment
                if args.algorithm_type == "projection" or args.algorithm_type == "iterative"  or ("discrete" in args.algorithm_type and config['GNN_observations']):
                    config.pop('logs_folder', None)  # to avoid recursion in the wrapper
                    environment.update_spec()
                    environment = AgentTrainingWrapper(env=environment, env_name=args.environment,
                                                   gnn_models=encoders, logs_folder=logs_folder,  **config)
                with open(os.path.join(envs_folder, f"{element}.pkl"),'wb') as f:  # saving into the destination folder
                    pickle.dump(environment, f)

        if config['verbose']:
            logger.info("Loaded and wrapped %d environments from the folder", config['num_environments'])

    # Splitting environment in case of holdout method
    yaml_split_path = os.path.join(envs_folder, "split.yaml")
    with open(yaml_split_path, 'r') as file:
        yaml_info = yaml.safe_load(file)
    if config['validation']:
        train_ids = []
        for elem in yaml_info['training_set']:
            train_ids.append(elem['id'])
        val_ids = []
        for elem in yaml_info['validation_set']:
            val_ids.append(elem['id'])
        test_ids = []
        for elem in yaml_info.get('test_set', []):
            test_ids.append(elem['id'])
    else:
        train_ids = []
        for elem in yaml_info['training_set']:
            train_ids.append(elem['id'])
        for elem in yaml_info['validation_set']:
            train_ids.append(elem['id'])
        val_ids = []
        test_ids = []
        for elem in yaml_info.get('test_set', []):
            test_ids.append(elem['id'])

    # save split information in logs
    save_yaml(yaml_info, logs_folder, "split.yaml")

    # map simple values to classes after saving the configuration file (to avoid saving the object)
    config['policy_kwargs'] = replace_with_classes(config['policy_kwargs'])

    return logger, logs_folder, envs_folder, config, train_ids, val_ids, test_ids



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL algorithm on a Graph-Based Environment!")
    parser.add_argument('-algo', '--algorithm', type=str, choices=['ppo', 'a2c', 'rppo', 'trpo', 'ddpg', 'sac', 'td3', 'tqc', 'maskable_ppo', 'idqn'], default='ppo',  help='RL algorithm to train')
    parser.add_argument('-at', '--algorithm_type', type=str, choices=['projection', 'iterative', 'discrete'], default='projection', help='Type of approximator to be used for training') # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument('-GO', '--GNN_observations', action='store_true', default=False, help='Whether to use graph-based observations for discrete algorithms (if not, use vectorized observations)')
    parser.add_argument('-s', '--semantic_ordering', action='store_true', default=False,
                        help='Whether to use semantic ordering for the action space')
    parser.add_argument('-e', '--environment', type=str, default='tsp',
                        help='Type of environment application to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument("--use_feature_vectors", action='store_true', default=False,
                        help="Whether to use feature vectors as action points in the case of continuous action spaces instead of embeddings")
    parser.add_argument('-ti', '--train_iterations', type=int, default=1000,
                        help='Number of training iterations overall')
    parser.add_argument("-pca", "--pca_percentage_target", type=float, default=1,
                        help="Do PCA dimensionality reduction on the action space")
    parser.add_argument("-ssa", "--sample_subset_actions", type=int, default=0,
                        help="Whether to sample a subset of actions at each step for the agent to choose from (only for continuous action spaces, and if > 0)")
    parser.add_argument("-pca_min", "--pca_minimum_without_loss", default=False, action='store_true',
                        help="Compute minimum number of components to not have loss in PCA reconstruction")
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--validation', action='store_true', default=False, help='Periodically evaluate on validation sets')
    parser.add_argument('--finetune_model', type=str, help='Path to the model to eventually finetune (relative to the logs folder)')
    parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping on the validation environments setting the number of patience runs')
    parser.add_argument('--name', default=False, help='Name of the logs folder related to the run')
    parser.add_argument('--load_envs', default=False, help='Path of the envs folder where the networks should be processed and loaded from')
    parser.add_argument('--load_envs_mode', default=False, choices=[False, 'extrapolation', 'interpolation', 'cartesian'],
                        help='If present, specifies mode for loading environments (e.g. other types of splits)')
    parser.add_argument('--static_seed', default=0, type=int, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config", help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seeds', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('-v', '--verbose', default=2, type=int, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('-f', '--logs_folder', type=str, default=None,
                        help='Path to the logs folder where to save training information')
    parser.add_argument("-approx", "--approximate_distance", default=False,
                        type=bool, help="Approximate distance for the projection approach")
    parser.add_argument('-gae', '--gae_folder', type=str, default=None,
                        help='Path to the gae folder to load, optional alternative to general config')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument('--no_save_switch_logs', action='store_false', dest='save_switch_logs',
                        default=True, help='Disable saving environment switch logs')
    parser.add_argument('--save_embeddings', action='store_true',
                        default=False, help='Save evolution of the observation vector periodically')
    parser.add_argument('--no_cuda', action='store_false', dest='cuda',
                        default=True, help='Disable use of cuda even if available')
    parser.add_argument('--train_config', type=str, default='config/train_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--algo_config', type=str, default='config/algo_config.yaml',
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    general_config = load_yaml(os.path.join(script_dir, "..", "config.yaml"))
    if not args.load_envs:
        if args.load_envs_mode == "extrapolation":
            args.load_envs = general_config[args.environment]['default_extrapolation_envs']
        elif args.load_envs_mode == "interpolation":
            args.load_envs = general_config[args.environment]['default_interpolation_envs']
        elif args.load_envs_mode == "cartesian":
            args.load_envs = general_config[args.environment]['default_cartesian_envs']
        else:
            args.load_envs = general_config[args.environment]['default_envs']

    # overwrite if defined
    if args.gae_folder:
        general_config[args.environment]['gae_path'] = args.gae_folder

    if args.algorithm_type == 'iterative':
        args.algorithm = 'idqn'  # to use the specific DQN with action reconstruction loss for that type of environment

    # Check consistency of arguments
    valid, message = check_args(args)
    if not valid:
        raise ValueError(message)

    logger, logs_folder, envs_folder, config, train_ids, val_ids, test_ids = setup_train_via_args(args, general_config, args.logs_folder)
    # Train the RL algorithm with the provided configuration
    train_rl_algorithm(args, logs_folder, envs_folder, config, train_ids, val_ids, test_ids, logger=logger, verbose=args.verbose)
