#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

"""
    test_agent.py
    Script to test a RL algorithm on a given environment using the trained model.
    Several options are present to assess the agent's performance.
"""

import numpy as np
import re
import datetime
import argparse
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import yaml
from datetime import datetime
import pickle
import math
from pathlib import Path
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
from tqdm import tqdm
from wrappers.agent_training_wrapper import AgentTrainingWrapper
from gae.model import GAEEncoder, VGAEEncoder
script_dir = Path(__file__).parent
from utils.train_utils import algorithm_models, derive_env_type_folder_name
from utils.file_utils import load_yaml, remove_folder_and_files
from utils.log_utils import setup_logging
from utils.math_utils import set_seeds
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
from utils.test_utils import calculate_average_performances, summarize_df, calculate_average_action_time
from envs.env_switcher import GraphEnvSwitcher

def parse_option(config, envs, logger, verbose):
    # Determine where the checkpoint to load should be a training or a validation one
    if config['val_checkpoints']:
        if verbose:
            logger.info("Focusing on validation checkpoints...")
    else:
        if verbose:
            logger.info("Focusing on training checkpoints...")
    checkpoints_folder = "checkpoints"
    all_outcomes = {}

    # Iterate over the different runs in the folder
    for internal_run in os.listdir(os.path.join(config['run_folder'], checkpoints_folder)):

        if config['val_checkpoints']:
            test_folder = os.path.join("test", config['test_folder'], "validation")
        else:
            test_folder = os.path.join("test", config['test_folder'], "train")

        config['run_id'] = internal_run

        if not os.path.exists(os.path.join(config['run_folder'], test_folder, str(config['run_id']))):
            os.makedirs(os.path.join(config['run_folder'], test_folder, str(config['run_id'])))

        # Load the proper checkpoint file(s)
        if config['val_checkpoints']:
            checkpoint_files = [file for file in os.listdir(
                os.path.join(config['run_folder'], checkpoints_folder, str(config['run_id']))) if
                                file.startswith("best_model_")]
            episode_pattern = re.compile(r'best_model_(-?\d+\.?\d*)')
            checkpoint_files.sort(key=lambda x: float(episode_pattern.search(x).group(1)))
        else:
            checkpoint_files = [file for file in os.listdir(
                os.path.join(config['run_folder'], checkpoints_folder, str(config['run_id']))) if
                                file.startswith("checkpoint_")]

            episode_pattern = re.compile(r'checkpoint_(-?\d+)')
            checkpoint_files.sort(key=lambda x: int(episode_pattern.search(x).group(1)))

        checkpoints = []

        # Case of a single checkpoint: the last one
        if config['val_checkpoints']:
            config['last_checkpoint'] = True # enforced due to merging reasons, otherwise we cannot merge checkpoints without knowing the timestep (which is available in the training case)

        if config['last_checkpoint']:
            # last checkpoint only
            if len(checkpoint_files) > 0:
                if verbose:
                    logger.info("Focusing on the last checkpoint only...")
                checkpoints.append(os.path.join(config['run_folder'], checkpoints_folder, str(config['run_id']),
                                                checkpoint_files[-1]))
        else:
            # use all checkpoints ordered
            if verbose:
                logger.info("Focusing on all checkpoints...")
            for checkpoint_file in checkpoint_files:
                checkpoints.append(
                    os.path.join(config['run_folder'], checkpoints_folder, str(config['run_id']), checkpoint_file))

        # Collect corresponding ProtoKNN checkpoints if enabled
        proto_knn_files = []
        if config['proto_knn']:
            proto_knn_files = sorted(
                [f for f in os.listdir(os.path.join(config['proto_knn_path'], "checkpoints", "1")) if f.startswith("checkpoint_")],
                key=lambda x: int(re.search(r'checkpoint_(\d+)', x).group(1)))
            if config['last_checkpoint']:
                proto_knn_files = [proto_knn_files[-1]]

        if len(checkpoints) == 0:
            if verbose:
                logger.error("No checkpoint to load from the folder")

        outcomes = {} # to be averaged with the other runs

        for checkpoint_index in range(len(checkpoints)):
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")

            checkpoint_path = checkpoints[checkpoint_index]
            checkpoint_short_name = "best_val"
            # Fail safe in case of GPU loading issues, try to load on GPU if available and enabled, otherwise load on CPU
            try:
                device = "cuda" if torch.cuda.is_available() and config['cuda'] else "cpu"
                model = algorithm_models[config['algorithm']].load(checkpoint_path, device=device)
            except Exception:
                print("GPU not available or error loading the model on GPU, loading on CPU instead...")
                model = algorithm_models[config['algorithm']].load(checkpoint_path, device="cpu")
            model.set_env(envs)
            if verbose:
                logger.info("Focusing on the checkpoint: %s", checkpoint_path)

            if config['proto_knn'] and len(proto_knn_files) > 0:
                # Pick proto checkpoint with closest or matching iteration
                proto_file = proto_knn_files[min(checkpoint_index, len(proto_knn_files) - 1)]
                proto_path = os.path.join(config['proto_knn_path'], "checkpoints", "1", proto_file)
                proto_knn = algorithm_models['idqn'].load(proto_path)
                proto_knn.set_env(envs)
                config['knn'] = config['proto_knn']
            else:
                # leave knn at its default value with optional heuristic
                proto_knn = None

            # Tune save folder name based on the test folder and checkpoint
            save_folder = os.path.join(config['run_folder'], test_folder, str(config['run_id']))
            if config['load_custom_test_envs'] or config['load_default_test_envs']:
                save_folder = os.path.join(save_folder, "test_set")
            elif config['load_custom_val_envs'] or config['load_default_val_envs']:
                save_folder = os.path.join(save_folder, "validation_set")
            elif config['load_custom_train_envs'] or config['load_default_train_envs']:
                save_folder = os.path.join(save_folder, "training_set")
            if len(checkpoints) > 1:
                save_folder = os.path.join(save_folder, f"checkpoint_{checkpoint_index}")

            if config['proto_knn']:
                save_folder = os.path.join(save_folder, f"proto_knn={config['proto_knn']}")

            if not config['custom_test_folder_name']:
                save_folder = os.path.join(save_folder, current_time)
            else:
                save_folder = os.path.join(save_folder, config['custom_test_folder_name'] + "_" + current_time)

            os.makedirs(save_folder, exist_ok=True)

            if config['option'] == "agent_performances":
                # Compute average performances and save them in a CSV file, also save a summary YAML file
                (df, metrics) = calculate_average_performances(
                    model, envs, config['seeds_test'], proto_knn, num_envs=config['num_envs'], num_episodes=config['num_episodes_per_checkpoint'], avoid_random=not config['add_random'], logger=logger, verbose=verbose)

                df.to_csv(os.path.join(save_folder, f"average_performances_{checkpoint_short_name}_{config['num_episodes_per_checkpoint']}.csv"), index=False)
                summarize_df(df, save_folder=save_folder)

                outcomes[checkpoint_short_name] = {}
                for metric in metrics:
                    outcomes[checkpoint_short_name][metric] = metrics[metric]

                all_outcomes[checkpoint_short_name] = {}
                for metric in metrics:
                    if metric not in all_outcomes[checkpoint_short_name]:
                        all_outcomes[checkpoint_short_name][metric] = metrics[metric]
                    else:
                        all_outcomes[checkpoint_short_name][metric].extend(metrics[metric])
            elif config['option'] == "action_time":
                # Compute average action time and episode time and save them in YAML files
                action_times, episode_times = calculate_average_action_time(model, envs, config, proto_knn,
                                              logger=logger, num_episodes=config['num_episodes_per_checkpoint'], verbose=verbose)

                with open(os.path.join(save_folder, f"action_times_{config['num_episodes_per_checkpoint']}.yaml"), 'w', newline='') as file:
                    yaml.dump(action_times, file)
                # condensed statistics
                with open(os.path.join(save_folder, f"action_times_summary_{config['num_episodes_per_checkpoint']}.yaml"), 'w', newline='') as file:
                    yaml.dump({
                        "mean": float(np.mean(action_times)),
                        "std": float(np.std(action_times, ddof=1)),
                        "max": float(np.max(action_times)),
                        "min": float(np.min(action_times))
                    }, file)
                with open(os.path.join(save_folder, f"episode_times_{config['num_episodes_per_checkpoint']}.yaml"), 'w', newline='') as file:
                    yaml.dump(episode_times, file)
                with open(os.path.join(save_folder, f"episode_times_summary_{config['num_episodes_per_checkpoint']}.yaml"), 'w', newline='') as file:
                    yaml.dump({
                        "mean": float(np.mean(episode_times)),
                        "std": float(np.std(episode_times, ddof=1)),
                        "max": float(np.max(episode_times)),
                        "min": float(np.min(episode_times))
                    }, file)
                print("Mean action time: ", np.mean(action_times))
                print("Mean episode time: ", np.mean(episode_times))
    return all_outcomes

# Load test environments and wrap it with the proper features
def load_test_envs(run_folder, args, train_config, test_config, logs_folder, logger=None, stable_baselines=True):
    test_ids = []
    envs_folder = None
    if args.load_default_test_envs:
        with open(os.path.join(run_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['test_set']:
            test_ids.append(str(elem['id']))
        envs_folder = os.path.join(run_folder, "envs")
    elif args.load_default_train_envs:
        with open(os.path.join(run_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['training_set']:
            test_ids.append(str(elem['id']))
        envs_folder = os.path.join(run_folder, "envs")
    elif args.load_default_val_envs:
        with open(os.path.join(run_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['validation_set']:
            test_ids.append(str(elem['id']))
        envs_folder = os.path.join(run_folder, "envs")
    elif args.load_custom_test_envs:
        envs_folder = os.path.join(script_dir, '..', 'data', 'env_samples', args.load_custom_test_envs)
        with open(os.path.join(envs_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['test_set']:
            test_ids.append(str(elem['id']))
    elif args.load_custom_val_envs:
        envs_folder = os.path.join(script_dir, '..', 'data', 'env_samples', args.load_custom_val_envs)
        with open(os.path.join(envs_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['validation_set']:
            test_ids.append(str(elem['id']))
    elif args.load_custom_train_envs:
        envs_folder = os.path.join(script_dir, '..', 'data', 'env_samples', args.load_custom_train_envs)
        with open(os.path.join(envs_folder, "split.yaml"), 'r') as file:
            yaml_info = yaml.safe_load(file)
        for elem in yaml_info['training_set']:
            test_ids.append(str(elem['id']))
    elif args.load_custom_envs:
        envs_folder = os.path.join(script_dir, '..', 'data', 'env_samples', args.load_custom_envs)
        for elem in os.listdir(envs_folder):
            if os.path.isdir(os.path.join(envs_folder, elem)):
                test_ids.append(str(elem))
    # Setting up the proper GAE
    encoders = {}
    if args.algorithm_type == "projection" or args.algorithm_type == "iterative" or (
            "discrete" in args.algorithm_type and train_config[
        'GNN_observations']):  # only if using graph-based observations, otherwise not needed

        general_config = load_yaml(os.path.join(script_dir, "..", "config.yaml"))
        if args.gae_folder:
            general_config[args.environment]['gae_path'] = args.gae_folder
        # overwrite if defined
        graph_names = []
        for element in os.listdir(
                os.path.join(script_dir, "..", "gae", "logs", general_config[args.environment]['gae_path'])):
            if os.path.isdir(os.path.join(script_dir, "..", "gae", "logs", general_config[args.environment]['gae_path'],
                                          element)) and element.endswith("graph"):
                graph_names.append(element)
        encoders_config = {}
        encoders_spec = {}
        encoders = {}
        for graph_name in graph_names:
            gae_config = load_yaml(
                os.path.join(script_dir, "..", "gae", "logs", general_config[args.environment]['gae_path'],
                             graph_name, "train_config.yaml"))
            if 'encoder_best_val.pth' in os.listdir(
                    os.path.join(script_dir, "..", "gae", "logs", general_config[args.environment]['gae_path'],
                                 graph_name)):
                graph_encoder_path = os.path.join(script_dir, "..", "gae", "logs",
                                                  general_config[args.environment]['gae_path'], graph_name,
                                                  "encoder_best_val.pth")
            else:
                graph_encoder_path = os.path.join(script_dir, "..", "gae", "logs",
                                                  general_config[args.environment]['gae_path'], graph_name,
                                                  "encoder_final.pth")
            graph_encoder_config_path = os.path.join(script_dir, "..", "gae", "logs",
                                                     general_config[args.environment]['gae_path'], graph_name, "train_config.yaml")
            graph_encoder_spec_path = os.path.join(script_dir, "..", "gae", "logs",
                                                   general_config[args.environment]['gae_path'], graph_name,
                                                   "model_spec.yaml")
            encoder_config = load_yaml(graph_encoder_config_path)
            encoder_spec = load_yaml(graph_encoder_spec_path)
            if gae_config['model_type'] == 'gae':
                graph_encoder = GAEEncoder(encoder_spec['node_feature_vector_size'],
                                           encoder_config['model_config']['layers'],
                                           encoder_spec['edge_feature_vector_size'])
            elif gae_config['model_type'] == 'vgae':
                graph_encoder = VGAEEncoder(encoder_spec['node_feature_vector_size'],
                                            encoder_config['model_config']['layers'],
                                            encoder_spec['edge_feature_vector_size'], latent_dim=32)
            else:
                raise ValueError("ERROR: Unknown GAE model type...")
            graph_encoder.load_state_dict(torch.load(str(graph_encoder_path)))
            graph_encoder.eval()
            encoders_config[graph_name] = encoder_config
            encoders_spec[graph_name] = encoder_spec
            encoders[graph_name] = graph_encoder
            train_config.update({f"graph_encoder_path_{graph_name}": graph_encoder_path})
            train_config.update({f"graph_encoder_config_path_{graph_name}": graph_encoder_config_path})
            train_config.update({f"graph_encoder_spec_path_{graph_name}": graph_encoder_spec_path})

    train_config.update({"distance_computation": test_config['distance_computation']})
    # map to classes after saving the configuration
    if args.load_custom_test_envs or args.load_custom_envs or args.load_custom_val_envs or args.load_custom_train_envs: # reload right elements from the custom folder
        # load in a temporary folder that will be then eliminated to not overload file system
        tmp_folder = os.path.join(script_dir, "tmp", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        for folder in tqdm(os.listdir(envs_folder), desc="Loading environments..."):
            if os.path.isdir(os.path.join(str(envs_folder), folder)) and folder.isdigit() and folder in test_ids:
                environment_folder = os.path.join(str(envs_folder), folder, f"env.pkl")
                with open(environment_folder, 'rb') as f:
                    environment = pickle.load(f)
                train_config.pop('verbose', None)

                environment.update_max_steps(test_config['max_steps_overall'], test_config['max_steps_coefficient'])
                if args.algorithm_type == "projection" or args.algorithm_type == "iterative":
                    environment.update_spec()
                    if args.proto_knn:
                        train_config['knn'] = args.proto_knn
                    environment = AgentTrainingWrapper(env=environment,
                                                       env_name=args.environment,
                                                       gnn_models=encoders, **train_config)
                # discrete has nothing to do
                with open(os.path.join(tmp_folder, f"{folder}.pkl"), 'wb') as f:
                     pickle.dump(environment, f)
        envs_folder = tmp_folder
    test_ids = [x for x in test_ids if f"{x}.pkl" in os.listdir(envs_folder)]
    if args.verbose:
        logger.info(f"Focusing on {len(test_ids)} test set environments...")
    envs = GraphEnvSwitcher(train_ids=test_ids, switch_strategy=test_config['switch_strategy'],
                            algorithm_type=args.algorithm_type, pca_percentage_target=train_config['pca_percentage_target'],
                            pca_minimum_without_loss=train_config.get('pca_minimum_without_loss', False),
                                         switch_interval=test_config['switch_interval'], envs_folder=envs_folder,
                                         pad_spaces=train_config['pad_spaces'],
                                         padding_config=train_config['padding_config'],
                                         save_switch_logs=False,
                                         GNN_observations=train_config['GNN_observations'],
                                         test_mode=True,
                                         log_path=logs_folder)
    if args.algorithm_type == "projection" or args.algorithm_type == "iterative":
        envs.set_knn(args.proto_knn if args.proto_knn else 1)
    envs.update_max_steps(test_config['max_steps_overall'], test_config['max_steps_coefficient'])
    envs.set_terminate_at_approximate_best(test_config['terminate_at_approximate_best'])
    if stable_baselines: # SB3 implementation requires DummyVecEnv and allows normalization
        envs = DummyVecEnv([lambda: Monitor(envs)])
        envs = VecNormalize(envs, norm_obs=train_config['norm_obs'], norm_reward=False) # reward not used in testing phase
    return envs, envs_folder

def main():
    parser = argparse.ArgumentParser(description='Test the trained RL agent on C-CyberBattleSim environments.')
    parser.add_argument('-f', '--logs_folder', required=False, help='Path to the specific logs folder with runs')
    parser.add_argument('-algo', '--algorithm', choices=['ppo', 'a2c', 'rppo', 'trpo', 'ddpg', 'sac', 'td3', 'tqc', 'maskable_ppo', 'idqn'], default='ppo', help='Algorithm to use ')
    parser.add_argument('-at', '--algorithm_type', type=str, choices=['projection', 'iterative', 'discrete'],
                        default='projection',
                        help='Type of approximator to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument("-approx", "--approximate_distance", default=False,
                        type=bool, help="Approximate distance for the projection approach")
    parser.add_argument('-s', '--semantic_ordering', action='store_true', default=False, help='Whether to use semantic ordering for the action space')
    parser.add_argument('-GO', '--GNN_observations', action='store_true', default=False,
                        help='Whether to use graph-based observations for discrete algorithms (if not, use vectorized observations)')
    parser.add_argument("-pca_min", "--pca_minimum_without_loss", default=False, action='store_true',
                        help="Compute minimum number of components to not have loss in PCA reconstruction")
    parser.add_argument("-ssa", "--sample_subset_actions", type=bool, default=False, help="Whether to sample a subset of actions instead of the whole action space for the action selection (only for projection algorithms)")
    parser.add_argument('-e', '--environment', type=str, default='sample',
                        help='Type of environment application to be used for training')  # to be extended in the future to LOCAL or DISCRETE or others
    parser.add_argument('-gae', '--gae_folder', type=str, default=None,
                        help='Path to the gae folder to load, optional alternative to general config')
    parser.add_argument('--load_default_test_envs', default=False, action="store_true", help='Load test environments using default location')
    parser.add_argument('--load_default_val_envs', default=False, action="store_true",
                        help='Load test environments using default val location')
    parser.add_argument('--load_default_train_envs', default=False, action="store_true",
                        help='Load test environments using default train location')
    parser.add_argument("-proto", "--proto_knn", type=int, default=0,
                        help="Add protoKNN training off-policy module with K neighbors (0 to disable)")
    parser.add_argument("-proto_path", "--proto_knn_path", type=int, default=0,
                        help="Add protoKNN training off-policy module with K neighbors (0 to disable)")
    parser.add_argument("-ne", "--num_episodes_per_checkpoint", required=True, type=int,
                        help="Number of episodes to run per checkpoint during testing")
    parser.add_argument('--load_custom_envs', required=False,
                        help='Path to the test folder customized (different from default test folder)')
    parser.add_argument('--load_custom_test_envs', required=False, help='Path to the test folder customized (different from default test folder), focusing only in its test set')
    parser.add_argument('--load_custom_val_envs', required=False,
                        help='Path to the test folder customized (different from default test folder), focusing only in its test set')
    parser.add_argument('--load_custom_train_envs', required=False,
                        help='Path to the test folder customized (different from default test folder), focusing only in its test set')
    parser.add_argument('--static_seed', default=0, type=int, help='Use a static seed for training')
    parser.add_argument('--load_seeds', default="config",
                        help='Path of the folder where the seeds.yaml should be loaded from (e.g. previous experiment)')
    parser.add_argument('--random_seed', action='store_true', default=False, help='Use random seeds for training')
    parser.add_argument('--last_checkpoint', default=False, action="store_true", help='Load the last checkpoint only (best for validation or last for training)')
    parser.add_argument('--val_checkpoints', default=False, action="store_true",
                        help='Use validation checkpoints instead of training checkpoints')
    parser.add_argument('-o', '--option', default='agent_performances', choices=['action_time', 'agent_performances'], help='Decide which statistics to plot')
    parser.add_argument('--add_random', default=False, action="store_true",
                        help='Avoid calculation of average performances for the random agent')
    parser.add_argument('--test_config', type=str, default='config/test_config.yaml', help='Path to the test configuration YAML file')
    parser.add_argument('-v', '--verbose', default=2, type=int, help='Verbose level: 0 - no output, 1 - training/validation information, 2 - episode level information, 3 - iteration level information')
    parser.add_argument('--no_save_log_file', action='store_false', dest='save_log_file',
                        default=True, help='Disable logging to file; log only to terminal')
    parser.add_argument("--custom_test_folder_name", type=str, default=None, help="Custom name for the test folder to save results, if not specified it will be inferred from the loaded environments (e.g. if loading custom_test_envs it will be the name of the folder)")
    parser.add_argument('--no_cuda', action='store_false', dest='cuda',
                        default=True, help='Disable use of cuda even if available')

    args = parser.parse_args()

    if not args.load_default_test_envs and not args.load_custom_test_envs and not args.load_custom_envs and not args.load_custom_val_envs and not args.load_custom_train_envs \
        and not args.load_default_val_envs and not args.load_default_train_envs:
        raise ValueError("ERROR: Need to specify either default or custom test environments...")

    # only one can be true
    flags = [
        args.load_default_test_envs,
        args.load_custom_test_envs,
        args.load_custom_envs,
        args.load_custom_val_envs,
        args.load_custom_train_envs,
        args.load_default_val_envs,
        args.load_default_train_envs,
    ]
    if sum(bool(f) for f in flags) > 1:
        raise ValueError("ERROR: Can only specify one type of test environment loading...")
    if not args.last_checkpoint and args.val_checkpoints:
        raise ValueError("ERROR: Can only use last checkpoint in case of validation checkpoints due to merging reason...")

    if args.algorithm_type == 'iterative':
        args.algorithm = 'idqn'  # to use the specific DQN with action reconstruction loss for that type of environment

    env_type_folder_name = derive_env_type_folder_name(args)

    base_logs_dir = os.path.join(
        script_dir,
        'logs',
        args.environment,
        env_type_folder_name
    )

    # If logs_folder is not specified → pick the most recent run that matches the algorithm
    if not args.logs_folder:
        run_dirs = [
            os.path.join(base_logs_dir, d)
            for d in os.listdir(base_logs_dir)
            if os.path.isdir(os.path.join(base_logs_dir, d))
        ]

        if not run_dirs:
            raise FileNotFoundError(f"No log folders found in {base_logs_dir}")

        # Filter only runs that have a subfolder starting with the algorithm name
        matching_runs = []
        for run_dir in run_dirs:
            subfolders = [
                f for f in os.listdir(run_dir)
                if os.path.isdir(os.path.join(run_dir, f))
            ]
            if args.algorithm == "maskable_ppo":
                algo_prefix = "PPO"
            elif args.algorithm == "idqn":
                algo_prefix = "IDQN"
            else:
                algo_prefix = args.algorithm.upper()
            if any(f.startswith(algo_prefix) for f in subfolders):
                matching_runs.append(run_dir)

        if not matching_runs:
            raise FileNotFoundError(f"No runs found for algorithm {args.algorithm}")

        # for each masking run, take the train_config.yaml file inside the folder and check the key algorithm
        refined_matching_runs = []
        for run_dir in matching_runs:
            train_config_path = os.path.join(run_dir, f"train_config.yaml")
            if os.path.exists(train_config_path):
                with open(train_config_path, 'r') as f:
                    train_config = yaml.safe_load(f)
                if train_config['algorithm'] == args.algorithm:
                    refined_matching_runs.append(run_dir)
        # Pick the most recently created matching run
        args.logs_folder = max(refined_matching_runs, key=os.path.getctime)
    else:
        args.logs_folder = os.path.join(base_logs_dir, args.logs_folder)

    # if protoKNN specified, load the most recent protoKNN run that matches the iterative algorithm and the environment if not specified
    if args.proto_knn and not args.proto_knn_path:
        proto_base_logs_dir = os.path.join(
            script_dir,
            'logs',
            args.environment,
            "iterative"
        )

        run_dirs = [
            os.path.join(proto_base_logs_dir, d)
            for d in os.listdir(proto_base_logs_dir)
            if os.path.isdir(os.path.join(proto_base_logs_dir, d))
        ]

        if not run_dirs:
            raise FileNotFoundError(f"No log folders found in {proto_base_logs_dir}")

        # Filter only runs that have a subfolder starting with the algorithm name
        matching_runs = []
        for run_dir in run_dirs:
            subfolders = [
                f for f in os.listdir(run_dir)
                if os.path.isdir(os.path.join(run_dir, f))
            ]
            algo_prefix = "IDQN"
            if any(f.startswith(algo_prefix) for f in subfolders):
                matching_runs.append(run_dir)

        if not matching_runs:
            raise FileNotFoundError(f"No runs found for algorithm {args.algorithm}")

        # for each masking run, take the train_config.yaml file inside the folder and check the key algorithm
        refined_matching_runs = []
        for run_dir in matching_runs:
            train_config_path = os.path.join(run_dir, f"train_config.yaml")
            if os.path.exists(train_config_path):
                with open(train_config_path, 'r') as f:
                    train_config = yaml.safe_load(f)
                if train_config['algorithm'] == "idqn":
                    refined_matching_runs.append(run_dir)
        # Pick the most recently created matching run
        args.proto_knn_path = max(refined_matching_runs, key=os.path.getctime)

    logger = setup_logging(args.logs_folder, args.save_log_file)

    # Read YAML configuration files
    with open(os.path.join(script_dir, args.test_config), 'r') as config_file:
        test_config = yaml.safe_load(config_file)

    test_config = {**test_config, **vars(args)} # overwrite with command line arguments if specified

    if args.load_custom_test_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_test_envs).split("/")[-1]
    elif args.load_custom_val_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_val_envs).split("/")[-1]
    elif args.load_custom_train_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_train_envs).split("/")[-1]
    elif args.load_custom_envs:
        test_config['test_folder'] = copy.deepcopy(args.load_custom_envs).split("/")[-1]
    else:
        test_config['test_folder'] = "default"


    # count number of envs to test to
    if args.load_custom_test_envs or args.load_default_test_envs:
        envs_folder = os.path.join(args.logs_folder, "envs") if args.load_default_test_envs else os.path.join(script_dir, '..', 'data', 'env_samples', args.load_custom_test_envs)
    elif args.load_custom_val_envs or args.load_default_val_envs:
        envs_folder = os.path.join(args.logs_folder, "envs") if args.load_default_val_envs else os.path.join(script_dir, '..', 'data', 'env_samples', args.load_custom_val_envs)
    elif args.load_custom_train_envs or args.load_default_train_envs:
        envs_folder = os.path.join(args.logs_folder, "envs") if args.load_default_train_envs else os.path.join(script_dir, '..', 'data', 'env_samples', args.load_custom_train_envs)
    else:
        envs_folder = None

    num_envs = 1
    if envs_folder is not None:
        num_envs = len([f for f in os.listdir(envs_folder) if os.path.isdir(os.path.join(envs_folder, f)) and f.endswith(".pkl")])
        # filter based on split.yaml
        if args.load_default_test_envs or args.load_default_val_envs or args.load_default_train_envs:
            with open(os.path.join(args.logs_folder, "split.yaml"), 'r') as file:
                yaml_info = yaml.safe_load(file)
            if args.load_default_test_envs:
                test_ids = [str(elem['id']) for elem in yaml_info['test_set']]
            elif args.load_default_val_envs:
                test_ids = [str(elem['id']) for elem in yaml_info['validation_set']]
            elif args.load_default_train_envs:
                test_ids = [str(elem['id']) for elem in yaml_info['training_set']]
            else:
                test_ids = []
            num_envs = len(test_ids)

    num_episodes_per_env = (
        math.ceil(args.num_episodes_per_checkpoint / num_envs)
        if envs_folder is not None
        else args.num_episodes_per_checkpoint
    )
    # Eventual seeds
    if args.static_seed:
        seed_test = [args.static_seed for _ in range(int(num_episodes_per_env))]
    elif args.random_seed:
        seed_test = [np.random.randint(1000) for i in range(int(num_episodes_per_env))]
    else: # args.load_seed, first seed
        if args.verbose:
            logger.info(f"Reading seeds from folder {args.load_seeds}")
        with open(os.path.join(script_dir, args.load_seeds, 'seeds.yaml'), 'r') as seeds_file:
            seeds_loaded = yaml.safe_load(seeds_file)
        seed_test = [seeds_loaded['seeds'][i % len(seeds_loaded['seeds'])]
                     for i in range(int(num_episodes_per_env))]
    test_config.update({"seeds_test": seed_test})
    test_config.update({"num_episodes_per_env": num_episodes_per_env})
    test_config.update({"num_envs": num_envs})
    set_seeds(seed_test[0])

    run_folder = os.path.join(args.logs_folder)
    train_config_file = os.path.join(run_folder, 'train_config.yaml')
    with open(train_config_file, 'r') as train_config_file:
        train_config = yaml.safe_load(train_config_file)
    if not args.algorithm:
        args.algorithm = train_config['algorithm']
        test_config['algorithm'] = train_config['algorithm']
    test_run_config = copy.deepcopy(test_config)
    test_run_config['run_folder'] = run_folder

    test_envs, envs_folder = load_test_envs(run_folder, args, train_config, test_config, args.logs_folder, logger=logger)
    _ = parse_option(test_run_config, test_envs, logger, args.verbose)
    # runs_outcomes.append(outcomes) # Potentially use runs_outcomes to average the outcomes in the future
    if args.load_custom_test_envs or args.load_custom_envs or args.load_custom_val_envs or args.load_custom_train_envs:
        remove_folder_and_files(envs_folder)


if __name__ == "__main__":
    main()
