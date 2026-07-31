#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

import gymnasium as gym
import pickle
import csv
import os
from utils.distance_utils import calculate_most_distant_point
from wrappers.agent_training_wrapper import GraphWrapper
from utils.file_utils import save_yaml, load_yaml
import numpy as np
from sklearn.decomposition import PCA
import json
from utils.padding_utils import pad_space, get_space_shape, max_shapes, pad_observation

class GraphEnvSwitcher(gym.Env):
    '''
        This environment wrapper allows switching between multiple graph-based environments during training, validation, and testing.
        It supports different switching strategies (random, random different, sequential) and can log environment switches and optionally save embeddings and step logs.
        It also handles padding of observation and action spaces if needed, and can compute PCA for continuous action spaces to reduce dimensionality while preserving variance.
    '''
    def __init__(self, envs_folder, train_ids, val_ids=None, test_ids=None,
                 algorithm_type=None,
                 GNN_observations=False,
                 pca_percentage_target=None,
                 pca_minimum_without_loss=False,
                 switch_interval=1, switch_strategy='random',
                 test_mode=False,
                 log_path=None,
                 pad_spaces=False,
                 padding_config=None,
                 save_switch_logs=True,
                 save_embeddings=False,
                 save_embeddings_interval_train=10,
                 save_embeddings_interval_val=1,
                 save_logs_transitions=False,
                 save_logs_interval_train=100,
                 save_logs_interval_val=1,
                 ):
        self.envs_folder = envs_folder
        self.train_ids = train_ids
        self.test_ids = test_ids if test_ids is not None else []
        self.val_ids = val_ids if val_ids is not None else []
        self.algorithm_type = algorithm_type
        self.GNN_observations = GNN_observations
        self.pca_percentage_target = pca_percentage_target
        self.pca_minimum_without_loss = pca_minimum_without_loss
        self.padding_config = padding_config
        self.switch_interval = switch_interval
        self.switch_strategy = switch_strategy
        self.reference_ids = self.train_ids
        self.current_env_index = self.reference_ids[0]  # Start with the first environment
        self.episode_id = 0
        self.global_step = 0  # Counting total steps, or could be used for switch moment
        self.mode = 'train' # mode affecting from which set to switch between environments
        self.log_path = log_path
        self.pad_spaces = pad_spaces
        self.save_logs_transitions = save_logs_transitions
        self.save_logs_interval_train = save_logs_interval_train
        self.save_logs_interval_val = save_logs_interval_val
        self.overall_step_logs = {}
        self.envs_cache = {}
        self.max_obs_shape = None
        self.max_action_shape = None
        self.save_switch_logs = save_switch_logs
        self.save_embeddings = save_embeddings
        self.save_embeddings_interval_train = save_embeddings_interval_train
        self.save_embeddings_interval_val = save_embeddings_interval_val
        self.knn = None
        self.proto_knn = None
        self.continuous_features_loaded = False
        self.terminate_at_approximate_best = False
        self.test_mode = test_mode
        # Initialize current environment
        if (self.algorithm_type == "projection" or self.algorithm_type == "iterative") and not self.test_mode:
            self._compute_general_continuous_features()
            self.continuous_features_loaded = True
        elif (self.algorithm_type == "projection" or self.algorithm_type == "iterative") and self.test_mode:
            self._load_general_continuous_features()
            self.continuous_features_loaded = True
        self.current_env = self.load_env(self.current_env_index)
        self._metrics = self.current_env.get_metrics()

        # derive padding differently whether it is training or testing
        if self.pad_spaces and not self.padding_config and ((self.algorithm_type == "discrete" and self.GNN_observations) or not isinstance(self.current_env, GraphWrapper)):
            self._compute_largest_spaces()
        elif self.pad_spaces and self.padding_config and ((self.algorithm_type == "discrete" and self.GNN_observations) or not isinstance(self.current_env, GraphWrapper)):
            self.max_obs_shape = self.padding_config['max_obs_shape']
            self.max_action_shape = self.padding_config['max_action_shape']

        if isinstance(self.current_env, GraphWrapper):
            # GNN embedding so no need to pad input
            self.observation_space = self.current_env.observation_space
            if self.algorithm_type == "discrete" and self.GNN_observations:
                # mixed case so pad action space
                self.action_space = pad_space(self.current_env.action_space, self.max_action_shape)
            else:
                self.action_space = self.current_env.action_space
        else:
            self.observation_space = pad_space(self.current_env.observation_space, self.max_obs_shape)
            self.action_space = pad_space(self.current_env.action_space, self.max_action_shape)
        self.metadata = {'render.modes': []}  # Or ['human'] if you support rendering
        # Initialize the CSV file with header if logging is enabled
        if self.save_switch_logs:
            if not os.path.exists(os.path.join(self.log_path, "switch_log.csv")):
                with open(os.path.join(self.log_path, "switch_log.csv"), mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['switch_moment', 'environment_id'])

            # Log initial environment
            if self.mode == 'train':
                self._log_switch(switch_moment=0, env_id=self.current_env_index)

    def get_padding_config(self):
        return self.max_obs_shape, self.max_action_shape

    def get_candidate_action_KNNs(self):
        # This method is only applicable for GraphWrapper environments, as it relies on the structure of the action space and the presence of KNN-based candidate actions.
        if isinstance(self.current_env, GraphWrapper):
            return self.current_env.get_candidate_action_KNNs()
        else:
            raise NotImplementedError("Candidate actions retrieval is only implemented for GraphWrapper environments.")

    def set_mode(self, mode):
        # Change the mode of the environment switcher, which determines from which set of environments to switch between.
        if mode == 'train':
            self.mode = 'train'
            self.reference_ids = self.train_ids
        elif mode == 'test':
            self.mode = 'test'
            if len(self.test_ids) == 0:
                raise ValueError("No test environments available.")
            self.reference_ids = self.test_ids
        elif mode == 'val':
            self.mode = 'val'
            if len(self.val_ids) == 0:
                raise ValueError("No validation environments available.")
            self.reference_ids = self.val_ids
        else:
            raise ValueError("Mode must be 'train', 'test', or 'val'.")
        self.current_env_index = self.reference_ids[0]  # Reset to the first environment
        self.current_env = self.load_env(self.current_env_index)
        self.current_env.reset()

        if self.save_switch_logs and self.mode == 'train':
            self._log_switch(switch_moment=0, env_id=self.current_env_index)

    def set_switch_interval(self, interval):
        # Keep track of the old switch interval to allow reverting back to it when needed (e.g., after temporarily disabling switching).
        if interval == False:
            self.switch_interval = self.old_switch_interval
        else:
            self.old_switch_interval = self.switch_interval
            self.switch_interval = interval

    def set_proto_knn_influence(self, proto_knn):
        if isinstance(self.current_env, GraphWrapper):
            self.current_env.set_proto_knn_influence(proto_knn)
        self.proto_knn = proto_knn

    def _maybe_save_embeddings(self, obs_vector, action_vector):
        if not self.save_embeddings:
            return

        # When we reach the save interval → write and reset buffer
        if self.episode_id != 0 and ((self.mode == 'train' and self.episode_id % self.save_embeddings_interval_train == 0)
            or (self.mode == 'val' and self.episode_id % self.save_embeddings_interval_val == 0)):
            # Initialize list if not present
            if not hasattr(self, "episode_embeddings"):
                self.episode_embeddings = []
            # Store step within episode
            self.episode_embeddings.append({
                "episode": self.episode_id,
                "environment_id": self.current_env_index,
                "step": len(self.episode_embeddings),
                "obs_vector": obs_vector,
                "action_vector": action_vector,
            })
            self.embeddings_file = os.path.join(self.log_path, f"episode_embeddings_{self.mode}.jsonl")
        else:
            if hasattr(self, "episode_embeddings"):
                # Save to JSONL file (append mode)
                with open(self.embeddings_file, "a") as f:
                    for entry in self.episode_embeddings:
                        # Safely serialize NumPy arrays to lists
                        json.dump(
                            entry,
                            f,
                            default=lambda o: o.tolist() if hasattr(o, "tolist") else o
                        )
                        f.write("\n")
                del self.episode_embeddings

    def get_current_env(self):
        return self.current_env

    def _compute_general_continuous_features(self):
        # Compute overall action bounds and discrete action embeddings across all environments in train and val sets, and apply PCA if specified (training time)

        all_ids = self.train_ids + self.val_ids
        overall_action_set = []
        env_ids = []
        env = None
        for env_id in all_ids:
            env = self.load_env(env_id)
            env.reset()
            if hasattr(env, 'action_set') and isinstance(env.action_set, dict):
                overall_action_set.extend(env.action_set.values())
                env_ids.extend([env_id] * len(env.action_set))
            else:
                print(f"Warning: Environment {env_id} has no valid action embeddings.")
        if overall_action_set:
            self.action_lower_bounds = np.min(overall_action_set, axis=0)
            self.action_upper_bounds = np.max(overall_action_set, axis=0)
            min_bound = np.min(self.action_lower_bounds)
            max_bound = np.max(self.action_upper_bounds)
            if len(env.get_discrete_actions()) > 0:
                for discrete_action in env.get_discrete_actions():
                    discrete_action_point = calculate_most_distant_point(overall_action_set, (min_bound, max_bound))
                    discrete_action_aid = ("global", discrete_action)
                    if not hasattr(self, "discrete_actions"):
                        self.discrete_actions = {}
                    self.discrete_actions[discrete_action_aid] = discrete_action_point
                    overall_action_set.append(discrete_action_point)
            # saving here because they may be modified by PCA
            action_stats = {
                        "action_lower_bounds": self.action_lower_bounds.tolist(),
                        "action_upper_bounds": self.action_upper_bounds.tolist(),
                        "discrete_actions": {str(k): v.tolist() for k, v in
                                             getattr(self, 'discrete_actions', {}).items()}
            }
            save_yaml(action_stats, self.log_path, "action_stats.yaml")

        reduced_action_set = overall_action_set
        original_dim = overall_action_set[0].shape[0]
        if self.pca_percentage_target is not None and self.pca_percentage_target < 1.0:
            self.pca = PCA(n_components=int(original_dim * self.pca_percentage_target))
            reduced_action_set = self.pca.fit_transform(overall_action_set)
            self.cumulative_explained_variance = np.sum(self.pca.explained_variance_ratio_)
            self.n_pca_components = self.pca.n_components_
            print("Using PCA to reduce action dimensionality from", original_dim, "to",
                  self.pca.n_components_, "explaining",
                  f"{self.cumulative_explained_variance * 100:.2f}%", "of variance.")
        elif self.pca_minimum_without_loss:
            n_components = min(original_dim, len(overall_action_set))
            pca_temp = PCA(n_components=n_components)
            pca_temp.fit(overall_action_set)
            cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            # Find number of components needed to explain ~100% variance
            n_components = np.searchsorted(cumulative_variance, 0.99) + 1  # 99.99%
            self.pca = PCA(n_components=n_components)
            reduced_action_set = self.pca.fit_transform(overall_action_set)
            self.cumulative_explained_variance = np.sum(self.pca.explained_variance_ratio_)
            assert 0.99 <= self.cumulative_explained_variance <= 1.0, "Explained variance should be ~100%."
            self.n_pca_components = self.pca.n_components_
            print(
                "Using PCA to reduce action dimensionality from", original_dim, "to",
                self.pca.n_components_, "explaining",
                f"{self.cumulative_explained_variance * 100:.2f}%", "of variance (minimum without loss)."
            )

        # Update action bounds if PCA was applied
        if hasattr(self, "pca") and self.pca is not None:
            self.action_lower_bounds = np.min(reduced_action_set, axis=0)
            self.action_upper_bounds = np.max(reduced_action_set, axis=0)
            pca_stats = {
                "original_dim": int(original_dim),
                "reduced_dim": int(self.pca.n_components_),
                "cumulative_explained_variance": float(self.cumulative_explained_variance),
                "percentage_target": getattr(self, "pca_percentage_target", None),
                "minimum_without_loss": getattr(self, "pca_minimum_without_loss", False)
            }
            save_yaml(pca_stats, self.log_path, "pca_action_stats.yaml")
            # Save PCA object
            with open(os.path.join(self.log_path, "pca_action_object.pkl"), 'wb') as f:
                pickle.dump(self.pca, f)
            # Save updated action stats after PCA transformation
            action_stats = {
                "action_lower_bounds": self.action_lower_bounds.tolist(),
                "action_upper_bounds": self.action_upper_bounds.tolist(),
                "discrete_actions": {str(k): v.tolist() for k, v in getattr(self, 'discrete_actions', {}).items()}
            }
            save_yaml(action_stats, self.log_path, "action_stats.yaml")

    def _load_general_continuous_features(self):
        # load action stats from file (testing time)
        action_stats = load_yaml(os.path.join(self.log_path, "action_stats.yaml"))
        self.action_lower_bounds = np.array(action_stats['action_lower_bounds'])
        self.action_upper_bounds = np.array(action_stats['action_upper_bounds'])
        self.discrete_actions = {}
        for k, v in action_stats.get('discrete_actions', {}).items():
            key_tuple = tuple(k.strip("()").replace("'", "").split(", "))
            self.discrete_actions[key_tuple] = np.array(v)

        pca_stats_path = os.path.join(self.log_path, "pca_action_stats.yaml")
        if os.path.exists(pca_stats_path):
            pca_stats = load_yaml(pca_stats_path)
            self.pca_percentage_target = pca_stats['percentage_target']
            self.pca_minimum_without_loss = pca_stats['minimum_without_loss']
            # load pca object
            with open(os.path.join(self.log_path, "pca_action_object.pkl"), 'rb') as f:
                self.pca = pickle.load(f)
        else:
            self.pca_percentage_target = None
            self.pca_minimum_without_loss = False

    def _compute_largest_spaces(self):
        # When computing largest, also use test set for determining the maximum shapes to ensure compatibility during testing
        all_ids = self.train_ids + self.val_ids + self.test_ids
        for env_id in all_ids:
            env = self.load_env(env_id)
            self.envs_cache[env_id] = env
            obs_shape = get_space_shape(env.observation_space)
            act_shape = get_space_shape(env.action_space)
            self.max_obs_shape = max_shapes(self.max_obs_shape, obs_shape)
            self.max_action_shape = max_shapes(self.max_action_shape, act_shape)

    def load_env(self, env_id):
        # Loading environment but also setting features to be shared across all
        env_path = f"{self.envs_folder}/{env_id}.pkl"
        with open(env_path, 'rb') as f:
            env = pickle.load(f)
        if self.knn is not None and isinstance(env, GraphWrapper):
            env.set_knn(self.knn)
        if self.proto_knn is not None and isinstance(env, GraphWrapper):
            env.set_proto_knn_influence(self.proto_knn)
        env.set_terminate_at_approximate_best(self.terminate_at_approximate_best)
        if (self.algorithm_type == "projection" or self.algorithm_type == "iterative") and self.continuous_features_loaded:
            if hasattr(self, 'discrete_actions'):
                env.set_discrete_actions(self.discrete_actions)
            env.set_continuous_bounds(self.action_lower_bounds, self.action_upper_bounds)
            if (self.pca_percentage_target is not None and self.pca_percentage_target < 1.0) or self.pca_minimum_without_loss:
                env.set_continuous_pca_object(self.pca if hasattr(self, 'pca') else None)
        return env

    def get_graphs(self):
        return self.current_env.get_graphs()

    def sample_valid_action(self):
        return self.current_env.sample_valid_action()

    def update_metrics(self):
        self.current_env.update_metrics()

    def get_metrics(self):
        return self._metrics

    def step(self, action):
        # Step inside and handle logic at a higher level for the switching strategy
        if isinstance(self.current_env, GraphWrapper):
            obs, reward, done, truncated, info = self.current_env.step(action)
        else:
            if isinstance(action, int) or isinstance(action, np.integer):
                action = (action,)
                obs, reward, done, truncated, info = self.current_env.step(*action)
            else:
                action = tuple(action)
                obs, reward, done, truncated, info = self.current_env.step(*action)
        self._maybe_save_embeddings(obs, action)
        self.global_step += 1

        if done or truncated:
            self.current_env.update_metrics()
            self._metrics = self.current_env.get_metrics()
            self.episode_id += 1
            if self.episode_id % self.switch_interval == 0:
                if self.save_logs_transitions:
                    self.old_step_logs = self.current_env.step_logs
                self._switch_env()

        # Pad observation if needed
        if self.pad_spaces and not isinstance(self.current_env, GraphWrapper):
            obs = pad_observation(obs, self.observation_space)
        return obs, reward, done, truncated, info

    def action_masks(self):
        # This method is only applicable for GraphWrapper environments, as it relies on the structure of the action space and the presence of an action_masks method.
        max_len = None
        if isinstance(self.action_space, gym.spaces.Discrete):
            max_len = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            # product of dimensions, e.g. [E, V] -> E * V
            max_len = self.action_space.nvec
        return self.current_env.action_masks(max_len)

    @property
    def unwrapped(self):
        return self.current_env.unwrapped

    def set_knn(self, k):
        if k is not None and isinstance(self.current_env, GraphWrapper):
            self.knn = k
            self.current_env.set_knn(k)

    def set_terminate_at_approximate_best(self, terminate):
        self.current_env.set_terminate_at_approximate_best(terminate)
        self.terminate_at_approximate_best = terminate

    def reset(self, **kwargs):
        # Call reset inside but also handle logic at a higher level for the switching strategy and logging
        if self.episode_id > 0:
            interval_logs = (
                self.save_logs_interval_train
                if self.mode == 'train'
                else self.save_logs_interval_val
            )
            if self.save_logs_transitions and self.episode_id % interval_logs == 0:
                step_logs = self.current_env.step_logs if len(self.current_env.step_logs) > 0 else self.old_step_logs
                logs_file = os.path.join(self.log_path, f"step_logs_{self.mode}.csv")
                file_exists = os.path.exists(logs_file)
                with open(logs_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    for step_log in step_logs:
                        if not file_exists:
                            header = ['episode_id'] + list(step_log.keys())
                            writer.writerow(header)
                            file_exists = True

                        row = [self.episode_id] + [step_log[k] for k in step_log.keys()]
                        writer.writerow(row)

        obs, info = self.current_env.reset(**kwargs)
        # Pad observation if needed
        obs = pad_observation(obs, self.observation_space)
        return obs, info

    def _switch_env(self):
        # Switching based on strategy and logging the switch if needed. Also resetting the new environment and updating max steps if applicable.
        if len(self.reference_ids) == 1:
            return
        if self.switch_strategy == 'random':
            self.current_env_index = np.random.choice(self.reference_ids)
        elif self.switch_strategy == 'random_different':
            copy_reference_ids = self.reference_ids.copy()
            copy_reference_ids.remove(self.current_env_index)
            self.current_env_index = np.random.choice(copy_reference_ids)
        elif self.switch_strategy == 'sequential':
            self.current_env_index = self.reference_ids[
                (self.reference_ids.index(self.current_env_index) + 1) % len(self.reference_ids)
            ]
        else:
            raise ValueError("Switch strategy must be 'random', 'random_different', or 'sequential'.")

        # Load the new environment
        self.current_env = self.load_env(self.current_env_index)
        self.current_env.reset()
        if hasattr(self, 'max_steps_overall'):
            self.current_env.update_max_steps(self.max_steps_overall, self.max_steps_coefficient)

        # Log the switch if path is provided
        if self.save_switch_logs and self.mode == 'train':
            self._log_switch(switch_moment=self.global_step, env_id=self.current_env_index)


    def update_max_steps(self, max_steps_overall, max_steps_coefficient):
        self.max_steps_overall = max_steps_overall
        self.max_steps_coefficient = max_steps_coefficient

    def _log_switch(self, switch_moment, env_id):
        with open(os.path.join(self.log_path, "switch_log.csv"), mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([switch_moment, env_id])
