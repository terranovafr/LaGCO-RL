#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

from typing import Any
import time
import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete


# DiscreteEnv class defining only the common logic for discrete action spaces, the specific environment should inherit from this class and implement the environment-specific logic, such as defining the action list, observation space, reward calculation, and done condition.
# This class also provides common features such as no-op action support, invalid action penalty, action masking, and logging.
class DiscreteEnv(gym.Env):
    # init function invariant to args and setting only relevant ones, and others will be used only by classes based on this
    def pre_init(self,
                **kwargs
                 ):
        # set default values for all possible attributes
        self.verbose = kwargs.get("verbose", False)
        self.max_steps_overall = kwargs.get("max_steps_overall", 100)
        self.max_steps_coefficient = kwargs.get("max_steps_coefficient", 1.0)
        self.no_action_support = kwargs.get("no_action", False)
        self.no_action_penalty_sum = kwargs.get("no_action_penalty_sum", -10)
        self.padding_invalid_action_penalty_sum = kwargs.get("padding_invalid_action_penalty_sum", -10)
        self.save_logs_transitions = kwargs.get("save_logs_transitions", False)
        self.remove_invalid_actions = kwargs.get("remove_invalid_actions", False)
        self.terminate_at_approximate_best = kwargs.get("terminate_at_approximate_best", False)
        self.semantic_ordering = False
        self.current_step = 0
        self.count_no_actions = 0
        self.padding_invalid_actions = 0
        self._metrics = {}
        self.scenario_size = 0
        self.termination_reason = 0
        self.action_list = []
        self.semantic_action_list = []

    def reconstruct_action_space(self):
        # called when action list changes based on re-parameterization of some hyper-parameters of the environment
        action_space_size = len(self.action_list)
        if self.no_action_support:
            action_space_size += 1
        self.action_space = Discrete(action_space_size)

    def post_init(self):
        # at the end of init function
        action_space_size = len(self.action_list)
        if self.no_action_support:
            action_space_size += 1
        self.action_space = Discrete(action_space_size)
        self.update_metrics()

    def pre_reset(
        self,
    ) -> tuple[ObsType, dict[str, Any]]:
        # before the env logic reset function
        self.current_step = 0
        self.count_no_actions = 0
        self.padding_invalid_actions = 0
        self.termination_reason = 0
        self.done = False
        self.truncated = False
        self.info = {}
        self.obs = []

    def post_reset(self):
        # after the env logic reset function, before returning the initial observation
        self.obs = self._pad_observation(self.obs)
        # padded observation is returned to ensure consistent observation shape across many scenario sizes
        return self.obs, self.info

    def pre_step(self, *action):
        # before the env logic step function, action is the raw action from the agent, this function decodes it to the actual action in the environment, and also calculates the reward for no-op and invalid actions, and also updates the step count and checks for truncation
        self.current_step += 1
        self.reward = 0
        self.invalid_action = False
        self.no_action = False
        self.step_start_time = time.time()

        if len(action) == 1 and isinstance(action[0], tuple):
            action = action[0]

        # remove None elements from tuple
        action = tuple(a for a in action if a is not None)
        decoded_action = None
        action_set = self.semantic_action_list if self.semantic_ordering else self.action_list

        if len(action) == 0:
            self.invalid_action = True

        elif action[0] == "no_action" and getattr(self, "no_action_support", False):
            self.no_action = True

        elif len(action) == 1:
            idx = action[0]
            if not isinstance(idx, int) and not isinstance(idx, np.integer) and not (isinstance(idx, float) and idx.is_integer()):
                self.invalid_action = True
            elif 0 <= idx < len(action_set):
                decoded_action = action_set[idx]
            elif getattr(self, "no_action_support", False) and idx == len(self.action_space) - 1:
                self.no_action = True
            else:
                self.invalid_action = True
        else:
            decoded_action = action

        if self.no_action:
            self.count_no_actions += 1
            self.reward = self.no_action_penalty_sum / min(
                self.max_steps_overall,
                self.max_steps_coefficient * self.scenario_size
            )
        elif self.invalid_action:
            self.padding_invalid_actions += 1
            self.reward = self.padding_invalid_action_penalty_sum / min(
                self.max_steps_overall,
                self.max_steps_coefficient * self.scenario_size
            )

        return decoded_action

    def post_step(self):
        # after the env logic step function, this function checks for truncation and updates the metrics, and also pads the observation for consistent shape, and also sets the termination reason for logging
        if self.save_logs_transitions:
            self.save_step_log()
        self.truncated = self.current_step >= self.max_steps_overall or ( self.scenario_size != 0 and \
                            self.current_step >= self.max_steps_coefficient * self.scenario_size)
        if self.done or self.truncated:
            self.update_metrics()
        self.step_time = time.time() - self.step_start_time
        self.obs = self._pad_observation(self.obs)
        if self.done:
            self.termination_reason = 1
        elif self.truncated:
            self.termination_reason = 2

    def update_max_steps(self, max_steps_overall, max_steps_coefficient):
        # they will determine truncate
        self.max_steps_overall = max_steps_overall
        self.max_steps_coefficient = max_steps_coefficient

    def set_semantic_ordering(self, bool_value):
        # ordering of the discrete action space by a task-specific heuristic
        self.semantic_ordering = bool_value

    def get_metrics(self):
        return self._metrics

    def update_config(self, config):
        self.max_steps_coefficient = config.get('max_steps_coefficient', self.max_steps_coefficient)
        self.max_steps_overall = config.get('max_steps_overall', self.max_steps_overall)
        self.verbose = config.get('verbose', self.verbose)
        self.no_action_support = config.get('no_action', self.no_action_support)
        self.no_action_penalty_sum = config.get('no_action_penalty_sum', self.no_action_penalty_sum)
        self.padding_invalid_action_penalty_sum = config.get('padding_invalid_action_penalty_sum', self.padding_invalid_action_penalty_sum)
        self.save_logs_transitions = config.get('save_logs_transitions', self.save_logs_transitions)
        self.remove_invalid_actions = config.get('remove_invalid_actions', self.remove_invalid_actions)

    def update_metrics(self):
        self._metrics = {
            'termination_reason': self.termination_reason,
            'scenario_size': self.scenario_size,
            'num_steps': self.current_step,
            'no_action_count': self.count_no_actions,
            'padding_invalid_action_count': self.padding_invalid_actions,
        }

    def save_step_log(self):
        if not hasattr(self, 'step_logs'):
            self.step_logs = []
        self.step_logs.append({
            'reward': self.reward,
            'done': self.done,
            'truncated': self.truncated
        })

    def _pad_observation(self, obs, space=None):
        # pad the observation to max_obs_shape if it's smaller, for compatibility with models that require fixed input size
        if not hasattr(self, 'max_obs_shape'):
            return obs  # No padding if max_obs_shape not set yet
        if space is None:
            space = self.observation_space

        # Box observation
        if isinstance(space, gym.spaces.Box):
            target_shape = space.shape
            if isinstance(obs, np.ndarray) and obs.shape != target_shape:
                padded_obs = np.zeros(target_shape, dtype=obs.dtype)
                slices = tuple(slice(0, min(obs_dim, pad_dim)) for obs_dim, pad_dim in zip(obs.shape, target_shape))
                padded_obs[slices] = obs[slices]
                return padded_obs
            return obs

        # Dict observation
        elif isinstance(space, gym.spaces.Dict):
            padded_dict = {}
            for k, subspace in space.spaces.items():

                padded_dict[k] = self._pad_observation(obs[k], subspace)
            return padded_dict

        # Tuple observation
        elif isinstance(space, gym.spaces.Tuple):
            return tuple(self._pad_observation(o, s) for o, s in zip(obs, space.spaces))

        # Discrete or other types: return as-is
        else:
            return obs

    def is_valid_action(self, *action):
        # check if the action is valid in the current state, this is used for action masking and invalid action penalty
        # by default, all actions are valid, override this function in specific envs to implement environment-specific logic
        return True

    def action_masks(self, max_len=None) -> np.ndarray:
        # action masks determined based on 'is_valid_action', to be overrided
        if max_len is None:
            max_len = (
                self.action_space.n
                if isinstance(self.action_space, Discrete)
                else self.action_space.shape[0]
            )

        actions = self.semantic_action_list if self.semantic_ordering else self.action_list
        mask = np.zeros(max_len, dtype=bool)

        for i, action in enumerate(actions):
            if self.remove_invalid_actions:
                # check action structure
                if isinstance(action, tuple):
                    mask[i] = self.is_valid_action(*action)
                else:
                    mask[i] = self.is_valid_action(action)
            else:
                mask[i] = True
        if self.no_action_support:
            mask[len(actions)] = True  # last action is no-op
        if not mask.any():
            raise RuntimeError("Empty action mask!")

        return mask

    def set_terminate_at_approximate_best(self, bool_value):
        # whether done should be set to true when we reach empirically known best
        self.terminate_at_approximate_best = bool_value