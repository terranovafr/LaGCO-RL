#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

# This file is derived from Stable-Baselines3 (SB3)
# Original source: https://github.com/DLR-RM/stable-baselines3
# Copyright (c) 2019 Antonin Raffin (SB3 Author)
# Licensed under the MIT License

# Modification: Added OffPolicyAlgorithmStateAction class that overrides the action sampling and rollout collection to handle state-action pairs, including storing the next action set in the replay buffer. This allows for algorithms that need to consider action masking or have a variable action space.

import random
from copy import deepcopy
from typing import Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from .buffers import DictReplayBufferWithNextActions

class OffPolicyAlgorithmStateAction(OffPolicyAlgorithm):
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:

        # This method is overridden to sample actions according to the exploration policy, which may involve sampling from the policy's probability distribution, adding noise for deterministic policies, or sampling random actions during the warm-up phase. It returns both the action to take in the environment and the scaled action that will be stored in the replay buffer, which may differ if the action space is not normalized
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            unscaled_action = np.array([random.choice(self.env.envs[i].get_candidate_action_KNNs()) for i in range(n_envs)])
        else:
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action = self.predict(self._last_obs, deterministic=False)
        buffer_action = unscaled_action
        action = buffer_action
        return action, buffer_action


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                self.actor.reset_noise(env.num_envs)  # type: ignore[operator]
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, n_envs=env.num_envs)
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
            self._update_info_buffer(infos, dones)

            # store also the set of valid next actions for each transition, which is useful for algorithms that need to consider action masking or have a variable action space
            self._store_transition(replay_buffer, buffer_actions, new_obs, env.envs[0].get_candidate_action_KNNs(), rewards, dones, infos)  # type: ignore[arg-type]
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


    def _store_transition(
            self,
            replay_buffer: DictReplayBufferWithNextActions,
            buffer_action: np.ndarray,
            new_obs: np.ndarray | dict[str, np.ndarray],
            new_action_set: np.ndarray,
            reward: np.ndarray,
            dones: np.ndarray,
            infos: list[dict[str, Any]],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        next_action_set = deepcopy(new_action_set)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(
                            next_obs[i, :])  # type: ignore[assignment]


        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            next_action_set # added next_action_set to the replay buffer
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_


