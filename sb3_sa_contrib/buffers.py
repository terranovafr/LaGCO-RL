#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

# This file is derived from Stable-Baselines3 (SB3)
# Original source: https://github.com/DLR-RM/stable-baselines3
# Copyright (c) 2019 Antonin Raffin (SB3 Author)
# Licensed under the MIT License

# Modification: Added next_action_set storage and sampling to DictReplayBuffer.
# This allows storing the set of valid next actions for each transition, which is useful for algorithms that need to consider action masking or have a variable action space.

from typing import Any, NamedTuple, Dict
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer

class DictReplayBufferSamplesStateAction(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    next_observations: Dict[str, th.Tensor]
    next_actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor | None = None

class DictReplayBufferWithNextActions(DictReplayBuffer):
    """
    DictReplayBuffer that also stores next_action_set for each next state.
    Returns DictReplayBufferSamplesStateAction when sampling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_action_set = [None] * self.buffer_size  # list to hold flexible arrays

    def add(
        self,
        obs: dict,
        next_obs: dict,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        next_action_set: np.ndarray = None,  # NEW
    ) -> None:
        # Call parent add
        super().add(obs, next_obs, action, reward, done, infos)

        if next_action_set is not None:
            # Store array directly in the list
            self.next_action_set[self.pos - 1] = next_action_set.copy()

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> DictReplayBufferSamplesStateAction:
        """
        Sample batch and return DictReplayBufferSamplesStateAction including next_actions.
        """
        # Randomly sample env indices for each batch element
        env_indices = np.random.randint(0, self.n_envs, size=(len(batch_inds),))

        # Normalize observations if env is provided
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env)

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)

        # Convert observations to torch tensors
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        # Convert next_action_set to torch tensor
        next_actions = [self.next_action_set[i] for i in batch_inds]

        return DictReplayBufferSamplesStateAction(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            next_actions=next_actions,
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )
