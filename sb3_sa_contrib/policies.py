#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

# This file is derived from Stable-Baselines3 (SB3)
# Original source: https://github.com/DLR-RM/stable-baselines3
# Copyright (c) 2019 Antonin Raffin (SB3 Author)
# Licensed under the MIT License

# Modification: Added StateActionQNetwork and StateActionDQNPolicy to implement a DQN variant that takes both state and action as input to the Q-network. This allows for more flexible action representations, such as continuous actions or action sets, rather than just discrete action indices, and support variable action sets.

import torch.nn as nn
from gymnasium.spaces import Box
import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.dqn.policies import QNetwork
import torch

def _dict_to_tensor(observation):
    # Convert each tensor in the dict to 1D and concatenate
    obs_list = []
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            obs_list.append(value.flatten())
        else:
            obs_list.append(torch.tensor(value).float().flatten())
    return torch.cat(obs_list)

class StateActionQNetwork(BasePolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch=None,
        activation_fn=nn.ReLU,
        normalize_images=False
    ):
        super().__init__(observation_space, action_space)

        if net_arch is None:
            net_arch = [256, 256]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        state_dim = get_flattened_state_dim(self.observation_space)
        self.action_dim = int(np.prod(self.action_space.shape))

        input_dim = state_dim + self.action_dim

        mlp_layers = create_mlp(
            input_dim,
            1,  # Q(s,a) -> scalar
            self.net_arch,
            self.activation_fn,
        )

        self.net = nn.Sequential(*mlp_layers)

    def set_candidate_actions(self, candidate_actions):
        self.candidate_actions = candidate_actions

    def forward(self, states, actions):
        if isinstance(states, dict):
            state_tensors = [states[k].float() for k in sorted(states.keys())]
            state_tensors = [
                v.view(v.shape[0], -1) if v.dim() > 2 else v
                for v in state_tensors
            ]
            states = th.cat(state_tensors, dim=1)
        else:
            states = states.float()

        qs = []
        if isinstance(actions, list):
            for i, action_set in enumerate(actions):
                action_set = th.as_tensor(action_set, device=states.device,
                                          dtype=th.float32)  # shape: [num_actions_i, action_dim]
                state_i = states[i].unsqueeze(0).expand(action_set.shape[0], -1)  # shape: [num_actions_i, state_dim]
                x = th.cat([state_i, action_set], dim=1)  # shape: [num_actions_i, state_dim + action_dim]
                q = self.net(x)  # shape: [num_actions_i, q_dim]
                qs.append(q)
        else:
            # --- Case: single action per batch element ---
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            actions = actions.float()
            x = th.cat([states, actions], dim=1)
            q = self.net(x)  # [batch_size, q_dim]
            return q
        return qs


    def set_exploration_rate(self, exploration_rate):
        self.exploration_rate = exploration_rate

    def _predict(self, observation):
        obs_tensor = _dict_to_tensor(observation).to(self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        batch_size = obs_tensor.shape[0]
        n_actions = len(self.candidate_actions)

        # Convert candidate_actions to tensor
        all_actions = th.tensor(self.candidate_actions, dtype=th.float32, device=self.device)  # (n_actions, action_dim)

        # Repeat actions for each element in the batch
        all_actions_expanded = all_actions.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, n_actions, action_dim)
        obs_expanded = obs_tensor.unsqueeze(1).repeat(1, n_actions, 1)  # (batch_size, n_actions, obs_dim)
        # Flatten both to feed Q-network
        obs_flat = obs_expanded.view(-1, obs_tensor.shape[1])  # (batch_size * n_actions, obs_dim)
        actions_flat = all_actions_expanded.view(-1, self.action_dim)  # (batch_size * n_actions, action_dim)
        # Concatenate for Q-network
        x = th.cat([obs_flat, actions_flat], dim=1)  # (batch_size * n_actions, obs_dim + action_dim)
        with th.no_grad():
            q_values = self.net(x)
            q_values = q_values.view(batch_size, n_actions)  # reshape back to (batch_size, n_actions)
        # Choose the action index for each observation in the batch
        action_idx = q_values.argmax(dim=1)  # (batch_size,)
        # Get the actual action vectors
        chosen_action = all_actions[action_idx]  # (batch_size, action_dim)
        return chosen_action


def get_flattened_state_dim(obs_space):
    total_dim = 0
    for key, space in obs_space.spaces.items():
        if isinstance(space, Box):
            total_dim += int(np.prod(space.shape))
        else:
            raise NotImplementedError(f"Space type {type(space)} not handled")
    return total_dim

class StateActionDQNPolicy(DQNPolicy):
    def make_q_net(self) -> QNetwork:
        return StateActionQNetwork(
            **self.net_args
        ).to(self.device)

    def _predict(self, obs: PyTorchObs) -> th.Tensor:
        return self.q_net._predict(obs)

    def predict(
            self,
            observation: np.ndarray | dict[str, np.ndarray],
            state: tuple[np.ndarray, ...] | None = None,
            episode_start: np.ndarray | None = None,
            deterministic: bool = False,
            no_exploration: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        self.set_training_mode(False)
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        with th.no_grad():
            actions = self._predict(obs_tensor)
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)  # type: ignore[assignment]
        return actions, state  # type: ignore[return-value]





