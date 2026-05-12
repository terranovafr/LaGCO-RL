#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

import os
from torch.utils.tensorboard import SummaryWriter
import collections
from stable_baselines3.common.callbacks import BaseCallback
from sb3_sa_contrib.dqn import iDQN
import numpy as np
import sb3_contrib


class TrainingCallback(BaseCallback):
    # tracks episode metrics and logs their mean over a sliding window of recent episodes
    def __init__(self, env, window_size=10):
        super().__init__()
        self.env = env
        self.window_size = window_size
        self.metric_queues = {}  # Dict[str, deque]

    def _on_training_start(self):
        self.metric_queues.clear()

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]
        if not done:
            return True

        # Fetch episode metrics from the environment
        metrics = self.env.envs[0].get_metrics()

        for key, value in metrics.items():
            if key not in self.metric_queues:
                self.metric_queues[key] = collections.deque(maxlen=self.window_size)
            self.metric_queues[key].append(value)

        # Log mean of each metric
        for key, values in self.metric_queues.items():
            print(f"[TrainingCallback] {key}: {values[-1]} (last {len(values)} values)")
            if len(values) > 0:
                mean_val = np.mean(values)
                self.logger.record(f"train/Mean {key}", mean_val)

        return True


class ValidationCallback(BaseCallback):
    # periodically runs validation episodes and logs metrics, supports early stopping based on a specified score key
    def __init__(self, val_envs, val_freq=10000, val_switch_interval=1, n_val_episodes=5, score_key="reward",
                 early_stopping=False, patience=5, gamma=1, log_dir="logs", save_dir="checkpoints", verbose=0):
        super().__init__(verbose)
        self.val_envs = val_envs if isinstance(val_envs, list) else [val_envs]
        self.val_freq = val_freq
        self.n_val_episodes = n_val_episodes
        self.val_switch_interval = val_switch_interval
        self.score_key = score_key
        self.gamma = gamma
        self.early_stopping = early_stopping
        self.patience = patience
        self.current_patience = 0
        self.best_score = -np.inf
        self.log_dir = log_dir
        self.save_dir = save_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.val_freq == 0:
            all_metrics = []
            self.val_envs[0].envs[0].set_mode('val')
            self.val_envs[0].envs[0].set_switch_interval(self.val_switch_interval)
            # Launch validation episodes
            for _ in range(self.n_val_episodes):
                done = False
                obs = self.val_envs[0].reset()
                num_steps = 0
                undiscounted_return = 0.0
                discounted_return = 0.0
                metrics = {}
                while not done:
                    num_steps += 1
                    # support for action masking if the model is MaskablePPO or iDQN
                    if isinstance(self.model, sb3_contrib.MaskablePPO):
                        mask = self.val_envs[0].envs[0].action_masks()
                        action, _ = self.model.predict(obs, deterministic=False, action_masks=mask)
                    elif isinstance(self.model, iDQN):
                        action = self.model.predict(obs, deterministic=False, no_exploration=True)
                    else:
                        action, _ = self.model.predict(obs, deterministic=False)

                    obs, reward, done, truncated = self.val_envs[0].step(action)
                    undiscounted_return += reward[0]
                    discounted_return += (self.gamma ** (num_steps - 1)) * reward[0]
                    if done[0] or truncated[0]['TimeLimit.truncated']:
                        break
                    else:
                        self.val_envs[0].envs[0].current_env.update_metrics()
                        metrics = self.val_envs[0].envs[0].get_metrics()

                all_metrics.append(metrics)
                all_metrics[-1]['num_steps'] = num_steps
                all_metrics[-1]['undiscounted_return'] = undiscounted_return
                all_metrics[-1]['discounted_return'] = discounted_return
            avg_metrics = {k: np.mean([m[k] for m in all_metrics if k in m]) for k in all_metrics[0]}

            for k, v in avg_metrics.items():
                self.writer.add_scalar(f"validation/{k}", v, self.num_timesteps)

            # only for undiscounted return we add also min and max
            self.writer.add_scalar(f"validation/undiscounted_return_min",
                                   np.min([m['undiscounted_return'] for m in all_metrics]),
                                   self.num_timesteps)

            self.writer.add_scalar(f"validation/undiscounted_return_max",
                                      np.max([m['undiscounted_return'] for m in all_metrics]),
                                      self.num_timesteps)
            self.writer.add_scalar(f"validation/discounted_return_min",
                                   np.min([m['discounted_return'] for m in all_metrics]),
                                   self.num_timesteps)
            self.writer.add_scalar(f"validation/discounted_return_max",
                                      np.max([m['discounted_return'] for m in all_metrics]),
                                      self.num_timesteps)

            # handle model saving and early stopping
            score = avg_metrics.get(self.score_key, None)
            if score is None:
                raise ValueError(f"Score key '{self.score_key}' not found in evaluation metrics.")

            if score > self.best_score:
                self.best_score = score
                self.current_patience = 0
                self.model.save(os.path.join(self.save_dir, f"best_model_{score:.4f}.zip"))
                if self.verbose:
                    print(f"[ValidationCallback] New best score: {score:.4f}. Model saved.")
            else:
                self.current_patience += 1
                if self.verbose:
                    print(f"[ValidationCallback] No improvement. Patience: {self.current_patience}/{self.patience}")

            if self.early_stopping and self.current_patience >= self.patience:
                if self.verbose:
                    print("[ValidationCallback] Early stopping triggered.")
                return False
            self.val_envs[0].envs[0].set_mode('train')
            self.val_envs[0].envs[0].set_switch_interval(False)
        return True

    def _on_training_end(self):
        self.writer.close()
