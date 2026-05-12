#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    test_utils.py
    This module provides utility functions for evaluating the performance of RL agents in various environments. It includes functions to compute average performances, action times, and to summarize results in a structured format
'''

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from collections import defaultdict
import sb3_contrib
from utils.math_utils import iqm, bootstrap_ci
import json
from utils.math_utils import set_seeds
from sb3_sa_contrib.dqn import iDQN
from pathlib import Path
import copy

def summarize_df(df, save_folder="summary_metrics_json"):
    # Create save folder if it doesn't exist
    save_path = Path(save_folder)
    save_path.mkdir(exist_ok=True, parents=True)

    summary_per_env = {}
    overall_envs = []

    # 1. Compute per environment
    for env_id, env_df in df.groupby("environment_ID"):
        summary_per_env[env_id] = {}

        for col in env_df.columns:
            if col in ["episode", "environment_ID"]:
                continue
            values = env_df[col].values
            # check if they are boolean and in case skip
            if isinstance(values[0], (bool, np.bool_)):
                continue
            summary_per_env[env_id][col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "max": float(np.max(values)),
                "min": float(np.min(values)),
                "iqm": float(iqm(values)),
                "ci95": bootstrap_ci(values)
            }

        # Save per environment JSON
        with open(save_path / f"summary_env_{env_id}.json", "w") as f:
            json.dump({env_id: summary_per_env[env_id]}, f, indent=4)

        overall_envs.append(env_df)

    # 2. Aggregate across environments
    all_env_df = pd.concat(overall_envs, ignore_index=True)
    summary_all_envs = {}
    for col in all_env_df.columns:
        if col in ["episode", "environment_ID"]:
            continue
        values = all_env_df[col].values
        summary_all_envs[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
            "iqm": float(iqm(values)),
            "ci95": bootstrap_ci(values)
        }

    with open(save_path / f"summary_all_envs.json", "w") as f:
        json.dump({"all_environments": summary_all_envs}, f, indent=4)

    return summary_per_env, summary_all_envs


def calculate_average_performances(model, gym_env, seeds, proto_knn, logger, num_envs=1, num_episodes=50, avoid_random=False, verbose=1):
    # Calculate average performances of the agent and random agent (if not avoided) across multiple episodes and environments. Returns a DataFrame with metrics for each episode and a dictionary of lists for each metric.
    episode_list = []
    metrics_lists = {}

    stats_data = []

    if verbose:
        logger.info("Computing average performances of the agent...")
        if not avoid_random:
            logger.info("Computing average performances of the random agent as well...")

    for episode in tqdm(range(num_episodes), desc='Saving performances of each episode...'):
        # determine which episode is with a certain environment and associate corresponding seed
        set_seeds(seeds[episode // num_envs])

        # Play random actions to calculate random agent performance
        if not avoid_random:
            stats_data.append({
                'episode': episode,
                "environment_ID": copy.deepcopy(gym_env.envs[0].current_env_index)
            })

            random_agent_metrics = play_random_agent_episode_until_done(gym_env)

            for key, value in random_agent_metrics.items():
                stats_data[-1][f"random_agent_{key}"] = value
                metrics_lists.setdefault(f"random_agent_{key}_list", []).append(value)

        stats_data.append({
            'episode': episode,
            "environment_ID": copy.deepcopy(gym_env.envs[0].current_env_index)
        })
        agent_metrics, _ = play_agent_episode_until_done(gym_env, model, proto_knn)

        episode_list.append(episode)
        for key, value in agent_metrics.items():

            metrics_lists.setdefault(f"agent_{key}_list", []).append(value)

        for key, value in agent_metrics.items():
            stats_data[-1][f"agent_{key}"] = value

    merged = defaultdict(dict)

    for row in stats_data:
        ep = row["episode"]
        merged[ep].update(row)

    # Convert to DataFrame
    df = pd.DataFrame(list(merged.values()))
    return (df, metrics_lists)


def calculate_average_action_time(model, gym_env, config, proto_knn, logger, num_episodes=50, verbose=1):
    # Calculate average action time of the agent across multiple episodes and environments. Returns a list of action times for each episode and a list of total episode times.
    episode_times = []
    action_times = []
    if verbose:
        logger.info("Computing average action time of the agent...")

    for _ in tqdm(range(num_episodes), desc='Saving action times of each episode...'):
        start_time = time.time()
        _, action_time = play_agent_episode_until_done(gym_env, model, proto_knn)
        end_time = time.time()
        action_times.extend(action_time)
        episode_times.append(end_time - start_time)
    return action_times, episode_times

def play_random_agent_episode_until_done(env):
    # # One episode of a random agent
    env.reset()
    while(True):
        action = env.action_space.sample()
        next_state, _, done, _ = env.step([action])
        if done:
            break
    return env.envs[0].get_metrics()

def play_agent_episode_until_done(env, model, proto_knn=None):
    # One episode of the agent, using ProtoKNN if provided to select the best action from the candidate actions generated by the RL model. Returns the metrics of the episode and a list of action times for each step.
    state = env.reset()
    num_steps = 0
    time_diffs = []
    while True:
        with (torch.no_grad()):
            start_time = time.time()
            if proto_knn is None:
                # Standard RL prediction
                if isinstance(model, sb3_contrib.MaskablePPO):
                    mask = env.envs[0].action_masks()
                    action, _ = model.predict(state, deterministic=False, action_masks=mask)
                elif isinstance(model, iDQN):
                    action = model.predict(state, deterministic=False, no_exploration=True)
                else:
                    action, _ = model.predict(state, deterministic=False)
                diff_time = time.time() - start_time
            else:
                # --- Use ProtoKNN to select the best action ---
                # Get candidate action vectors from the RL model
                if isinstance(model, sb3_contrib.MaskablePPO):
                    mask = env.action_masks()
                    candidate_action, _ = model.predict(state, deterministic=False, action_masks=mask)
                elif isinstance(model, iDQN):
                    candidate_action = model.predict(state, deterministic=False, no_exploration=True)
                else:
                    candidate_action, _ = model.predict(state, deterministic=False)

                diff_time = time.time() - start_time
                if proto_knn:
                    env.envs[0].current_env.store_KNNs(candidate_action[0], no_normalization=True)
                    new_start_time = time.time()
                    action = proto_knn.predict(state, deterministic=False, no_exploration=True)
                    diff_time += time.time() - new_start_time
                    action = [env.envs[0].current_env.convert_action_to_normalized_space(action[0])]

        # Step environment with chosen action
        next_state, reward, done, _ = env.step(action)
        time_diffs.append(diff_time)
        state = next_state
        num_steps += 1
        if done:
            break
    metrics = env.envs[0].get_metrics()
    metrics['num_steps'] = num_steps
    return metrics, time_diffs
