#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    train_utils.py
    This module contains utility functions for training reinforcement learning agents using the Stable Baselines3 library. It includes mappings of algorithm types to their corresponding model classes, checks for argument compatibility, and functions to replace string representations of activation functions and optimizers with their actual classes and vice versa.
'''

import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, TD3, SAC, DDPG, DQN
from stable_baselines3.a2c import A2C
from sb3_contrib import RecurrentPPO, TRPO, TQC, MaskablePPO
from sb3_sa_contrib.dqn import iDQN

# Mapping algorithm type to model class and additional parameters
algorithm_models = {
    'ppo': PPO,
    'a2c': A2C,
    'rppo': RecurrentPPO,
    'trpo': TRPO,
    'ddpg': DDPG,
    'sac': SAC,
    'td3': TD3,
    'tqc': TQC,
    'dqn': DQN,
    'idqn': iDQN,
    'maskable_ppo': MaskablePPO  # Placeholder for MaskablePPO
}

recurrent_algorithms = ["rppo"]

activation_functions = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "ELU": nn.ELU
    }

def check_args(args):
    # Check for compatibility of arguments based on the algorithm type and other options. For example, if semantic ordering is enabled, the algorithm type must be "discrete". If saving embeddings is enabled, the algorithm type must be "projection" or "iterative". If using feature vectors as action points is enabled, the algorithm type must be "projection" or "iterative". Return a tuple of (is_valid, message) where is_valid is a boolean indicating whether the arguments are valid and message is a string describing the reason if they are not valid or "OK" if they are valid.
    if args.semantic_ordering and args.algorithm_type != "discrete":
        return False, "Semantic ordering is only compatible with discrete action spaces"
    if args.save_embeddings and (args.algorithm_type != "projection" and args.algorithm_type != "iterative"):
       return False, "Saving embeddings is only available for continuous environments!"
    if args.use_feature_vectors and (args.algorithm_type != "projection" and args.algorithm_type != "iterative"):
        return False, "The option to use feature vectors as action points is only compatible with continuous action spaces"
    return True, "OK"

def derive_env_type_folder_name(args):
    # Derive the environment type folder name based on the algorithm type and other options. For example, if the algorithm type is "discrete" and semantic ordering is enabled, the folder name should be "DO_discrete_semantic". If the algorithm type is "projection" and saving embeddings is enabled, the folder name should be "projection_embeddings". The function should check for the presence of certain substrings in the algorithm type and other options to construct the folder name accordingly
    env_type_folder_name = args.algorithm_type
    if args.semantic_ordering:
        env_type_folder_name += "_semantic"
    if args.algorithm == 'maskable_ppo':
        env_type_folder_name += "_M"
    if args.algorithm_type == "discrete" and args.GNN_observations:
        env_type_folder_name = "GO_" + env_type_folder_name
    elif args.algorithm_type == "discrete":
        env_type_folder_name = "DO_" + env_type_folder_name
    if args.approximate_distance:
        env_type_folder_name += "_approx"
    if args.sample_subset_actions:
        env_type_folder_name += "_sample"
    if args.pca_minimum_without_loss:
        env_type_folder_name += "_pca"
    return env_type_folder_name


def replace_with_classes(policy_kwargs):
    # Function to replace the strings with the actual classes
    global activation_functions
    if 'activation_fn' in policy_kwargs:
        policy_kwargs['activation_fn'] = activation_functions[policy_kwargs['activation_fn']]
    # Handle optimizers
    optimizers = {
        "Adam": optim.Adam,
        "SGD": optim.SGD
    }
    if 'optimizer_class' in policy_kwargs:
        policy_kwargs['optimizer_class'] = optimizers[policy_kwargs['optimizer_class']]

    return policy_kwargs


def replace_with_strings(policy_kwargs):
    # Function to replace the actual classes with their string representations for saving in config files
    activation_functions = {
        nn.ReLU: "ReLU",
        nn.LeakyReLU: "LeakyReLU",
        nn.Tanh: "Tanh",
        nn.Sigmoid: "Sigmoid",
        nn.ELU: "ELU"
    }
    if 'activation_fn' in policy_kwargs:
        policy_kwargs['activation_fn'] = activation_functions[policy_kwargs['activation_fn']]
    # Handle optimizers
    optimizers = {
        optim.Adam: "Adam",
        optim.SGD: "SGD"
    }
    if 'optimizer_class' in policy_kwargs:
        policy_kwargs['optimizer_class'] = optimizers[policy_kwargs['optimizer_class']]
    return policy_kwargs
