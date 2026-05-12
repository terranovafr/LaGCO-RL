#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    file_utils.py
    Utility functions for file handling, including JSON and YAML operations, as well as TensorBoard log extraction.
'''

import yaml
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import json
import shutil
import re
from datetime import datetime
from pathlib import Path


def sanitize_filename(name):
    # replace slash and spaces and remove weird chars
    name = name.replace("/", "_").replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name

def load_latest_yaml_from_split_folder(run_dir: Path, split: str):
    target_dir = run_dir / "test" / split
    if not target_dir.exists():
        return None

    yamls = sorted(target_dir.glob("compression_*.yaml"))
    if not yamls:
        return None

    return yamls[-1]

def load_json(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)

def save_json(data, folder_path, file_name="config.json"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data_path = os.path.join(folder_path, file_name)
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_yaml(file_path):
    with open(file_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def save_yaml(data, folder_path, file_name=None):
    if file_name is not None:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_path = os.path.join(folder_path, file_name)
    else:
        data_path = folder_path
    with open(data_path, 'w') as config_file:
        yaml.safe_dump(data, config_file)

# Extract a specific metric from a tensorboard log (x and y axis)
def extract_metric_data(log_dir, metric_name): # from tensorboard log
    # check if log_dir is valid
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} does not exist.")
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()
    if metric_name in event_acc.Tags()['scalars']:
        metric_events = event_acc.Scalars(metric_name)
        times = [event.step for event in metric_events]
        values = [event.value for event in metric_events]
        return np.array(times), np.array(values)
    else:
        raise ValueError(f"Metric {metric_name} not found in TensorBoard logs.")


# Read split file to determine training and validation sets (used during training)
def read_split_file(holdout, nets_folder, logs_folder):
    with open(os.path.join(nets_folder, "split.yaml"), 'r') as file:
        yaml_info = yaml.safe_load(file)
    train_ids = []
    for elem in yaml_info['training_set']:
        train_ids.append(elem['id'])
    val_ids = []
    for elem in yaml_info['validation_set']:
        val_ids.append(elem['id'])

    with open(os.path.join(logs_folder, "split.yaml"), 'w') as f:
        yaml.dump(yaml_info, f)
    if not holdout:
        val_ids = []
    return train_ids, val_ids


def remove_folder_and_files(folder):
    if os.path.exists(os.path.join(folder)):
        shutil.rmtree(folder)

def extract_timestamp(name):
    m = re.search(r"\d{14}", name)
    if m:
        return datetime.strptime(m.group(), "%Y%m%d%H%M%S")
    return datetime.min