#!/bin/bash

ENVS=("maxcut" "mvc" "ospf_engineering" "traffic_engineering" "vmp")

for ENV in "${ENVS[@]}"; do
  echo "Running for environment: $ENV"

  # Group of 3 runs
  python plot_tensorboard_metric.py -f logs_nonrevisiting -e "$ENV" -s DO_discrete GO_discrete --cases largest smallest --spread std

  python plot_tensorboard_metric.py -f logs_nonrevisiting -e "$ENV" -s DO_discrete GO_discrete --cases mean random_pct --spread std

  python plot_tensorboard_metric.py -f logs_nonrevisiting -e "$ENV" -s DO_discrete_M GO_discrete_M --cases largest smallest --spread std

  python plot_tensorboard_metric.py -f logs_nonrevisiting -e "$ENV" -s DO_discrete_M GO_discrete_M --cases mean random_pct --spread std

  python plot_tensorboard_metric.py -f logs_nonrevisiting -e "$ENV" -s projection iterative --cases largest smallest --spread std

  python plot_tensorboard_metric.py -f logs_nonrevisiting -e "$ENV" -s projection iterative --cases mean random_pct --spread std

done