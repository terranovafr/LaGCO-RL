# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

ENVS=("tsp" "maxcut" "mvc" "ospf_engineering" "cyberattack" "traffic_engineering" "vmp")

for ENV in "${ENVS[@]}"; do
  echo "Running for environment: $ENV"

  # Group of 3 runs
  python plot_tensorboard_metric.py -f logs -e "$ENV" -s DO_discrete GO_discrete --cases largest smallest --spread std

  python plot_tensorboard_metric.py -f logs -e "$ENV" -s DO_discrete GO_discrete --cases mean random_pct --spread std

  python plot_tensorboard_metric.py -f logs -e "$ENV" -s DO_discrete_M GO_discrete_M --cases largest smallest --spread std

  python plot_tensorboard_metric.py -f logs -e "$ENV" -s DO_discrete_M GO_discrete_M --cases mean random_pct --spread std

  python plot_tensorboard_metric.py -f logs -e "$ENV" -s projection iterative --cases largest smallest --spread std

  python plot_tensorboard_metric.py -f logs -e "$ENV" -s projection iterative --cases mean random_pct --spread std

done