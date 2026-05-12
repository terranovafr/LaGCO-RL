#!/bin/bash

CONFIG_FILE="config.yaml"

# Create file if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    touch "$CONFIG_FILE"
    echo "Created $CONFIG_FILE"
fi

# Ask user for Hugging Face key
read -p "Enter your Hugging Face API key: " HF_KEY

# Overwrite config with clean structure
cat > "$CONFIG_FILE" <<EOF
tsp:
  default_envs: null

huggingface:
  key: "$HF_KEY"
EOF

echo "Config written to $CONFIG_FILE"