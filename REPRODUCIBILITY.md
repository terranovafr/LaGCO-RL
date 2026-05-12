# Reproducibility Guide

This document provides step-by-step instructions to reproduce all experiments and results presented in our paper. The process includes scenario generation, optional hyperparameter optimization, training, evaluation, and plotting.

---

## Data Availability

All datasets, logs, and results are available on Zenodo: https://doi.org/10.5281/zenodo.20019625

The repository already includes:
- Generated scenarios with proper metadata to be used to support different training and splitting strategies
- Training/hyperopt logs
- Evaluation outputs

You can either:
- Directly reuse the provided data (recommended for quick reproduction)
- Fully regenerate all experiments from scratch

---

## Setup & Installation
Start with the setup of proper files needed by the library:
```
chmod +x setup.sh
./setup.sh
```
Install dependencies using:
```bash
conda env create -f environment.yml
```

## 1. Hyperparameter Optimization (Optional)

You may either:
- Use hyperparameters reported in the paper (Appendix C)
- Re-run hyperparameter optimization

In the paper, hyperparameter tuning was performed only on TSP for simplicity, then transferred to other benchmarks.

---

### Generate Hyperopt Set with Validation Set (E.g., TSP only)
```
cd generators
```
```
python3 random_generator.py \
    -n 10 \
    -e tsp \
    -ms 10000 \
    -s random \
    --force_saving \
    --train_pct 0.1 \
    --val_pct 0.9 \
    --test_pct 0
```
---
Maximum Sweeps (-ms) controls the number of heuristic iterations used to approximate optimal solutions.
- Higher values → better normalization of performance scores
- Lower values → faster generation
Change hyper-parameters accordingly to customize this stage (e.g., faster generation with lower number of sweeps and instances).

### GAE Hyperparameter Optimization
```
cd gae
```
```
python3 hyperopt_gae.py \
    -e tsp \
    -ti 10000 \
    --optimization_type tpe \
    --validation \
    --num_trials 25
```
---

### Agent Hyperparameter Optimization
Train a simple GAE used to generate embeddings for the hyperparameter optimization of agents, then run the optimization for latent algorithm types:
```
cd gae
```
```
python3 train_gae.py \
    -e tsp \
    -ti 10000 
```
Set the trained GAE as default for this benchmark once finished.
Start the hyperparameter optimization for the different algorithm types:
```
cd agents
```
```
TYPES=(discrete projection iterative)
ALGOS=(maskable_ppo ppo idqn)

for i in "${!TYPES[@]}"
do
    TYPE=${TYPES[$i]}
    ALGO=${ALGOS[$i]}

    python3 hyperopt_agent.py \
        -e tsp \
        -at $TYPE \
        -ti 100000 \
        -algo $ALGO \
        --optimization_type tpe \
        --num_trials 25 \
        --validation
done
```
---

## 2. Scenario Generation for the Generalization Study

Two options are available for the generation of scenarios to be used in the generalization study.

### Option 1 — Regenerate Scenarios

With this option, scenarios are regenerated based on a proper custom distribution of parameters. 
We recommend using the same configuration ranges as in the paper, present in the data repository in `config/*_ranges.yaml` and `config/*_config.yaml`.

Run:
```
for BENCHMARK_NAME in tsp mvc maxcut vmp cyberattack ospf_engineering traffic_engineering
do
    python3 random_generator.py \
        -n 101 \
        -e $BENCHMARK_NAME \
        -ms 5000 \
        -s smallest_train \
        --force_saving \
        --train_pct 0.01 \
        --val_pct 0 \
        --test_pct 0.99
done
```


Values used in the paper:
- tsp: 10000  
- mvc: 10000  
- maxcut: 2000  
- vmp: 5000  
- cyberattack: not required  
- ospf_engineering: 2000  
- traffic_engineering: 2000  

---

### Option 2 — Use Pre-generated Scenarios

1. Copy files from Zenodo: `generalization/envs/*` → `data/env_samples/`
2. Update ROOT `config.yaml` by setting the environment folder path to each benchmark:
```
BENCHMARK_NAME:
  default_envs: FOLDER_NAME
...
```

Example:
```
cyberattack:
  default_envs: cyberattack_20261004
```

---

## 3. Generalization Study

A script support the generalization experiment by automating GAE training, agent training, and evaluation.

```
cd experiments
```
```
for BENCHMARK in tsp mvc maxcut vmp cyberattack ospf_engineering traffic_engineering
do
    python3 assess_agent_generalization.py \
        -e $BENCHMARK \
        -ni_rl 100000 \
        -algo ppo \
        -ne_test 5 \
        -ne_train 25 \
        -at DO_discrete DO_discrete_valid GO_discrete GO_discrete_valid projection iterative \
        -ni_gae 5000 \
        --skip_generation \
        --num_runs 5 \
        --env_sampling smallest mean largest
done
```
Notes:

- ppo is automatically replaced:
  - maskable_ppo for discrete variants relying on masking
  - idqn for iterative variants
- All parameters can be customized to more accurate experiments with larger number of runs per strategy or any desired customization

---

### K-Fold Generalization
```
for BENCHMARK in tsp mvc maxcut vmp cyberattack ospf_engineering traffic_engineering
do
    python3 assess_agent_generalization.py \
        -e $BENCHMARK \
        -ni_rl 100000 \
        -algo ppo \
        -ne_test 5 \
        -ne_train 25 \
        -at DO_discrete DO_discrete_valid GO_discrete GO_discrete_valid projection iterative \
        -ni_gae 5000 \
        --skip_generation \
        --num_runs 5 \
        --env_sampling random_pct \
        -rsp 20
done
```
This creates 5 folds, each with 20 percent of the data.

---

## 4. Plotting Generalization Comparison

```
cd plotting
```
```
for BENCHMARK in tsp mvc maxcut vmp cyberattack ospf_engineering traffic_engineering
do
    python3 plot_agent_comparison.py \
        -f logs \
        -e $BENCHMARK \
        --cases smallest largest mean random_pct \
        -p boxplot \
        -i max \
        --no_training
done 
```
Options:
- Remove --no_training to include training curves
- To generate publication-ready tables
  - --print_format latex
  - --print_format latex_extended
- Change -i to overall to not take the maximum performance per instance but rather average all runs

---

## 5. Action-Time Scalability Experiment

### Scenario Generation

The repository supports the generation of cartesian evolutions of instances to study action-selection time scaling with instance size. To generate these scenarios, run:

```
cd generators
```
```
for BENCHMARK in tsp mvc maxcut vmp cyberattack ospf_engineering traffic_engineering
do
    python3 random_cartesian_scalability_generator.py \
        -e $BENCHMARK \
        --min_size 10 \
        --max_size 150 \
        --interval 20 \
        --num_envs 1 \
        -ms 2 \
        --force_saving
done
```

Notes:

- Adjust sizes and interval depending on benchmark complexity (e.g., traffic engineering has path-based actions and hence grows faster than TSP)
- You can use small -ms (e.g., 2) and no specific GAE since solution quality is not relevant; focus is on action-space scaling

---

### Training Configuration

Before running, update `agents/config/train_config.yaml` to set a reasonable checkpoint saving frequency to have a neural network, in order to have it soon possible to use for evaluating action-selection time:

`train_config.yaml → checkpoint-save-freq: 2000`

Subsequently, compare all possible algorithm type on this cartesian scenario set to have a comprehensive view of action-selection time scaling with instance size across all approaches.
```
cd experiments
```
```
for BENCHMARK in tsp mvc maxcut vmp cyberattack ospf_engineering traffic_engineering
do
    python3 assess_agent_cartesian.py \
        -e $BENCHMARK \
        -at projection iterative \
        -ne 10 \
        -algo ppo \
        -es smallest \
        --skip_gae \
        --num_iter_rl 2000 \
        -o action_time
done 
```
---

### Plot Action-Time Results

The repository supports plotting of action-selection time scaling results with instance size, comparing all approaches. To generate these plots, run:

```
cd plotting
```
```
python3 plot_action_time_selection.py \
    -f logs \
    -e BENCHMARK \
    --solutions projection iterative
```

Outputs:
- Boxplots of action selection time
- Polynomial regression scaling coefficients

---

## Summary

To fully reproduce the paper:

1. Generate or load scenarios  
2. (Optional) Run hyperparameter optimization  
3. Train GAE and agents  
4. Run generalization experiments   
5. Run scalability experiments  

Additional scripts are available for training curves (TensorBoard logs), GAE loss curves, and other appendix figures and can be generated using scripts in the `plotting/` directory.

---

## Reproducibility Notes

- Ensure consistent seeds when comparing results  
- Hardware differences may slightly affect runtime performance  
- Hyperparameter transfer from TSP to other benchmarks is intentional for simplifying this heavy process  
- Using Zenodo data without regeneration of scenarios ensures exact reproducibility of reported results  