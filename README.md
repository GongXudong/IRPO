# IRPO
Official code for "Iterative Regularized Policy Optimization with Imperfect Demonstrations" (ICML2024).

## Prepare python environment

```bash
conda create --name IRPO python=3.8
conda activate IRPO
conda install --file requirements.txt
```

## Generate Demonstrations

### Generate demonstrations with PID controller

Sample from $ V \times \Mu \times \Chi = [v_{min}:v_{max}:v_{interval}] \times [mu_{min}:mu_{max}:mu_{interval}] \times [chi_{min}:chi_{max}:chi_{interval}] $ with PID controller and save sampled trajectories in demonstrations/data/\{step-frequence\}hz\_\{$v_{interval}$\}\_\{$mu_{interval}$\}\_\{$chi_{interval}$\}\_\{data-dir-suffix\}.

```bash
# sample trajectories by single-processing
python demonstrations/rollout_trajs/rollout_by_pid.py --data-dir-suffix v0 --step-frequence 10 --v-min 100 --v-max 300 --v-interval 10 --mu-min -85 --mu-max 85 --mu-interval 5 --chi-min -170 --chi-max 170 --chi-interval 5

# sample trajectories by multi-processing (base on Ray)
python demonstrations/rollout_trajs/rollout_by_pid_parallel.py --data-dir-suffix v1 --step-frequence 10 --v-min 100 --v-max 300 --v-interval 10 --mu-min -85 --mu-max 85 --mu-interval 5 --chi-min -170 --chi-max 170 --chi-interval 5
```

### Update demonstrations with policy

Update demonstrations in {demos-dir} with policy in {policy-ckpt-dir}.

```bash
python demonstrations/rollout_trajs/rollout_by_policy_and_update_demostrations.py --policy-ckpt-dir checkpoints/sac_her/best_model --env-config-dir configs/env/env_config_for_ppo.json --demos-dir demonstrations/data/10hz_10_5_5_v1
```

### Augment demonstrations

Augment trajectories based on $\chi$'s symmetry.

```bash
python demonstrations/utils/augment_trajs.py --demos-dir demonstrations/data/10hz_10_5_5_v1
```

## Train policies

### Pre-train policy with Behavioral Cloning

Train policy by Behaviroal Cloning with config file in {config-file-name}

```bash
python train_scripts/train_with_bc_ppo.py --config-file-name configs/train/iteration_1/lambda_1e-3/ppo_bc_config_10hz_128_128_1.json
```

### Train policy with PPO

Train policy by PPO with config file in {config-file-name}

```bash
python train_scripts/train_with_rl_ppo.py --config-file-name configs/train/iteration_1/lambda_1e-3/ppo_bc_config_10hz_128_128_1.json
```

### Fine-tune a pre-trained policy with PPO

Fine-tune a pre-trained policy with PPO with config file in {config-file-name}

```bash
python train_scripts/train_with_rl_bc_ppo.py --config-file-name configs/train/iteration_1/lambda_1e-3/ppo_bc_config_10hz_128_128_1.json
```

## Citation

```bibtex
@inproceedings{gong2024iterative,
  title={Iterative Regularized Policy Optimization with Imperfect Demonstrations},
  author={Xudong, Gong and Dawei, Feng and Kele, Xu and Yuanzhao, Zhai and ChengKang, Yao and Weijia, Wang and Bo, Ding and Huaimin, Wang},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```
