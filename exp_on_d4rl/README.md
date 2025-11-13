# Experiments on D4RL

## Prepare Python Environment

Refer to [D4RL](https://github.com/Farama-Foundation/D4RL) for setting up the RL environments, the other python packages are the same as `requirements.txt`.

_**Note: all the following scripts are run in the directory of `exp_on_d4rl`. Previously, these experiments were in a separate repository. I have now moved them to a subdirectory under the IRPO repository, and I haven't run the scripts to check. So, there might be some path errors during execution, please check carefully._

```bash
# in the root dir of IRPO, then
cd exp_on_d4rl
```

## Prepare Demonstrations

BC training scripts will download demonstrations automatically with the help of D4RL.

## Pre-train policies by BC

```bash
# HalfCheetah
bash scripts/train_bc/medium/iter_1/halfcheetah.sh

# Hopper
bash scripts/train_bc/medium/iter_1/hopper.sh
```

## Fine-tune policies by RL

```bash
# HalfCheetah
bash scripts/train_rl_after_bc/medium/iter_1/halfcheetah.sh

# Hopper
bash scripts/train_rl_after_bc/medium/iter_1/hopper.sh
```

## Train policies from scratch by RL

```bash
# HalfCheetah
bash scripts/train_rl/halfcheetah.sh

# Hopper
bash scripts/train_rl/hopper.sh
```

## Update demonstrations

Refer to the Jupyter scripts in `rollout/rollout.ipynb` and `rollout/rollout_by_multi_policies.ipynb`