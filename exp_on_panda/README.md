# Experiments on Panda-Gym

## Prepare Python Environment

Install [Panda-Gym](https://github.com/qgallouedec/panda-gym)

```bash
pip install panda-gym
```

_**Note**: all the following scripts are run in the directory of `exp_on_panda`. Previously, these experiments were in a separate repository. I have now moved them to a subdirectory under the IRPO repository, and I haven't run the scripts to check. So, there might be some path errors during execution, please check carefully._

```bash
# in the root dir of IRPO, then
cd exp_on_panda
```

## Prepare Demonstrations

Rollout demonstrations by a [PID controller](https://panda-gym.readthedocs.io/en/latest/usage/manual_control.html), refer to the Jupyter script `rollout/rollout_reach_by_pid.ipynb`.

## Pre-train policies by BC

```bash
bash scripts/train_bc/iter_1.sh
```

## Fine-tune policies by RL

```bash
bash scripts/train_rl_after_bc/iter_1_kl_1e-1.sh
```

## Train policies from scratch by RL

```bash
bash scripts/train_rl/iter_1.sh
```

## Update demonstrations

Refer to the Jupyter scripts in `rollout/rollout_my_reach_by_rl_bc.ipynb`.