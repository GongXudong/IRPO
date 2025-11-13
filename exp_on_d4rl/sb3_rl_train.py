import gym
import d4rl
import numpy as np
from pathlib import Path
import logging
import torch as th
import argparse
from copy import deepcopy
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize, sync_envs_normalization
from stable_baselines3.common.env_util import make_vec_env

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.sb3_schedule import linear_schedule
from configs.load_config import load_config
from utils.sb3_env_utils import make_env
from utils.sb3_callbacks import SaveVecNormalizeCallback

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="iter_1/seed1/halfcheetah_medium_256_256.json")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

ENV_NAME = custom_config["env"]["name"]
ENV_NORMALIZE = custom_config["env"].get("normalize", False)

GAMMA = custom_config["rl_bc"].get("gamma", 0.99)
GAE_LAMBDA = custom_config["rl_bc"].get("gae_lambda", 0.95)
NET_ARCH = custom_config["rl_bc"]["net_arch"]

RL_EXPERIMENT_NAME = custom_config["rl"]["experiment_name"]
SEED = custom_config["rl"]["seed"]
RL_TRAIN_STEPS = custom_config["rl"].get("train_steps", 1e6)
PPO_BATCH_SIZE = custom_config["rl"].get("batch_size", 64)
RL_ENT_COEF = custom_config["rl"].get("ent_coef", 0.0)
ROLLOUT_PROCESS_NUM = custom_config["rl"].get("rollout_process_num", 1)
N_STEPS = custom_config["rl"].get("n_steps", 512)
N_EPOCHS = custom_config["rl"].get("n_epochs", 5)
LEARNING_RATE = custom_config["rl"].get("learning_rate", 1e-4)
VF_COEF = custom_config["rl"].get("vf_coef", 0.5)
MAX_GRAD_NORM = custom_config["rl"].get("max_grad_norm", 0.5)
CLIP_RANGE = custom_config["rl"].get("clip_range", 0.2)
EVAL_FREQ = custom_config["rl"].get("eval_freq", 256)

np.seterr(all="raise")  # 检查nan

def get_ppo_algo(env):
    policy_kwargs = dict(
        # full_std=True,  # 使用state dependant exploration
        # # squash_output=True,  # 使用state dependant exploration
        # net_arch=dict(
        #     pi=NET_ARCH,
        #     vf=deepcopy(NET_ARCH)
        # ),
        # activation_fn=th.nn.Tanh,
        # ortho_init=True,
        # optimizer_class=th.optim.Adam,
        # optimizer_kwargs={
        #     "eps": 1e-5
        # }
        log_std_init=-2,
        ortho_init=False,
        activation_fn=th.nn.ReLU,
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        ),
    )

    return PPO(
        policy=MlpPolicy, 
        env=env, 
        seed=SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=N_STEPS,  # 采样时每个环境采样的step数
        n_epochs=N_EPOCHS,  # 采样的数据在训练中重复使用的次数
        ent_coef=RL_ENT_COEF,
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration,
        normalize_advantage=True,
        # learning_rate=linear_schedule(3e-4),
        learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE, 
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF,
    )

def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "rl_single" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    # 1.准备环境

    # 训练使用的环境
    vec_env = SubprocVecEnv([make_env(env_id=ENV_NAME, rank=i, scale_obs=False) for i in range(ROLLOUT_PROCESS_NUM)])
    # vec_env = make_vec_env(env_id=ENV_NAME, n_envs=ROLLOUT_PROCESS_NUM, vec_env_cls=SubprocVecEnv)
    # evaluate_policy使用的测试环境
    env_num_used_in_eval = 10
    eval_env = SubprocVecEnv([make_env(env_id=ENV_NAME, rank=i, scale_obs=False) for i in range(env_num_used_in_eval)])
    # eval_env = make_vec_env(env_id=ENV_NAME, n_envs=env_num_used_in_eval, vec_env_cls=SubprocVecEnv)
    # 回调函数中使用的测试环境
    env_num_used_in_callback = 10
    eval_env_in_callback = SubprocVecEnv([make_env(env_id=ENV_NAME, rank=i, scale_obs=False) for i in range(env_num_used_in_callback)])
    # eval_env_in_callback = make_vec_env(env_id=ENV_NAME, n_envs=env_num_used_in_callback, vec_env_cls=SubprocVecEnv)

    if ENV_NORMALIZE:
        vec_env = VecNormalize(venv=vec_env, norm_obs=True, norm_reward=True, gamma=GAMMA)
        eval_env = VecNormalize(venv=eval_env, norm_obs=True, norm_reward=False, gamma=GAMMA, training=False)
        # callback在调用的时候会自动的同步training_env和eval_env的normalize的相关参数！！！
        eval_env_in_callback = VecNormalize(venv=eval_env_in_callback, norm_obs=True, norm_reward=False, gamma=GAMMA, training=False)


    algo_ppo = get_ppo_algo(vec_env)
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    if ENV_NORMALIZE:
        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME).absolute()))
        eval_callback = EvalCallback(
            eval_env_in_callback,
            callback_on_new_best=save_vec_normalize,
            best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME).absolute()),
            log_path=str((PROJECT_ROOT_DIR / "logs" / "rl_single" / RL_EXPERIMENT_NAME).absolute()), 
            eval_freq=EVAL_FREQ,
            n_eval_episodes=1*env_num_used_in_callback,
            deterministic=True, 
            render=False,
        )
    else:
        eval_callback = EvalCallback(
            eval_env_in_callback,
            best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME).absolute()),
            log_path=str((PROJECT_ROOT_DIR / "logs" / "rl_single" / RL_EXPERIMENT_NAME).absolute()), 
            eval_freq=EVAL_FREQ,
            n_eval_episodes=1*env_num_used_in_callback,
            deterministic=True, 
            render=False,
        )

    algo_ppo.learn(total_timesteps=RL_TRAIN_STEPS, callback=eval_callback, log_interval=10)

    # evaluate
    if ENV_NORMALIZE:
        # eval_env = VecNormalize.load(load_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME / "vecnormalize.pkl").absolute()), venv=eval_env)
        # eval_env.training = False
        # eval_env.norm_reward = False
        sync_envs_normalization(vec_env, eval_env)
    mean_reward, _ = evaluate_policy(algo_ppo.policy, eval_env, 1*env_num_used_in_eval)
    sb3_logger.info(f"Reward after RL: {mean_reward}")

    return sb3_logger, eval_env

# python sb3_rl_train.py --config_file_name iter_1/seed1/halfcheetah_medium_256_256.json
if __name__ == "__main__":
    sb3_logger, eval_env = train()

    # test best policy saved during training
    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME
    algo_ppo = PPO.load(str((policy_save_dir / "best_model").absolute()))

    eval_env = VecNormalize.load(load_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME / "vecnormalize.pkl").absolute()), venv=eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    mean_reward, _ = evaluate_policy(algo_ppo.policy, eval_env, 10)
    sb3_logger.info(f"最优策略得分: {mean_reward}")
