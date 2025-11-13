import gymnasium as gym
import panda_gym
import numpy as np
from pathlib import Path
import logging
from time import time
from copy import deepcopy
import argparse
import sys
import torch as th

from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize, sync_envs_normalization
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import PPO

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from models.sb3_model import PPOWithBCLoss
from configs.load_config import load_config
from utils.sb3_schedule import linear_schedule
from utils.sb3_env_utils import make_env
from utils.register_env import register_my_env
from utils.sb3_callbacks import SaveVecNormalizeCallback
from utils.sb3_evaluate_policy import evaluate_policy_with_success_rate

register_my_env(goal_range=0.3, distance_threshold=0.01, max_episode_steps=100)  # 注意此处：max_episode_steps, 根据环境文件的配置修改此值！！！！

np.seterr(all="raise")  # 检查nan

def evaluate(
        env_id: str, 
        evaluate_algo: str, 
        evaluate_exp_name: str, 
        evaluate_process_num: int, 
        evaluate_num: int, 
        seed: int=0, 
        wrap_env_with_nmr: bool=False, 
        nmr_waypoint: bool=False,
        nmr_waypoint_delta: np.ndarray=np.array([0., 0., 0.1]),
        nmr_jump: bool=False, nmr_jump_cnt: int=2, nmr_length: int=10, 
        use_original_reward: bool=False, non_terminal_reward: float=0.0, terminal_reward: float=1.0
    ):

    print(f"测试中使用的env_id: {env_id}, use nmr: {wrap_env_with_nmr}, nmr waypoint: {nmr_waypoint}, nmr waypoint delta: {nmr_waypoint_delta}, nmr jump: {nmr_jump_cnt}")

    # load algorithm and prepare evaluation environment
    if evaluate_algo == "rl":
        eval_env = SubprocVecEnv([make_env(
            env_id=env_id, rank=i, seed=seed, 
            flattern_obs=ENV_FLATTERN_OBS,
            wrap_with_nmr=wrap_env_with_nmr, 
            nmr_waypoint=nmr_waypoint, nmr_waypoint_delta=nmr_waypoint_delta,
            nmr_jump=nmr_jump, nmr_jump_cnt=nmr_jump_cnt, nmr_length=nmr_length, 
            nmr_use_original_reward=use_original_reward, nmr_non_terminal_reward=non_terminal_reward, nmr_terminal_reward=terminal_reward, 
            scale_obs=False,
        ) for i in range(evaluate_process_num)])
        if ENV_NORMALIZE:
            eval_env = VecNormalize.load(load_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / evaluate_exp_name / "vecnormalize.pkl").absolute()), venv=eval_env)
            eval_env.training = False
            eval_env.norm_reward = False

        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / evaluate_exp_name
        policy_save_name = "best_model"
        algo_ppo = PPO.load(str((policy_save_dir / policy_save_name).absolute()))

    elif evaluate_algo == "rl_bc":
        eval_env = SubprocVecEnv([make_env(
            env_id=env_id, rank=i, seed=seed, 
            flattern_obs=ENV_FLATTERN_OBS,
            wrap_with_nmr=wrap_env_with_nmr, 
            nmr_waypoint=nmr_waypoint, nmr_waypoint_delta=nmr_waypoint_delta,
            nmr_jump=nmr_jump, nmr_jump_cnt=nmr_jump_cnt, nmr_length=nmr_length, 
            nmr_use_original_reward=use_original_reward, nmr_non_terminal_reward=non_terminal_reward, nmr_terminal_reward=terminal_reward, 
            scale_obs=True, expert_data_dir=BC_EXPERT_DATA_DIR,
        ) for i in range(evaluate_process_num)])
        if ENV_NORMALIZE:
            eval_env = VecNormalize.load(load_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl" / evaluate_exp_name / "vecnormalize.pkl").absolute()), venv=eval_env)
            eval_env.training = False
            eval_env.norm_reward = False

        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl" / evaluate_exp_name
        policy_save_name = "best_model"
        algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / policy_save_name).absolute()))
    elif evaluate_algo == "bc":
        eval_env = SubprocVecEnv([make_env(
            env_id=env_id, rank=i, seed=seed, 
            flattern_obs=ENV_FLATTERN_OBS,
            wrap_with_nmr=wrap_env_with_nmr, 
            nmr_waypoint=nmr_waypoint, nmr_waypoint_delta=nmr_waypoint_delta,
            nmr_jump=nmr_jump, nmr_jump_cnt=nmr_jump_cnt, nmr_length=nmr_length, 
            nmr_use_original_reward=use_original_reward, nmr_non_terminal_reward=non_terminal_reward, nmr_terminal_reward=terminal_reward, 
            scale_obs=True, expert_data_dir=BC_EXPERT_DATA_DIR
        ) for i in range(evaluate_process_num)])

        policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / evaluate_exp_name
        policy_save_name = "bc_checkpoint"
        algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / policy_save_name).absolute()))
    elif evaluate_algo == "random":
        eval_env = SubprocVecEnv([make_env(
            env_id=env_id, rank=i, seed=seed, 
            flattern_obs=ENV_FLATTERN_OBS,
            wrap_with_nmr=wrap_env_with_nmr, 
            nmr_waypoint=nmr_waypoint, nmr_waypoint_delta=nmr_waypoint_delta,
            nmr_jump=nmr_jump, nmr_jump_cnt=nmr_jump_cnt, nmr_length=nmr_length, 
            nmr_use_original_reward=use_original_reward, nmr_non_terminal_reward=non_terminal_reward, nmr_terminal_reward=terminal_reward, 
            scale_obs=False,
        ) for i in range(evaluate_process_num)])

        algo_ppo = PPO(
            policy=MultiInputPolicy,
            env=eval_env,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256]
                ),
            ),
        )

    rewards, lengths, success_rate = evaluate_policy_with_success_rate(
        model=algo_ppo.policy,
        env=eval_env,
        n_eval_episodes=evaluate_num,
        deterministic=True,
        return_episode_rewards=True,
    )
    print(rewards, lengths)

    return success_rate, np.mean(rewards), np.mean(lengths)


# waypoint [0.0, 0.0, 0.0]和原始环境一样，测试waypoint wrapper的正确性
# python evaluate.py --config-file-name configs/iter_1/seed1/reacher_256_256_kl_1e-1.json --test-algo bc --eval-process-num 16 --eval-episode-num 96 --env-id my-reach --wrap-env-with-nmr --nmr-waypoint --nmr-waypoint-delta 0.0 0.0 0.0 --non-terminal-reward -1.0 --terminal-reward 0.0 --seed 5

# waypoint [0.0, 0.0, 0.1]
# python evaluate.py --config-file-name configs/iter_1/seed1/reacher_256_256_kl_1e-1.json --test-algo bc --eval-process-num 16 --eval-episode-num 96 --env-id my-reach --wrap-env-with-nmr --nmr-waypoint --nmr-waypoint-delta 0.0 0.0 0.1 --non-terminal-reward -1.0 --terminal-reward 0.0 --seed 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="configs/pointmaze-diverse-g-dense/seed5/sparse_256_256_lambda_1e-1.json")
    parser.add_argument("--test-algo", type=str, help="要测试的算法, rl, rl_bc, bc, random", default="rl")
    parser.add_argument("--eval-process-num", type=int, help="测试使用的进程数", default=1)
    parser.add_argument("--eval-episode-num", type=int, help="测试的episode数", default=1)
    parser.add_argument("--env-id", type=str, help="测试使用的env的id", default="none")
    parser.add_argument("--wrap-env-with-nmr", action="store_true", help="环境是否设置为非马尔科夫奖励")
    parser.add_argument("--nmr-waypoint", action="store_true", help="是否使用waypoint类型的NMR")
    parser.add_argument("--nmr-waypoint-delta", type=float, nargs='+', default=[0.0, 0.0, 0.1], help="waypoint_delta")
    parser.add_argument("--nmr-jump", action="store_true", help="是否使用jump类型的NMR")
    parser.add_argument("--nmr-jump-cnt", type=int, default=2, help="NMR Jump的目标次数")
    parser.add_argument("--nmr-length", type=int, default=10, help="NMR的长度")
    parser.add_argument("--use-original-reward", action="store_true", help="是否使用环境的原始奖励")
    parser.add_argument("--non-terminal-reward", type=float, default=0.0)
    parser.add_argument("--terminal-reward", type=float, default=1.0)
    parser.add_argument("--seed", type=int, help="测试使用的seed", default=10)
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    ENV_NAME = custom_config["env"]["name"]
    ENV_NORMALIZE = custom_config["env"].get("normalize", False)
    ENV_MAX_EPISODE_STEPS = custom_config["env"].get("max_episode_steps", 50)
    ENV_FLATTERN_OBS = custom_config["env"].get("flattern_obs", True)
    ENV_NMR_WAYPOINT = custom_config["env"].get("nmr_waypoint", False)
    ENV_NMR_WAYPOINT_DELTA = np.array(custom_config["env"].get("nmr_waypoint_delta", [0.0, 0.0, 0.1]))
    

    BC_EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
    BC_POLICY_FILE_NAME = custom_config["bc"]["policy_file_save_name"]
    BC_POLICY_AFTER_VALUE_HEAD_TRAINED_FILE_NAME = custom_config["bc"]["policy_after_value_head_trained_file_save_name"]
    BC_EXPERT_DATA_DIR = custom_config["bc"].get("data_cache_dir", "cache")

    RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
    SEED = custom_config["rl_bc"]["seed"]
    SEED_FOR_LOAD_ALGO = custom_config["rl_bc"]["seed_for_load_algo"]
    NET_ARCH = custom_config["rl_bc"]["net_arch"]
    PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
    GAMMA = custom_config["rl_bc"]["gamma"]
    GAE_LAMBDA = custom_config["rl_bc"]["gae_lambda"]
    ACTIVATE_VALUE_HEAD_TRAIN_STEPS = custom_config["rl_bc"]["activate_value_head_train_steps"]
    RL_TRAIN_STEPS = custom_config["rl_bc"]["train_steps"]
    RL_ENT_COEF = custom_config["rl_bc"].get("ent_coef", 0.0)
    RL_LR_RATE = custom_config["rl_bc"].get("lr", 3e-4)
    ROLLOUT_PROCESS_NUM = custom_config["rl_bc"]["rollout_process_num"]
    N_STEPS = custom_config["rl_bc"]["n_steps"]
    N_EPOCHS = custom_config["rl_bc"]["n_epochs"]
    KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]
    KL_ANNEALING = custom_config["rl_bc"].get("kl_annealing", False)
    EVAL_FREQ = custom_config["rl_bc"]["eval_freq"]

    RL_SINGLE_EXPERIMENT_NAME = custom_config["rl"].get("experiment_name", "xx")

    tmp_exp_name = ""
    if args.test_algo == "rl":
        tmp_exp_name = RL_SINGLE_EXPERIMENT_NAME
    elif args.test_algo == "rl_bc":
        tmp_exp_name = RL_EXPERIMENT_NAME
    elif args.test_algo == "bc":
        tmp_exp_name = BC_EXPERIMENT_NAME
    elif args.test_algo == "random":
        tmp_exp_name = "random"

    tmp_env_id = args.env_id if args.env_id != "none" else ENV_NAME

    print(
        evaluate(
            env_id=tmp_env_id, 
            evaluate_algo=args.test_algo, 
            evaluate_exp_name=tmp_exp_name, 
            evaluate_process_num=args.eval_process_num,
            evaluate_num=args.eval_episode_num,
            seed=args.seed,
            wrap_env_with_nmr=args.wrap_env_with_nmr,
            nmr_waypoint=args.nmr_waypoint,
            nmr_waypoint_delta=args.nmr_waypoint_delta,
            nmr_jump=args.nmr_jump,
            nmr_jump_cnt=args.nmr_jump_cnt,
            nmr_length=args.nmr_length,
            use_original_reward=args.use_original_reward,
            non_terminal_reward=args.non_terminal_reward,
            terminal_reward=args.terminal_reward
        )
    )
