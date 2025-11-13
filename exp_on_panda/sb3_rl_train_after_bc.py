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

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize, sync_envs_normalization
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import EvalCallback

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

def get_ppo_algo(env):
    policy_kwargs = dict(
        full_std=True,  # 使用state dependant exploration
        # squash_output=True,  # 使用state dependant exploration
        net_arch=dict(
            pi=NET_ARCH,
            vf=deepcopy(NET_ARCH)
        ),
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs={
            "eps": 1e-5
        }
    )

    return PPOWithBCLoss(
        policy=MlpPolicy, 
        env=env, 
        seed=SEED,
        kl_coef_with_bc=linear_schedule(KL_WITH_BC_MODEL_COEF) if KL_ANNEALING else KL_WITH_BC_MODEL_COEF,
        batch_size=PPO_BATCH_SIZE,  # PPO Mini Batch Size, PPO每次更新使用的数据量
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=RL_ENT_COEF,
        n_steps=N_STEPS,  # 采样时每个环境采样的step数，PPO每次训练收集的数据量是n_steps * num_envs
        n_epochs=N_EPOCHS,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration,
        normalize_advantage=True,
        learning_rate=linear_schedule(RL_LR_RATE),
    )


def train():
    
    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "rl" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    # 1.准备环境

    # 训练使用的环境
    vec_env = SubprocVecEnv([make_env(
        env_id=ENV_NAME, rank=i, seed=SEED_IN_TRAINING_ENV,
        flattern_obs=ENV_FLATTERN_OBS,
        wrap_with_nmr=ENV_USE_NMR, 
        nmr_waypoint=ENV_NMR_WAYPOINT, nmr_waypoint_delta=ENV_NMR_WAYPOINT_DELTA,
        nmr_jump=ENV_NMR_JUMP, nmr_jump_cnt=ENV_NMR_JUMP_CNT, nmr_length=ENV_NMR_LENGTH, 
        nmr_use_original_reward=ENV_USE_ORIGINAL_REWARD, nmr_non_terminal_reward=ENV_NON_TERMINAL_REWARD, nmr_terminal_reward=ENV_TERMINAL_REWARD,
        scale_obs=True, expert_data_dir=BC_EXPERT_DATA_DIR
        ) for i in range(ROLLOUT_PROCESS_NUM)
    ])
    # evaluate_policy使用的测试环境
    env_num_used_in_eval = 16
    eval_env = SubprocVecEnv([make_env(
        env_id=ENV_NAME, rank=i, seed=SEED_IN_EVAL_ENV,
        flattern_obs=ENV_FLATTERN_OBS,
        wrap_with_nmr=ENV_USE_NMR, 
        nmr_waypoint=ENV_NMR_WAYPOINT, nmr_waypoint_delta=ENV_NMR_WAYPOINT_DELTA,
        nmr_jump=ENV_NMR_JUMP, nmr_jump_cnt=ENV_NMR_JUMP_CNT, nmr_length=ENV_NMR_LENGTH, 
        nmr_use_original_reward=ENV_USE_ORIGINAL_REWARD, nmr_non_terminal_reward=ENV_NON_TERMINAL_REWARD, nmr_terminal_reward=ENV_TERMINAL_REWARD,
        scale_obs=True, expert_data_dir=BC_EXPERT_DATA_DIR
        ) for i in range(env_num_used_in_eval)
    ])
    # 回调函数中使用的测试环境
    env_num_used_in_callback = 16
    eval_env_in_callback = SubprocVecEnv([make_env(
        env_id=ENV_NAME, rank=i, seed=SEED_IN_CALLBACK_ENV, 
        flattern_obs=ENV_FLATTERN_OBS,
        wrap_with_nmr=ENV_USE_NMR, 
        nmr_waypoint=ENV_NMR_WAYPOINT, nmr_waypoint_delta=ENV_NMR_WAYPOINT_DELTA,
        nmr_jump=ENV_NMR_JUMP, nmr_jump_cnt=ENV_NMR_JUMP_CNT, nmr_length=ENV_NMR_LENGTH, 
        nmr_use_original_reward=ENV_USE_ORIGINAL_REWARD, nmr_non_terminal_reward=ENV_NON_TERMINAL_REWARD, nmr_terminal_reward=ENV_TERMINAL_REWARD,
        scale_obs=True, expert_data_dir=BC_EXPERT_DATA_DIR
        ) for i in range(env_num_used_in_callback)
    ])

    # TODO: normalize reward!!!
    if ENV_NORMALIZE:
        vec_env = VecNormalize(venv=vec_env, norm_obs=False, norm_reward=True, gamma=GAMMA)
        eval_env = VecNormalize(venv=eval_env, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)
        # callback在调用的时候会自动的同步training_env和eval_env的normalize的相关参数！！！
        eval_env_in_callback = VecNormalize(venv=eval_env_in_callback, norm_obs=False, norm_reward=False, gamma=GAMMA, training=False)

    # 2.load model
    bc_policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / BC_EXPERIMENT_NAME
    algo_ppo_for_kl_loss = PPOWithBCLoss.load(str((bc_policy_save_dir / BC_POLICY_FILE_NAME).absolute()))
    algo_ppo_for_kl_loss.policy.set_training_mode(False)
    algo_ppo = PPOWithBCLoss.load(
        str((bc_policy_save_dir / BC_POLICY_FILE_NAME).absolute()), 
        env=vec_env, 
        seed=SEED_FOR_LOAD_ALGO,
        custom_objects={
            "bc_trained_algo": algo_ppo_for_kl_loss,
            "learning_rate": linear_schedule(RL_LR_RATE),
        },
        ent_coef=RL_ENT_COEF,
        kl_coef_with_bc=KL_WITH_BC_MODEL_COEF,
        gamma=GAMMA,
        n_steps=N_STEPS,
        n_epochs=N_EPOCHS,
    )
    sb3_logger.info(str(algo_ppo.policy))

    # set sb3 logger
    algo_ppo.set_logger(sb3_logger)

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, eval_env, 3*env_num_used_in_eval)
    sb3_logger.info(f"Before RL, success rate: {success_rate}, reward: {reward}")

    # 3.train value head
    for k, v in algo_ppo.policy.named_parameters():
        # print(k)
        if any([ x in k.split('.') for x in ['shared_net', 'policy_net', 'action_net']]):  # 还有一个log_std
            v.requires_grad = False
    # exit(0)
    # for k, v in algo_ppo.policy.named_parameters():
    #     print(k, v.requires_grad)
    
    start_time = time()
    algo_ppo.learn(total_timesteps=ACTIVATE_VALUE_HEAD_TRAIN_STEPS, log_interval=10)
    sb3_logger.info(f"training value head time: {time() - start_time}(s).")

    # save model, value head训练完的
    algo_ppo.save(bc_policy_save_dir / BC_POLICY_AFTER_VALUE_HEAD_TRAINED_FILE_NAME)

    # evaluate
    reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, eval_env, 3*env_num_used_in_eval)
    sb3_logger.info(f"After training value head, success rate: {success_rate}, reward: {reward}")

    # 4.continue training
    for k, v in algo_ppo.policy.named_parameters():
        if any([ x in k.split('.') for x in ['shared_net', 'policy_net', 'action_net']]):
            v.requires_grad = True

    # for k, v in algo_ppo.policy.named_parameters():
    #     print(k, v.requires_grad)

    # sb3自带的EvalCallback根据最高平均reward保存最优策略；改成MyEvalCallback，根据最高胜率保存最优策略
    if ENV_NORMALIZE:
        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME).absolute()))
        eval_callback = EvalCallback(
            eval_env_in_callback, 
            callback_on_new_best=save_vec_normalize,
            best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME).absolute()),
            log_path=str((PROJECT_ROOT_DIR / "logs" / "rl" / RL_EXPERIMENT_NAME).absolute()), 
            eval_freq=EVAL_FREQ,  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
            n_eval_episodes=1*env_num_used_in_callback,  # 每次评估使用多少条轨迹
            deterministic=True, 
            render=False,
        )
    else:
        eval_callback = EvalCallback(
            eval_env_in_callback, 
            best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME).absolute()),
            log_path=str((PROJECT_ROOT_DIR / "logs" / "rl" / RL_EXPERIMENT_NAME).absolute()), 
            eval_freq=EVAL_FREQ,  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
            n_eval_episodes=1*env_num_used_in_callback,  # 每次评估使用多少条轨迹
            deterministic=True, 
            render=False,
        )

    # 用了eval_callback，不用再设置log_interval参数
    algo_ppo.learn(total_timesteps=RL_TRAIN_STEPS, callback=eval_callback, log_interval=10)

    # evaluate
    if ENV_NORMALIZE:
        sync_envs_normalization(vec_env, eval_env)

    reward, _ = evaluate_policy(algo_ppo.policy, vec_env, 3*env_num_used_in_eval)
    sb3_logger.info(f"Reward after RL: {reward}")

    # save model
    # rl_policy_save_dir = Path(__file__).parent / "checkpoints_sb3" / "rl" / RL_EXPERIMENT_NAME
    # algo_ppo.save(str(rl_policy_save_dir / RL_POLICY_FILE_NAME))

    return sb3_logger, eval_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="configs/iter_1/seed1/reacher_256_256.json")
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    ENV_NAME = custom_config["env"]["name"]
    ENV_NORMALIZE = custom_config["env"].get("normalize", False)
    ENV_MAX_EPISODE_STEPS = custom_config["env"].get("max_episode_steps", 50)
    ENV_FLATTERN_OBS = custom_config["env"].get("flattern_obs", True)
    ENV_USE_NMR = custom_config["env"].get("use_nmr", False)
    ENV_NMR_WAYPOINT = custom_config["env"].get("nmr_waypoint", False)
    ENV_NMR_WAYPOINT_DELTA = np.array(custom_config["env"].get("nmr_waypoint_delta", [0.0, 0.0, 0.1]))
    ENV_NMR_JUMP = custom_config["env"].get("nmr_jump", False)
    ENV_NMR_JUMP_CNT = custom_config["env"].get("nmr_jump_cnt", 2)
    ENV_NMR_LENGTH= custom_config["env"].get("nmr_length", 50)
    ENV_USE_ORIGINAL_REWARD = custom_config["env"].get("use_original_reward", False)
    ENV_NON_TERMINAL_REWARD = custom_config["env"].get("non_terminal_reward", 0.)
    ENV_TERMINAL_REWARD = custom_config["env"].get("terminal_reward", 0.)

    BC_EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
    BC_POLICY_FILE_NAME = custom_config["bc"]["policy_file_save_name"]
    BC_POLICY_AFTER_VALUE_HEAD_TRAINED_FILE_NAME = custom_config["bc"]["policy_after_value_head_trained_file_save_name"]
    BC_EXPERT_DATA_DIR = custom_config["bc"].get("data_cache_dir", "cache")

    RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]
    SEED = custom_config["rl_bc"]["seed"]
    SEED_FOR_LOAD_ALGO = custom_config["rl_bc"]["seed_for_load_algo"]
    SEED_IN_TRAINING_ENV = custom_config["rl_bc"]["seed_in_train_env"]
    SEED_IN_EVAL_ENV = custom_config["rl_bc"]["seed_in_eval_env"]
    SEED_IN_CALLBACK_ENV = custom_config["rl_bc"]["seed_in_callback_env"]
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

    sb3_logger, eval_env = train()

    # test best policy saved during training
    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / "best_model").absolute()))

    if ENV_NORMALIZE:
        eval_env = VecNormalize.load(load_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME / "vecnormalize.pkl").absolute()), venv=eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    for i in range(5):
        mean_reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, eval_env, 6*16)
        sb3_logger.info(f"最优策略, success rate: {success_rate}, reward: {mean_reward}")
