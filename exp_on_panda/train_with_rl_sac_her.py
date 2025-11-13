import gymnasium as gym
import numpy as np
from pathlib import Path
import os
import sys
import torch as th
import argparse

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.vec_env import sync_envs_normalization

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.sb3_env_utils import make_env
from configs.load_config import load_config
from utils.sb3_eval_callback import MyEvalCallback
from utils.sb3_env_wrappers import ScaledObservationWrapper
from utils.sb3_evaluate_policy import evaluate_policy_with_success_rate
from utils.register_env import register_my_env
from utils.sb3_callbacks import SaveVecNormalizeCallback

register_my_env(goal_range=0.3, distance_threshold=0.01, max_episode_steps=100)  # 注意此处：max_episode_steps, 根据环境文件的配置修改此值！！！！

np.seterr(all="raise")  # 检查nan


def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "rl_single" / RL_EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    vec_env = SubprocVecEnv([make_env(
        env_id=ENV_NAME, rank=i, seed=SEED_IN_TRAINING_ENV,
        flattern_obs=ENV_FLATTERN_OBS,
        wrap_with_nmr=ENV_USE_NMR, 
        nmr_waypoint=ENV_NMR_WAYPOINT, nmr_waypoint_delta=ENV_NMR_WAYPOINT_DELTA,
        nmr_jump=ENV_NMR_JUMP, nmr_jump_cnt=ENV_NMR_JUMP_CNT, nmr_length=ENV_NMR_LENGTH, 
        nmr_use_original_reward=ENV_USE_ORIGINAL_REWARD, nmr_non_terminal_reward=ENV_NON_TERMINAL_REWARD, nmr_terminal_reward=ENV_TERMINAL_REWARD,
        scale_obs=False
        ) for i in range(RL_TRAIN_PROCESS_NUM)
    ])
    # 回调函数中使用的测试环境
    eval_env_in_callback = SubprocVecEnv([make_env(
        env_id=ENV_NAME, rank=i, seed=SEED_IN_CALLBACK_ENV,
        flattern_obs=ENV_FLATTERN_OBS,
        wrap_with_nmr=ENV_USE_NMR, 
        nmr_waypoint=ENV_NMR_WAYPOINT, nmr_waypoint_delta=ENV_NMR_WAYPOINT_DELTA,
        nmr_jump=ENV_NMR_JUMP, nmr_jump_cnt=ENV_NMR_JUMP_CNT, nmr_length=ENV_NMR_LENGTH, 
        nmr_use_original_reward=ENV_USE_ORIGINAL_REWARD, nmr_non_terminal_reward=ENV_NON_TERMINAL_REWARD, nmr_terminal_reward=ENV_TERMINAL_REWARD,
        scale_obs=False
        ) for i in range(CALLBACK_PROCESS_NUM)
    ])

    # SAC hyperparams:
    sac_algo = SAC(
        "MultiInputPolicy",
        vec_env,
        seed=SEED,
        replay_buffer_class=HerReplayBuffer if USE_HER else DictReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ) if USE_HER else None,
        verbose=1,
        buffer_size=int(BUFFER_SIZE),
        learning_starts=int(LEARNING_STARTS),
        gradient_steps=int(GRADIENT_STEPS),
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=int(BATCH_SIZE),
        policy_kwargs=dict(
            net_arch=NET_ARCH,
            activation_fn=th.nn.Tanh
        ),
    )

    sac_algo.set_logger(sb3_logger)

    # callback: evaluate, save best
    eval_callback = MyEvalCallback(
        eval_env_in_callback, 
        best_model_save_path=str((PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME).absolute()),
        log_path=str((PROJECT_ROOT_DIR / "logs" / "rl_single" / RL_EXPERIMENT_NAME).absolute()), 
        eval_freq=EVAL_FREQ,  # 多少次env.step()评估一次，此处设置为1000，因为VecEnv有72个并行环境，所以实际相当于72*1000次step，评估一次
        n_eval_episodes=N_EVAL_EPISODES,  # 每次评估使用多少条轨迹
        deterministic=True, 
        render=False,
    )

    sac_algo.learn(
        total_timesteps=int(RL_TRAIN_STEPS), 
        callback=eval_callback
    )
    # sac_algo.save(str(PROJECT_ROOT_DIR / "checkpoints" / RL_EXPERIMENT_NAME))

    return sb3_logger, vec_env, eval_env_in_callback

def test_single_traj():
    # Load saved model
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    
    env = make_env(
        env_id=ENV_NAME, rank=0, 
        flattern_obs=ENV_FLATTERN_OBS,
        wrap_with_nmr=ENV_USE_NMR, 
        nmr_waypoint=ENV_NMR_WAYPOINT, nmr_waypoint_delta=ENV_NMR_WAYPOINT_DELTA,
        nmr_jump=ENV_NMR_JUMP, nmr_jump_cnt=ENV_NMR_JUMP_CNT, nmr_length=ENV_NMR_LENGTH, 
        nmr_use_original_reward=ENV_USE_ORIGINAL_REWARD, nmr_non_terminal_reward=ENV_NON_TERMINAL_REWARD, nmr_terminal_reward=ENV_TERMINAL_REWARD,
        scale_obs=False
    )()

    model = SAC.load(
        str(PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME / "best_model"), 
        env=env
    )

    obs, info = env.reset()

    # Evaluate the agent
    episode_reward = 0
    for _ in range(400):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated or info.get("is_success", False):
            print("Reward:", episode_reward, "Success?", info.get("is_success", False))
            episode_reward = 0.0
            obs, info = env.reset()

def test_multi_traj(eval_env, sb3_logger, seed=0):
    
    vec_env = SubprocVecEnv([make_env(
        env_id=ENV_NAME, rank=i, seed=seed,
        flattern_obs=ENV_FLATTERN_OBS,
        wrap_with_nmr=ENV_USE_NMR, 
        nmr_waypoint=ENV_NMR_WAYPOINT, nmr_waypoint_delta=ENV_NMR_WAYPOINT_DELTA,
        nmr_jump=ENV_NMR_JUMP, nmr_jump_cnt=ENV_NMR_JUMP_CNT, nmr_length=ENV_NMR_LENGTH, 
        nmr_use_original_reward=ENV_USE_ORIGINAL_REWARD, nmr_non_terminal_reward=ENV_NON_TERMINAL_REWARD, nmr_terminal_reward=ENV_TERMINAL_REWARD,
        scale_obs=False
        ) for i in range(RL_TRAIN_PROCESS_NUM)
    ])

    sac_algo = SAC.load(
        str(PROJECT_ROOT_DIR / "checkpoints" / "rl_single" / RL_EXPERIMENT_NAME / "best_model"), 
        env=vec_env,
        custom_objects={
            "observation_space": vec_env.observation_space,
            "action_space": vec_env.action_space
        }
    )

    for i in range(5):
        reward, _, success_rate = evaluate_policy_with_success_rate(sac_algo.policy, vec_env, 100)

        sb3_logger.info(f"Training finished, best model, success rate: {success_rate}, reward: {reward}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="sac_config_10hz_128_128_1.json")
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    ENV_NAME = custom_config["env"]["name"]
    ENV_NORMALIZE = custom_config["env"].get("normalize", False)  # 该脚本没用VecNormalize!!!!!!!!!!!
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

    SEED = custom_config["rl"].get("seed")
    SEED_IN_TRAINING_ENV = custom_config["rl"].get("seed_in_train_env")
    SEED_IN_CALLBACK_ENV = custom_config["rl"].get("seed_in_callback_env")

    RL_EXPERIMENT_NAME = custom_config["rl"]["experiment_name"]
    NET_ARCH = custom_config["rl"]["net_arch"]
    RL_TRAIN_STEPS = custom_config["rl"]["train_steps"]
    GAMMA = custom_config["rl"].get("gamma", 0.995)
    BUFFER_SIZE = custom_config["rl"].get("buffer_size", 1e6)
    BATCH_SIZE = custom_config["rl"].get("batch_size", 1024)
    LEARNING_STARTS = custom_config["rl"].get("learning_starts", 10240)
    RL_TRAIN_PROCESS_NUM = custom_config["rl"].get("rollout_process_num", 32)
    RL_EVALUATE_PROCESS_NUM = custom_config["rl"].get("evaluate_process_num", 32)
    CALLBACK_PROCESS_NUM = custom_config["rl"].get("callback_process_num", 32)
    GRADIENT_STEPS = custom_config["rl"].get("gradient_steps", 2)
    LEARNING_RATE = custom_config["rl"].get("learning_rate", 3e-4)

    USE_HER = custom_config["rl"].get("use_her", True)

    EVAL_FREQ = custom_config["rl"].get("eval_freq", 1000)
    N_EVAL_EPISODES = custom_config["rl"].get("n_eval_episodes", CALLBACK_PROCESS_NUM*10)

    sb3_logger, train_env, eval_env = train()

    # test_single_traj()
    test_multi_traj(eval_env, sb3_logger, seed=39870)
