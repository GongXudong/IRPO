import sys
from pathlib import Path

import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3.common import policies
from gymnasium.wrappers import FlattenObservation

PROJECT_ROOT_DIR = Path(__file__).absolute().parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.sb3_env_wrappers import ScaledObservationWrapper


def log_to_dict(log_dict: dict, obs: np.ndarray, goal: np.ndarray, action: np.ndarray):
    log_dict["s_x"].append(obs[0])
    log_dict["s_y"].append(obs[1])
    log_dict["s_z"].append(obs[2])
    log_dict["s_v_x"].append(obs[3])
    log_dict["s_v_y"].append(obs[4])
    log_dict["s_v_z"].append(obs[5])

    log_dict["s_g_x"].append(goal[0])
    log_dict["s_g_y"].append(goal[1])
    log_dict["s_g_z"].append(goal[2])
    
    log_dict["a_x"].append(action[0])
    log_dict["a_y"].append(action[1])
    log_dict["a_z"].append(action[2])

def rollout_by_goal_with_policy(env: ScaledObservationWrapper, goal: np.ndarray, policy: policies.ActorCriticPolicy):
    """_summary_

    Args:
        env (ScaledObservationWrapper): 按如下顺序wrap：gym.Env -> FlattenObservation -> ScaledObservationWrapper
        goal (np.ndarray): _description_
        policy (policies.ActorCriticPolicy): _description_

    Returns:
        _type_: _description_
    """
    tmp_logs = {
        "s_x": [],
        "s_y": [],
        "s_z": [],
        "s_v_x": [],
        "s_v_y": [],
        "s_v_z": [],
        "s_g_x": [],
        "s_g_y": [],
        "s_g_z": [],
        "a_x": [],
        "a_y": [],
        "a_z": []
    }

    # env = gym.make("PandaReach-v3", render_mode="rgb_array")
    original_env: gym.Env = env.unwrapped
    flatten_env: FlattenObservation = env.env
    observation, info = env.reset()
    original_env.task.goal = goal
    original_env.task.sim.set_base_pose("target", goal, np.array([0.0, 0.0, 0.0, 1.0]))
    
    original_observation = {
        "achieved_goal": original_env.task.get_achieved_goal().astype(np.float32),
        "desired_goal": original_env.task.get_goal().astype(np.float32),
        "observation": original_env.robot.get_obs().astype(np.float32)
    }

    observation = env.observation(flatten_env.observation(original_observation))

    terminated, truncated = False, False
    step_cnt = 0

    while(not (terminated or truncated)):
        action, _ = policy.predict(observation=observation)
        step_cnt += 1

        unscaled_obs = env.inverse_scale_state(observation)

        # FlattenObservation的顺序：achieved_goal(3维), desired_goal(3维), observation(6维)
        log_to_dict(tmp_logs, unscaled_obs[6:12], unscaled_obs[3:6], action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            if terminated:
                print(f"\033[32m success. goal: ({goal[0]}, {goal[1]}, {goal[2]}), steps: {step_cnt} \033[0m")
            else:
                print(f"\033[31m truncated. goal: ({goal[0]}, {goal[1]}, {goal[2]}), steps: {step_cnt} \033[0m")
            break

    return terminated, truncated, tmp_logs