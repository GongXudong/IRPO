import gymnasium as gym
import panda_gym
import numpy as np

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


def rollout(total_timesteps: int=1e4, goal_range: float=0.1, speed: float=1.0):

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

    env = gym.make("PandaReach-v3", render_mode="rgb_array")
    env.task.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
    env.task.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])

    observation, info = env.reset()

    episode_cnt = 0
    episode_len = []
    reward_sum = []
    tmp_episode_len = 0
    tmp_reward = 0
    for _ in range(total_timesteps):
        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        action = speed * (desired_position - current_position)

        log_to_dict(tmp_logs, observation["observation"], observation["desired_goal"], action)

        observation, reward, terminated, truncated, info = env.step(action)
        tmp_reward += reward
        tmp_episode_len += 1

        if terminated or truncated:
            reward_sum.append(tmp_reward)
            tmp_reward = 0
            episode_len.append(tmp_episode_len)
            tmp_episode_len = 0
            observation, info = env.reset()
            episode_cnt += 1

    print(np.mean(reward_sum))
    print(np.mean(episode_len))
    print(episode_cnt)
    env.close()
    print(len(tmp_logs["s_x"]))
    return tmp_logs

def rollout_by_goal_with_pid(env: gym.Env, goal: np.ndarray, speed: float=1.0):
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
    observation, info = env.reset()
    env.task.goal = goal
    env.task.sim.set_base_pose("target", goal, np.array([0.0, 0.0, 0.0, 1.0]))
    
    observation["desired_goal"] = env.task.get_goal().astype(np.float32)

    # robot_obs = env.robot.get_obs().astype(np.float32)  # robot state
    # task_obs = env.task.get_obs().astype(np.float32)  # object position, velococity, etc...
    # observation = np.concatenate([robot_obs, task_obs])
    # achieved_goal = env.task.get_achieved_goal().astype(np.float32)
    # observation = {
    #     "observation": observation,
    #     "achieved_goal": achieved_goal,
    #     "desired_goal": env.task.get_goal().astype(np.float32),
    # }

    terminated, truncated = False, False
    step_cnt = 0

    while(not (terminated or truncated)):
        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        action = speed * (desired_position - current_position)
        step_cnt += 1

        log_to_dict(tmp_logs, observation["observation"], observation["desired_goal"], action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            if terminated:
                print(f"\033[32m success. goal: ({goal[0]}, {goal[1]}, {goal[2]}), steps: {step_cnt} \033[0m")
            else:
                print(f"\033[31m truncated. goal: ({goal[0]}, {goal[1]}, {goal[2]}), steps: {step_cnt} \033[0m")
            break

    return terminated, truncated, tmp_logs


def rollout_by_waypoint_and_goal_with_pid(env: gym.Env, waypoint: np.ndarray, goal: np.ndarray, speed: float=1.0):
    """

    Args:
        env (gym.Env): 注意，实际应该传utils.non_markovian_reard_waypoint_wrapper.NMRWayPointWrapper类型
        waypoint (np.ndarray): _description_
        goal (np.ndarray): _description_
        speed (float, optional): _description_. Defaults to 1.0.

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
    observation, info = env.reset()
    env.unwrapped.task.goal = goal
    env.unwrapped.task.sim.set_base_pose("target", goal, np.array([0.0, 0.0, 0.0, 1.0]))
    env.waypoint = waypoint
    env.reach_way_point = False
    
    observation["desired_goal"] = env.task.get_goal().astype(np.float32)

    # robot_obs = env.robot.get_obs().astype(np.float32)  # robot state
    # task_obs = env.task.get_obs().astype(np.float32)  # object position, velococity, etc...
    # observation = np.concatenate([robot_obs, task_obs])
    # achieved_goal = env.task.get_achieved_goal().astype(np.float32)
    # observation = {
    #     "observation": observation,
    #     "achieved_goal": achieved_goal,
    #     "desired_goal": env.task.get_goal().astype(np.float32),
    # }

    terminated, truncated = False, False
    step_cnt = 0

    while(not (terminated or truncated)):
        current_position = observation["observation"][0:3]

        # 判断当前desired_position为waypoint还是desired_goal
        if not env.reach_way_point:
            desired_position = waypoint
        else:
            desired_position = observation["desired_goal"][0:3]
        action = speed * (desired_position - current_position)
        step_cnt += 1

        print(step_cnt, desired_position, observation["observation"][:3])

        log_to_dict(tmp_logs, observation["observation"], observation["desired_goal"], action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            if terminated:
                print(f"\033[32m success. goal: ({goal[0]}, {goal[1]}, {goal[2]}), steps: {step_cnt} \033[0m")
            else:
                print(f"\033[31m truncated. goal: ({goal[0]}, {goal[1]}, {goal[2]}), steps: {step_cnt} \033[0m")
            break

    return terminated, truncated, tmp_logs