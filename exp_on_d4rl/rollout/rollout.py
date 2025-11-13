import numpy as np
import torch as th
import h5py
import argparse
import sys 
from pathlib import Path

import gym
import d4rl
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import BasePolicy

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from models.sb3_model import PPOWithBCLoss
from utils.sb3_env_wrappers import ScaledObservationWrapper
from configs.load_config import load_config
from utils.load_data import load_data, scale_obs, split_data


def get_reset_data():
    data = dict(
        observations = [],
        next_observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = [],
        logprobs = [],
        qpos = [],
        qvel = []
    )
    return data


def rollout(policy: BasePolicy, env: ScaledObservationWrapper, max_path: int=1000, num_data: int=1000_000, deterministic: bool=False):

    data = get_reset_data()
    traj_data = get_reset_data()

    _returns = 0
    t = 0 
    done = False
    s = env.reset()
    while len(data['rewards']) < num_data:

        # torch.from_numpy(*args, **kwargs).float().to(device)
        torch_s = th.from_numpy(np.expand_dims(s, axis=0)).float().to("cuda")
        # action, _ = policy.predict(observation=s, deterministic=True)
        actions, values, log_prob = policy.forward(obs=torch_s, deterministic=deterministic)  # 确定性采样时，基于数据集做BC效果很差

        a = actions.to('cpu').detach().numpy().squeeze()
        log_prob = log_prob.to('cpu').detach().numpy().squeeze()


        #mujoco only
        qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel().copy()

        ns, rew, done, infos = env.step(a)

        _returns += rew

        t += 1
        timeout = False
        terminal = False
        if t == max_path:
            timeout = True
        elif done:
            terminal = True

        # 需要先还原s
        origin_s = env.inverse_scale_state(s)
        origin_ns = env.inverse_scale_state(ns)

        traj_data['observations'].append(origin_s)
        traj_data['actions'].append(a)
        traj_data['next_observations'].append(origin_ns)
        traj_data['rewards'].append(rew)
        traj_data['terminals'].append(terminal)
        traj_data['timeouts'].append(timeout)
        traj_data['logprobs'].append(log_prob)
        traj_data['qpos'].append(qpos)
        traj_data['qvel'].append(qvel)

        s = ns
        if terminal or timeout:
            print('Finished trajectory. Len=%d, Returns=%f. Progress:%d/%d' % (t, _returns, len(data['rewards']), num_data))
            s = env.reset()
            t = 0
            _returns = 0
            for k in data:
                data[k].extend(traj_data[k])
            traj_data = get_reset_data()
    
    new_data = dict(
        observations=np.array(data['observations']).astype(np.float32),
        actions=np.array(data['actions']).astype(np.float32),
        next_observations=np.array(data['next_observations']).astype(np.float32),
        rewards=np.array(data['rewards']).astype(np.float32),
        terminals=np.array(data['terminals']).astype(np.bool),
        timeouts=np.array(data['timeouts']).astype(np.bool)
    )
    new_data['infos/action_log_probs'] = np.array(data['logprobs']).astype(np.float32)
    new_data['infos/qpos'] = np.array(data['qpos']).astype(np.float32)
    new_data['infos/qvel'] = np.array(data['qvel']).astype(np.float32)

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data

def save_data_to_h5_file(data: dict, h5_file_path: str):
    hfile = h5py.File(h5_file_path, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')

    hfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config_file_name", type=str, help="配置文件名", default="iter_1/seed1/kl_1/medium_halfcheetah_256_256_kl1.json")
    parser.add_argument("--deterministic", action="store_true", help="是否确定性采样")
    args = parser.parse_args()

    custom_config = load_config(args.config_file_name)

    if args.deterministic:
        print("确定性采样。")
    else:
        print("根据分布采样！！！！！！！！！！！！！！！！")

    ENV_NAME = custom_config["env"]["name"]
    RL_EXPERIMENT_NAME = custom_config["rl_bc"]["experiment_name"]

    # prepare env
    origin_env = gym.make(ENV_NAME)

    # TODO: 选择从哪加载数据
    obs, acts, infos = load_data(origin_env)

    scaled_obs, scaler = scale_obs(obs)
    env = ScaledObservationWrapper(env=origin_env, scaler=scaler)

    # load policy
    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "rl" / RL_EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / "best_model").absolute()))
    algo_ppo.policy.set_training_mode(False)

    dataset = rollout(algo_ppo.policy, env, max_path=1000, num_data=1_000_000, deterministic=args.deterministic)

    save_data_to_h5_file(dataset, str(PROJECT_ROOT_DIR / "rollout" / "data" / (RL_EXPERIMENT_NAME + ".hdf5")))