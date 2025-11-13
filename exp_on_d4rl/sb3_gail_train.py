import sys
from pathlib import Path
import numpy as np
from copy import deepcopy
import argparse
import gym
import d4rl
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.util.logger import HierarchicalLogger

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from models.sb3_model import PPOWithBCLoss
from utils.sb3_env_wrappers import ScaledObservationWrapper
from configs.load_config import load_config
from utils.sb3_schedule import linear_schedule
from utils.load_data_with_dones_and_next_obs import load_data, scale_obs, split_data
from utils.sb3_env_utils import make_env

parser = argparse.ArgumentParser(description="传入配置文件")
parser.add_argument("--config_file_name", type=str, help="配置文件名", default="iter_1/seed1/test_gail_human_door_256_256.json")
args = parser.parse_args()

custom_config = load_config(args.config_file_name)

ENV_NAME = custom_config["env"]["name"]

EXPERIMENT_NAME = custom_config["gail"]["experiment_name"]
POLICY_FILE_SAVE_NAME = custom_config["gail"]["policy_file_save_name"]
DATASET_SPLIT = custom_config["bc"].get("dataset_split", [0.96, 0.02, 0.02])
SEED = custom_config["gail"]["seed"]

RL_SEED = custom_config["rl"]["seed"]
NET_ARCH = custom_config["rl_bc"]["net_arch"]
PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
GAMMA = custom_config["rl_bc"]["gamma"]
KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]


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
        bc_trained_algo=None,
        kl_coef_with_bc=KL_WITH_BC_MODEL_COEF,
        seed=RL_SEED,
        batch_size=PPO_BATCH_SIZE,
        gamma=GAMMA,
        n_steps=128,  # 采样时每个环境采样的step数
        n_epochs=5,  # 采样的数据在训练中重复使用的次数
        policy_kwargs=policy_kwargs,
        use_sde=True,  # 使用state dependant exploration
        normalize_advantage=True,
        learning_rate=linear_schedule(3e-4),
    )

def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "gail" / EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    origin_env = gym.make(ENV_NAME)
    
    obs, acts, infos, next_obs, dones = load_data(origin_env)
    scaled_obs, scaler = scale_obs(obs)

    env = ScaledObservationWrapper(env=origin_env, scaler=scaler)

    vec_env = SubprocVecEnv([make_env(env_id=ENV_NAME, rank=i, scale_obs=True) for i in range(8)])
    eval_env = SubprocVecEnv([make_env(env_id=ENV_NAME, rank=i, scale_obs=True) for i in range(8)])

    algo_ppo = get_ppo_algo(vec_env)
    algo_ppo.bc_trained_algo = algo_ppo  # TODO:
    sb3_logger.info(str(algo_ppo.policy))
    # print(algo_ppo.policy)

    rng = np.random.default_rng(SEED)
    
    
    train_transitions, validation_transitions, test_transitions = split_data(
        obs=scaled_obs, 
        acts=acts,
        infos=infos,
        next_obs=next_obs, 
        dones=dones,
        train_size=DATASET_SPLIT[0],
        validation_size=DATASET_SPLIT[1],
        test_size=DATASET_SPLIT[2],
        shuffle=True,
    )

    sb3_logger.info(f"train_set: obs size, {train_transitions.obs.shape}, act size, {train_transitions.acts.shape}")
    sb3_logger.info(f"validation_set: obs size, {validation_transitions.obs.shape}, act size, {validation_transitions.acts.shape}")
    sb3_logger.info(f"test_set: obs size, {test_transitions.obs.shape}, act size, {test_transitions.acts.shape}")

    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=train_transitions,
        demo_batch_size=128,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=vec_env,
        gen_algo=algo_ppo,
        reward_net=reward_net,
        custom_logger=HierarchicalLogger(sb3_logger)
    )

    # evaluate the learner before training
    eval_env.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        algo_ppo.policy, eval_env, 10, return_episode_rewards=True,
    )

    # train the learner and evaluate again
    gail_trainer.train(
        total_timesteps=1000_000,
    )  # Train for 800_000 steps to match expert.
    eval_env.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        algo_ppo.policy, eval_env, 10, return_episode_rewards=True,
    )

    print("mean reward after training:", np.mean(learner_rewards_after_training))
    print("mean reward before training:", np.mean(learner_rewards_before_training))

    # save policy
    checkpoint_save_dir: Path = PROJECT_ROOT_DIR / "checkpoints" / "gail" / EXPERIMENT_NAME
    if not checkpoint_save_dir.exists():
        checkpoint_save_dir.absolute().mkdir()

    algo_ppo.save(str((checkpoint_save_dir / POLICY_FILE_SAVE_NAME).absolute()))

    return sb3_logger, validation_transitions, test_transitions, gail_trainer, origin_env, env


if __name__ == "__main__":
    sb3_logger, validation_transitions, test_transitions, gail_trainer, origin_env, env = train()


