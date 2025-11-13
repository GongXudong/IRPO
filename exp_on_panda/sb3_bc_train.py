import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import panda_gym
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
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy
from imitation.algorithms import bc
from imitation.util.logger import HierarchicalLogger
from imitation.util import util
from imitation.data.types import TransitionsMinimal

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from models.sb3_model import PPOWithBCLoss
from utils.sb3_env_wrappers import ScaledObservationWrapper
from configs.load_config import load_config
from utils.sb3_schedule import linear_schedule
from utils.load_data import load_data, split_data
from utils.register_env import register_my_env
from utils.sb3_env_utils import make_env
from utils.sb3_evaluate_policy import evaluate_policy_with_success_rate


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


# strategy for save policy，有两个选择：（1）根据最大的prob_true_act保存；（2）根据最小的loss保存。
# 因为bc.train()方法的on_batch_end是没有参数的回调函数，所以这里使用闭包，通过一个外部变量记录最高的prob_true_act
def on_best_act_prob_save(algo: PPO, validation_transitions: TransitionsMinimal, sb3_logger: Logger):
    best_prob = PROB_TRUE_ACT_THRESHOLD  # 预估一个初始值，不然训练开始阶段会浪费大量的时间在存储模型上！！！
    def calc_func():
        algo.policy.set_training_mode(mode=False)
        
        nonlocal best_prob
        
        obs = util.safe_to_tensor(validation_transitions.obs).to("cuda")
        acts = util.safe_to_tensor(validation_transitions.acts).to("cuda")
        _, log_prob, entropy = algo.policy.evaluate_actions(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        if prob_true_act > best_prob:
            sb3_logger.info(f"update prob true act from {best_prob} to {prob_true_act}!")
            best_prob = prob_true_act

            # save policy
            checkpoint_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / EXPERIMENT_NAME
            if not checkpoint_save_dir.exists():
                checkpoint_save_dir.mkdir()

            algo.save(str(checkpoint_save_dir / POLICY_FILE_SAVE_NAME))

        algo.policy.set_training_mode(mode=True)
    return calc_func

def on_best_loss_save(algo: PPO, validation_transitions: TransitionsMinimal, loss_calculator: bc.BehaviorCloningLossCalculator, sb3_logger: Logger):
    min_loss = LOSS_THRESHOLD  # 预估一个初始值，不然训练开始阶段会浪费大量的时间在存储模型上！！！
    def calc_func():
        algo.policy.set_training_mode(mode=False)
        
        nonlocal min_loss
        
        obs = util.safe_to_tensor(validation_transitions.obs).to("cuda")
        acts = util.safe_to_tensor(validation_transitions.acts).to("cuda")
        
        metrics: bc.BCTrainingMetrics = loss_calculator(policy=algo.policy, obs=obs, acts=acts)
        cur_loss = metrics.loss
        if cur_loss < min_loss:
            sb3_logger.info(f"update loss from {min_loss} to {cur_loss}!")
            min_loss = cur_loss

            # save policy
            checkpoint_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / EXPERIMENT_NAME
            if not checkpoint_save_dir.exists():
                checkpoint_save_dir.mkdir()

            algo.save(str(checkpoint_save_dir / POLICY_FILE_SAVE_NAME))

        algo.policy.set_training_mode(mode=True)
    return calc_func


def train():

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / "bc" / EXPERIMENT_NAME).absolute()), format_strings=['stdout', 'log', 'csv', 'tensorboard'])

    env = make_env(
        env_id=ENV_NAME, rank=0, seed=0,
        flattern_obs=ENV_FLATTERN_OBS,
        wrap_with_nmr=ENV_USE_NMR, 
        nmr_waypoint=ENV_NMR_WAYPOINT, nmr_waypoint_delta=ENV_NMR_WAYPOINT_DELTA,
        nmr_jump=ENV_NMR_JUMP, nmr_jump_cnt=ENV_NMR_JUMP_CNT, nmr_length=ENV_NMR_LENGTH, 
        nmr_use_original_reward=ENV_USE_ORIGINAL_REWARD, nmr_non_terminal_reward=ENV_NON_TERMINAL_REWARD, nmr_terminal_reward=ENV_TERMINAL_REWARD,
        scale_obs=True, expert_data_dir=EXPERT_DATA_CACHE_DIR
    )()

    data_file: Path = PROJECT_ROOT_DIR / EXPERT_DATA_CACHE_DIR
    sb3_logger.info(f"load data from {str(data_file.absolute())}")
    scaled_obs, acts, infos, obs_scaler = load_data(data_file)

    algo_ppo = get_ppo_algo(env)
    sb3_logger.info(str(algo_ppo.policy))

    rng = np.random.default_rng(SEED)
    
    train_transitions, validation_transitions, test_transitions = split_data(
        obs=scaled_obs, 
        acts=acts,
        infos=infos,
        train_size=DATASET_SPLIT[0],
        validation_size=DATASET_SPLIT[1],
        test_size=DATASET_SPLIT[2],
        shuffle=True,
    )

    sb3_logger.info(f"train_set: obs size, {train_transitions.obs.shape}, act size, {train_transitions.acts.shape}")
    sb3_logger.info(f"validation_set: obs size, {validation_transitions.obs.shape}, act size, {validation_transitions.acts.shape}")
    sb3_logger.info(f"test_set: obs size, {test_transitions.obs.shape}, act size, {test_transitions.acts.shape}")

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=algo_ppo.policy,
        batch_size=BC_BATCH_SIZE,
        ent_weight=BC_ENT_WEIGHT,
        l2_weight=BC_L2_WEIGHT,
        demonstrations=train_transitions,
        rng=rng,
        custom_logger=HierarchicalLogger(sb3_logger)
    )

    # train
    bc_trainer.train(
        n_epochs=TRAIN_EPOCHS,
        # on_batch_end=on_best_act_prob_save(algo_ppo, validation_transitions, sb3_logger),
        on_batch_end=on_best_loss_save(algo_ppo, validation_transitions, bc_trainer.loss_calculator, sb3_logger) if USE_LOSS_CALLBACK else on_best_act_prob_save(algo_ppo, validation_transitions, sb3_logger),
    )

    # evaluate with environment
    reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, env, n_eval_episodes=10)
    sb3_logger.info(f"After BC, success rate: {success_rate}, reward: {reward}")

    # 最终的policy在测试集上的prob_true_act / loss
    if USE_LOSS_CALLBACK:
        test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "训练结束时的策略", "测试集")
    else:
        test_on_prob_true_act(algo_ppo.policy, test_transitions, sb3_logger, "训练结束时的策略", "测试集")

    return sb3_logger, validation_transitions, test_transitions, bc_trainer, env


def test_on_prob_true_act(
        policy: MlpPolicy, 
        test_transitions: TransitionsMinimal, 
        sb3_logger: Logger, 
        policy_descreption: str, dataset_descreption: str
    ):
    policy.set_training_mode(mode=False)

    obs = util.safe_to_tensor(test_transitions.obs).to("cuda")
    acts = util.safe_to_tensor(test_transitions.acts).to("cuda")
    _, log_prob, entropy = policy.evaluate_actions(obs, acts)
    prob_true_act = th.exp(log_prob).mean()
    sb3_logger.info(f"{policy_descreption}在{dataset_descreption}上的prob_true_act: {prob_true_act}.")


def test_on_loss(
        policy: MlpPolicy, 
        test_transitions: TransitionsMinimal, 
        loss_calculator: bc.BehaviorCloningLossCalculator, 
        sb3_logger: Logger, 
        policy_descreption: str, dataset_descreption: str
    ):
    policy.set_training_mode(mode=False)

    obs = util.safe_to_tensor(test_transitions.obs).to("cuda")
    acts = util.safe_to_tensor(test_transitions.acts).to("cuda")
    
    metrics: bc.BCTrainingMetrics = loss_calculator(policy=policy, obs=obs, acts=acts)
    sb3_logger.info(f"{policy_descreption}在{dataset_descreption}上的loss: {metrics.loss}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="传入配置文件")
    parser.add_argument("--config-file-name", type=str, help="配置文件名", default="iter_1/seed1/reacher_256_256.json")
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

    EXPERIMENT_NAME = custom_config["bc"]["experiment_name"]
    SEED = custom_config["bc"]["seed"]
    POLICY_FILE_SAVE_NAME = custom_config["bc"]["policy_file_save_name"]
    TRAIN_EPOCHS = custom_config["bc"]["train_epochs"]
    BC_BATCH_SIZE = custom_config["bc"]["batch_size"]
    BC_L2_WEIGHT = custom_config["bc"].get("l2_weight", 0.0)
    BC_ENT_WEIGHT = custom_config["bc"].get("ent_weight", 1e-3)
    EXPERT_DATA_CACHE_DIR = custom_config["bc"]["data_cache_dir"]
    USE_LOSS_CALLBACK = custom_config["bc"]["use_loss_callback"]
    PROB_TRUE_ACT_THRESHOLD = custom_config["bc"]["prob_true_act_threshold"]  # validate的时候，当prob_true_act大于这个值的时候，开始保存prob_true_act最优的policy
    LOSS_THRESHOLD = custom_config["bc"]["loss_threshold"]
    DATASET_SPLIT = custom_config["bc"].get("dataset_split", [0.96, 0.02, 0.02])

    RL_SEED = custom_config["rl"]["seed"]
    NET_ARCH = custom_config["rl_bc"]["net_arch"]
    PPO_BATCH_SIZE = custom_config["rl_bc"]["batch_size"]
    GAMMA = custom_config["rl_bc"]["gamma"]
    KL_WITH_BC_MODEL_COEF = custom_config["rl_bc"]["kl_with_bc_model_coef"]

    register_my_env(goal_range=0.3, distance_threshold=0.01, max_episode_steps=ENV_MAX_EPISODE_STEPS)

    sb3_logger, validation_transitions, test_transitions, bc_trainer, env = train()

    policy_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "bc" / EXPERIMENT_NAME
    algo_ppo = PPOWithBCLoss.load(str((policy_save_dir / POLICY_FILE_SAVE_NAME).absolute()))

    if USE_LOSS_CALLBACK:
        test_on_loss(algo_ppo.policy, validation_transitions, bc_trainer.loss_calculator, sb3_logger, "最优策略", "验证集")
        test_on_loss(algo_ppo.policy, test_transitions, bc_trainer.loss_calculator, sb3_logger, "最优策略", "测试集")
    else:
        test_on_prob_true_act(algo_ppo.policy, validation_transitions, sb3_logger, "最优策略", "验证集")
        test_on_prob_true_act(algo_ppo.policy, test_transitions, sb3_logger, "最优策略", "测试集")
    
    for i in range(5):
        reward, _, success_rate = evaluate_policy_with_success_rate(algo_ppo.policy, env, n_eval_episodes=100)
        sb3_logger.info(f"最优策略, success rate: {success_rate}, reward: {reward}")
