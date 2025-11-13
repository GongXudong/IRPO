import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import panda_gym
import numpy as np
import sys
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed
from utils.sb3_env_wrappers import ScaledObservationWrapper
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils import load_data
from utils.non_markovian_reward_jump_wrapper import NMRJumpWrapper
from utils.non_markovian_reward_waypoint_wrapper import NMRWayPointWrapper
from utils.non_markovian_reward_wrapper import NMRWrapper

def make_env(
        env_id: str, 
        rank: int, 
        seed: int = 0, 
        flattern_obs: bool = True,
        wrap_with_nmr: bool = False, 
        nmr_waypoint: bool = False,
        nmr_waypoint_delta: np.ndarray = np.array([0., 0., 0.1]),
        nmr_jump: bool = True,
        nmr_jump_cnt: int = 2,
        nmr_length: int = 10, 
        nmr_use_original_reward: float = 0., nmr_non_terminal_reward: float = 0., nmr_terminal_reward: float = 1.,
        scale_obs: bool = False, 
        expert_data_dir: str = "cache"
    ):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id)

        if wrap_with_nmr:
            if nmr_waypoint:
                print(f"wrap env with NMR Waypoint, {nmr_waypoint_delta}")
                env = NMRWayPointWrapper(env, target_waypoint_delta=nmr_waypoint_delta, use_original_reward=nmr_use_original_reward, non_terminal_reward=nmr_non_terminal_reward, terminal_reward=nmr_terminal_reward)
            else:
                if nmr_jump:
                    print(f"wrap env with NMR Jump, {nmr_jump_cnt}, {nmr_use_original_reward}, {nmr_non_terminal_reward}, {nmr_terminal_reward}")
                    env=NMRJumpWrapper(env, jump_cnt_target=nmr_jump_cnt, use_original_reward=nmr_use_original_reward, non_terminal_reward=nmr_non_terminal_reward, terminal_reward=nmr_terminal_reward)
                else:
                    print(f"wrap env with NMR, {nmr_length}, {nmr_use_original_reward}, {nmr_non_terminal_reward}, {nmr_terminal_reward}")
                    env = NMRWrapper(env, nmr_length=nmr_length, use_original_reward=nmr_use_original_reward, non_terminal_reward=nmr_non_terminal_reward, terminal_reward=nmr_terminal_reward)

        if scale_obs:
            data_file: Path = PROJECT_ROOT_DIR / expert_data_dir
            print(f"\033[31m load data from {str(data_file.absolute())} \033[0m")
            scaled_obs, acts, infos, obs_scaler = load_data.load_data(data_file)
            if flattern_obs:
                env = FlattenObservation(env)
            env = ScaledObservationWrapper(env=env, scaler=obs_scaler)
        else:
            if flattern_obs:
                env = FlattenObservation(env)

        env.reset(seed=seed + rank)
        # env.reset()
        return env
    set_random_seed(seed)
    return _init
