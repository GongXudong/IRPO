import gym
import d4rl
import sys
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed
from utils.sb3_env_wrappers import ScaledObservationWrapper
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils import load_data
from rollout.load_data import load_data as load_data_from_hd5


def make_env(env_id: str, rank: int, seed: int = 0, scale_obs: bool = False, expert_data_dir: str = "cache"):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        if scale_obs:
            if expert_data_dir == "cache":
                
                print("\033[32m load data from d4rl cache. \033[0m")
                obs, acts, infos = load_data.load_data(env)
            else:
                data_file: Path = PROJECT_ROOT_DIR / expert_data_dir
                print(f"\033[31m load data from {str(data_file.absolute())} \033[0m")
                dataset = load_data_from_hd5(str(data_file.absolute()))
                obs, acts, infos = load_data.load_data_from_my_cache(dataset)
            
            scaled_obs, scaler = load_data.scale_obs(obs)
            env = ScaledObservationWrapper(env=env, scaler=scaler)
        # env.reset(seed=seed + rank)
        env.reset()
        return env
    set_random_seed(seed)
    return _init

