import unittest
import numpy as np
from pathlib import Path
import sys
from functools import partial
import gymnasium as gym
from collections import namedtuple

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.non_markovian_reward_waypoint_wrapper import NMRWayPointWrapper
from utils.register_env import register_my_env
from utils.sb3_env_utils import make_env

register_my_env(goal_range=0.3, distance_threshold=0.01)


class NMRWrapperTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.env = make_env(
            env_id="my-reach",
            rank=0,
            seed=0,
            flattern_obs=False,
            wrap_with_nmr=True,
            nmr_waypoint=True, 
            nmr_waypoint_delta=np.array([0., 0., 0.1]),
            nmr_use_original_reward=False,
            nmr_non_terminal_reward=-1.0,
            nmr_terminal_reward=0.0
        )()
    
    def test_reset(self):
        for i in range(5):
            self.env.reset()
            print(self.env.unwrapped.task.goal, self.env.waypoint)

    def test_step(self):
        self.env.reset()
        for i in range(5):
            obs, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            print(obs, reward, terminated, truncated, info)

if __name__ == "__main__":
    unittest.main()