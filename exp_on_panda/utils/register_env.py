from pathlib import Path
import sys
from gymnasium.envs.registration import register

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

def register_my_env(goal_range: float=0.3, distance_threshold: float=0.01, max_episode_steps: int=50):
    register(
        id="my-reach",
        entry_point=f"my_reach_env:MyPandaReachEnv",
        kwargs={"reward_type": "sparse", "control_type": "ee", "goal_range": goal_range, "distance_threshold": distance_threshold},
        max_episode_steps=max_episode_steps,
    )   