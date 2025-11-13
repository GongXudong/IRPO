import os
import tempfile
import time
from copy import deepcopy
from functools import wraps
from threading import Thread
from typing import Optional, Type, Union

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import VecEnv


class EmaBCPolicyCallback2(BaseCallback):

    def __init__(self, ema_gamma: float = 0.99, verbose: int = 0):
        super().__init__(verbose)
        self.ema_gamma = ema_gamma

    def _on_step(self) -> bool:
        print("update bc model!!!!!!!!!!!!!!")
        print(f"n_calls: {self.n_calls}")
        polyak_update(
            self.model.policy.parameters(), 
            self.locals["self"].bc_trained_algo.policy.parameters(), 
            1.0-self.ema_gamma
        )
        return True

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # make mypy happy
        assert self.model is not None

        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)  # type: ignore[union-attr]
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True