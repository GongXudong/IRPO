from gym import ObservationWrapper, ActionWrapper, Env, spaces
import d4rl
from sklearn.preprocessing import StandardScaler
from typing import TypeVar
import numpy as np
from pathlib import Path
import sys

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")


class ScaledObservationWrapper(ObservationWrapper):
    
    def __init__(self, env: Env, scaler: StandardScaler):
        super().__init__(env)

        # 缩放与仿真器无关，只在学习器中使用
        # 送进策略网络的观测，各分量的取值都在[-inf, inf]之间，但是做了标准化        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_space.shape)
        self.state_scalar = scaler
    
    def scale_state(self, state_var: np.ndarray) -> np.ndarray:
        """将仿真器返回的state缩放到[0, 1]之间
        """
        if len(state_var.shape) == 1:
            tmp_state_var = state_var.reshape((1, -1))
            return self.state_scalar.transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            return self.state_scalar.transform(state_var)
        else:
            raise TypeError("state_var只能是1维或者2维！")

    def observation(self, observation: ObsType) -> WrapperObsType:
        # 检查observation类型
        if type(observation) == np.ndarray:
            return self.scale_state(observation)
        else:
            return self.scale_state(np.array(observation))
    
    def inverse_scale_state(self, state_var: np.ndarray) -> np.ndarray:
        """将[0, 1]之间state变回仿真器定义的原始state。用于测试！！！
        """
        if len(state_var.shape) == 1:
            tmp_state_var = state_var.reshape((1, -1))
            return self.state_scalar.inverse_transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            return self.state_scalar.inverse_transform(state_var)
        else:
            raise TypeError("state_var只能是1维或者2维！")