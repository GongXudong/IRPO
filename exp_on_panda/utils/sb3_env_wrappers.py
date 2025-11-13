from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces
import panda_gym
from sklearn.preprocessing import StandardScaler
from typing import TypeVar
import numpy as np
from pathlib import Path
import sys
from typing import Union

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


class ScaledDictObservationWrapper(ObservationWrapper):
    """用于dict观测类型的环境，key包括observation，desired_goal，achieved_goal
    """
    def __init__(self, env: Env, obs_scaler: StandardScaler, achieved_goal_scaler: StandardScaler, desired_goal_scaler: StandardScaler):
        super().__init__(env)

        # 缩放与仿真器无关，只在学习器中使用
        # 送进策略网络的观测，各分量的取值都在[-inf, inf]之间，但是做了标准化        
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=env.observation_space["observation"].shape, dtype=np.float32),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=env.observation_space["desired_goal"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=env.observation_space["achieved_goal"].shape, dtype=np.float32),
            )
        )
        self.obs_scalar: StandardScaler = obs_scaler
        self.achieved_goal_scaler: StandardScaler = achieved_goal_scaler
        self.desired_goal_scaler: StandardScaler = desired_goal_scaler

    def _scale_single_state(self, state_var: dict) -> dict:

        assert "observation" in state_var, "state_var中必须包括键：observation！！！"
        assert "achieved_goal" in state_var, "state_var中必须包括键：achieved_goal！！！"
        assert "desired_goal" in state_var, "state_var中必须包括键：desired_goal！！！"
        assert type(state_var["observation"]) == np.array, "observation必须是np.ndarray类型！！！"
        assert type(state_var["achieved_goal"]) == np.array, "achieved_goal必须是np.ndarray类型！！！"
        assert type(state_var["desired_goal"]) == np.array, "desired_goal必须是np.ndarray类型！！！"

        tmp_obs_var = state_var["observation"].reshape((1, -1))
        tmp_a_goal_var = state_var["achieved_goal"].reshape((1, -1))
        tmp_d_goal_var = state_var["desired_goal"].reshape((1, -1))
        
        return {
            "observation": self.obs_scalar.transform(tmp_obs_var).reshape((-1)),
            "achieved_goal": self.achieved_goal_scaler.transform(tmp_a_goal_var).reshape((-1)),
            "desired_goal": self.desired_goal_scaler.transform(tmp_d_goal_var).reshape((-1)),
        }

    def scale_state(self, state_var: Union[dict, np.ndarray]) -> Union[dict, np.ndarray]:
        """将仿真器返回的state缩放到[0, 1]之间
        """
        if type(state_var) == dict:
            return self._scale_single_state(state_var)
        elif type(state_var) == np.ndarray:
            np.array([self._scale_single_state(item) for item in state_var])
        else:
            raise TypeError("state_var只能是dict或者np.array类型！")

    def observation(self, observation: ObsType) -> WrapperObsType:
        # 检查observation类型
        assert type(observation) == dict, "observation只能是dict类型"
        return self.scale_state(observation)
    
    def _inverse_scale_single_state(self, state_var: dict) -> dict:
        assert "observation" in state_var, "state_var中必须包括键：observation！！！"
        assert "achieved_goal" in state_var, "state_var中必须包括键：achieved_goal！！！"
        assert "desired_goal" in state_var, "state_var中必须包括键：desired_goal！！！"
        assert type(state_var["observation"]) == np.array, "observation必须是np.ndarray类型！！！"
        assert type(state_var["achieved_goal"]) == np.array, "achieved_goal必须是np.ndarray类型！！！"
        assert type(state_var["desired_goal"]) == np.array, "desired_goal必须是np.ndarray类型！！！"

        tmp_obs_var = state_var["observation"].reshape((1, -1))
        tmp_a_goal_var = state_var["achieved_goal"].reshape((1, -1))
        tmp_d_goal_var = state_var["desired_goal"].reshape((1, -1))

        return {
            "observation": self.obs_scalar.inverse_transform(tmp_obs_var).reshape((-1)),
            "achieved_goal": self.achieved_goal_scaler.inverse_transform(tmp_a_goal_var).reshape((-1)),
            "desired_goal": self.desired_goal_scaler.inverse_transform(tmp_d_goal_var).reshape((-1)),
        }

    def inverse_scale_state(self, state_var: Union[dict, np.ndarray]) -> Union[dict, np.ndarray]:
        """将[0, 1]之间state变回仿真器定义的原始state。用于测试！！！
        """
        if type(state_var) == dict:
            return self._inverse_scale_single_state(state_var)
        elif type(state_var) == np.ndarray:
            return np.array([self._inverse_scale_single_state(item) for item in state_var])
        else:
            raise TypeError("state_var只能是dict或者np.array类型！")