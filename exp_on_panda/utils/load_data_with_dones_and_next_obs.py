from imitation.data.types import TransitionsMinimal, Transitions
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import sys


def load_data(env):
    dataset = env.get_dataset()
    print(dataset.keys())
    obs = np.array(dataset["observations"])
    act = np.array(dataset["actions"])
    tmp = list(dataset["observations"][1:])
    tmp.append(dataset["observations"][-1])
    next_obs = np.array(tmp)
    dones = np.array(dataset["terminals"])
    if "infos/action_log_probs" in dataset:
        infos = np.array(dataset["infos/action_log_probs"])
    else:
        infos = np.array([None] * len(obs))
    return obs, act, infos, next_obs, dones

def scale_obs(obs: np.array) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaled_obs = scaler.fit_transform(obs)
    return scaled_obs, scaler

def split_data(
        obs: np.ndarray, 
        acts: np.ndarray, 
        infos: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        train_size: float=0.9, 
        validation_size: float=0.05, 
        test_size: float=0.05,
        shuffle: bool=True,
    ) -> Tuple[TransitionsMinimal, TransitionsMinimal, TransitionsMinimal]:

    # 训练集、验证集、测试集划分
    train_data, tmp_data, train_labels, tmp_labels, train_infos, tmp_infos, train_next_obs, tmp_next_obs, train_dones, tmp_dones = train_test_split(
        obs, acts, infos, next_obs, dones,
        train_size=train_size, 
        test_size=validation_size + test_size, 
        shuffle=shuffle,
        random_state=0,  # 保证每次得到的结果是一样的
    )

    validation_data, test_data, validation_labels, test_labels, validation_infos, test_infos, validation_next_obs, test_next_obs, validation_dones, test_dones = train_test_split(
        tmp_data, tmp_labels, tmp_infos, tmp_next_obs, tmp_dones,
        train_size=validation_size/(validation_size + test_size),
        test_size=test_size/(validation_size + test_size),
        shuffle=shuffle,
        random_state=0,
    )

    return (
        Transitions(obs=train_data, acts=train_labels, infos=train_infos, next_obs=train_next_obs, dones=train_dones),
        Transitions(obs=validation_data, acts=validation_labels, infos=validation_infos, next_obs=validation_next_obs, dones=validation_dones),
        Transitions(obs=test_data, acts=test_labels, infos=test_infos, next_obs=test_next_obs, dones=test_dones)
    )