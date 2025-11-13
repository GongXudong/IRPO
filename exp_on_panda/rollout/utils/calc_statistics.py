from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

def is_goal_in_goal_list(goal: list, goal_list: List[Dict]) -> int:
    """判断goal是否在goal_list中

    Args:
        goal (list): [goal_x, goal_y, goal_z]
        goal_list (List[Dict]): Dict的key包括goal_x, goal_y, goal_z, length, rollout_by

    Returns:
        _type_: 返回goal在goal_list中的index，如果不存在，返回-1
    """
    for index, item_goal in enumerate(goal_list):
        if item_goal["goal_x"] == goal[0] and item_goal["goal_y"] == goal[1] and item_goal["goal_z"] == goal[2]:
            return index
    return -1



def get_statistics_of_one_file(csv_file_path: Path, rollout_by: str) -> pd.DataFrame:
    """
    goal_list中的item的key包括：goal_x, goal_y, goal_z, length, rollout_by

    Args:
        csv_file_path (Path): _description_
        rollout_by (str): 来自rollout/cache/内的哪个文件，，格式为“bc/文件名”或“rl_bc/文件名”
    """
    goal_list: List[Dict] = []
    data_df = pd.read_csv(csv_file_path)

    for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
        goal = [row["s_g_x"], row["s_g_y"], row["s_g_z"]]
        goal_index = is_goal_in_goal_list(goal, goal_list)
        if goal_index == -1: 
            
            this_goal_length = data_df[(data_df["s_g_x"] == goal[0]) & (data_df["s_g_y"] == goal[1]) & (data_df["s_g_z"] == goal[2])]

            tmp_goal_info = {
                "goal_x": goal[0],
                "goal_y": goal[1],
                "goal_z": goal[2],
                "length": len(this_goal_length),
                "rollout_by": rollout_by
            }

            goal_list.append(tmp_goal_info)

    return pd.DataFrame(goal_list)

def merge_goal_list(goal_list_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """合并goal_list，相同目标对应的多条轨迹，取短的

    Args:
        goal_list_dict (Dict[str, pd.DataFrame]): key对应get_statistics_of_one_file()中的rollout_by参数，value对应get_statistics_of_one_file()返回的pd.DataFrame
    """
    goal_list_total: List[Dict] = []
    for goal, goal_list in goal_list_dict.items():
        print(f"process {goal}...")
        for index, row in tqdm(goal_list.iterrows(), total=len(goal_list)):
            this_goal = [row["goal_x"], row["goal_y"], row["goal_z"]]
            goal_index = is_goal_in_goal_list(this_goal, goal_list_total)
            if goal_index == -1:
                goal_list_total.append({
                    "goal_x": this_goal[0],
                    "goal_y": this_goal[1],
                    "goal_z": this_goal[2],
                    "length": row["length"],
                    "rollout_by": row["rollout_by"]
                })
            else:
                if goal_list_total[goal_index]["length"] > row["length"]:
                    goal_list_total[goal_index]["length"] = row["length"]
                    goal_list_total[goal_index]["rollout_by"] = row["rollout_by"]
    
    return pd.DataFrame(goal_list_total)

def load_data_by_goal_list(goal_list: pd.DataFrame) -> pd.DataFrame:
    """接收merge_goal_list()返回的结果，从相应文件加载轨迹数据

    Args:
        goal_list (pd.DataFrame): 包括的列：goal_x, goal_y, goal_z, length, rollout_by

    Returns:
        _type_: _description_
    """
    data_df = pd.DataFrame(columns=['s_x' ,'s_y', 's_z', 's_v_x', 's_v_y', 's_v_z', 's_g_x', 's_g_y', 's_g_z', 'a_x', 'a_y', 'a_z'])

    for tmp_file in tqdm(goal_list["rollout_by"].unique()):
        data_file: Path = PROJECT_ROOT_DIR / "rollout" / "cache" / tmp_file
        tmp_df = pd.read_csv(data_file)
        
        for index, row in goal_list[goal_list["rollout_by"] == tmp_file].iterrows():
            
            this_goal_df = tmp_df[(tmp_df["s_g_x"] == row["goal_x"]) & (tmp_df["s_g_y"] == row["goal_y"]) & (tmp_df["s_g_z"] == row["goal_z"])]
            data_df = pd.concat([data_df, this_goal_df])
    
    return data_df