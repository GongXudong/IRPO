import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

def load_config(config_name: str):
    
    with open(str(CONFIG_DIR / config_name), "r") as f:
        return json.load(f)


if __name__ == "__main__":
    res = load_config("config1.json")
    print(res)
    print(type(res["rl"]["net_arch"]), res["rl"]["net_arch"])