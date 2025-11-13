#!/bin/bash

python sb3_rl_train_after_bc.py --config_file_name iter_2/seed1/kl_0/medium_hopper_256_256_kl0.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed2/kl_0/medium_hopper_256_256_kl0.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed3/kl_0/medium_hopper_256_256_kl0.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed4/kl_0/medium_hopper_256_256_kl0.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed5/kl_0/medium_hopper_256_256_kl0.json