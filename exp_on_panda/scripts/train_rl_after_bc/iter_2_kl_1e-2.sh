#!/bin/bash

python sb3_rl_train_after_bc.py --config_file_name iter_2/seed1/reacher_256_256_kl_1e-2.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed2/reacher_256_256_kl_1e-2.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed3/reacher_256_256_kl_1e-2.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed4/reacher_256_256_kl_1e-2.json
python sb3_rl_train_after_bc.py --config_file_name iter_2/seed5/reacher_256_256_kl_1e-2.json