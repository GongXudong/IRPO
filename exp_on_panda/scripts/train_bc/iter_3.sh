#!/bin/bash

python sb3_bc_train.py --config_file_name iter_3/seed1/reacher_256_256_kl_1e0.json
python sb3_bc_train.py --config_file_name iter_3/seed2/reacher_256_256_kl_1e0.json
python sb3_bc_train.py --config_file_name iter_3/seed3/reacher_256_256_kl_1e0.json
python sb3_bc_train.py --config_file_name iter_3/seed4/reacher_256_256_kl_1e0.json
python sb3_bc_train.py --config_file_name iter_3/seed5/reacher_256_256_kl_1e0.json