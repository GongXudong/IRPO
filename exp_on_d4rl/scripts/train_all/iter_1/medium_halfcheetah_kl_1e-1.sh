#!/bin/bash

python sb3_bc_train.py --config_file_name iter_1/seed1/kl_1e-1/medium_halfcheetah_256_256_kl1e-1.json

python sb3_rl_train_after_bc.py --config_file_name iter_1/seed1/kl_1e-1/medium_halfcheetah_256_256_kl1e-1.json

python rollout/rollout.py --config_file_name iter_1/seed1/kl_1e-1/medium_halfcheetah_256_256_kl1e-1.json