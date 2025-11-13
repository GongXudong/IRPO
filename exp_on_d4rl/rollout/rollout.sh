#!/bin/bash

# iter_1
python rollout/rollout.py --config_file_name iter_1/seed1/kl_1/medium_halfcheetah_256_256_kl1.json
python rollout/rollout.py --config_file_name iter_1/seed1/kl_1/medium_hopper_256_256_kl1.json
python rollout/rollout.py --config_file_name iter_1/seed1/kl_1/medium_walker2d_256_256_kl1.json
