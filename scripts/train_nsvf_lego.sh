#!/bin/bash

set -euo pipefail

export DATA_DIR=./Synthetic_NeRF

python3 train.py \
    --root_dir $DATA_DIR/Lego \
    --exp_name Lego --perf \
    --num_epochs 100 --batch_size 4 --lr 1e-2 --half2_opt\
