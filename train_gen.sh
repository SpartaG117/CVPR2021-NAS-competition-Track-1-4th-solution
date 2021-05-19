#!/bin/bash

python gen_numpy_data.py
python train.py --model_type 0 --output_dir ws --lcm 32 --epochs 300
python train.py --model_type 1 --output_dir op --lcm 32 --epochs 1000 --fixlr --learning_rate 0.001 --pretrain ./checkpoints/ws/weights.pth
python submit_sh.py --num_gpu 4 --n_p 16 --output_dir op --model_name weights.pth
bash ./checkpoints/op/submission/run_0_50000.sh
python merge.py ./checkpoints/op/submission
