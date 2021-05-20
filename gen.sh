#!/bin/bash

python submit_sh.py --num_gpu 4 --n_p 16 --output_dir op --model_name weights.pth
bash ./checkpoints/op/submission/run_0_50000.sh
python merge.py ./checkpoints/op/submission
