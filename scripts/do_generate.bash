#!/bin/bash
#   bash scripts/do_generate.bash 0 100 0 &
#   bash scripts/do_generate.bash 100 200 1 &
#   bash scripts/do_generate.bash 101 150 2 &
#   bash scripts/do_generate.bash 151 200 3 & 

# bash scripts/do_generate.bash 180 200 3 & 
start_idx=$1
end_idx=$2
GPU=0

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=$GPU python ./datagenerator/do_generate.py \
    --start_idx=$start_idx \
    --end_idx=$end_idx\
    --prompot_seed 'sft'

