#!/bin/bash
# Run evaluation pipeline: attack the target model and save trajectories.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.bash
#
# After pipeline finishes, score the results with:
#   CUDA_VISIBLE_DEVICES=0 python evaluate/get_score.py \
#       --dir data/result/<method>-<target> --evaluate

export PYTHONPATH=$(pwd):$PYTHONPATH

# CUDA_VISIBLE_DEVICES=0 python evaluate/pipeline.py --method "MartinJYHuang/Jailbreak-agent-temp"
# CUDA_VISIBLE_DEVICES=0 python evaluate/pipeline.py --method "MartinJYHuang/JA-Llama-3"
CUDA_VISIBLE_DEVICES=0 python evaluate/pipeline.py --method "MartinJYHuang/JA-v2"
