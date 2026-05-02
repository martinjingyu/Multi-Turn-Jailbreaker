#!/bin/bash
# Run evaluation: attack the target model and save trajectories to data/result/{org}/{model}-{target}/
#
# Step 1 — run attack pipeline:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.bash
#
# Step 2 — score results with LlamaGuard (run on a free GPU after step 1):
#   CUDA_VISIBLE_DEVICES=0 python evaluate/get_score.py \
#       --dir data/result/MartinJYHuang/JA-v2-Meta-Llama-3-8B-Instruct \
#       --evaluate
#
# For a quick score-only pass (reward >= 4.0 or 5.0), use:
#   python utils/cal_asr.py

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python evaluate/pipeline.py --method "MartinJYHuang/Jailbreak-agent-temp"
# CUDA_VISIBLE_DEVICES=0 python evaluate/pipeline.py --method "MartinJYHuang/JA-Llama-3"
# CUDA_VISIBLE_DEVICES=0 python evaluate/pipeline.py --method "MartinJYHuang/JA-v2"
