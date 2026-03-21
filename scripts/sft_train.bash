export PYTHONPATH=$(pwd):$PYTHONPATH
# CUDA_VISIBLE_DEVICES=0,1 python ./trainer/sft_trainer.py
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 ./trainer/sft_trainer.py