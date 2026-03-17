from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import traceback
from trainer.ppo_utils import load_seeds, save_trees
from trainer.grpo_vllm_one import gen_worker
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_path = "MartinJYHuang/Jailbreak-agent-temp"
gen_device = 0    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.04
all_steps = 1000
Q_batch_size = 5
train_batch_size = 8
gen_update_steps = 16
save_steps = 200
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def load_data():
    from datagenerator.mcts_utils import MCTSNode
    from pathlib import Path
    import json
    data_dir = Path("ppo_data/JA-advBench/depth1/")
    json_files = list(data_dir.glob("*.json"))

    root_list = []
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            root_node = MCTSNode.load_from_json(data)
            root_list.append(root_node)
    return root_list

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

if __name__ == '__main__':
    
    gen_worker(Q=None, physics_device=0)
