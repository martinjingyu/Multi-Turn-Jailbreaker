import os 
import json
import hydra
import datasets
import logging
import logging
import yaml
import random
import torch
from tqdm import tqdm
from types import SimpleNamespace
import argparse

from generate_utils import Node, TreeGenerator
from model.Attacker import AttackAgent
from model.Evaluator.llamaJedge import LlamaGuardModeration
from model.Target.target_model import TargetModel
from utils import upload_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    with open("config/generate_trajdata.yaml", "r") as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))
        
    with open("config/target_config.yaml", "r") as f:
        target_cfg = SimpleNamespace(**yaml.safe_load(f))

    with open("config/attacker_config.yaml", "r") as f:
        attacker_config = SimpleNamespace(**yaml.safe_load(f))

        
    attack_agent = AttackAgent(attacker_config)
    target = TargetModel(target_cfg)
    # evaluator = Evaluator(target.model, target.tokenizer, target.sampling_params)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    torch.cuda.set_device(0)
    evaluator = LlamaGuardModeration()
    generator = TreeGenerator(cfg, attack_agent, target, evaluator)
    
    
    try:
        if (cfg.seed_prompts_file.endswith(".parquet")):
            dataset = datasets.load_dataset("parquet", data_files=cfg.seed_prompts_file)
            prompts_data = dataset["train"]
            prompts_data_list = []
            for i in range(len(prompts_data)):
                prompts_data_list.append(prompts_data[i])
            prompts_data = prompts_data_list
        
        
        else:
            prompts_data_list = []
            with open(cfg.seed_prompts_file, "r") as f:
                prompts_data = json.load(f)
                for prompt in prompts_data:
                    
                    prompts_data_list.append(prompt["goal"])

    except Exception as e:
        print(f"Error loading prompts data: {e}")
        # with open(cfg.seed_prompts_file, "r") as f:
        #     prompts_data = [json.loads(line) for line in f.readlines()]
        exit()
        
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    cfg.output_path = cfg.output_path.format(task_name=cfg.task_name)
    
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path, exist_ok=True)

    logging.info(f"Load {len(prompts_data)} Prompts for generating trajectory data!")
    
    for i in range(args.start_idx, args.end_idx, cfg.trees_per_batch):

        sub_prompt_seeds = prompts_data_list[i:i+cfg.trees_per_batch]
        root_node_list = grow_trees(cfg, sub_prompt_seeds, generator)
        
        
        for j, root_node in enumerate(root_node_list):
            root_node.save_tree(f"{cfg.output_path}/{i}_{j}.json")
        
        upload_data()
        
            
        

def grow_trees(config, prompt_list:list[str], generator: TreeGenerator) -> list[Node]:

    root_list = []
    for propmt in prompt_list:
        root = Node("root",None, propmt, None, 0, None, None, None)
        root_list.append(root)
        
        
    generator.build_tree_to_depth(root_list)
    
    return root_list




if __name__ == "__main__":
    main()