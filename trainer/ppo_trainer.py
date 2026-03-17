from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.optim import AdamW
from utils import get_config
import torch
from torch.utils.data import DataLoader
import json
from utils import preprocess, IGNORE_INDEX, get_config, seed_torch, safe_save_model_for_hf_trainer
from trainer.ppo_utils import MCSTreeGenerator, save_trees
from datagenerator.mcts_utils import MCTSNode
from model.Attacker.attack_agent import AttackAgent
from model.Target.target_model import TargetModel
from dataset.ppoDataset import PPODataset
import random
import os
from trl import PPOTrainer, PPOConfig
import torch.distributed as dist

def model_init(model_args):
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        model_max_length=model_args.model_max_length,
        padding_side="left",
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else training_args.bf32,
        
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    model.to(device)

    print(f"[DEBUG] Rank {dist.get_rank()} using device: {device}")
    print(f"[DEBUG] Model device: {next(model.parameters()).device}")
    return model, tokenizer

def get_evaluator(target_model):

    # if "gpt" in config.reward_url:
    #     from model.Evaluator.gpt4o import OpenAI_Models
    #     llm = OpenAI_Models(if_attack=False)
    # if "claud" in config.reward_url:
    #     from model.Evaluator.claude import ClaudeModel
    #     llm = ClaudeModel()
    # else:
    from model.Evaluator.local import Evaluator
    llm = Evaluator(target_model.model, target_model.tokenizer)
    return llm

def load_data(cfg):
    if "advBench" in cfg.data_path:
        with open(cfg.data_path,"r") as f:
            data = json.load(f)
            
    data_list = []
    for prompt in data:
        data_list.append(prompt["prompt"])
        
    return data_list

def collate(batch):
    prompts = [b["instruction"] for b in batch]
    responses = [b["output"] for b in batch]
    rewards = [b["reward"] for b in batch]
    return prompts, responses, rewards

if __name__ == "__main__":

    
   
    config, target_cfg = get_config("ppo_config.yaml")
    
    model_args, data_args, training_args, generate_cfg  = config
    
    seed_torch(training_args.seed)
    
    model, tokenizer = model_init(model_args)
    
    attacker = AttackAgent(model_args, model = model, tokenizer = tokenizer)
    target = TargetModel(target_cfg)
    evaluator = get_evaluator(target)
    
    
    # print(generate_cfg)
    # exit()
    generator = MCSTreeGenerator(generate_cfg, attacker, target, evaluator)

    
    prompt_seeds = load_data(data_args)
    for prompt in range(generate_cfg.rollout_num):
        shuffled_seeds = random.sample(prompt_seeds, 20)

        root_list = []
        for seed in shuffled_seeds:
            root = MCTSNode("root",None, seed, None, 0, None, None, None)
            root_list.append(root)
        
        # === Step 1: Rollout Tree ===
        # generator.build_tree_to_depth(root_list)

 


        # save_trees(root_list, generate_cfg)

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
        # === Step 2: Evaluate Reward ===
        # generator.compute_tree_reward(root_list)

        
        # === Step 3: Collect training samples ===
        dataset = PPODataset(data_args, root_list, tokenizer)
        print(f"[INFO] Dataset size: {len(dataset)}")
        print(f"[INFO] Example data point: {dataset[0]}")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)
        # === Step 4: PPO update ===
        
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=4,
        )

        # 3. PPO trainer
        ppo_trainer = PPOTrainer(config, model, tokenizer)
        
        for prompts, responses, rewards in dataloader:
            queries = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
            responses = tokenizer(responses, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
            rewards = torch.tensor(rewards).cuda()

            ppo_trainer.step([queries], [responses], [rewards])