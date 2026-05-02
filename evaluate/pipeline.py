import yaml
import json
import argparse
import os
from types import SimpleNamespace

from model.Attacker import AttackAgent
from model.Target.target_model import TargetModel
from model.Evaluator import Evaluator, FakeEvaluator
from datagenerator.generate_utils import TreeGenerator, Node


def make_unique_dir(base_dir):
    counter = 1
    new_dir = base_dir
    while os.path.exists(new_dir):
        new_dir = f"{base_dir}-{counter}"
        counter += 1
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate jailbreak agent on HarmBench")
    parser.add_argument('--method', type=str, required=True, help='Attacker model HuggingFace ID, e.g. MartinJYHuang/JA-v2')
    args = parser.parse_args()

    with open("config/target_config.yaml", "r") as f:
        target_cfg = SimpleNamespace(**yaml.safe_load(f))

    with open("config/grpo_generate.yaml", "r") as f:
        generator_cfg = SimpleNamespace(**yaml.safe_load(f))
        generator_cfg.child_num_list = [1, 1, 1, 1, 1]
        generator_cfg.trees_per_batch = 200

    with open("config/attacker_config.yaml", "r") as f:
        attacker_config = SimpleNamespace(**yaml.safe_load(f))
        attacker_config.model = args.method

    attacker = AttackAgent(attacker_config)
    vllm_target = TargetModel(target_cfg)

    if "vicuna" in target_cfg.model:
        evaluator = Evaluator(vllm_target.model, sampling_params=vllm_target.sampling_params, tokenizer=vllm_target.tokenizer, type="vicuna")
    elif "llama" in target_cfg.model:
        evaluator = Evaluator(vllm_target.model, sampling_params=vllm_target.sampling_params, tokenizer=vllm_target.tokenizer, type="llama")
    else:
        evaluator = FakeEvaluator()

    generator = TreeGenerator(generator_cfg, attacker, vllm_target, evaluator)

    with open("data/test.json", "r") as f:
        prompts_data = json.load(f)
    inputs = [p.get("prompt") or p.get("question") for p in prompts_data]

    # Result path: data/result/{org}/{model_name}-{target_name}
    # e.g. data/result/MartinJYHuang/JA-v2-Meta-Llama-3-8B-Instruct
    org = args.method.split('/')[0]
    model_name = args.method.split('/')[1]
    target_name = target_cfg.model.split('/')[-1]
    path = make_unique_dir(f"data/result/{org}/{model_name}-{target_name}")

    root_list = [Node("root", None, seed, None, 0, None, None, None) for seed in inputs]
    for i in range(0, len(root_list), generator_cfg.trees_per_batch):
        sub_root = root_list[i: i + generator_cfg.trees_per_batch]
        generator.build_tree_to_depth(sub_root, radical=True)
        for j, root in enumerate(sub_root):
            root.save_tree(f"{path}/{i + j}.json")
