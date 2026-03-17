
import yaml
import json
import os, re
from collections import deque
from tqdm import tqdm
import argparse
from types import SimpleNamespace
import torch.nn.functional as F
from model.Attacker import AttackAgent
from model.Target.target_model import TargetModel
from sentence_transformers import SentenceTransformer, util
from model.Attacker.template import first_prompt_wo_strategy, user_message_template, agent_message_template, system_prompt, target_system_prompt
from model.Attacker import AttackAgent
from model.Target import TargetModel
from model.Evaluator.local import Evaluator
from datagenerator.generate_utils import Node

class MCSTreeGenerator():
    def __init__(self, cfg, agent, target, evaluator):
        self.cfg = cfg
        self.attacker = agent
        self.target = target
        self.evaluator = evaluator
        self.sentence_embedding = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    def build_tree_to_depth(self, root_nodes: list[Node], **cfg):

        current_nodes = root_nodes

        for child_num in tqdm(self.cfg.child_num_list,desc="Generate Tree"):

            cache_list = [node for node in current_nodes if node.state != "leaf"]
            if not cache_list:
                break 

            self.expand(cache_list) 

            next_nodes = []
            for node in cache_list:
                next_nodes.extend(node.children)
            current_nodes = next_nodes
    
    def expand(self, node_list:list[Node]):
        expand_node_list =[]
        for node in node_list:
            expand_node_list.extend([node]*self.cfg.child_num_list[node.depth])

        outputs = self.pipeline(expand_node_list)
        
        for node, output in zip(expand_node_list, outputs):
            if float(output["score"]) == 5.0:
                state = "leaf"
            elif node.depth == len(self.cfg.child_num_list)-1:
                state = "leaf"
            else:
                state = "internal"
            
            child_node = Node(state, node, output["prompt"], output["response"],node.depth+1,None,output["reasoning"], None, reward = output["score"]if output["score"] else None, origin_output= output["origin_output"])
            node.add_child(child_node)
            
    
    
    def pipeline(self, node_list):

        
        outputs_list = []

        for i in tqdm(range(0, len(node_list), self.cfg.generate_batch_size),desc="Processing batches"):
            
            
            messages_list = self.get_messages_list(node_list)
            
            attacker_inputs = self.get_attacker_input(messages_list)
            outputs = self.attacker.batch_infer(attacker_inputs)

            
            actions = []
            reasonings = []
            for output in outputs:
                
                reasoning, action = self.attacker.extract_output_1(output)
                actions.append(action)
                reasonings.append(reasoning)
            
            target_input = self.get_target_input(messages_list, actions)
            target_responses = self.target.batch_response(target_input)
            
          
            scores = self.evaluator.eval_batch(target_responses)
            
            
            for k in range(len(node_list)):
                outputs_list.append({"prompt": actions[k],"reasoning":reasonings[k],"response":target_responses[k],"score":scores[k], "origin_output": outputs[k]})
            
        return outputs_list
    
    
    def get_messages_list(self, node_list):
        messages_list = []
        for node in node_list:
            traj = []
            while node != None:
                traj.append(node)
                node = node.parent
            traj.reverse()
            messages = []
            messages.append({"role":"system","content": system_prompt})
            messages.append({"role":"user","content": first_prompt_wo_strategy.format(seed_prompt = traj[0].prompt)})
            for internal_node in traj[1:]:
                messages.append({"role":"assistant", "reasoning":internal_node.reasoning, "action": internal_node.prompt})
                messages.append({"role":"user", "resposne":internal_node.response, "score": internal_node.reward})
            messages_list.append(messages)
            
        return messages_list
    
    def get_attacker_input(self, messages_list):
        
        attacker_inputs = []
        
        for messages in messages_list:
            attacker_input = []
            attacker_input.append(messages[0])
            attacker_input.append(messages[1])
            for message in messages[2:]:
                if message["role"] == "assistant":
                    attacker_input.append({"role":message["role"],"content":agent_message_template.format(analysis=message["reasoning"], action=message["action"])})
                else:
                    attacker_input.append({"role":message["role"],"content":user_message_template.format(environment=message["resposne"], score=message["score"])})
            
            
        
            attacker_inputs.append(attacker_input)
        return attacker_inputs
    
    def get_target_input(self, messages_list, actions):

        processed_batch = []
        for i, messages in enumerate(messages_list):
            list = [{"role": "system", "content": target_system_prompt}]
            for message in messages[2:]:
                if message["role"] == "assistant":
                    list.append({"role": "user","content":message["action"]})
                if message["role"] == "user":
                    list.append({"role": "assistant","content":message["resposne"]})
            list.append({"role": "user","content":actions[i]})
            processed_batch.append(list)
        return processed_batch
        
    
    def process_attacker_batch(self, batch):

        processed_batch = []
        for messages in batch:
            list = [{"role":message["role"],"content":message["content"]} for message in messages[:2]]
            for message in messages[2:]:
                if message["role"] == "assistant":
                    list.append({"role":message["role"],"content":agent_message_template.format(analysis=message["reasoning"], action=message["content"])})
                else:
                    list.append({"role":message["role"],"content":user_message_template.format(environment=message["content"], score=message["score"])})
            
            processed_batch.append(list)
            
        return processed_batch

    def precess_target_batch(self, batch, actions):

        from model.Attacker.template import user_message_template
        processed_batch = []
        for i, messages in enumerate(batch):
            list = [{"role": "system", "content": self.target_model.system_prompt}]
            for message in messages[2:]:
                if message["role"] == "assistant":
                    list.append({"role": "user","content":message["content"]})
                if message["role"] == "user":
                    list.append({"role": "assistant","content":message["content"]})
            list.append({"role": "user","content":actions[i]})
            processed_batch.append(list)
        return processed_batch
    
    def compute_tree_reward(self, root_list: list[Node]):
        for root in root_list:
            node_list = root.get_all_nodes()
            for node in node_list:
                
                
                if node.children == []:
                    continue
                # Success_reward
                leaf_nodes = node.get_all_leaf_nodes()
                rewards = []
                for leaf in leaf_nodes:
                    reward = 5 if leaf.reward == "none" else float(leaf.reward)
                    rewards.append(reward)
                reward = sum(rewards)
                
                suc_reward = reward / len(leaf_nodes)
                
                # Diversity_reward
                div_reward = self.compute_diversity_reward(node)
                
                # Format reward
                answer = node.origin_output
                
                if answer is None:
                    format_reward = -1
                else:
                    pattern = r"^<analysis>.*?</analysis>[\n ]*<action>.*?</action>[\n ]*$"
                    think_count = answer.count("<analysis>") + answer.count("</analysis>")
                    answer_count = answer.count("<action>") + answer.count("</action>")
                    if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2:
                        format_reward = 1
                    
                    else:
                        format_reward = -1
                
                # Get total reward
                
                # suc_reward = 0
                # div_reward = 0
                # format_reward = 0
                
                
                
                total_reward = self.cfg.success_weight * suc_reward + self.cfg.diversity_weight * div_reward + self.cfg.format_weight * format_reward
                # total_reward = format_reward
                node.reward = total_reward


    def compute_diversity_reward(self, node:Node):
        brother_nodes = []
        if node.parent:
            for brother in node.parent.children:
                if brother != node:
                    brother_nodes.append(brother)
        if brother_nodes == []:
            return 0
        else:
            emb_text = self.sentence_embedding.encode(node.origin_output, normalize_embeddings=True)
            emb_refs = self.sentence_embedding.encode([brother.origin_output for brother in brother_nodes], normalize_embeddings=True)
            
            sims = [util.cos_sim(emb_text, e)[0][0].item() for e in emb_refs]
            avg_sim = sum(sims) / len(sims)
            
            diversity = 1 - avg_sim

            return diversity
def load_seeds():
    path = "data/train_attack_target.json"
    with open(path,"r") as f:
        data = json.load(f)
            
    data_list = []
    for prompt in data:
        data_list.append(prompt["goal"])
    
    import random    
    random.shuffle(data_list)
    return data_list

def save_trees(root_list: list[Node], generate_cfg, physics_device):
    
    path = os.path.join(generate_cfg.output_path,str(physics_device))
    os.makedirs(path, exist_ok=True)
    existing_dirs = [
        int(name) for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name)) and name.isdigit()
        ]
    next_index = max(existing_dirs) + 1 if existing_dirs else 0
    next_output_dir = os.path.join(path, str(next_index))
    
    os.makedirs(next_output_dir, exist_ok=True)
    for i, root in enumerate(root_list):
        total_nodes = []

        cache_list =[root]
        while cache_list: 
            current_node = cache_list.pop(0) 
            total_nodes.append(current_node)
            if current_node.children:
                cache_list.extend(current_node.children)
        

        index = 0
        for node in total_nodes:
            node.index = index
            index += 1
        

        json_path = os.path.join(next_output_dir, f"{i}.json")
        with open(json_path, "w") as f:
            outputs = []
            for node in total_nodes:
                datapoint = {}
                datapoint["id"] = node.index
                datapoint["value"] = node.value
                datapoint["visits"] = node.visits
                datapoint["reward"] = node.reward
                datapoint["attacker"] = node.prompt
                datapoint["target"] = node.response
                datapoint["parent"] = node.parent.index if node.parent != None else None
                datapoint["origin"] = node.origin_output
                children = []
                for child in node.children:
                    subnode={}
                    subnode["id"] = child.index
                    children.append(subnode)
                datapoint["children"] =children 
                datapoint["reasoning"] = node.reasoning
                outputs.append(datapoint)
                
            json.dump(outputs, f, indent=4)

def upload_model_folder(model_path, repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        path_in_repo=".",   
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*"]
    )
    
    
if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description="Evaluate pipeline")
    # parser.add_argument('--method', type=str, required=True, help='Evaluation method to use')
    # args = parser.parse_args()
    # with open("config/evaluation/base.yaml", "r") as f:
        
    #     cfg = SimpleNamespace(**yaml.safe_load(f)) 
    from glob import glob
    def find_latest_checkpoint(base_dir="."):
        checkpoints = glob(os.path.join(base_dir, "step_*"))
        if not checkpoints:
            raise ValueError("No checkpoint directories found.")
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
        return checkpoints[-1]  
    save_dir_base = "."
    latest_model_path = find_latest_checkpoint(save_dir_base)
    upload_model_folder(latest_model_path,"MartinJYHuang/jailbreakAgent-grpo")
        
    