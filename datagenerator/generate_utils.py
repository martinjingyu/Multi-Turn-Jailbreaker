"""
Monte Carlo Tree Search (MCTS) implementation for mathematical reasoning
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import json
import random
import time
import copy
import re
from model.Attacker.template import first_prompt_wo_strategy, user_message_template, agent_message_template, system_prompt, first_prompt
from sentence_transformers import SentenceTransformer, util
import asyncio
class Node:
    def __init__(self, 
                 state: str=None, 
                 parent: Optional['Node'] = None, 
                 prompt: Optional[str] = None,
                 response: Optional[str] = None,
                 depth: Optional[int] = None,
                 index: Optional[int] = None,
                 reasoning: Optional[str] = "",
                 strategy: Optional[str] = "",
                 reward: Optional[float]= None,
                 origin_output: Optional[str] = None
        ):
        self.state = state
        self.parent = parent
        self.prompt = prompt
        self.response = response
        self.children = []
        self.reward = reward
        self.index = index
        self.depth = depth
        self.reasoning = reasoning
        self.strategy = strategy
        self.origin_output = origin_output
        self.data_for_training = {}
        
    def add_child(self, child) -> 'Node':
        """Add a child node with the given action and state."""
        self.children.append(child)

    def get_all_nodes(self) -> List['Node']:

        cache_list = []
        cache_list.extend(self.children)
        nodes = []
        nodes.append(self)
        while cache_list != []:
            node = cache_list.pop(0)
            nodes.append(node)
            cache_list.extend(node.children)
        return nodes

    def get_all_leaf_nodes(self) -> List['Node']:

        cache_list = []
        leaf_nodes = []
        if self.children == []:
            leaf_nodes.append(self)
            return leaf_nodes
        
        cache_list.extend(self.children)
        while cache_list != []:
            node = cache_list.pop(0)
            if node.children == []:
                leaf_nodes.append(node)
            cache_list.extend(node.children)
        return leaf_nodes
    
    def get_trajectory(self) -> List['Node']:
        trajectory = []
        node = self
        while node is not None:
            trajectory.append(node)
            node = node.parent
        trajectory.reverse()
        return trajectory
    
    def get_agent_input_messages(self) -> List[Dict[str, str]]:
        messages = []
        trajctory = self.get_trajectory()
        messages.append({"role": "system", "content":system_prompt})
        
        if self.strategy != None:
            messages.append({"role": "user", "content": first_prompt.format(seed_prompt=trajctory[0].prompt,strategy_description = self.strategy)})
            
        else :
            messages.append({"role": "user", "content": first_prompt_wo_strategy.format(seed_prompt=trajctory[0].prompt)})
            
        for node in trajctory[1:-1]:
            messages.append({"role": "assistant", "content": agent_message_template.format(analysis=node.reasoning, action=node.prompt)})
            
            messages.append({"role": "user", "content": user_message_template.format(environment=node.response, score=node.reward)})
            
        return messages
    
    def get_target_input_messages(self) -> List[Dict[str, str]]:
        messages = []
        trajectory = self.get_trajectory()
        
        messages.append({"role": "system", "content":"You are a helpful assistant."})
        for node in trajectory[1:-1]:
            messages.append({"role": "user", "content": node.prompt})
            messages.append({"role": "assistant", "content": node.response})
            
        messages.append({"role": "user", "content": trajectory[-1].prompt})
        
        return messages

    def load_from_json(dict_list:List[Dict[str, Any]]):
        node_list = []
        for item in dict_list:
            node = Node(
                item["state"],
                item["parent"],
                item['attacker'],
                item['target'],
                item['depth'],
                item['id'],
                item['reasoning'],
                None,
                item['reward'] if item['reward'] is not None else None,
                item['origin']
            )
            node_list.append(node)
        for item, node in zip(dict_list, node_list):
            if item["parent"] is not None:
                node.parent = node_list[item["parent"]]
                node.parent.add_child(node)
        return node_list[0]
    
    def save_tree(self, path):
        
        total_nodes = []
        cache_list =[self]
        
        while cache_list: 
            current_node = cache_list.pop(0) 
            total_nodes.append(current_node)
            if current_node.children:
                cache_list.extend(current_node.children)
        

        index = 0
        for node in total_nodes:
            node.index = index
            index += 1
    
        with open(path, "w") as f:
            outputs = []
            for node in total_nodes:
                node:Node
                datapoint = {}
                datapoint["id"] = node.index
                datapoint["state"] = node.state
                datapoint["depth"] = node.depth
                # handle both dict and string cases
                if node.depth == 1:
                    if isinstance(node.strategy, dict):
                        datapoint["strategy"] = node.strategy.get("strategy", None)
                    elif isinstance(node.strategy, str):
                        datapoint["strategy"] = node.strategy     
                    else:
                        datapoint["strategy"] = node.strategy            
                
                if node.reward != None:
                    try:
                        datapoint["reward"] = str(float(node.reward))
                    except:
                        datapoint["reward"] = node.reward
                else:
                    datapoint["reward"] = None
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

class TreeGenerator():
    def __init__(self, cfg, agent, target, evaluator):
        self.cfg = cfg
        self.attacker = agent
        self.target = target
        self.evaluator = evaluator
        self.sentence_embedding = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    def build_tree_to_depth(self, root_nodes: list[Node], **cfg):
        if cfg.get("radical") is not None:
            self.radical = True
        current_nodes = root_nodes

        for child_num in tqdm(self.cfg.child_num_list,desc="Trees Growing"):

            cache_list = [node for node in current_nodes if node.state != "leaf"]
            
            if not cache_list:
                break 

            child_node_list = self.expand(cache_list)
            
            current_nodes = child_node_list
            

    
    def expand(self, node_list:list[Node]):
        
        expand_node_list =[]
        child_node_list=[]
        for node in node_list:
            # print(node.depth)
            expand_node_list.extend([node]*self.cfg.child_num_list[node.depth])
            
        for node in expand_node_list:
            child_node = Node()
            child_node_list.append(child_node)
            node.add_child(child_node)
            child_node.parent = node
        
        # Assign strategy
        if self.cfg.use_strategy == True and node_list[0].depth == 0:
            strategy_list = random.choices(self.attacker.jailbreak_strategy_libray, k=len(expand_node_list))

            for node, strategy in zip(child_node_list, strategy_list):
                node.strategy = strategy
        elif self.cfg.use_strategy == True:
            for node in child_node_list:
                node.strategy = node.parent.strategy
        else:
            pass
            
        self.pipeline(child_node_list)
        
        return child_node_list
        
        
            
    
    
    def pipeline(self, node_list: list[Node]):


        # for i in tqdm(range(0, len(node_list), self.cfg.batch_size), desc="Processing batches"):
        #     sub_node_list = node_list[i:i+self.cfg.batch_size]
            
        for i in tqdm(range(0, len(node_list), len(node_list)), desc="Processing batches"):
            sub_node_list = node_list
            attacker_messages_list = []
            for node in sub_node_list:
                attacker_messages_list.append(node.get_agent_input_messages())
            # messages_list = self.get_messages_list(sub_node_list, self.cfg.whether_sft)
            # attacker_inputs = self.get_attacker_input(messages_list)
            outputs = self.attacker.batch_infer(attacker_messages_list)

            for output, node in zip(outputs, sub_node_list):
                node: Node
                reasoning, action = self.attacker.extract_output_1(output)
                node.prompt = action
                node.reasoning =reasoning
                
                node.origin_output = output
                
                
            target_messages_list=[]
            for node in sub_node_list:
                target_messages_list.append(node.get_target_input_messages())
            if self.target.config.model in ["Openai/gpt","Google/gemini","Amazon/claude"]:
                gemini_messages_list = []
                for messages_list in target_messages_list:
                    gemini_messages_list.append(messages_list[1:])
                    
                target_responses = asyncio.run(self.target.batch_response_close(gemini_messages_list))
            else:
                target_responses = self.target.batch_response(target_messages_list)
            
            
            # else:
            scores = self.evaluator.eval_batch(target_responses, target_messages_list)
            
            # print(scores)
            for node, response, score in zip(sub_node_list, target_responses, scores):
                node.depth = node.parent.depth+1
                node.response = response
                node.reward = score
                if hasattr(self, "radical") and node.depth != None:
                    if node.depth >= 3:
                        node.reward = max(float(node.reward), 3)
                    if node.depth >= 4:
                        node.reward = max(float(node.reward), 4)
               
                if score == "None":
                    state = "internal"
                elif float(score) == 5.0:
                    state = "leaf"
                elif node.depth == len(self.cfg.child_num_list):
                    state = "leaf"
                else:
                    state = "internal"
                node.state = state
                
        return node_list

    def pruning(self, root_list: list[Node]):
        for root in root_list:
            root.reward = "0"
            node_list = root.get_all_nodes()
            found = False
            max_node = root
            
            for node in node_list:
                if float(node.reward) > 3:
                    found = True
                    break
                else:
                    if float(node.reward) > float(max_node.reward):
                        max_node = node
                        
            if found:
                for node in node_list[1:]:
                    subtree = node.get_all_nodes()
                    if not any(float(n.reward) > 3 for n in subtree):
                        node.children = []
            else:
                for node in node_list[1:]:
                    subtree = node.get_all_nodes()
                    if max_node not in subtree:
                        node.children = []
                
                            
    def prune_all_children(self,node):
        for child in node.children:
            self.prune_all_children(child)
        node.children = []
    
    def compute_tree_reward(self, root_list: list[Node]):
        for root in root_list:
            node_list = root.get_all_nodes()
            for node in node_list[1:]:
                if node.children == []:
                    continue
                # Success_reward: fraction of subtree (including self) with raw reward >= 3
                # nodes[0] == node itself, so self is already included
                nodes = node.get_all_nodes()
                suc_reward = sum(1 for sub_node in nodes if float(sub_node.reward) >= 3)
                suc_reward = suc_reward / len(nodes)

                
                # Diversity_reward
                div_reward = self.compute_diversity_reward(node)
                
                # Format reward
                answer = node.origin_output
                
                if answer is None:
                    format_reward = 0
                else:
                    pattern = r"^<analysis>.*?</analysis>[\n ]*<action>.*?</action>[\n ]*$"
                    think_count = answer.count("<analysis>") + answer.count("</analysis>")
                    answer_count = answer.count("<action>") + answer.count("</action>")
                    if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2:
                        format_reward = 1
                    else:
                        format_reward = 0
                
                    
                # # Encourage Hitting
                # credit = 0
                # if node.depth >=3 and "[Hitting Mode]" in answer:
                #     credit = 1
                    
                # Get total reward
                total_reward =  self.cfg.success_weight * suc_reward + self.cfg.diversity_weight * div_reward + self.cfg.format_weight * format_reward
                # total_reward = format_reward

                node.reward = total_reward
            
            # Normalization    
            
            for node in node_list:
                if node.children == []:
                    continue
                reward_list = []
                for child in node.children:
                    reward_list.append(float(child.reward))
                reward_array = np.array(reward_list, dtype=np.float32)
                normalized_rewards = (reward_array - reward_array.mean()) / (reward_array.std() + 1e-4)
                for child, norm_r in zip(node.children, normalized_rewards):
                    child.reward = norm_r
                    

    def compute_diversity_reward(self, node: Node):
        brother_nodes = []
        if node.parent:
            for brother in node.parent.children:
                if brother != node:
                    brother_nodes.append(brother)
        if brother_nodes == []:
            return 0
        else:
            emb_text = self.sentence_embedding.encode(node.prompt, normalize_embeddings=True)
            emb_refs = self.sentence_embedding.encode([brother.prompt for brother in brother_nodes], normalize_embeddings=True)
            
            sims = [util.cos_sim(emb_text, e)[0][0].item() for e in emb_refs]
            avg_sim = sum(sims) / len(sims)
            
            diversity = 1 - avg_sim

            return diversity