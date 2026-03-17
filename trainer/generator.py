
import yaml
import json
import os 
from collections import deque
from tqdm import tqdm
import argparse
from types import SimpleNamespace

from model.Attacker import AttackAgent
from model.Target.target_model import TargetModel

from model.Attacker.template import first_prompt_wo_strategy, user_message_template, agent_message_template, system_prompt, target_system_prompt
from model.Attacker import AttackAgent
from model.Target import TargetModel
from model.Evaluator.local import Evaluator
from datagenerator.mcts_utils import MCTSNode, MCTS

class MCSTreeGenerator():
    def __init__(self, cfg, attacker, target, evaluator):
        self.cfg = cfg
        self.attacker = attacker
        self.target = target
        self.evaluator = evaluator
        
    
    
    # Given a list of root nodes, build the MCTS tree to the specified depth
    def build_tree_to_depth(self, root_nodes: list[MCTSNode], **cfg):

        current_nodes = root_nodes

        for depth in tqdm(self.cfg.depth_list,desc="Generate Tree"):

            cache_list = [node for node in current_nodes if node.state != "leaf"]
            if not cache_list:
                break 

            self.expand(cache_list) 

            next_nodes = []
            for node in cache_list:
                next_nodes.extend(node.children)
            current_nodes = next_nodes
    
    def expand(self, node_list:list[MCTSNode]):
        node_list = node_list * self.cfg.child_num
        outputs = self.pipeline(node_list)
        
        for node, output in zip(node_list, outputs):
            if float(output["score"]) == 5.0:
                state = "leaf"
            elif node.depth == self.cfg.max_depth:
                state = "leaf"
            else:
                state = "internal"
            
            child_node = MCTSNode(state, node, output["prompt"], output["response"],node.depth+1,None,output["reasoning"], None, reward = output["score"]if output["score"] else None, origin_output= output["origin_output"])
            node.add_child(child_node)
            
            
    # def get_inputs_from_nodes(node_list):
    #     inputs_list = []
    #     for node in node_list:
            
    #     inputs_list = [node.prompt for node in node_list]
        
    #     return inputs_list
    
    
    def pipeline(self, node_list):

        
        outputs_list = []

        for i in tqdm(range(0, len(node_list), self.cfg.generate_batch_size),desc="Processing batches"):
            
            
            
            # batch = []
            # for node in node_list[i:i + self.cfg.generate_batch_size]:
            #     batch.append(self.get_traj(node))
                
            # batch = [[{"role": "system", "content": system_prompt},{"role":"user","content":first_prompt_wo_strategy.format(seed_prompt=item)}] for item in batch]

            messages_list = self.get_messages_list(node_list)
            

            attacker_inputs = self.get_attacker_input(messages_list)
            outputs = self.attacker.batch_infer(attacker_inputs)
            # print(11111)
            # print(outputs)
            # exit()
            
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
    
    def compute_tree_reward(self, root_list: list[MCTSNode]):
        for root in root_list:
            node_list = []
            cache_list = deque()
            cache_list.extend(root.children)
            while cache_list:
                current_node = cache_list.popleft()
                node_list.append(current_node)
                if current_node.children:
                    cache_list.extend(current_node.children)
            for node in node_list:
                reward = 0
                leaf_num = 0
                cache_list = deque()
                cache_list.append(node)
                while cache_list:
                    current_node = cache_list.popleft()
                    if current_node.children:
                        cache_list.extend(current_node.children)
                    else:
                        reward += current_node.reward
                        leaf_num += 1
                if leaf_num > 0:
                    node.reward = reward / (len(node.children) if node.children else 1)
            
            cache_list.append(root)
            while cache_list:
                current_node = cache_list.popleft()
                if current_node.children:
                    for child in current_node.children:
                        cache_list.append(child)
                    current_node.reward = sum([child.reward for child in current_node.children]) / len(current_node.children)
        
        