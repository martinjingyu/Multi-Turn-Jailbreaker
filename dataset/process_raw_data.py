import os
import json
import numpy as np
import hydra
import re

from transformers import AutoTokenizer
from model.Attacker.template import first_prompt_wo_strategy, system_prompt, user_message_template, agent_message_template
from datagenerator.generate_utils import Node
def preprocess(root_node: Node):
    # preprocess the raw data
    # Got the trajectory for each leaf node
    # Return a list of trajectory 
    
    processed_data = []
    success_node_set = set()
    all_nodes = root_node.get_all_nodes()

    for node in all_nodes:
        node: Node
        
        if node.reward is not None and (float(node.reward) > 3):
            # print(node.reward)
            traj = node.get_trajectory()
            for node in traj[1:]:
                success_node_set.add(node)
        
    
    for node in success_node_set:

        node:Node
        
        data_sample = process_and_filter(node)

        if data_sample != None:
            processed_data.append(data_sample)
    
    return processed_data


def process_and_filter(node: Node):
    data = {}
    
    if "<analysis>" not in node.origin_output or "</analysis>" not in node.origin_output or "<action>" not in node.origin_output or "</action>" not in node.origin_output:
        return None
    
    messages = node.get_agent_input_messages()
    
    data["input"] = messages
    response = node.origin_output
    data["response"] = response
    
    return data

@hydra.main(version_base=None, config_path="./../config", config_name="data_process")
def main(cfg):
    print(cfg)
    
    
def test():
    path = "data/raw_data/JA-advBench/rollout64/tree_0.json"
    with open(path, "r") as file:
        raw_data = json.load(file)
    
    preprocess(raw_data)
if __name__ == "__main__":
    test()