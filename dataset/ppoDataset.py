import json
import random
import logging
import transformers
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from model.Attacker.template import first_prompt_wo_strategy, system_prompt, user_message_template, agent_message_template

class PPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, root_list, tokenizer):
        super(PPODataset, self).__init__()
        logging.warning("Loading data...")
        
        data_list = []
        for root in root_list:
            
            node_list = root.get_all_nodes()
            for node in node_list:
                if node.children == []:
                    continue
                for child in node.children:
                    if child.response is not None and child.reasoning is not None:
                        data_list.append({
                            "instruction": tokenizer.apply_chat_template(node.get_agent_input_messages(),
                                                                add_generation_prompt=True,
                                                                    enable_thinking=False,
                                                                    tokenize=False
                                                                ),
                            "output": agent_message_template.format(analysis=child.reasoning, action=child.response),
                            "reward": child.reward
                        })
                
                        

        print(f"Total data loaded: {len(data_list)}")
        
        # exit() # Checkpoint

        sources = []
        for example in tqdm(data_list):
            if example['instruction'] == '':
                sources.append('')
            else:
                sources.append(example["instruction"])
                

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in data_list]

        self.sources = sources
        self.targets = targets
        self.rwards = [float(example["reward"]) for example in data_list]
        
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        
        return dict(input_ids=self.sources[i], labels=self.targets[i], reward=self.rwards[i])
    