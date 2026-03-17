import json
import random
import logging
import transformers
import io
import os
from tqdm import tqdm
from datagenerator.generate_utils import Node
from torch.utils.data import Dataset
from .process_raw_data import preprocess
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_args.data_path
        
        list_data_dict = []
        data_list = []
        for file in os.listdir(data_path):
            if file.endswith(".json") == False:
                continue
            print(f"Loading file: {file}")
            path = os.path.join(data_path, file)
            
            with open(path, "r") as file:
                data = json.load(file)
                root_node:Node = Node.load_from_json(data)
                

                data = preprocess(root_node)
                print(f"Data after preprocess: {len(data)}")
                for i, item in enumerate(data):
                    messages = item["input"]
                    data[i]["input"] = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        tokenize=False
                    )
                data_list.extend(data)

        # print(data_list[0]["input"])
        print(f"Total data loaded: {len(data_list)}")
        
        # exit() # Checkpoint
        list_data_dict = random.sample(data_list,  len(data_list))

        list_data_dict = [{'instruction':data['input'], 'output':data['response']} for data in list_data_dict]
        
        sources = []
        
        for example in tqdm(list_data_dict):
            sources.append(example["instruction"])
                

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets
        
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        
        return dict(input_ids=self.sources[i], labels=self.targets[i])
    