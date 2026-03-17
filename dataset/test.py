import sys
sys.path.append("./")
from dataset.sftDataset import SupervisedDataset

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
class data_args:
    data_path = "data/processed/Qwen8B-wildjailbreak-llama-8B"

train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)

lengths = []
max_len = 0
for data in train_dataset:
    # print(tokenizer(data['input_ids'], return_tensors="pt").input_ids[0])
    new_len = len(tokenizer(data['input_ids'], return_tensors="pt").input_ids[0])+len(tokenizer(data['labels'], return_tensors="pt").input_ids[0])
    if max_len < new_len:
        max_len = new_len
        print(max_len)