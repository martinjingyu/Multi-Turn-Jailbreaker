import torch
import numpy as np
import random
import transformers

from transformers import Trainer, AutoTokenizer
from dataset.sftDataset import SupervisedDataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from utils import preprocess, IGNORE_INDEX, get_config, seed_torch, safe_save_model_for_hf_trainer

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        sources = []
        targets = []

        for instance in instances:
            

            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    
    config = get_config("sft_config.yaml")
    
    model_args, data_args, training_args = config
    print(model_args)
    print(data_args)
    print(training_args)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    training_args.seed = seed
    training_args.learning_rate = float(training_args.learning_rate)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    
    def model_init():
        import os
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.gradient_checkpointing_enable()
        print("Model hidden size:", model.config.hidden_size)
        print("Model num params:", sum(p.numel() for p in model.parameters()) / 1e9, "B parameters")

        return model

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    
    
    trainer = Trainer(model_init=model_init, tokenizer=tokenizer, args=training_args, **data_module)
    learning_rate=(training_args.learning_rate)
    print(f"Learning rate: {learning_rate} ({type(learning_rate)})")
    trainer.train()  # resume
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    
    
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
        

if __name__ == "__main__":
    data = None
    train()