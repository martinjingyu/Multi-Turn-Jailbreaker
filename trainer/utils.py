from typing import Dict, Sequence, Optional
from types import SimpleNamespace
from dataclasses import dataclass, field
import transformers
import copy
import random
import torch
import os
import yaml
IGNORE_INDEX = -100
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)

    input_ids = examples_tokenized["input_ids"]
    
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    return dict(input_ids=input_ids, labels=labels)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [truncate(tokenized.input_ids[0],8192) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
def truncate(tensor, max_length):
    if tensor.size(0) > max_length:
        return tensor[:max_length]
    return tensor

@dataclass 
class ModelArguments:
    model_name: Optional[str] = field(default="Qwen/Qwen3-8B")
    batch_size: Optional[int] = field(default=20)
    temperature: Optional[float] = 0.9
    max_new_tokens: int = 4096
    top_p: float = 0.8
    top_k: int = 20
    do_sample: bool = True
    half_precision: bool = True
    dtype: str ="bfloat16"

@dataclass
class DataArguments:
    data_path: str = field(default="data/processed/test")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    optim: str = "adamw_torch"
    deepspeed: Optional[str] = None
    model_max_length: int = 8192
    per_device_train_batch_size: int = 1
    overwrite_output_dir: bool = True
    bf16: bool = True
    
    output_dir: str = "./checkpoints" 
    num_train_epochs: int = 3  
    gradient_accumulation_steps: int = 8 
    save_strategy: str = "steps" 
    save_steps: int = 500  
    save_total_limit: int = 5
    evaluation_strategy: str = "no" 
    logging_steps: int = 50
    logging_dir: str = "./logs"
    report_to: str = "tensorboard"  

    seed: int = 42  
    learning_rate: float = 5e-5
    disable_tqdm: bool = False

@dataclass
class PPOTrainingArguments():
    # 1. Generate enough data for one ppo training （1 rollout） 
    #  
    
    # 
    optim: str = "adamw_torch"
    deepspeed: Optional[str] = None
    
    per_device_train_batch_size: int = 1
    overwrite_output_dir: bool = True
    bf16: bool = True
    
    output_dir: str = "./checkpoints_ppo" 
    num_train_epochs: int = 3  
    gradient_accumulation_steps: int = 8 
    save_strategy: str = "steps" 
    save_steps: int = 500  
    save_total_limit: int = 5
    evaluation_strategy: str = "no" 
    logging_steps: int = 50
    logging_dir: str = "./logs"
    report_to: str = "tensorboard"  

    seed: int = 42  
    learning_rate: float = 5e-5
    num_tree_per_rollout: int = 10    
    ppo_batch_size: int = 20               
    clip_range: float= 0.2                      
    kl_penalty_beta: float = 0.1      

@dataclass
class GenerateArguments():
    seed_prompts_file: str = None
    task_name: str = None
    output_path: str = None
    max_depth: int = 5
    child_num: int = 3
    generate_batch_size: int = 20
    rollout_num : int = 20

    def __post_init__(self):

        if self.output_path:
            self.output_path = self.output_path.format(
                task_name=self.task_name,
                max_depth=self.max_depth
            )

def get_config(name):
    if "ppo" in name:
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, PPOTrainingArguments, GenerateArguments))
        config = parser.parse_yaml_file(f"config/{name}")
        with open("config/target_config.yaml", "r") as f:
            target_cfg = SimpleNamespace(**yaml.safe_load(f))
        
        return config,target_cfg

    else:
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        config = parser.parse_yaml_file(f"config/{name}")

    return config

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)
    # torch.use_deterministic_algorithms(True)
    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        
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