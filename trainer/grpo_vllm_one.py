from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
from glob import glob
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm import tqdm

from trainer.ppo_utils import load_seeds, save_trees, upload_model_folder
from trainer.gen_worker import gen_worker
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# model_path = "MartinJYHuang/Jailbreak-agent-temp"
model_path =  "MartinJYHuang/JA-v1"
# model_path = "tmp/"
gen_device = 1    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.03
max_save_total = 5
save_dir_base = "checkpoints/"
all_steps = 500
train_batch_size = 1
gen_update_steps = 200
save_steps = 400
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"


from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 8,
    # "optimizer": {
    #     "type": "DeepSpeedCPUAdam",
    #     "params": { "lr": 1e-6 }
    # },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}
def find_latest_checkpoint(base_dir="."):
            checkpoints = glob(os.path.join(base_dir, "step_*"))
            if not checkpoints:
                raise ValueError("No checkpoint directories found.")
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
            return checkpoints[-1]  
        
def get_batch():
    try:
        r = requests.get(f"{ref_server}/get",proxies={"http": None, "https": None}).content

        if r == b'empty': return None
    except Exception as e: 
        print("Exception when getting batch:", e)
        return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def GRPO_step(batch):
    prompt_length = batch['plen']
    
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    
    # print(inputs.shape)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss





tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()
    start_1=0
    end_1=200
    
    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn', force=True)
        Q = mp.Queue(maxsize=1)
        p = mp.Process(target=gen_worker, args=(Q, 1, start_1, end_1))
        
        p.start()


    model = AutoModelForCausalLM.from_pretrained(model_path, 
            dtype=torch.bfloat16, _attn_implementation="sdpa")

    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-6)
    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model,
                optimizer=optimizer,
            model_parameters=model.parameters())


    

    progress = range(0, all_steps)
    if dist.get_rank() == 0: progress = tqdm(progress)
    
    for step in progress:
        
        batch = get_batch()
        
        while batch is None:
            print('waiting for batch...'); time.sleep(60)
            batch = get_batch()
        while batch is not None:
        # print("start training step")
        # print("inputs:", batch['inputs'].shape)
        # print("rewards:", batch['rewards'].shape)
        # print("refs:", batch['refs'].shape)
        # if 'gen_logps' in batch:
        #     print("gen_logps:", batch['gen_logps'].shape)
            loss = GRPO_step(batch)
            engine.backward(loss)
            engine.step()
            batch = get_batch()
            
        dist.barrier()
        if dist.get_rank() == 0:
            while(True):
                print("checking trainer weight")
                with open("generator.txt", "r") as file:
                    content = file.read().strip()
                    print(f"[Debug] File content: '{content}'")  
                    print(f"[Debug] Step: '{step}'")  
                if content == str(step):
                    break
                else:
                    time.sleep(30)
                
            state_dict = engine.module.state_dict()
            os.makedirs("tmp/", exist_ok=True)
            engine.module.save_pretrained("tmp/", state_dict=state_dict)
            tokenizer.save_pretrained("tmp/")
            with open("trainer.txt", "w") as file:
                file.write(f"{step+1}")
        dist.barrier()
        
        
        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")



        dist.barrier()
        if dist.get_rank() == 0:
            print(f'saving model at step {step}')
            save_name = os.path.join(save_dir_base, f"step_{step}")


            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)

            all_checkpoints = sorted(
                glob(os.path.join(save_dir_base, "step_*")),
                key=os.path.getmtime
            )
            if len(all_checkpoints) > max_save_total:
                checkpoints_to_delete = all_checkpoints[:len(all_checkpoints) - max_save_total]
                for ckpt in checkpoints_to_delete:
                    print(f"Removing old checkpoint: {ckpt}")
                    shutil.rmtree(ckpt)
        dist.barrier()
            
            
        # Upload Model
        dist.barrier()
        if step%4 == 0:
            if dist.get_rank() == 0:
                latest_model_path = find_latest_checkpoint(save_dir_base)
                upload_model_folder(latest_model_path,"MartinJYHuang/JA-v2")
        dist.barrier()

