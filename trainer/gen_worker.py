import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, yaml
import traceback
import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
import contextlib
import gc
import math
import torch
from vllm.distributed import (destroy_distributed_environment,
                            destroy_model_parallel)
from vllm import LLM
from transformers import AutoTokenizer
from datagenerator.generate_utils import Node, TreeGenerator

from types import SimpleNamespace
from model.Evaluator.local import Evaluator
from model.Evaluator.llamaJedge import LlamaGuardModeration
from model.Target.target_model import TargetModel
from model.Attacker.attack_agent import AttackAgent
from vllm.distributed.parallel_state import destroy_model_parallel
import torch.distributed as dist

from vllm import SamplingParams
model_path = "MartinJYHuang/Jailbreak-agent-temp"
gen_device = 1    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.04
max_save_total = 10
save_dir_base = "checkpoint/"
train_batch_size = 1
gen_update_steps = 16

max_token_len = 6144

compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

from trainer.ppo_utils import load_seeds, save_trees


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

def load_data():
    from pathlib import Path
    import json
    data_dir = Path("ppo_data/cache")
    json_files = list(data_dir.glob("*.json"))

    root_list = []
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            root_node = Node.load_from_json(data)
            root_list.append(root_node)
    return root_list

def gen_worker(Q, physics_device, start, end):
    try:
        with open("config/target_config.yaml", "r") as f:
            target_cfg = SimpleNamespace(**yaml.safe_load(f))


        with open("config/grpo_generate.yaml", "r") as f:
            generator_cfg = SimpleNamespace(**yaml.safe_load(f))
        
        
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
        torch.cuda.set_device(0)
        print(f"Generation worker process uses GPU {physics_device}")

        ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        prompt_seeds = load_seeds()
        
        gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

        # Load attacker model
        with open("config/attacker_config.yaml", "r") as f:
            attacker_config = SimpleNamespace(**yaml.safe_load(f))
        attacker = AttackAgent(attacker_config)
        
        # Load target model
        vllm_target = TargetModel(target_cfg)
        
        # Load evaluator
        # evaluator = Evaluator(vllm_target.model, tokenizer=vllm_target.tokenizer, sampling_params=vllm_target.sampling_params)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        torch.cuda.set_device(0)
        evaluator = LlamaGuardModeration()
        #Load generator
        generator = TreeGenerator(generator_cfg, attacker, vllm_target, evaluator)

        

        def gen_samples(prompts, it):
            
            if it == -1:
                root_list = load_data()
                
            else:
                root_list: list[Node]= [Node("root", None, seed, None, 0, 0, None, None, None, None) for seed in prompts]
                generator.build_tree_to_depth(root_list)
                generator.pruning(root_list)
                generator.compute_tree_reward(root_list)
                
                os.makedirs("ppo_data/sample/", exist_ok=True)
                for i, root in enumerate(root_list):
                    
                    root.save_tree(f"ppo_data/sample/{physics_device}_{it}_{i}.json")

            messages_list = []
            answers = []
            ans_token_ids = []
            rewards = []
            for i, root_node in enumerate(root_list):
                nodes = root_node.get_all_nodes()
                for node in nodes:
                    if node.children != []:
                        input_messages = node.get_agent_input_messages()
                        for child in node.children:
                            child: Node
                            messages_list.append(input_messages)
                            answers.append(child.origin_output)
                            rewards.append(float(child.reward))
                            if "ans_token_ids" not in child.data_for_training:
                                child.data_for_training["ans_token_ids"] = tokenizer(child.origin_output, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                            ans_token_ids.append(child.data_for_training["ans_token_ids"])
            
            prompts_text = tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True, enable_thinking = False)
            
            return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids
    except Exception as e:
        print("=" * 80)
        print("[ERROR] Exception before gen_worker loop!")
        traceback.print_exc()        
        print("=" * 80)
        sys.stdout.flush()
        time.sleep(0.5)
        raise       
    
    num_batches = math.ceil(len(prompt_seeds) / generator_cfg.trees_per_batch)
    for it in range(num_batches):
        try:
            print(f'===== Generation Iteration {it} =====')
            
            if it == 0:
                pass
            else: 
                while(True):
                    print("waiting new weight")
                    with open("trainer.txt","r") as file:
                        content = file.read().strip()

                    if content == str(it):
                            
                        del attacker.model.llm_engine
                        del attacker.model
                        destroy_model_parallel()
                        destroy_distributed_environment()
                        with contextlib.suppress(AssertionError):
                            torch.distributed.destroy_process_group()
                        gc.collect()
                        torch.cuda.empty_cache()
                        with open("config/attacker_config.yaml", "r") as f:
                            attacker_config = SimpleNamespace(**yaml.safe_load(f))
                            attacker_config.model = "tmp/"
                        attacker.model = LLM(model="tmp/",gpu_memory_utilization=attacker_config.gpu_memory_utilization)
                        print("finish update_model")
                            
                        with open("generator.txt","w") as file:
                            file.write(f"{it}")
                        break
                    else:
                        time.sleep(30)
                    
                    
                    print("Start update_model")
                    
                
            
       
            # seed_batch = random.sample(prompt_seeds[start:end], generator_cfg.trees_per_batch)
            seed_batch = prompt_seeds[it*generator_cfg.trees_per_batch: it*generator_cfg.trees_per_batch+generator_cfg.trees_per_batch]
            prompt_inputs, rewards, answers, ans_token_ids = gen_samples(seed_batch, it)
            
            # print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
            # print(len(prompt_inputs), len(rewards), len(answers), len(ans_token_ids))
            # if it % 5 == 0: print('answers:', answers[0])
            # print("Prompt inputs example:", prompt_inputs[0])
            prompt_ids = tokenizer(prompt_inputs, return_tensors="pt", add_special_tokens=False,padding=True)["input_ids"]
            plen = prompt_ids.shape[1]
            curr_answers = answers
            curr_ans_ids = ans_token_ids
            curr_rewards = rewards
            print("Start uploading...")
        
            # reward, input, answer
            prompt_inputs, curr_rewards, curr_answers = zip(*sorted(zip(prompt_inputs, curr_rewards, curr_answers), key=lambda x: len(x[0])))
            curr_rewards = torch.stack(curr_rewards)
            
            for ii in range(0, len(curr_answers), train_batch_size):
                
                sub_prompt_inputs = prompt_inputs[ii:ii+train_batch_size]
                sub_propmt_ids = tokenizer(sub_prompt_inputs, return_tensors="pt", add_special_tokens=False,padding=True)["input_ids"]
                
                plen = sub_propmt_ids.shape[1]
                
                if plen > max_token_len:
                    plen = max_token_len
                    sub_propmt_ids = sub_propmt_ids[:,-max_token_len:]
                    
                sub_rewards = curr_rewards[ii:ii+train_batch_size]
                sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                
                tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                merged_ids = torch.cat([sub_propmt_ids, output_ids], dim=1)
                
                
                data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

                if compute_gen_logps:
                    complet = tokenizer.batch_decode(merged_ids)
                    zz = attacker.model.generate(complet, sampling_params=gen_logps_sp, use_tqdm=False)
                    
                    zz = [xx.prompt_logprobs[plen:] for xx in zz]

                    gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                    data.append(tensor_to_bytes(gen_logps))

                xdata = make_bytes_list(data)
                
                r = requests.post(f"{ref_server}/upload", data=xdata, proxies={"http": None, "https": None})
                if r.content == b'string': ref_server_ver = 'string'
                    
            # elif ref_server_ver == 'string':
            #     xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
            #                             tensor_to_bytes(curr_rewards)])
            #     r = requests.post(f"{ref_server}/upload", data=xdata, proxies={"http": None, "https": None})
            #     if r.content == b'tensor': ref_server_ver = 'tensor'
            
                
        except Exception as e:
            print("=" * 80)
            print("[ERROR] Exception in gen_worker loop!")
            traceback.print_exc()        
            print("=" * 80)
            sys.stdout.flush()
            time.sleep(0.5)
            raise                    
        
        
if __name__ == '__main__':
    

    # import deepspeed
    # deepspeed.init_distributed()
    start_1=0
    end_1=1000

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method("spawn", force=True)
        Q = mp.Queue()
        p1 = mp.Process(target=gen_worker, args=(Q, 0, start_1, end_1))
        # p2 = mp.Process(target=gen_worker, args=(Q, 2, 5, 10))
        p1.start()
        # p2.start()
        
    
        
    time.sleep(9999999)