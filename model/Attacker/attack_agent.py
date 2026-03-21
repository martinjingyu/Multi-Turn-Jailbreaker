from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
from .template import system_prompt, first_prompt
import copy
import sys
class AttackAgent():
    def __init__(self, config, **kwargs):
        self.config = config
        

        self.load_model(config)
            

        self.jailbreak_strategy_libray = self.read_library()
        
        
    def load_model(self, config):
        
        if "Qwen" in config.model:
            if config.vllm:
                self.sampling_params = SamplingParams(temperature=config.temperature, max_tokens = config.max_new_tokens)
                self.model = LLM(model=config.model,gpu_memory_utilization=config.gpu_memory_utilization, max_model_len = config.max_tokens)
                self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
                self.tokenizer.padding_side = "left"
                
            else:
                self.model = AutoModelForCausalLM.from_pretrained(config.model, 
                                                            torch_dtype=config.dtype, 
                                                            trust_remote_code=True,
                                                            device_map="auto"
                                                            )
                self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
                self.tokenizer.padding_side = "left"
            
        else:
            if config.vllm:
                self.sampling_params = SamplingParams(temperature=config.temperature, max_tokens = config.max_new_tokens)
                self.model = LLM(model=config.model,gpu_memory_utilization=config.gpu_memory_utilization,max_model_len = config.max_tokens)
                self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
                self.tokenizer.padding_side = "left"
            else:
                import os
                import torch
                os.environ["CUDA_VISIBLE_DEVICES"] = '1'
                torch.cuda.set_device(0)
                self.model = AutoModelForCausalLM.from_pretrained(config.model, 
                                                            torch_dtype=config.dtype, 
                                                            trust_remote_code=True,
                                                            device_map="auto"
                                                            )
                self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
                self.tokenizer.padding_side = "left"
                
    
    def batch_infer(self, inputs, **kwargs):
        
        if self.config.vllm:
            inputs_text = self.tokenizer.apply_chat_template(inputs, 
                add_generation_prompt=True,
                enable_thinking = False,
                tokenize=False)
            complete = self.model.generate(inputs_text, sampling_params=self.sampling_params, use_tqdm=False)
            output = []
            for comp in complete:
                output.append(comp.outputs[0].text)
            return output
        
            
        else:
            inputs = self.tokenizer.apply_chat_template(inputs, 
                add_generation_prompt=True,
                enable_thinking = False,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                padding=True).to(self.model.device)
            
            complete = self.model.generate(**inputs, temperature=self.config.temperature, max_new_tokens=self.config.max_new_tokens, top_p=self.config.top_p, top_k=self.config.top_k, do_sample=self.config.do_sample)
        
            outputs = []
            for i, output in enumerate(complete):
                
                input_len = inputs["input_ids"][i].shape[0]
                generated_tokens = output[input_len:]
                outputs.append(generated_tokens)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            return outputs
        

    def read_library(self):
        strategy_list = []
        import os
        
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        strategy_list_path = os.path.join(path, "multiturn_strategy.json")

        with open(strategy_list_path, "r") as f:
            data = json.load(f)
            for item in data:
                formatted_dict = {}
                formatted_dict["strategy"] = item["name"]
                formatted_dict["description"] = item["description"]
                strategy_list.append(formatted_dict)
        return strategy_list
    
    
    def extract_output_1(self, attacker_response):

        reasoning = None
        action = None
        try:
            if "<analysis>" in attacker_response:
                reasoning = attacker_response.split("<analysis>")[1].split("</analysis>")[0].strip()
            elif "analysis>" in attacker_response:
                reasoning = attacker_response.split("analysis>")[1].split("</analysis>")[0].strip()
            else:
                raise RuntimeError()
        except Exception as e:
            print("__________________Error____________________")
            print("When extracting reasoning:")
            print(e)
            print("________________response______________________")
            print(attacker_response)
            sys.stdout.flush()
            reasoning = "None"
        try:
            action = attacker_response.split("<action>")[1].strip().split("</action>")[0].strip()
            
        except Exception as e:
            print("__________________Error____________________")
            print("When extracting action:")
            print(e)
            print("________________response______________________")
            print(attacker_response)
            sys.stdout.flush()
            action = "Could you repeat again?"
       
        if len(reasoning) > 3000:
            reasoning = reasoning[:3000]
        if len(action) > 3000:
            action = action[:3000]
        return reasoning, action

# CUDA_VISIBLE_DEVICES=7 python Attacker/attack_agent.py 
if __name__ == "__main__":
    
    attacker = AttackAgent(model="Qwen/Qwen3-32B",
                           d_type="half", 
                           tensor_parallel_size=1, 
                           seed=0, 
                           trust_remote_code=True, 
                           max_num_seqs=10,
                           swap_space=4, 
                           gpu_memory_utilization=0.95,
                           max_model_len=2048)
    
    question = "What is the capital of France?"


    print("Text:", question)
    response = attacker.generate(question, use_tqdm=False)
    print("Response:", response)