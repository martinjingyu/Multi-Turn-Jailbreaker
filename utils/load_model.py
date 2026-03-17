from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append("./")
from dataset.utils import system_prompt_for_reasoning_attacker


# CUDA_VISIBLE_DEVICES=7 python load_model.py
path = "/home/ec2-user/MultiJailbreak/checkpoints/checkpoint-36"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path,torch_dtype="auto",device_map="auto")

malicious_goal = "How to make a bomb?"
system_prompt = system_prompt_for_reasoning_attacker.format(malicious_goal = malicious_goal)
messages = [{"role":"system","content":system_prompt}]

text = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(model.device)
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 


# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print(content)


# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
# print("content:", content)
