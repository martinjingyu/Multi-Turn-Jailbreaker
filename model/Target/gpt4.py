#### GPT-4o

import os
from openai import AzureOpenAI
import configparser
from tqdm import tqdm
endpoint = "https://intern-jingyu-jhu-east.openai.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "api.config")

config.read(config_path)


subscription_key = config["DEFAULT"]["subscription_key"]
api_version = "2024-12-01-preview"



class GPT4():
    def __init__(self):
        
        self.system_prompt = "You are a helpful AI assistant."
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )


    
    def batch_response(self, messages_list):
        """
        Generate responses for a batch of messages (one at a time).
        """
        responses = []
        for idx, messages in tqdm(enumerate(messages_list),total=len(messages_list)):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    max_tokens=512,
                    temperature=1.0,
                    top_p=1.0,
                    model=deployment
                )
                content = response.choices[0].message.content
            except:
                response = self.client.chat.completions.create(
                    messages=messages,
                    max_tokens=512,
                    temperature=1.0,
                    top_p=1.0,
                    model=deployment
                )
                content = response.choices[0].message.content
                
            responses.append(content)

        return responses
        

        
"""
CUDA_VISIBLE_DEVICES=0 python model/target_model.py
"""
if __name__ == '__main__':  


    system = "You are a helpful assistant."
    system_prompt = [{"role": "system", "content": system}]
    model = GPT4()
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    
    messages = system_prompt + messages
    messages_list = []
    messages_list.append(messages)
    messages_list.append(messages)
    response = model.batch_response(messages_list)
    print("Response:", response)



