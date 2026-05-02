#### GPT-5.4

import os
from openai import AzureOpenAI
import configparser
from tqdm import tqdm

endpoint = "https://intern-jingyu-jhu-east.openai.azure.com/"
model_name = "gpt-5.4"
deployment = "gpt-5.4"

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "api.config")
config.read(config_path)

subscription_key = config["DEFAULT"]["subscription_key"]
api_version = "2024-12-01-preview"


class GPT54():
    def __init__(self):
        self.system_prompt = "You are a helpful AI assistant."
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

    def batch_response(self, messages_list):
        responses = []
        for idx, messages in tqdm(enumerate(messages_list), total=len(messages_list)):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    max_completion_tokens=16384,
                    model=deployment
                )
                content = response.choices[0].message.content
            except:
                response = self.client.chat.completions.create(
                    messages=messages,
                    max_completion_tokens=16384,
                    model=deployment
                )
                content = response.choices[0].message.content
            responses.append(content)
        return responses
