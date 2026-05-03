#### GPT-5.4

import os
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI, BadRequestError

endpoint = "https://intern-jingyu-jhu-east.openai.azure.com/"
model_name = "gpt-5.4"
deployment = "gpt-5.4"

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "api.config")
config.read(config_path)

subscription_key = config["DEFAULT"]["subscription_key"]
api_version = "2024-12-01-preview"

MAX_WORKERS = 16


class GPT54():
    def __init__(self):
        self.system_prompt = "You are a helpful AI assistant."
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

    def _single_request(self, messages):
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                max_completion_tokens=16384,
                model=deployment
            )
            return response.choices[0].message.content
        except BadRequestError:
            return "I'm sorry, I cannot assist with that request."
        except Exception:
            response = self.client.chat.completions.create(
                messages=messages,
                max_completion_tokens=16384,
                model=deployment
            )
            return response.choices[0].message.content

    def batch_response(self, messages_list):
        results = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self._single_request, messages): idx
                for idx, messages in enumerate(messages_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results
