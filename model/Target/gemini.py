import time
from typing import List
from google import genai
import re

class Gemini:
    def __init__(self, model: str = "models/gemini-2.5-flash", display_name: str = "gemini-batch-job"):
        self.client = genai.Client(api_key="")
        self.model = model
        self.display_name = display_name

    def _convert_messages_to_contents(self, messages: List[dict]) -> List[dict]:
        """
        将多轮 messages (role, content) 转换为 Gemini 格式的 contents
        e.g. [{"role": "user", "content": "hi"}] -> [{"role": "user", "parts": [{"text": "hi"}]}]
        """
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            contents.append({"role": role, "parts": [{"text": text}]})
        return contents

    def batch_response(self, messages_list: List[List[dict]]) -> List[str]:
        """
        批量推理：输入 list[list[dict]]，输出 list[str]
        """
        inline_requests = []
        for messages in messages_list:
            inline_requests.append({
                "contents": self._convert_messages_to_contents(messages),
            })

        job = self.client.batches.create(
            model=self.model,
            src=inline_requests,
            config={"display_name": self.display_name,
                    },
            
        )

        job_name = job.name
        print(f"✅ Job created: {job_name}")

        # --- 轮询状态 ---
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
        print("⏳ Waiting for batch job to complete...")
        while True:
            job = self.client.batches.get(name=job_name)
            state = job.state.name
            if state in completed_states:
                break
            print(f"Current state: {state}, sleeping 30s...")
            time.sleep(30)

        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"❌ Gemini batch job failed: {job.state.name}")

        outputs = []
        for i, resp in enumerate(job.dest.inlined_responses):
            if resp.response:
                try:
                    raw_text = resp.response.text.strip()
                    trimmed_text = truncate_to_token_limit(raw_text, max_tokens=540)
                    outputs.append(trimmed_text)
                except AttributeError:
                    outputs.append(str(resp.response))
            elif resp.error:
                outputs.append(f"[Error]: {resp.error.message}")
            else:
                outputs.append("[Unknown error or empty result]")

        return outputs


def rough_token_count(text: str) -> int:
    # 粗略统计英文 token（空格/标点划分），可根据实际 tokenizer 替换
    return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))

def truncate_to_token_limit(text: str, max_tokens: int = 540) -> str:
    # 非精确切词，但在无法加载官方 tokenizer 的情况下有效
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    truncated_tokens = tokens[:max_tokens]
    return ' '.join(truncated_tokens)


if __name__ == "__main__":
    prompts = [
        "用一句话解释什么是量子纠缠。",
        "请简述马格丽塔披萨的主要材料。",
        "请写一首关于猫的短诗。"
    ]

    gemini = Gemini()
    results = gemini.batch_response(prompts)

    for i, r in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {r}")