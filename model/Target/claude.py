import os
import uuid
from anthropic import AsyncAnthropic
import asyncio


class Claude:
    def __init__(self, api_key: str = "", model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in ANTHROPIC_API_KEY environment variable.")
        self.model = model
        self.client = AsyncAnthropic(api_key=self.api_key)

    async def batch_response(self, messages_list: list[list[dict]], max_tokens: int = 1024, temperature: float = 1) -> list[str]:
        # Step 1: Prepare batch request with custom IDs
        requests = []
        id_to_index = {}
        for idx, messages in enumerate(messages_list):
            cid = f"req-{uuid.uuid4()}"
            id_to_index[cid] = idx
            requests.append({
                "custom_id": cid,
                "params": {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages
                }
            })

        # Step 2: Submit batch
        batch = await self.client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"✅ Batch created: {batch_id}")

        # Step 3: Stream results
        results = []
        print("⏳ Waiting for results...")
        
        status_url = f"/v1/messages/batches/{batch_id}"
        while True:
            status_resp = await self.client.messages.batches.retrieve(batch_id)
            status = status_resp.processing_status
            if status == "ended":
                break
            elif status in ("canceling", "failed"):
                raise RuntimeError(f"Batch ended with failure state: {status}")
            print(f"⏳ Still processing... Status = {status}")
            await asyncio.sleep(1)
            
        result_stream = await self.client.messages.batches.results(batch_id)
        async for entry in result_stream:
            if entry.result.type == "succeeded":
                content_blocks = entry.result.message.content
                # print(content_blocks)
                text = "".join(block.text for block in content_blocks if block.type == "text")
                results.append(text.strip())

        return results

async def main():
    claude = Claude(api_key="",model="claude-3-haiku-20240307") 
    inputs = [
        [{"role": "user", "content": "What is LLM"}],
        [{"role": "user", "content": "What is async and await？"}]
    ]
    outputs = await claude.batch_response(inputs)
    for i, out in enumerate(outputs):
        print(f"\nPrompt {i + 1}: {inputs[i][0]['content']}")
        print(f"Response: {out}")


# asyncio.run(main())