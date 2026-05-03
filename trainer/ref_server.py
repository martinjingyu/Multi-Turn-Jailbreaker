from __future__ import annotations

import io
import json
import os
import queue
import threading
import uuid
from typing import Any

import torch


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def bytes_to_tensor(buffer_bytes: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(buffer_bytes), weights_only=True)


def make_bytes_list(byte_list: list[bytes]) -> bytes:
    buffer = io.BytesIO()
    buffer.write(len(byte_list).to_bytes(4, "big"))
    for item in byte_list:
        buffer.write(len(item).to_bytes(4, "big"))
        buffer.write(item)
    return buffer.getvalue()


def bytes_list_to_list(buffer_bytes: bytes) -> list[bytes]:
    buffer = io.BytesIO(buffer_bytes)
    num_items = int.from_bytes(buffer.read(4), "big")
    byte_list = []
    for _ in range(num_items):
        item_len = int.from_bytes(buffer.read(4), "big")
        byte_list.append(buffer.read(item_len))
    return byte_list


def get_per_token_logps(model, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, 1:]

    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(
            log_probs,
            dim=1,
            index=input_ids_row.unsqueeze(1),
        ).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


class RefService:
    def __init__(self, model_path: str, enable_judge: bool = False):
        from transformers import AutoModelForCausalLM

        self.model_path = model_path
        self.enable_judge = enable_judge

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa",
        ).to("cuda")
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        self.judge = None
        if enable_judge:
            from model.Evaluator.llamaJedge import LlamaGuardModeration

            self.judge = LlamaGuardModeration()

        self.ref_request_queue: queue.LifoQueue[dict[str, Any]] = queue.LifoQueue()
        self.ref_result_queue: queue.LifoQueue[bytes] = queue.LifoQueue()
        self.judge_request_queue: queue.LifoQueue[dict[str, Any]] = queue.LifoQueue()
        self.judge_results: dict[str, bytes] = {}
        self.judge_lock = threading.Lock()

    def handle_ref_upload(self, payload: bytes) -> bytes:
        items = bytes_list_to_list(payload)
        if len(items) not in (3, 4):
            return b"tensor"

        data = {"base": json.loads(items[0])}
        data["inputs"] = bytes_to_tensor(items[1])
        data["rewards"] = bytes_to_tensor(items[2])
        if len(items) == 4:
            data["gen_logps"] = bytes_to_tensor(items[3])
        self.ref_request_queue.put(data)
        return b"tensor"

    def handle_ref_get(self) -> bytes:
        if self.ref_result_queue.empty():
            return b"empty"
        return self.ref_result_queue.get()

    def handle_judge_upload(self, payload: dict[str, Any]) -> bytes:
        request_id = payload.get("request_id")
        if not request_id:
            request_id = str(uuid.uuid4())
            payload["request_id"] = request_id
        self.judge_request_queue.put(payload)
        return json.dumps({"request_id": request_id}).encode()

    def handle_judge_get(self, request_id: str) -> bytes:
        with self.judge_lock:
            if request_id not in self.judge_results:
                return b"empty"
            return self.judge_results.pop(request_id)

    def run_ref_loop(self) -> None:
        while True:
            data = self.ref_request_queue.get()
            prompt_length = data["base"]["plen"]

            with torch.inference_mode():
                per_token_logps = get_per_token_logps(
                    self.ref_model,
                    data["inputs"].to(self.ref_model.device),
                )

            per_token_logps = per_token_logps[:, prompt_length - 1 :]
            result_items = [
                json.dumps(data["base"]).encode(),
                tensor_to_bytes(data["inputs"]),
                tensor_to_bytes(data["rewards"]),
                tensor_to_bytes(per_token_logps),
            ]
            if "gen_logps" in data:
                result_items.append(tensor_to_bytes(data["gen_logps"]))
            self.ref_result_queue.put(make_bytes_list(result_items))

    def run_judge_loop(self) -> None:
        if self.judge is None:
            return

        while True:
            data = self.judge_request_queue.get()
            request_id = data["request_id"]
            histories = data["histories"]
            responses = data["responses"]
            scores = self.judge.eval_batch(responses, histories)
            with self.judge_lock:
                self.judge_results[request_id] = json.dumps({"scores": scores}).encode()


def main() -> int:
    from bottle import request
    import bottle

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_path = os.environ.get("REF_MODEL_PATH", "MartinJYHuang/Jailbreak-agent-temp")
    port = int(os.environ.get("REF_SERVER_PORT", "59875"))
    enable_judge = os.environ.get("ENABLE_LLAMA_GUARD", "0") == "1"
    bottle.BaseRequest.MEMFILE_MAX = int(
        os.environ.get("BOTTLE_MEMFILE_MAX_BYTES", str(16 * 1024 * 1024))
    )

    service = RefService(model_path=model_path, enable_judge=enable_judge)
    app = bottle.Bottle()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path": model_path,
            "enable_judge": enable_judge,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    @app.post("/upload")
    def upload() -> bytes:
        return service.handle_ref_upload(request.body.read())

    @app.get("/get")
    def get_result() -> bytes:
        return service.handle_ref_get()

    @app.post("/judge/upload")
    def judge_upload() -> bytes:
        payload = request.json
        if not isinstance(payload, dict):
            bottle.response.status = 400
            return b"invalid-json"
        if "histories" not in payload or "responses" not in payload:
            bottle.response.status = 400
            return b"missing-fields"
        return service.handle_judge_upload(payload)

    @app.get("/judge/get")
    def judge_get() -> bytes:
        request_id = request.query.get("request_id")
        if not request_id:
            bottle.response.status = 400
            return b"missing-request-id"
        return service.handle_judge_get(request_id)

    threading.Thread(target=service.run_ref_loop, daemon=True).start()
    if enable_judge:
        threading.Thread(target=service.run_judge_loop, daemon=True).start()

    bottle.run(app, host="0.0.0.0", port=port, server="tornado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
