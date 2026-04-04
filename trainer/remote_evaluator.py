from __future__ import annotations

import json
import time
from typing import Any

import requests


class RemoteLlamaGuardEvaluator:
    """
    Adapter that preserves the local evaluator interface:
    `eval_batch(target_responses, chat_histories) -> list[float]`
    while delegating the actual judging work to the service worker.
    """

    def __init__(
        self,
        service_url: str,
        poll_interval_seconds: float = 0.2,
        request_timeout_seconds: float = 30.0,
    ):
        self.service_url = service_url.rstrip("/")
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds

    def eval_batch(self, target_responses: list[str], chat_histories: list[list[dict[str, Any]]]) -> list[float]:
        payload = {
            "responses": target_responses,
            "histories": chat_histories,
        }

        response = requests.post(
            f"{self.service_url}/judge/upload",
            json=payload,
            proxies={"http": None, "https": None},
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()
        request_id = response.json()["request_id"]

        while True:
            result = requests.get(
                f"{self.service_url}/judge/get",
                params={"request_id": request_id},
                proxies={"http": None, "https": None},
                timeout=self.request_timeout_seconds,
            )
            result.raise_for_status()

            if result.content == b"empty":
                time.sleep(self.poll_interval_seconds)
                continue

            body = json.loads(result.content)
            return body["scores"]
