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
        max_request_bytes: int = 512 * 1024,
    ):
        self.service_url = service_url.rstrip("/")
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.max_request_bytes = max_request_bytes

    def _payload_size_bytes(
        self,
        target_responses: list[str],
        chat_histories: list[list[dict[str, Any]]],
    ) -> int:
        payload = {
            "responses": target_responses,
            "histories": chat_histories,
        }
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _split_into_chunks(
        self,
        target_responses: list[str],
        chat_histories: list[list[dict[str, Any]]],
    ) -> list[tuple[list[str], list[list[dict[str, Any]]]]]:
        if len(target_responses) != len(chat_histories):
            raise ValueError("target_responses and chat_histories must have the same length.")

        chunks: list[tuple[list[str], list[list[dict[str, Any]]]]] = []
        current_responses: list[str] = []
        current_histories: list[list[dict[str, Any]]] = []

        for response, history in zip(target_responses, chat_histories):
            test_responses = [*current_responses, response]
            test_histories = [*current_histories, history]

            if current_responses and self._payload_size_bytes(test_responses, test_histories) > self.max_request_bytes:
                chunks.append((current_responses, current_histories))
                current_responses = [response]
                current_histories = [history]
            else:
                current_responses = test_responses
                current_histories = test_histories

        if current_responses:
            chunks.append((current_responses, current_histories))

        return chunks

    def _eval_chunk(
        self,
        target_responses: list[str],
        chat_histories: list[list[dict[str, Any]]],
    ) -> list[float]:
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

    def eval_batch(
        self,
        target_responses: list[str],
        chat_histories: list[list[dict[str, Any]]],
    ) -> list[float]:
        all_scores: list[float] = []
        chunks = self._split_into_chunks(target_responses, chat_histories)

        for chunk_responses, chunk_histories in chunks:
            chunk_scores = self._eval_chunk(chunk_responses, chunk_histories)
            all_scores.extend(chunk_scores)

        return all_scores
