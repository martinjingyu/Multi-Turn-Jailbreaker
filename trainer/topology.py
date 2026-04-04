from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def parse_gpu_ids(gpus: str | Iterable[int]) -> list[int]:
    """
    Parse GPU ids from either:
    - a comma-separated string such as "0,1,2,3"
    - an iterable of integers
    """
    if isinstance(gpus, str):
        gpu_ids = []
        for item in gpus.split(","):
            item = item.strip()
            if not item:
                continue
            gpu_ids.append(int(item))
    else:
        gpu_ids = [int(x) for x in gpus]

    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided.")

    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU ids must be unique, got: {gpu_ids}")

    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"GPU ids must be non-negative, got: {gpu_ids}")

    return gpu_ids


@dataclass(frozen=True)
class ClusterTopology:
    """
    Phase-based topology for GRPO training.

    Design:
    - One dedicated service GPU:
      loads ref model + llama guard and serves reward / ref logprobs.
    - Remaining GPUs are actor GPUs during rollout.
    - The same actor GPUs become the trainer group during GRPO training.
    """

    all_gpus: tuple[int, ...]
    service_gpu: int
    actor_gpus: tuple[int, ...]

    @property
    def num_total_gpus(self) -> int:
        return len(self.all_gpus)

    @property
    def num_actor_gpus(self) -> int:
        return len(self.actor_gpus)

    @property
    def train_world_size(self) -> int:
        return len(self.actor_gpus)

    @property
    def supports_training(self) -> bool:
        return self.train_world_size > 0

    @property
    def rollout_world_size(self) -> int:
        return len(self.actor_gpus)

    def actor_gpu_for_rank(self, rank: int) -> int:
        if rank < 0 or rank >= len(self.actor_gpus):
            raise IndexError(
                f"Actor rank {rank} is out of range for actor_gpus={list(self.actor_gpus)}"
            )
        return self.actor_gpus[rank]

    def as_dict(self) -> dict:
        return {
            "all_gpus": list(self.all_gpus),
            "service_gpu": self.service_gpu,
            "actor_gpus": list(self.actor_gpus),
            "num_total_gpus": self.num_total_gpus,
            "num_actor_gpus": self.num_actor_gpus,
            "rollout_world_size": self.rollout_world_size,
            "train_world_size": self.train_world_size,
            "supports_training": self.supports_training,
        }

    def pretty(self) -> str:
        lines = [
            "ClusterTopology(",
            f"  all_gpus={list(self.all_gpus)},",
            f"  service_gpu={self.service_gpu},",
            f"  actor_gpus={list(self.actor_gpus)},",
            f"  rollout_world_size={self.rollout_world_size},",
            f"  train_world_size={self.train_world_size},",
            ")",
        ]
        return "\n".join(lines)


def build_topology(
    gpus: str | Iterable[int],
    service_gpu: int | None = None,
) -> ClusterTopology:
    """
    Build a simple and stable topology.

    Default policy:
    - the first GPU is reserved as the service GPU
    - all remaining GPUs become actor / trainer GPUs

    Examples:
    - "0,1"     -> service=0, actors=[1]
    - "0,1,2"   -> service=0, actors=[1,2]
    - "2,3,5,7" -> service=2, actors=[3,5,7]
    """
    gpu_ids = parse_gpu_ids(gpus)

    if service_gpu is None:
        service_gpu = gpu_ids[0]

    if service_gpu not in gpu_ids:
        raise ValueError(
            f"service_gpu={service_gpu} must be one of the provided GPUs: {gpu_ids}"
        )

    actor_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id != service_gpu]

    return ClusterTopology(
        all_gpus=tuple(gpu_ids),
        service_gpu=service_gpu,
        actor_gpus=tuple(actor_gpus),
    )


def visible_device_env(physical_gpu_id: int) -> dict[str, str]:
    """
    Build the minimal CUDA environment for a single-GPU worker process.

    After setting CUDA_VISIBLE_DEVICES to one physical GPU id, the worker
    should use local cuda:0 inside that process.
    """
    return {"CUDA_VISIBLE_DEVICES": str(physical_gpu_id)}


def local_cuda_device() -> str:
    """
    The device each isolated worker should use after CUDA_VISIBLE_DEVICES
    has been restricted to a single GPU.
    """
    return "cuda:0"


def infer_worker_role(topology: ClusterTopology, physical_gpu_id: int) -> str:
    if physical_gpu_id == topology.service_gpu:
        return "service"
    if physical_gpu_id in topology.actor_gpus:
        return "actor"
    raise ValueError(
        f"GPU {physical_gpu_id} is not part of topology {list(topology.all_gpus)}"
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect GRPO cluster topology.")
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help='Comma-separated GPU ids, for example "0,1,2,3".',
    )
    parser.add_argument(
        "--service-gpu",
        type=int,
        default=None,
        help="Optional override for the dedicated service GPU.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print topology as JSON instead of a human-readable summary.",
    )
    args = parser.parse_args()

    topology = build_topology(args.gpus, service_gpu=args.service_gpu)
    if args.json:
        print(json.dumps(topology.as_dict(), indent=2))
    else:
        print(topology.pretty())
