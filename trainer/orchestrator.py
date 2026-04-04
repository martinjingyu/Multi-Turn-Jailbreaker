from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from topology import ClusterTopology, build_topology, visible_device_env


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    role: str
    physical_gpu_id: int
    cmd: list[str]
    cwd: str
    env: dict[str, str]

    def pretty(self) -> str:
        joined_cmd = " ".join(self.cmd)
        return (
            f"{self.name}: role={self.role}, gpu={self.physical_gpu_id}, "
            f"cwd={self.cwd}, cmd={joined_cmd}"
        )


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def build_service_spec(
    topology: ClusterTopology,
    port: int = 59875,
    script_path: Path | None = None,
) -> ProcessSpec:
    script_path = script_path or (THIS_DIR / "ref_server.py")
    env = _base_env()
    env.update(visible_device_env(topology.service_gpu))
    env["REF_SERVER_PORT"] = str(port)
    env["ENABLE_LLAMA_GUARD"] = "1"

    return ProcessSpec(
        name="service",
        role="service",
        physical_gpu_id=topology.service_gpu,
        cmd=[sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def build_rollout_specs(
    topology: ClusterTopology,
    ref_server_url: str,
    script_path: Path | None = None,
) -> list[ProcessSpec]:
    script_path = script_path or (THIS_DIR / "gen_worker.py")
    specs: list[ProcessSpec] = []

    for actor_rank, gpu_id in enumerate(topology.actor_gpus):
        env = _base_env()
        env.update(visible_device_env(gpu_id))
        env["ROLE"] = "actor"
        env["ACTOR_RANK"] = str(actor_rank)
        env["PHYSICAL_GPU_ID"] = str(gpu_id)
        env["REF_SERVER_URL"] = ref_server_url
        env["USE_REMOTE_JUDGE"] = "1"

        specs.append(
            ProcessSpec(
                name=f"rollout-{actor_rank}",
                role="actor",
                physical_gpu_id=gpu_id,
                cmd=[sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        )

    return specs


def build_train_spec(
    topology: ClusterTopology,
    ref_server_url: str,
    model_path: str,
    save_dir: str,
    ds_config_path: str,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    all_steps: int,
    save_every: int,
    max_save_total: int,
    idle_seconds: int,
    poll_interval: int,
    lr: float,
    beta: float,
    clip_param: float,
    upload_hf_repo: str,
    script_path: Path | None = None,
) -> ProcessSpec:
    script_path = script_path or (THIS_DIR / "grpo_train_phase.py")
    deepspeed_exe = Path(sys.executable).parent / "deepspeed"
    env = _base_env()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in topology.actor_gpus)
    env["REF_SERVER_URL"] = ref_server_url

    cmd = [
        str(deepspeed_exe),
        "--num_gpus",
        str(topology.train_world_size),
        str(script_path),
        "--model-path",
        model_path,
        "--ref-server",
        ref_server_url,
        "--save-dir",
        save_dir,
        "--ds-config",
        ds_config_path,
        "--train-batch-size",
        str(train_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--all-steps",
        str(all_steps),
        "--save-every",
        str(save_every),
        "--max-save-total",
        str(max_save_total),
        "--idle-seconds",
        str(idle_seconds),
        "--poll-interval",
        str(poll_interval),
        "--lr",
        str(lr),
        "--beta",
        str(beta),
        "--clip-param",
        str(clip_param),
    ]
    if upload_hf_repo:
        cmd.extend(["--upload-hf-repo", upload_hf_repo])

    return ProcessSpec(
        name="trainer",
        role="trainer",
        physical_gpu_id=-1,
        cmd=cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def print_plan(topology: ClusterTopology, specs: list[ProcessSpec]) -> None:
    print(topology.pretty())
    print("")
    print("Launch plan:")
    for spec in specs:
        print(f"  - {spec.pretty()}")


def launch_processes(specs: list[ProcessSpec]) -> list[subprocess.Popen]:
    processes: list[subprocess.Popen] = []
    for spec in specs:
        print(f"[orchestrator] starting {spec.pretty()}")
        proc = subprocess.Popen(
            spec.cmd,
            cwd=spec.cwd,
            env=spec.env,
        )
        processes.append(proc)
    return processes


def terminate_processes(processes: list[subprocess.Popen], grace_seconds: float = 5.0) -> None:
    if not processes:
        return

    for proc in processes:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)

    deadline = time.time() + grace_seconds
    for proc in processes:
        remaining = max(0.0, deadline - time.time())
        if proc.poll() is None:
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()


def run_rollout_phase(
    topology: ClusterTopology,
    ref_server_port: int,
    dry_run: bool,
) -> int:
    ref_server_url = f"http://127.0.0.1:{ref_server_port}"
    service_spec = build_service_spec(topology, port=ref_server_port)
    rollout_specs = build_rollout_specs(topology, ref_server_url=ref_server_url)
    all_specs = [service_spec, *rollout_specs]

    print_plan(topology, all_specs)

    if dry_run:
        print("")
        print("[orchestrator] dry-run mode enabled, no processes were started.")
        return 0

    processes: list[subprocess.Popen] = []
    try:
        processes = launch_processes(all_specs)
        print("")
        print("[orchestrator] rollout phase launched. Press Ctrl+C to stop.")
        while True:
            time.sleep(5)
            for proc, spec in zip(processes, all_specs):
                if proc.poll() is not None:
                    print(
                        f"[orchestrator] process {spec.name} exited with code {proc.returncode}"
                    )
                    return proc.returncode or 0
    except KeyboardInterrupt:
        print("\n[orchestrator] received Ctrl+C, shutting down child processes.")
        return 0
    finally:
        terminate_processes(processes)


def run_train_phase(
    topology: ClusterTopology,
    ref_server_port: int,
    model_path: str,
    save_dir: str,
    ds_config_path: str,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    all_steps: int,
    save_every: int,
    max_save_total: int,
    idle_seconds: int,
    poll_interval: int,
    lr: float,
    beta: float,
    clip_param: float,
    upload_hf_repo: str,
    dry_run: bool,
) -> int:
    ref_server_url = f"http://127.0.0.1:{ref_server_port}"
    service_spec = build_service_spec(topology, port=ref_server_port)
    train_spec = build_train_spec(
        topology=topology,
        ref_server_url=ref_server_url,
        model_path=model_path,
        save_dir=save_dir,
        ds_config_path=ds_config_path,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        all_steps=all_steps,
        save_every=save_every,
        max_save_total=max_save_total,
        idle_seconds=idle_seconds,
        poll_interval=poll_interval,
        lr=lr,
        beta=beta,
        clip_param=clip_param,
        upload_hf_repo=upload_hf_repo,
    )
    all_specs = [service_spec, train_spec]

    print_plan(topology, all_specs)

    if dry_run:
        print("")
        print("[orchestrator] dry-run mode enabled, no processes were started.")
        return 0

    processes: list[subprocess.Popen] = []
    try:
        processes = launch_processes(all_specs)
        print("")
        print("[orchestrator] training phase launched. Press Ctrl+C to stop.")
        while True:
            time.sleep(5)
            for proc, spec in zip(processes, all_specs):
                if proc.poll() is not None:
                    print(
                        f"[orchestrator] process {spec.name} exited with code {proc.returncode}"
                    )
                    return proc.returncode or 0
    except KeyboardInterrupt:
        print("\n[orchestrator] received Ctrl+C, shutting down child processes.")
        return 0
    finally:
        terminate_processes(processes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-based GRPO orchestrator.")
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
        "--ref-server-port",
        type=int,
        default=59875,
        help="Port used by the service worker.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="rollout",
        choices=["rollout", "train"],
        help="Which phase to run.",
    )
    parser.add_argument("--model-path", type=str, default="MartinJYHuang/JA-v1")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--ds-config", type=str, default="config/ds_config.json")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--all-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-save-total", type=int, default=5)
    parser.add_argument("--idle-seconds", type=int, default=600)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--upload-hf-repo", type=str, default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the launch plan without starting any processes.",
    )
    args = parser.parse_args()

    topology = build_topology(args.gpus, service_gpu=args.service_gpu)

    if not topology.supports_training:
        raise ValueError(
            "At least 2 GPUs are required: 1 for service and 1 for actor/training."
        )

    if args.phase == "rollout":
        return run_rollout_phase(
            topology=topology,
            ref_server_port=args.ref_server_port,
            dry_run=args.dry_run,
        )
    if args.phase == "train":
        return run_train_phase(
            topology=topology,
            ref_server_port=args.ref_server_port,
            model_path=args.model_path,
            save_dir=args.save_dir,
            ds_config_path=args.ds_config,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            all_steps=args.all_steps,
            save_every=args.save_every,
            max_save_total=args.max_save_total,
            idle_seconds=args.idle_seconds,
            poll_interval=args.poll_interval,
            lr=args.lr,
            beta=args.beta,
            clip_param=args.clip_param,
            upload_hf_repo=args.upload_hf_repo,
            dry_run=args.dry_run,
        )

    raise ValueError(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    raise SystemExit(main())
