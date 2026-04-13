from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from glob import glob
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
    model_path: str = "",
    rollout_iter: int = 0,
    trees_per_worker: int = 5,
    script_path: Path | None = None,
) -> list[ProcessSpec]:
    script_path = script_path or (THIS_DIR / "gen_worker.py")
    num_actor_workers = len(topology.actor_gpus)
    specs: list[ProcessSpec] = []

    for actor_rank, gpu_id in enumerate(topology.actor_gpus):
        env = _base_env()
        env.update(visible_device_env(gpu_id))
        env["ROLE"] = "actor"
        env["ACTOR_RANK"] = str(actor_rank)
        env["PHYSICAL_GPU_ID"] = str(gpu_id)
        env["REF_SERVER_URL"] = ref_server_url
        env["USE_REMOTE_JUDGE"] = "1"
        env["ROLLOUT_ITER"] = str(rollout_iter)
        env["NUM_ACTOR_WORKERS"] = str(num_actor_workers)
        env["TREES_PER_WORKER"] = str(trees_per_worker)
        if model_path:
            env["ATTACKER_MODEL_PATH"] = model_path

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


def get_num_seeds() -> int:
    """Read the RL seed file and return the number of prompts."""
    seeds_path = PROJECT_ROOT / "data" / "rl.json"
    with open(seeds_path) as f:
        return len(json.load(f))


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


def find_latest_checkpoint(save_dir: str) -> str:
    checkpoints = glob(os.path.join(save_dir, "step_*"))
    if not checkpoints:
        raise ValueError(f"No checkpoint directories found in {save_dir}.")
    return max(checkpoints, key=lambda x: int(x.rsplit("_", 1)[-1]))


def run_loop(
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
    trees_per_worker: int,
    iterations: int,
    dry_run: bool,
) -> int:
    """
    Full GRPO training loop.

    Seeds are partitioned across both workers and iterations so that each
    worker generates exactly `trees_per_worker` trees per iteration:

        global_batch = rollout_iter * num_actor_workers + actor_rank
        seeds[global_batch * trees_per_worker : (global_batch+1) * trees_per_worker]

    Total iterations are auto-computed as ceil(num_seeds / trees_per_worker /
    num_actor_gpus).  Pass --iterations to cap or override that number.

    The ref_server is started once and kept alive for all iterations.
    """
    ref_server_url = f"http://127.0.0.1:{ref_server_port}"
    service_spec = build_service_spec(topology, port=ref_server_port)

    num_seeds = get_num_seeds()
    num_actor_workers = topology.num_actor_gpus
    auto_iterations = math.ceil(num_seeds / trees_per_worker / num_actor_workers)
    total_iterations = iterations if iterations > 0 else auto_iterations

    print(
        f"[orchestrator] seeds={num_seeds}, trees_per_worker={trees_per_worker}, "
        f"actor_workers={num_actor_workers} → "
        f"auto_iterations={auto_iterations}, running={total_iterations}"
    )

    if dry_run:
        rollout_specs = build_rollout_specs(
            topology, ref_server_url,
            model_path=model_path,
            rollout_iter=0,
            trees_per_worker=trees_per_worker,
        )
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
        print_plan(topology, [service_spec, *rollout_specs, train_spec])
        print("")
        print(f"[orchestrator] dry-run: {total_iterations} iteration(s) planned, no processes started.")
        return 0

    # Start the ref_server once; it persists across all rollout+train iterations.
    print("[orchestrator] Starting ref_server (kept alive for all iterations)...")
    service_proc = launch_processes([service_spec])[0]
    current_model_path = model_path

    try:
        for i in range(total_iterations):
            print(f"\n[orchestrator] ========== GRPO Iteration {i + 1}/{total_iterations} ==========\n")

            # -- Rollout phase --
            rollout_specs = build_rollout_specs(
                topology, ref_server_url,
                model_path=current_model_path,
                rollout_iter=i,
                trees_per_worker=trees_per_worker,
            )
            print(
                f"[orchestrator] Rollout {i + 1}: each worker generates seeds "
                f"[{i * num_actor_workers * trees_per_worker} .. "
                f"{(i + 1) * num_actor_workers * trees_per_worker - 1}], "
                f"model={current_model_path}"
            )
            rollout_procs = launch_processes(rollout_specs)

            # Wait for every gen_worker to finish uploading its batches.
            for proc, spec in zip(rollout_procs, rollout_specs):
                ret = proc.wait()
                if ret != 0:
                    print(f"[orchestrator] {spec.name} exited with code {ret}, aborting loop.")
                    return ret

            if service_proc.poll() is not None:
                print(f"[orchestrator] ref_server exited unexpectedly "
                      f"(code {service_proc.returncode}), aborting.")
                return service_proc.returncode or 1

            print(f"[orchestrator] Rollout phase {i + 1} complete.")

            # -- Train phase --
            train_spec = build_train_spec(
                topology=topology,
                ref_server_url=ref_server_url,
                model_path=current_model_path,
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
            print(f"[orchestrator] Starting train phase {i + 1}...")
            train_proc = launch_processes([train_spec])[0]
            ret = train_proc.wait()
            if ret != 0:
                print(f"[orchestrator] trainer exited with code {ret}, aborting loop.")
                return ret

            if service_proc.poll() is not None:
                print(f"[orchestrator] ref_server exited unexpectedly "
                      f"(code {service_proc.returncode}), aborting.")
                return service_proc.returncode or 1

            print(f"[orchestrator] Train phase {i + 1} complete.")

            # -- Update model path for next rollout --
            try:
                current_model_path = find_latest_checkpoint(save_dir)
                print(f"[orchestrator] Updated attacker model → {current_model_path}")
            except ValueError as exc:
                print(f"[orchestrator] Warning: {exc}  Keeping previous model path.")

        print(f"\n[orchestrator] GRPO loop finished ({total_iterations} iteration(s)).")
        return 0

    except KeyboardInterrupt:
        print("\n[orchestrator] received Ctrl+C, shutting down.")
        return 0
    finally:
        terminate_processes([service_proc])


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
        choices=["rollout", "train", "loop"],
        help=(
            "Which phase to run.  "
            "'rollout' starts gen_workers + ref_server;  "
            "'train' starts trainer + ref_server;  "
            "'loop' runs N full rollout→train iterations automatically "
            "(use --iterations to set N)."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=-1,
        help=(
            "Number of rollout→train iterations (--phase loop only).  "
            "Default -1 = auto-compute from ceil(num_seeds / trees_per_worker / num_actor_gpus)."
        ),
    )
    parser.add_argument(
        "--trees-per-worker",
        type=int,
        default=5,
        help="Trees each gen_worker generates per iteration (default: 5).",
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
    if args.phase == "loop":
        return run_loop(
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
            trees_per_worker=args.trees_per_worker,
            iterations=args.iterations,
            dry_run=args.dry_run,
        )

    raise ValueError(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    raise SystemExit(main())
