from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from glob import glob
from pathlib import Path

import requests
import torch
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from trainer.ppo_utils import upload_model_folder
from trainer.ref_server import bytes_list_to_list, bytes_to_tensor


DEFAULT_MODEL_PATH = "MartinJYHuang/JA-v1"
DEFAULT_REF_SERVER = "http://localhost:59875"
DEFAULT_SAVE_DIR = "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure GRPO training phase.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ref-server", type=str, default=DEFAULT_REF_SERVER)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--ds-config", type=str, default="config/ds_config.json")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--all-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-save-total", type=int, default=5)
    parser.add_argument("--idle-seconds", type=int, default=600)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument(
        "--max-train-seq-len",
        type=int,
        default=int(os.environ.get("TRAIN_MAX_SEQ_LEN", "4096")),
        help="Skip queued batches whose total input length exceeds this value.",
    )
    parser.add_argument(
        "--compute-gen-logps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether gen_workers computed per-token logprobs at rollout time (default: True).",
    )
    parser.add_argument("--upload-hf-repo", type=str, default="")
    args, _ = parser.parse_known_args()
    return args


def load_ds_config(config_path: str, train_batch_size: int, grad_acc_steps: int) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    config["train_micro_batch_size_per_gpu"] = train_batch_size
    config["gradient_accumulation_steps"] = grad_acc_steps
    # Remove "auto" so DeepSpeed computes train_batch_size = micro_batch * grad_acc * world_size
    config.pop("train_batch_size", None)
    return config


def ref_server_exhausted(ref_server: str) -> bool:
    """Returns True when the ref server has no more batches to deliver."""
    try:
        resp = requests.get(
            f"{ref_server}/health",
            proxies={"http": None, "https": None},
            timeout=10,
        )
        data = resp.json()
        return data.get("ref_queue_size", 1) == 0 and data.get("result_queue_size", 1) == 0
    except Exception:
        return False


def get_batch(ref_server: str) -> dict | None:
    try:
        response = requests.get(
            f"{ref_server}/get",
            proxies={"http": None, "https": None},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.content
        if payload == b"empty":
            return None
    except Exception as exc:
        print(f"[grpo_train_phase] Exception when getting batch: {exc}")
        return None

    items = bytes_list_to_list(payload)
    data = json.loads(items[0])
    data["inputs"] = bytes_to_tensor(items[1])
    data["rewards"] = bytes_to_tensor(items[2])
    data["refs"] = bytes_to_tensor(items[3])
    if len(items) == 5:
        data["gen_logps"] = bytes_to_tensor(items[4])
    return data


def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        # Process in chunks along the sequence dimension so we never materialize
        # the full (seq_len × vocab_size) float32 tensor at once (~1 GiB for 151K vocab).
        token_log_probs = []
        for i in range(0, logits_row.shape[0], chunk_size):
            chunk = logits_row[i : i + chunk_size].float()
            log_probs = chunk.log_softmax(dim=-1)
            gathered = torch.gather(
                log_probs, 1, input_ids_row[i : i + chunk_size].unsqueeze(1)
            ).squeeze(1)
            token_log_probs.append(gathered)
            del chunk, log_probs
        per_token_logps.append(torch.cat(token_log_probs))
    return torch.stack(per_token_logps)


def grpo_step(
    batch: dict,
    engine,
    tokenizer,
    beta: float,
    clip_param: float,
    compute_gen_logps: bool,
) -> torch.Tensor:
    prompt_length = batch["plen"]
    inputs = batch["inputs"].to(engine.device)
    advantages = batch["rewards"].to(engine.device).unsqueeze(1)
    advantages = advantages.clamp(-2.0, 2.0)

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length - 1 :]

    ref_per_token_logps = batch["refs"].to(per_token_logps.device)
    # Clamp log-ratio to prevent exp() overflow → Inf → NaN
    kl_log_ratio = (ref_per_token_logps - per_token_logps).clamp(-10, 10)
    per_token_kl = torch.exp(kl_log_ratio) - kl_log_ratio - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if "gen_logps" in batch:
        log_ratio = (per_token_logps - batch["gen_logps"].to(engine.device)).clamp(-10, 10)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        # For positive advantages: conservative = min; for negative: conservative = max
        per_token_loss = torch.where(advantages >= 0, torch.min(surr1, surr2), torch.max(surr1, surr2))
    else:
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    # torch.where avoids 0 * NaN = NaN that occurs with plain multiplication
    mask_bool = completion_mask.bool()
    per_token_loss = torch.where(mask_bool, per_token_loss, torch.zeros_like(per_token_loss))
    mask_sum = completion_mask.sum(dim=1).clamp(min=1)
    loss = (per_token_loss.sum(dim=1) / mask_sum).mean()
    return loss


def batch_debug_summary(batch: dict) -> str:
    inputs = batch["inputs"]
    refs = batch["refs"]
    gen_logps = batch.get("gen_logps")
    parts = [
        f"plen={batch['plen']}",
        f"inputs_shape={tuple(inputs.shape)}",
        f"refs_shape={tuple(refs.shape)}",
    ]
    if gen_logps is not None:
        parts.append(f"gen_logps_shape={tuple(gen_logps.shape)}")
    return ", ".join(parts)


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "cuda=not-used"
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    return (
        f"cuda_allocated={allocated:.2f}GiB, "
        f"cuda_reserved={reserved:.2f}GiB, "
        f"cuda_max_allocated={max_allocated:.2f}GiB"
    )


def find_latest_checkpoint(base_dir: str) -> str:
    checkpoints = glob(os.path.join(base_dir, "step_*"))
    if not checkpoints:
        raise ValueError(f"No checkpoint directories found in {base_dir}.")
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
    return checkpoints[-1]


def save_checkpoint(
    engine,
    tokenizer,
    save_dir: str,
    step: int,
    max_save_total: int,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, f"step_{step}")

    state_dict = engine.module.state_dict()
    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
    engine.module.save_pretrained(save_name, state_dict=state_dict)
    tokenizer.save_pretrained(save_name)

    all_checkpoints = sorted(glob(os.path.join(save_dir, "step_*")), key=os.path.getmtime)
    if len(all_checkpoints) > max_save_total:
        checkpoints_to_delete = all_checkpoints[: len(all_checkpoints) - max_save_total]
        for ckpt in checkpoints_to_delete:
            print(f"[grpo_train_phase] Removing old checkpoint: {ckpt}")
            shutil.rmtree(ckpt)

    return save_name


def main() -> int:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import deepspeed

    deepspeed.init_distributed()
    rank = dist.get_rank()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ds_config = load_ds_config(
        args.ds_config,
        train_batch_size=args.train_batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        _attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=args.lr)
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
    )

    progress = range(args.all_steps)
    if rank == 0:
        progress = tqdm(progress)

    idle_start = None
    last_loss = None
    total_batches = 0

    for step in progress:
        batch = get_batch(args.ref_server)

        # Synchronize batch availability across all ranks.
        # If any rank got no batch, all ranks skip this step to avoid deadlock in
        # collective ops (engine.backward/step require all ranks to participate).
        has_batch = torch.tensor([0 if batch is None else 1], dtype=torch.int, device=engine.device)
        dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)

        if not has_batch.item():
            if idle_start is None:
                idle_start = time.time()

            # Exit immediately if the ref server has no more batches queued.
            is_done = ref_server_exhausted(args.ref_server) if rank == 0 else False
            exhausted = torch.tensor([1 if is_done else 0], dtype=torch.int, device=engine.device)
            dist.all_reduce(exhausted, op=dist.ReduceOp.MAX)
            if exhausted.item():
                if rank == 0:
                    print("[grpo_train_phase] Ref server queues empty, finishing training phase.")
                break

            idle_elapsed = time.time() - idle_start
            if idle_elapsed >= args.idle_seconds:
                if rank == 0:
                    print(
                        "[grpo_train_phase] No new batches arrived within idle timeout. "
                        "Finishing training phase."
                    )
                break

            if rank == 0:
                print(
                    f"[grpo_train_phase] waiting for batch... "
                    f"idle={int(idle_elapsed)}s/{args.idle_seconds}s"
                )
            time.sleep(args.poll_interval)
            continue

        idle_start = None

        local_seq_len = int(batch["inputs"].shape[1])
        too_long = torch.tensor(
            [1 if local_seq_len > args.max_train_seq_len else 0],
            dtype=torch.int,
            device=engine.device,
        )
        dist.all_reduce(too_long, op=dist.ReduceOp.MAX)
        if too_long.item():
            if rank == 0:
                print(
                    "[grpo_train_phase] Skipping overlong batch: "
                    f"max_train_seq_len={args.max_train_seq_len}, "
                    f"{batch_debug_summary(batch)}, "
                    f"{cuda_memory_summary(engine.device)}"
                )
            continue

        if rank == 0 and total_batches % 10 == 0:
            print(
                "[grpo_train_phase] Training batch: "
                f"{batch_debug_summary(batch)}, "
                f"{cuda_memory_summary(engine.device)}"
            )

        local_oom = torch.tensor([0], dtype=torch.int, device=engine.device)
        loss = None
        try:
            loss = grpo_step(
                batch=batch,
                engine=engine,
                tokenizer=tokenizer,
                beta=args.beta,
                clip_param=args.clip_param,
                compute_gen_logps=args.compute_gen_logps,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            local_oom.fill_(1)
            if rank == 0:
                print(
                    "[grpo_train_phase] CUDA OOM during forward/backward prep: "
                    f"{batch_debug_summary(batch)}, "
                    f"{cuda_memory_summary(engine.device)}"
                )
            torch.cuda.empty_cache()

        dist.all_reduce(local_oom, op=dist.ReduceOp.MAX)
        if local_oom.item():
            continue
        assert loss is not None

        # Skip non-finite loss to prevent NaN from corrupting model weights.
        finite = torch.tensor([1 if torch.isfinite(loss) else 0], dtype=torch.int, device=engine.device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            if rank == 0:
                print(f"[grpo_train_phase] Non-finite loss ({loss.item():.4f}), skipping batch.")
            continue

        engine.backward(loss)
        engine.step()

        last_loss = loss.item()
        total_batches += 1

        dist.barrier()

        if rank == 0:
            if last_loss is not None:
                progress.set_description(
                    f"Loss: {last_loss:.6f} | Batches: {total_batches}"
                )

            should_save = ((step + 1) % args.save_every == 0) or (step == args.all_steps - 1)
            if should_save:
                print(f"[grpo_train_phase] saving model at step {step}")
                save_checkpoint(
                    engine=engine,
                    tokenizer=tokenizer,
                    save_dir=args.save_dir,
                    step=step,
                    max_save_total=args.max_save_total,
                )

                if args.upload_hf_repo:
                    latest_model_path = find_latest_checkpoint(args.save_dir)
                    upload_model_folder(latest_model_path, args.upload_hf_repo)

        dist.barrier()

    if rank == 0:
        print(
            f"[grpo_train_phase] finished. total_batches={total_batches}, "
            f"last_loss={last_loss}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
