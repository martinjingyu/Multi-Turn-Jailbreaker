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
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--all-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-save-total", type=int, default=5)
    parser.add_argument("--idle-seconds", type=int, default=600)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--compute-gen-logps", action="store_true", default=True)
    parser.add_argument("--upload-hf-repo", type=str, default="")
    return parser.parse_args()


def load_ds_config(config_path: str, train_batch_size: int, grad_acc_steps: int) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    config["train_micro_batch_size_per_gpu"] = train_batch_size
    config["gradient_accumulation_steps"] = grad_acc_steps
    return config


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


def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
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

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length - 1 :]

    ref_per_token_logps = batch["refs"].to(per_token_logps.device)
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps)
        - 1
    )
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if "gen_logps" in batch:
        ratio = torch.exp(per_token_logps - batch["gen_logps"].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


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

        if batch is None:
            if idle_start is None:
                idle_start = time.time()

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

        while batch is not None:
            loss = grpo_step(
                batch=batch,
                engine=engine,
                tokenizer=tokenizer,
                beta=args.beta,
                clip_param=args.clip_param,
                compute_gen_logps=args.compute_gen_logps,
            )
            engine.backward(loss)
            engine.step()

            last_loss = loss.item()
            total_batches += 1
            batch = get_batch(args.ref_server)

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
