from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import re
import requests
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

from datagenerator.generate_utils import Node, TreeGenerator
from model.Attacker.attack_agent import AttackAgent
from model.Evaluator.llamaJedge import LlamaGuardModeration
from model.Target.target_model import TargetModel
from trainer.ppo_utils import load_seeds
from trainer.remote_evaluator import RemoteLlamaGuardEvaluator
from trainer.ref_server import make_bytes_list, tensor_to_bytes


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = "MartinJYHuang/Jailbreak-agent-temp"
TRAIN_BATCH_SIZE = 1
MAX_TOKEN_LEN = int(os.environ.get("ROLLOUT_MAX_PROMPT_TOKENS", "4096"))
COMPUTE_GEN_LOGPS = True
DEFAULT_REF_SERVER = "http://localhost:59875"
DEFAULT_MODEL_PATH = MODEL_PATH


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _str_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_yaml_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        return SimpleNamespace(**yaml.safe_load(f))


def load_data() -> list[Node]:
    data_dir = PROJECT_ROOT / "ppo_data" / "cache"
    json_files = list(data_dir.glob("*.json"))

    root_list = []
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            root_node = Node.load_from_json(data)
            root_list.append(root_node)
    return root_list


def reload_attacker_model(attacker: AttackAgent, model_path: str) -> None:
    if hasattr(attacker.model, "llm_engine"):
        del attacker.model.llm_engine
    del attacker.model

    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError, RuntimeError):
        torch.distributed.destroy_process_group()

    gc.collect()
    torch.cuda.empty_cache()

    attacker.model = LLM(
        model=model_path,
        gpu_memory_utilization=attacker.config.gpu_memory_utilization,
        max_model_len=attacker.config.max_tokens,
    )


def build_generator(
    ref_server_url: str,
    use_remote_judge: bool,
    model_path: str = "",
) -> tuple[AutoTokenizer, list[str], SamplingParams, TreeGenerator, SimpleNamespace]:
    target_cfg = load_yaml_config(str(PROJECT_ROOT / "config" / "target_config.yaml"))
    generator_cfg = load_yaml_config(str(PROJECT_ROOT / "config" / "grpo_generate.yaml"))
    attacker_config = load_yaml_config(str(PROJECT_ROOT / "config" / "attacker_config.yaml"))

    # Allow the orchestrator to override the attacker model (e.g. latest checkpoint).
    effective_model_path = model_path if model_path else attacker_config.model
    attacker_config.model = effective_model_path

    tokenizer = AutoTokenizer.from_pretrained(effective_model_path)
    prompt_seeds = load_seeds()
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    attacker = AttackAgent(attacker_config)
    vllm_target = TargetModel(target_cfg)

    if use_remote_judge:
        evaluator = RemoteLlamaGuardEvaluator(ref_server_url)
    else:
        evaluator = LlamaGuardModeration()
    generator = TreeGenerator(generator_cfg, attacker, vllm_target, evaluator)
    return tokenizer, prompt_seeds, gen_logps_sp, generator, attacker_config


def gen_samples(
    prompts: list[str],
    iteration: int,
    generator: TreeGenerator,
    tokenizer: AutoTokenizer,
    worker_label: str,
) -> tuple[list[str], torch.Tensor, list[str], list[torch.Tensor]]:
    if iteration == -1:
        root_list = load_data()
    else:
        root_list = [
            Node("root", None, seed, None, 0, 0, None, None, None, None) for seed in prompts
        ]
        generator.build_tree_to_depth(root_list)
        generator.pruning(root_list)
        generator.compute_tree_reward(root_list)

        sample_dir = PROJECT_ROOT / "ppo_data" / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for sample_index, root in enumerate(root_list):
            root.save_tree(str(sample_dir / f"{worker_label}_{iteration}_{sample_index}.json"))

    _FORMAT_PATTERN = re.compile(
        r"^<analysis>.*?</analysis>[\n ]*<action>.*?</action>[\n ]*$",
        re.DOTALL,
    )

    def _is_well_formed(text: str | None) -> bool:
        if not text:
            return False
        if text.count("<analysis>") + text.count("</analysis>") != 2:
            return False
        if text.count("<action>") + text.count("</action>") != 2:
            return False
        return bool(_FORMAT_PATTERN.match(text))

    messages_list = []
    answers = []
    ans_token_ids = []
    rewards = []
    skipped = 0
    for root_node in root_list:
        nodes = root_node.get_all_nodes()
        for node in nodes:
            if not node.children:
                continue
            input_messages = node.get_agent_input_messages()
            for child in node.children:
                if not _is_well_formed(child.origin_output):
                    skipped += 1
                    continue
                messages_list.append(input_messages)
                answers.append(child.origin_output)
                rewards.append(float(child.reward))
                if "ans_token_ids" not in child.data_for_training:
                    child.data_for_training["ans_token_ids"] = tokenizer(
                        child.origin_output,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                ans_token_ids.append(child.data_for_training["ans_token_ids"])
    if skipped:
        print(f"[gen_worker] Filtered {skipped} malformed samples (missing <analysis>/<action> tags).")

    if not messages_list:
        print("[gen_worker] No valid samples after filtering. Returning empty batch.")
        return [], torch.tensor([], dtype=torch.float32), [], []

    prompts_text = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids


def wait_for_new_weight(iteration: int, attacker: AttackAgent) -> None:
    if iteration == 0:
        return

    trainer_marker = PROJECT_ROOT / "trainer.txt"
    generator_marker = PROJECT_ROOT / "generator.txt"

    while True:
        print("[gen_worker] waiting for new weight")
        with open(trainer_marker, "r") as file:
            content = file.read().strip()

        if content == str(iteration):
            reload_attacker_model(attacker, model_path=str(PROJECT_ROOT / "tmp"))
            print("[gen_worker] finish update_model")
            with open(generator_marker, "w") as file:
                file.write(f"{iteration}")
            return

        time.sleep(30)


def upload_training_batches(
    prompt_inputs: list[str],
    rewards: torch.Tensor,
    answers: list[str],
    ans_token_ids: list[torch.Tensor],
    tokenizer: AutoTokenizer,
    attacker: AttackAgent,
    gen_logps_sp: SamplingParams,
    ref_server: str,
) -> None:
    prompt_inputs, rewards, answers = zip(
        *sorted(zip(prompt_inputs, rewards, answers), key=lambda x: len(x[0]))
    )
    rewards = torch.stack(rewards)

    for batch_start in range(0, len(answers), TRAIN_BATCH_SIZE):
        sub_prompt_inputs = prompt_inputs[batch_start : batch_start + TRAIN_BATCH_SIZE]
        sub_prompt_ids = tokenizer(
            sub_prompt_inputs,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )["input_ids"]

        prompt_length = sub_prompt_ids.shape[1]
        if prompt_length > MAX_TOKEN_LEN:
            prompt_length = MAX_TOKEN_LEN
            sub_prompt_ids = sub_prompt_ids[:, -MAX_TOKEN_LEN:]

        sub_rewards = rewards[batch_start : batch_start + TRAIN_BATCH_SIZE]
        sub_ans_ids = ans_token_ids[batch_start : batch_start + TRAIN_BATCH_SIZE]

        tensor_list = [torch.tensor(token_ids) for token_ids in sub_ans_ids]
        output_ids = pad_sequence(
            tensor_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        merged_ids = torch.cat([sub_prompt_ids, output_ids], dim=1)

        data = [
            json.dumps({"plen": prompt_length}).encode(),
            tensor_to_bytes(merged_ids),
            tensor_to_bytes(sub_rewards),
        ]

        if COMPUTE_GEN_LOGPS:
            # Cap total sequence length to avoid OOM: vLLM materializes
            # (seq_len × vocab_size) float32 logits for prompt_logprobs.
            MAX_LOGPS_SEQ_LEN = 2048
            completion_len = output_ids.shape[1]
            keep_prompt = max(0, MAX_LOGPS_SEQ_LEN - completion_len)
            if sub_prompt_ids.shape[1] > keep_prompt:
                logps_ids = torch.cat(
                    [sub_prompt_ids[:, -keep_prompt:], output_ids], dim=1
                )
                logps_prompt_len = keep_prompt
            else:
                logps_ids = merged_ids
                logps_prompt_len = prompt_length

            completions = tokenizer.batch_decode(logps_ids)
            outputs = attacker.model.generate(
                completions,
                sampling_params=gen_logps_sp,
                use_tqdm=False,
            )
            prompt_logprobs = [item.prompt_logprobs[logps_prompt_len:] for item in outputs]
            gen_logps = torch.tensor(
                [[list(x.values())[0].logprob for x in item] for item in prompt_logprobs]
            ).clamp(min=-100)
            data.append(tensor_to_bytes(gen_logps))

        xdata = make_bytes_list(data)
        requests.post(
            f"{ref_server}/upload",
            data=xdata,
            proxies={"http": None, "https": None},
        )


def run_worker(
    actor_rank: int,
    num_actor_workers: int,
    physical_gpu_id: int,
    ref_server: str,
    use_remote_judge: bool,
    model_path: str = "",
    rollout_iter: int = 0,
    trees_per_worker: int = 5,
) -> None:
    try:
        print(
            f"[gen_worker] actor_rank={actor_rank}/{num_actor_workers}, "
            f"physical_gpu_id={physical_gpu_id}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"use_remote_judge={use_remote_judge}, "
            f"rollout_iter={rollout_iter}, trees_per_worker={trees_per_worker}, "
            f"model_path={model_path or '(default)'}"
        )
        torch.cuda.set_device(0)

        tokenizer, prompt_seeds, gen_logps_sp, generator, _ = build_generator(
            ref_server_url=ref_server,
            use_remote_judge=use_remote_judge,
            model_path=model_path,
        )
        attacker = generator.attacker
        worker_label = f"actor{actor_rank}_gpu{physical_gpu_id}"

        # Each worker is responsible for exactly one contiguous slice of seeds.
        # Seeds are partitioned across both iterations and workers:
        #   global_batch = rollout_iter * num_actor_workers + actor_rank
        #   seeds[global_batch * trees_per_worker : (global_batch+1) * trees_per_worker]
        global_batch = rollout_iter * num_actor_workers + actor_rank
        start = global_batch * trees_per_worker
        seed_batch = prompt_seeds[start : start + trees_per_worker]

        if not seed_batch:
            print(
                f"[gen_worker] No seeds remaining for rollout_iter={rollout_iter}, "
                f"actor_rank={actor_rank} (start={start} >= {len(prompt_seeds)}). Exiting."
            )
            return

        print(
            f"[gen_worker] Generating {len(seed_batch)} trees "
            f"(prompt indices {start}–{start + len(seed_batch) - 1} of {len(prompt_seeds)})..."
        )

        try:
            prompt_inputs, rewards, answers, ans_token_ids = gen_samples(
                seed_batch,
                rollout_iter,
                generator,
                tokenizer,
                worker_label,
            )

            if not answers:
                print("[gen_worker] Empty batch, skipping upload.")
                return

            print("[gen_worker] Uploading batches...")
            upload_training_batches(
                prompt_inputs=prompt_inputs,
                rewards=rewards,
                answers=answers,
                ans_token_ids=ans_token_ids,
                tokenizer=tokenizer,
                attacker=attacker,
                gen_logps_sp=gen_logps_sp,
                ref_server=ref_server,
            )
            print("[gen_worker] Done.")
        except Exception:
            print("=" * 80)
            print("[ERROR] Exception in gen_worker!")
            traceback.print_exc()
            print("=" * 80)
            sys.stdout.flush()
            raise
    except Exception:
        print("=" * 80)
        print("[ERROR] Exception in gen_worker setup!")
        traceback.print_exc()
        print("=" * 80)
        sys.stdout.flush()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-GPU rollout worker.")
    parser.add_argument("--actor-rank", type=int, default=_int_env("ACTOR_RANK", 0))
    parser.add_argument("--physical-gpu-id", type=int, default=_int_env("PHYSICAL_GPU_ID", 0))
    parser.add_argument(
        "--ref-server",
        type=str,
        default=_str_env("REF_SERVER_URL", DEFAULT_REF_SERVER),
    )
    parser.add_argument(
        "--use-remote-judge",
        action="store_true",
        default=_bool_env("USE_REMOTE_JUDGE", False),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=_str_env("ATTACKER_MODEL_PATH", ""),
        help="Path to the attacker model.  Overrides the model field in attacker_config.yaml.",
    )
    parser.add_argument(
        "--rollout-iter",
        type=int,
        default=_int_env("ROLLOUT_ITER", 0),
        help="Global rollout iteration index (set by orchestrator).",
    )
    parser.add_argument(
        "--num-actor-workers",
        type=int,
        default=_int_env("NUM_ACTOR_WORKERS", 1),
        help="Total number of gen_worker processes in this rollout phase.",
    )
    parser.add_argument(
        "--trees-per-worker",
        type=int,
        default=_int_env("TREES_PER_WORKER", 5),
        help="Number of trees this worker generates per iteration (default: 5).",
    )
    args = parser.parse_args()

    run_worker(
        actor_rank=args.actor_rank,
        num_actor_workers=args.num_actor_workers,
        physical_gpu_id=args.physical_gpu_id,
        ref_server=args.ref_server,
        use_remote_judge=args.use_remote_judge,
        model_path=args.model_path,
        rollout_iter=args.rollout_iter,
        trees_per_worker=args.trees_per_worker,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
