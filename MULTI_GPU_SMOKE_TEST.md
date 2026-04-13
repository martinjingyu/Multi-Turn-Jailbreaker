# Multi-GPU Smoke Test & GRPO Training Guide

This file covers GPU topology validation, smoke testing, and running the full GRPO training loop.

## Architecture Overview

The training system uses a **phase-based** layout with three roles:

| Role | GPU | Responsibility |
|------|-----|----------------|
| `service` | GPU 0 | Hosts ref model + LlamaGuard; serves ref logprobs and remote judge requests |
| `actor` | GPU 1..N | Rollout workers: load attacker + target, generate tree data, upload batches to service |
| `trainer` | GPU 1..N | DeepSpeed training group; pulls processed batches from service, runs GRPO updates |

The actor GPUs and trainer GPUs are the **same physical GPUs** used in different phases.

### Full GRPO Loop (`--phase loop`)

```
for each iteration:
    1. Rollout phase  — gen_workers generate tree data → upload to ref_server
    2. Train phase    — trainer drains ref_server queue → GRPO gradient updates → save checkpoint
    3. Update model   — latest checkpoint becomes the attacker for the next rollout
```

The ref_server is started **once** and kept alive across all iterations.

---

## Files Involved

- [trainer/topology.py](trainer/topology.py)
- [trainer/orchestrator.py](trainer/orchestrator.py)
- [trainer/ref_server.py](trainer/ref_server.py)
- [trainer/gen_worker.py](trainer/gen_worker.py)
- [trainer/grpo_train_phase.py](trainer/grpo_train_phase.py)
- [trainer/remote_evaluator.py](trainer/remote_evaluator.py)

---

## Before You Start

Work from the project root:

```bash
cd /home/jingyuh/Multi-Turn-Jailbreaker
```

Recommended environment variables:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Part 1: Topology Smoke Test

### Step 1: Dry-run the topology

Check that process planning looks right before launching anything:

```bash
# 2-GPU setup
python trainer/orchestrator.py --gpus 0,1 --phase rollout --dry-run

# 4-GPU setup
python trainer/orchestrator.py --gpus 0,1,2,3 --phase rollout --dry-run
```

Expected for 4 GPUs:

- `service_gpu=0`
- `actor_gpus=[1, 2, 3]`
- one `service` process
- three `rollout-*` processes

### Step 2: Start the service worker alone

In terminal A:

```bash
CUDA_VISIBLE_DEVICES=0 ENABLE_LLAMA_GUARD=1 REF_SERVER_PORT=59875 python trainer/ref_server.py
```

### Step 3: Check the health endpoint

In terminal B:

```bash
curl http://127.0.0.1:59875/health
```

Expected response fields:

- `"status": "ok"`
- `"enable_judge": true`
- `"cuda_visible_devices": "0"`

If this endpoint is not reachable, the service process may have crashed, the port may be in use, or models are still loading.

### Step 4: Start one actor worker alone

In terminal C:

```bash
CUDA_VISIBLE_DEVICES=1 ACTOR_RANK=0 PHYSICAL_GPU_ID=1 \
  REF_SERVER_URL=http://127.0.0.1:59875 USE_REMOTE_JUDGE=1 \
  python trainer/gen_worker.py
```

Useful log patterns to look for:

- **Good**: `actor_rank=0`, `CUDA_VISIBLE_DEVICES=1`, `use_remote_judge=True`
- **Suspicious**: anything that re-assigns `CUDA_VISIBLE_DEVICES` at runtime, or references a hard-coded GPU id

To use a specific checkpoint (e.g. after training):

```bash
CUDA_VISIBLE_DEVICES=1 ACTOR_RANK=0 PHYSICAL_GPU_ID=1 \
  REF_SERVER_URL=http://127.0.0.1:59875 USE_REMOTE_JUDGE=1 \
  ATTACKER_MODEL_PATH=checkpoints/step_49 \
  python trainer/gen_worker.py
```

### Step 5: Start rollout through orchestrator

Once steps 2–4 look healthy, stop them and try:

```bash
python trainer/orchestrator.py --gpus 0,1 --phase rollout
```

### Rollout smoke test success criteria

- `--dry-run` prints the expected layout
- `ref_server.py` starts and responds to `/health`
- `gen_worker.py` starts as a single-GPU actor without cross-GPU switching
- Actor uses remote judge mode without immediate errors

---

## Part 2: Training Phase Smoke Test

### Step 6: Dry-run the training plan

```bash
# 2-GPU
python trainer/orchestrator.py --gpus 0,1 --phase train --dry-run

# 4-GPU
python trainer/orchestrator.py --gpus 0,1,2,3 --phase train --dry-run
```

Expected for 4 GPUs:

- `GPU 0` — service worker
- `GPU 1,2,3` — training GPUs (`deepspeed --num_gpus 3`)
- **The service GPU must not appear in the deepspeed group**

### Step 7: Start service for training

In terminal A:

```bash
CUDA_VISIBLE_DEVICES=0 ENABLE_LLAMA_GUARD=1 REF_SERVER_PORT=59875 python trainer/ref_server.py
```

### Step 8: Start the trainer alone

For a 2-GPU setup (training on GPU 1 only):

```bash
CUDA_VISIBLE_DEVICES=1 deepspeed --num_gpus 1 trainer/grpo_train_phase.py \
  --model-path MartinJYHuang/JA-v1 \
  --ref-server http://127.0.0.1:59875
```

For a 4-GPU setup (training on GPUs 1,2,3):

```bash
CUDA_VISIBLE_DEVICES=1,2,3 deepspeed --num_gpus 3 trainer/grpo_train_phase.py \
  --model-path MartinJYHuang/JA-v1 \
  --ref-server http://127.0.0.1:59875
```

It is normal to see `waiting for batch...` messages here — the trainer is up but no rollout data has been produced yet.

### Step 9: Start training through orchestrator

```bash
python trainer/orchestrator.py --gpus 0,1,2,3 --phase train \
  --model-path MartinJYHuang/JA-v1
```

### Training phase success criteria

- `--phase train --dry-run` shows the correct GPU split
- Service starts on the dedicated service GPU
- Trainer starts on actor GPUs only
- DeepSpeed initializes successfully
- Trainer can poll `/get` from the service worker
- No dependency on `trainer.txt` or `generator.txt`

---

## Part 3: Full GRPO Loop

This is the main training entrypoint. It runs N rollout → train iterations automatically, updating the attacker model with the latest checkpoint before each rollout.

### Step 10: Dry-run the loop

```bash
python trainer/orchestrator.py \
  --gpus 0,1,2,3 \
  --phase loop \
  --iterations 5 \
  --model-path MartinJYHuang/JA-v1 \
  --save-dir checkpoints/ \
  --dry-run
```

### Step 11: Launch full training

```bash
python trainer/orchestrator.py \
  --gpus 0,1,2,3 \
  --phase loop \
  --iterations 10 \
  --model-path MartinJYHuang/JA-v1 \
  --save-dir checkpoints/ \
  --all-steps 500 \
  --save-every 50 \
  --lr 1e-6 \
  --beta 0.03
```

Key parameters:

| Flag | Default | Meaning |
|------|---------|---------|
| `--iterations` | `1` | Number of rollout→train rounds |
| `--all-steps` | `500` | Max gradient steps per train phase |
| `--idle-seconds` | `600` | Trainer exits after this many seconds with no new batches |
| `--save-every` | `50` | Save checkpoint every N steps |
| `--max-save-total` | `5` | Keep only the latest N checkpoints |
| `--beta` | `0.03` | KL penalty weight |
| `--clip-param` | `0.2` | PPO clipping epsilon |

### What happens each iteration

1. **Rollout**: All gen_workers start with `ATTACKER_MODEL_PATH` set to the current model, generate trees for all seed prompts, upload batches to the ref_server, then exit.
2. **Train**: Trainer drains the ref_server queue, runs GRPO updates, saves checkpoints, exits when the queue is idle for `--idle-seconds`.
3. **Checkpoint update**: Orchestrator scans `--save-dir` for the latest `step_*` checkpoint and passes it as the attacker model to the next rollout.

---

## Recommended Debugging Order

Always narrow down the problem in this order:

1. `topology.py` dry-run
2. `ref_server.py` alone + `/health` endpoint
3. `gen_worker.py` alone
4. `orchestrator.py --phase rollout`
5. `grpo_train_phase.py` alone
6. `orchestrator.py --phase train`
7. `orchestrator.py --phase loop --iterations 1`

---

## Common Failure Modes

### A. Model load fails immediately

- Check the exact traceback and which model was loading
- Verify GPU memory is sufficient
- Confirm the model has been downloaded

### B. `/health` does not respond

- Service may still be loading models (LlamaGuard takes time)
- Port may already be in use: `lsof -i :59875`
- Check service process logs for crash traces

### C. Actor cannot connect to judge service

- Confirm `REF_SERVER_URL` is correct
- Check that the service is running and `/health` returns `"enable_judge": true`
- Verify `/judge/upload` and `/judge/get` are reachable

### D. Actor switches GPUs at runtime

- Check actor startup logs for any line that re-sets `CUDA_VISIBLE_DEVICES`
- `AttackAgent` in non-vLLM mode contains a hard-coded `CUDA_VISIBLE_DEVICES='1'` assignment — this only triggers when `config.vllm = False`; ensure `attacker_config.yaml` has `vllm: True`

### E. DeepSpeed starts on the wrong GPUs

- Check `CUDA_VISIBLE_DEVICES` and `--num_gpus` in the trainer launch command
- The `--dry-run` output shows exactly what command is used

### F. Trainer never sees batches

During isolated testing this is expected — rollout workers have not uploaded anything yet.  
During a full loop run, if the trainer idles out immediately, check that gen_workers completed successfully before the train phase started.

### G. `trainer.txt` / `generator.txt` blocking

These files were used by the old monolithic script `grpo_vllm_one.py`.  
They are **not used** by the current phase-based system.  
If you see code waiting on these files, you are running the wrong entry point.
