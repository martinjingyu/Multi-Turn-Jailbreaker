# Multi-GPU Smoke Test

This file records the first round of validation for the new phase-based multi-GPU layout.

Current goal:
- do not aim for full GRPO training yet
- first confirm process roles, GPU placement, and service connectivity
- make sure the new `topology -> orchestrator -> service/actor worker` path is wired correctly

## 1. What We Are Testing

Current intended layout:

- `GPU 0`: service worker
  - loads `ref model`
  - loads `LlamaGuard`
  - serves ref logprobs
  - serves remote judge requests
- `GPU 1..N-1`: actor workers
  - load attacker
  - load target
  - generate rollout data
  - call the service worker for remote judging

For now we are only doing a smoke test, not a full end-to-end training run.

Success means:

- topology is printed correctly
- service worker starts on the expected GPU
- actor worker starts on its own single visible GPU
- remote judge endpoints are reachable
- actor no longer tries to switch GPUs inside one process

## 2. Files Involved

- [trainer/topology.py](/home/jhuang664/Multi-Turn-Jailbreaker/trainer/topology.py)
- [trainer/orchestrator.py](/home/jhuang664/Multi-Turn-Jailbreaker/trainer/orchestrator.py)
- [trainer/ref_server.py](/home/jhuang664/Multi-Turn-Jailbreaker/trainer/ref_server.py)
- [trainer/gen_worker.py](/home/jhuang664/Multi-Turn-Jailbreaker/trainer/gen_worker.py)
- [trainer/remote_evaluator.py](/home/jhuang664/Multi-Turn-Jailbreaker/trainer/remote_evaluator.py)

## 3. Before You Start

Work from the project root:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
```

Recommended environment variables:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If you are inside a cluster job, also make sure the GPUs you want are actually visible to the job.

## 4. Step 1: Dry Run the Topology

Check that process planning looks right before launching anything:

```bash
python trainer/orchestrator.py --gpus 0,1 --dry-run
```

Expected result:

- `service_gpu=0`
- `actor_gpus=[1]`
- one `service` process
- one `rollout-0` process

You can also test 4-GPU planning:

```bash
python trainer/orchestrator.py --gpus 0,1,2,3 --dry-run
```

Expected result:

- `service_gpu=0`
- `actor_gpus=[1, 2, 3]`
- one service process
- three rollout processes

## 5. Step 2: Start the Service Worker Alone

In terminal A:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
CUDA_VISIBLE_DEVICES=0 ENABLE_LLAMA_GUARD=1 REF_SERVER_PORT=59875 python trainer/ref_server.py
```

What this should do:

- lock the process to physical GPU 0
- start the ref service
- start the judge service
- listen on port `59875`

If it fails here, do not move on yet.

## 6. Step 3: Check the Health Endpoint

In terminal B:

```bash
curl http://127.0.0.1:59875/health
```

Expected output should include fields like:

- `"status": "ok"`
- `"enable_judge": true`
- `"cuda_visible_devices": "0"`

If this endpoint is not reachable:

- the service process may have crashed
- the port may already be in use
- the process may still be loading models

## 7. Step 4: Start One Actor Worker Alone

In terminal C:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
CUDA_VISIBLE_DEVICES=1 ACTOR_RANK=0 PHYSICAL_GPU_ID=1 REF_SERVER_URL=http://127.0.0.1:59875 USE_REMOTE_JUDGE=1 python trainer/gen_worker.py
```

What this should confirm:

- the actor prints its rank and GPU assignment
- the actor uses only one visible GPU
- the actor does not try to reassign CUDA to another physical GPU
- the actor uses remote judge mode

Useful log patterns:

- good:
  - `actor_rank=0`
  - `CUDA_VISIBLE_DEVICES=1`
  - `use_remote_judge=True`
- suspicious:
  - code tries to set `CUDA_VISIBLE_DEVICES` again during runtime
  - code tries to use `GPU 3` or another hard-coded GPU

## 8. Step 5: Start Through Orchestrator

After steps 2-4 look healthy, stop them and try the orchestrator:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
python trainer/orchestrator.py --gpus 0,1
```

Expected behavior:

- orchestrator starts one service process on GPU 0
- orchestrator starts one actor process on GPU 1
- logs from both children begin to appear

For planning only, use:

```bash
python trainer/orchestrator.py --gpus 0,1 --dry-run
```

## 9. Success Criteria for This Stage

This stage is successful even if training is not complete yet.

We only need to confirm most of the following:

- `orchestrator --dry-run` prints the expected layout
- `ref_server.py` starts and responds to `/health`
- `gen_worker.py` starts as a single-GPU actor
- actor uses remote judge mode without immediate request errors
- no more cross-GPU switching inside one actor process

## 10. Common Failure Modes

### A. Model load fails immediately

Possible reasons:

- not enough GPU memory
- model not downloaded yet
- incompatible environment or package versions

What to check:

- exact traceback
- which model was loading
- whether failure happened in service or actor

### B. `/health` does not respond

Possible reasons:

- service process crashed
- service is still loading large models
- wrong port
- another process already occupies port `59875`

Useful check:

```bash
lsof -i :59875
```

### C. Actor cannot connect to judge service

Possible reasons:

- service is not running
- wrong `REF_SERVER_URL`
- `/judge/upload` or `/judge/get` path not reachable

Check:

- actor logs
- service logs
- the value of `REF_SERVER_URL`

### D. Actor still behaves like a multi-GPU process

Possible reasons:

- some old hard-coded device logic remains
- a model class internally overrides device placement

Check:

- actor startup log
- any line mentioning hard-coded GPU ids
- whether `CUDA_VISIBLE_DEVICES` is set before process start

### E. Process blocks waiting on `trainer.txt` or `generator.txt`

This is expected for now in some paths.

Reason:

- old checkpoint handoff logic is still present
- we have not rewritten the rollout/train phase handshake yet

This does not necessarily mean the new GPU topology is broken.

## 11. Recommended Order When Debugging

Always debug in this order:

1. `topology.py` dry run
2. `ref_server.py` alone
3. `/health` endpoint
4. `gen_worker.py` alone
5. `orchestrator.py`

This keeps the problem small and prevents confusing multi-process failures.

## 12. Next Planned Refactor

After this smoke test is stable, the next big step is:

- separate rollout phase and train phase more clearly
- stop relying on `trainer.txt` and `generator.txt`
- refactor `grpo_vllm_one.py` into a dedicated training-phase script

## 13. Training Phase Smoke Test

We now also have a separate training-phase entrypoint:

- [trainer/grpo_train_phase.py](/home/jhuang664/Multi-Turn-Jailbreaker/trainer/grpo_train_phase.py)

And `orchestrator.py` now supports:

- `--phase rollout`
- `--phase train`

Important:

- the trainer is now meant to be a pure training process
- it should not start rollout workers by itself
- service GPU is still separated from training GPUs

## 14. Step 6: Dry Run the Training Plan

For 2 GPUs:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
python trainer/orchestrator.py --gpus 0,1 --phase train --dry-run
```

Expected result:

- `GPU 0` is the service worker
- `GPU 1` is the training worker group
- trainer command uses `deepspeed --num_gpus 1`

For 4 GPUs:

```bash
python trainer/orchestrator.py --gpus 0,1,2,3 --phase train --dry-run
```

Expected result:

- `GPU 0` is the service worker
- `GPU 1,2,3` are the training GPUs
- trainer command uses `deepspeed --num_gpus 3`

This is the key thing to verify:

- service GPU is not included in the deepspeed group

## 15. Step 7: Start Service for Training

In terminal A:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
CUDA_VISIBLE_DEVICES=0 ENABLE_LLAMA_GUARD=1 REF_SERVER_PORT=59875 python trainer/ref_server.py
```

This is the same service process used by rollout, but now the trainer will pull batches from it.

Health check:

```bash
curl http://127.0.0.1:59875/health
```

## 16. Step 8: Start the Trainer Alone

In terminal B, for a 2-GPU setup where training should use only physical GPU 1:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
CUDA_VISIBLE_DEVICES=1 deepspeed --num_gpus 1 trainer/grpo_train_phase.py --model-path MartinJYHuang/JA-v1 --ref-server http://127.0.0.1:59875
```

For a 4-GPU setup where training should use physical GPUs 1,2,3:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
CUDA_VISIBLE_DEVICES=1,2,3 deepspeed --num_gpus 3 trainer/grpo_train_phase.py --model-path MartinJYHuang/JA-v1 --ref-server http://127.0.0.1:59875
```

What this should confirm:

- the trainer can initialize DeepSpeed
- only actor GPUs are used for training
- the trainer can poll `ref_server` without startup errors
- no rollout worker is started from inside the trainer

At this stage it is okay if the trainer prints waiting messages such as:

- `waiting for batch`
- `No new batches arrived within idle timeout`

This simply means the training process is up, but no rollout batches have been produced yet.

## 17. Step 9: Start Training Through Orchestrator

Once the separate service and trainer both look healthy, try:

```bash
cd /home/jhuang664/Multi-Turn-Jailbreaker
python trainer/orchestrator.py --gpus 0,1 --phase train
```

For a larger setup:

```bash
python trainer/orchestrator.py --gpus 0,1,2,3 --phase train
```

Expected behavior:

- orchestrator starts the service on the service GPU
- orchestrator starts DeepSpeed on actor GPUs only
- trainer no longer launches `gen_worker`

## 18. Training Phase Success Criteria

This stage is successful if most of the following are true:

- `--phase train --dry-run` prints the expected GPU split
- service starts on the dedicated service GPU
- trainer starts on actor GPUs only
- DeepSpeed initializes successfully
- trainer can poll `/get` from the service worker
- trainer does not depend on `trainer.txt` or `generator.txt`

## 19. Training Phase Failure Modes

### A. DeepSpeed starts on the wrong GPUs

Check:

- `CUDA_VISIBLE_DEVICES`
- `--num_gpus`
- whether the service GPU accidentally appears in the trainer launch command

### B. Trainer starts but never sees batches

This may be normal during isolated testing.

Reason:

- rollout is not producing batches yet
- service queue is empty

This is still a useful test because it confirms the training process can start and wait correctly.

### C. Trainer still behaves like the old combined script

This should not happen with:

- [trainer/grpo_train_phase.py](/home/jhuang664/Multi-Turn-Jailbreaker/trainer/grpo_train_phase.py)

If you see behavior like:

- trainer spawning generation workers
- trainer reading `generator.txt`
- trainer writing `trainer.txt`

then you are probably still running the old:

- [trainer/grpo_vllm_one.py](/home/jhuang664/Multi-Turn-Jailbreaker/trainer/grpo_vllm_one.py)

### D. Service and trainer both start, but nothing progresses

That usually means the phase split is working, but no rollout data has been produced yet.

This is acceptable during a smoke test.

The purpose of this phase is to confirm:

- process boundaries are clean
- GPU assignment is clean
- DeepSpeed launch is clean
