# Docker workflow for Qwen3-ASR fine-tuning

A persistent, GPU-enabled training container. You can SSH in, start training, disconnect, come back hours later — training keeps running.

## Why Docker (vs native uv)

The host has three friction points this image resolves:

| Host problem | Inside this image |
|---|---|
| No ffmpeg, no sudo → PyAV `.libs/` symlink hack for `torchcodec` | `apt install ffmpeg` — no symlinks |
| `nvcc 13.0` vs `torch cu12.8` mismatch → flash-attn build fails | `nvidia/cuda:12.8.0-cudnn-devel` base → matching nvcc |
| Other users' PIDs eating GPU budget | Container is pinned to one NVIDIA device (`--device nvidia.com/gpu=0`) — `--gpus all` is avoided because the daemon's multi-vendor CDI scan errors on this host |
| Default `nofile=1024` causes `Too many open files` mid-eval (torchcodec leaks fds) | `--ulimit nofile=65535:65535` set by `up.sh` |

## One-time setup

```bash
bash scripts/docker/build.sh
```
~20–30 min. Most time is flash-attn compilation. Cached after first build.

Verify everything landed:
```bash
docker run --rm --device nvidia.com/gpu=0 syntts-asr:latest \
  python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"
```

## Daily usage

```bash
# 1. Start the container (stays alive, survives your disconnects)
bash scripts/docker/up.sh

# 2. Launch training inside a tmux session
bash scripts/docker/train.sh pt        # picks scripts/train/qwen_pt.sh
bash scripts/docker/train.sh nl        # picks scripts/train/qwen_nl.sh
#    You're now attached to tmux watching training output.
#    Detach with Ctrl+b then d. Training keeps running.

# 3. Disconnect your SSH / close laptop — training keeps going.

# 4. Later — check progress
bash scripts/docker/attach.sh         # re-attach to tmux (full interactive view)
bash scripts/docker/logs.sh           # or just tail the log file
bash scripts/docker/shell.sh          # drop into a bash shell in the container
                                      # (e.g. `nvidia-smi`, `ls results/`, etc.)

# 5. When training is done and you've pushed to HF
bash scripts/docker/down.sh           # stop + remove container (caches + image remain)
```

**The `--restart unless-stopped` policy** means the container auto-restarts on Docker daemon restart or host reboot. If you explicitly `down.sh` it, it stays down.

## What's mounted from the host

Read-write bind mounts — changes are visible on both sides:

| Host path | Container path | Why |
|---|---|---|
| `./src` | `/workspace/src` | Edit code on host, see changes in container without rebuild |
| `./scripts` | `/workspace/scripts` | Same |
| `./prompts` | `/workspace/prompts` | Same |
| `./results` | `/workspace/results` | Checkpoints + `test_results.json` persisted on host |
| `./pyproject.toml` + `./uv.lock` | `/workspace/*` (read-only) | Reference only; venv lives in `/opt/venv` inside image |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | Don't re-download CV22 (~6 GB) or base model (~5 GB) each run |
| `~/.cache/syntts_asr` | `/root/.cache/syntts_asr` | Extracted CV22 audio cache |
| `./.env` | container env | HF_API_KEY, WANDB_API_KEY, OPENAI_API_KEY |

Notably **NOT** mounted: `.venv/` stays in the image at `/opt/venv` so the host's host-built venv can't mask the Docker-built one. `results/wandb/` and `results/checkpoint-*/` persist to host through the `results` mount.

## Inspecting things while training runs

```bash
# GPU usage
docker exec qwen-training nvidia-smi

# Current training step / live progress
bash scripts/docker/attach.sh

# Python env inside
docker exec qwen-training python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"
```

## Common operations

```bash
# Override default GPU (share with other workloads)
GPU_DEVICE=nvidia.com/gpu=1 bash scripts/docker/up.sh

# Use a different container name (run multiple experiments in parallel)
CONTAINER_NAME=qwen-experiment-2 bash scripts/docker/up.sh

# Use a different training script (default = qwen_${LANG_KEY}.sh)
TRAIN_SCRIPT=scripts/train/some_other.sh bash scripts/docker/train.sh

# Force rebuild after changing the Dockerfile
docker build --no-cache -t syntts-asr:latest .
```

## Tradeoffs vs native uv

- **Pro**: everything just works — ffmpeg, CUDA 12.8, flash-attn are all matched.
- **Pro**: reproducible across machines. `docker build` on a "new computer" gets the same env.
- **Pro**: training isolated from the host's package conflicts and other users' processes.
- **Con**: ~15–20 GB image on disk.
- **Con**: first build is slow (~25 min) due to flash-attn compile.
- **Con**: source edits via bind-mount, not `uv run python ...` from host. You must go through `bash scripts/docker/shell.sh` (or the other helpers) to run anything.

Use native uv for quick ad-hoc scripts and Jupyter. Use Docker for long training runs that need to survive disconnects + require flash-attn.
