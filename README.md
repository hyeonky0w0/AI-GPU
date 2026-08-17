# AI-GPU — Git push → RunPod GPU → GitHub Artifact

## What this repository does

`feature/**` or `experiment/**` push
→ GitHub Actions
→ package the current branch's `src/` + `config/`
→ RunPod Queue endpoint
→ GPU executes the bundled `train.py`
→ `metrics.json`, `train.log`, `error.log`
→ GitHub Actions Artifact.

The RunPod worker image is fixed. Experiment branches do **not** require rebuilding the worker image because the branch's experiment code is bundled into each job.

## One-time RunPod setup

The endpoint must use the Docker image built from this repository's `Dockerfile`.

Use a Queue-based endpoint. The worker image is based on RunPod's official `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image. It includes PyTorch 2.8.0 and CUDA 12.8.1; the image also configures pip for system-package installation.

## GitHub Secrets

Repository → Settings → Secrets and variables → Actions:

- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`

Never commit the API key.

## Test directly in RunPod

Use Requests → Run in browser:

```json
{
  "input": {
    "experiment_id": "manual-test",
    "branch": "manual"
  }
}
```

The worker will execute the built-in `/app/src/train.py`.

## Branch experiments

Push:

```text
feature/mlp
experiment/catboost
experiment/feature-engineering
```

Each job receives the exact `src/` and `config/` from that commit. The endpoint itself does not need to be rebuilt for normal experiment-code changes.

Only rebuild the RunPod image when changing:

- Dockerfile
- requirements.txt
- handler.py
- system-level dependencies
- Python/runtime dependencies

## Result files

Each successful job produces:

- `metrics.json`
- `train.log`
- `error.log`

They are uploaded as a GitHub Actions Artifact for 30 days.

## Payload limits

RunPod's asynchronous `/run` operation has a 10 MB payload limit. The repository intentionally keeps the experiment bundle below 6 MiB to leave room for JSON/base64 overhead. Large datasets, model checkpoints, or huge prediction files should use RunPod Network Volumes or object storage instead of the request/response payload.
