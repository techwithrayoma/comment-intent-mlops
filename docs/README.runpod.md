# RunPod GPU Worker

GPU training worker for comment-intent fine-tuning using LLaMA-Factory + Qwen2.5-1.5B.

---

## Pod Specs

| Resource | Value |
|---|---|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| vCPU | 16 (AMD EPYC 75F3) |
| RAM | 62 GB |
| Volume | `/workspace` (persistent across stops) |

---

## First-Time Setup

### 1. Add SSH key to RunPod

```bash
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
# Paste into RunPod pod → SSH keys
```

### 2. Connect

```bash
ssh root@<pod-ip> -p <port>
```

### 3. Clone repo

```bash
cd ~
git clone https://github.com/your-org/comment-intent-mlops.git
cd comment-intent-mlops
```

### 4. Add environment file

Create `src/.env`:

```dotenv

```

### 5. Run the setup script

```bash
bash src/scripts/setup_gpu_worker.sh
```

This installs all requirements, sets up LLaMA-Factory, and starts the Celery worker. You're done.


## Triggering Training

You never trigger training from RunPod. The worker listens — you trigger from your laptop:

```bash
curl -X POST "http://localhost:8000/training/run?version=v1&model_name=comment-intent"
```

Flow: Laptop → CloudAMQP → RunPod GPU worker picks up task automatically.

---

## What Happens During Training

1. Worker receives `task_run_model_training` from `gpu` queue
2. Downloads `train.jsonl`, `val.jsonl`, `dataset_info.json` from S3
3. Downloads `final_config.yaml` from S3
4. Runs `llamafactory-cli train` with Qwen2.5-1.5B-Instruct + LoRA
5. Logs params, metrics, and costs to MLflow
6. Uploads adapter files to `s3://your-bucket/training/model_output/`
7. Runs evaluation against benchmark and saves metrics to DB

Approximate time: ~6 minutes on RTX 4090 for ~900 samples, 5 epochs.
Approximate cost: $0.50–1.00 per run.

---

## Cost Management

| Action | Effect |
|---|---|
| **Stop** pod | Preserves `/workspace` and `~/` — pay for storage only |
| **Terminate** pod | Deletes everything — clean slate |

Always **stop**, never terminate unless you need a reset.

---

## Verify GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 4090
```

---

## Troubleshooting

**Worker not picking up tasks**
```bash
python -c "from src.app.celery_app import celery_app; print(celery_app.control.inspect().ping())"
```

**CUDA not available** — restart the pod, then rerun the script.

**MLflow not reachable** — make sure ngrok is running on your laptop before triggering training, and update `MLFLOW_TRACKING_URI` in `src/.env` with the latest ngrok URL.