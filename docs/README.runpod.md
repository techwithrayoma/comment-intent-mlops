# 🖥️ RunPod GPU Worker Setup

This document covers how to set up and run the GPU training worker on RunPod.

---

## Pod Specs Used

| Resource | Value |
|----------|-------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| vCPU | 16 (AMD EPYC 75F3) |
| RAM | 62 GB |
| Container Disk | 20 GB |
| Volume | `/workspace` (persistent) |

---

## First-Time Setup

### 1. Add your SSH key to RunPod

On your laptop:
```bash
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
# Copy this output → paste into RunPod pod SSH keys
```

### 2. Connect to pod

```bash
ssh root@<your-pod-ip> -p <port>
```

### 3. Clone your repo

```bash
cd ~
git clone https://github.com/your-org/barq.git
cd barq
```

### 4. Install dependencies

```bash
# Core project requirements
pip install --no-cache-dir -r src/requirements.txt

# LLM fine-tuning stack
pip install --no-cache-dir \
  "transformers>=4.49.0,<=4.57.1,!=4.52.0,!=4.57.0" \
  "peft>=0.14.0,<=0.17.1" \
  "datasets>=2.16.0,<=4.0.0" \
  "accelerate>=1.3.0,<=1.11.0" \
  scikit-learn tqdm "pandas>=2.0.0" \
  bitsandbytes

# LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e . && cd ..
```

### 5. Set environment variables

```bash
set -a && source src/.env && set +a
export PYTHONPATH=/root/src
```

### 6. Start the GPU worker

```bash
python -m celery -A src.app.celery_app:celery_app worker \
  --loglevel=info \
  --queues=gpu \
  --hostname=gpu-worker@%h \
  --concurrency=1 \
  --pool=solo
```

---

## Subsequent Runs (Pod Already Set Up)

If you stopped the pod (not terminated), `/workspace` and `~/barq` are preserved.

```bash
cd ~/barq
git pull                          # pull latest code changes
set -a && source docker/env/.env.gpu_worker && set +a
export PYTHONPATH=/root/barq

python -m celery -A ladybug.app.celery_app:celery_app worker \
  --loglevel=info \
  --queues=gpu \
  --hostname=gpu-worker@%h \
  --concurrency=1 \
  --pool=solo
```

---

## Environment File: `.env.gpu_worker`

```dotenv
# Celery
CELERY_BROKER_URL=amqps://...@capybara.lmq.cloudamqp.com:5671/...
CELERY_RESULT_BACKEND=rediss://default:...@intimate-raptor-72645.upstash.io:6379/

# AWS S3
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name

# MLflow
MLFLOW_TRACKING_URI=https://your-ngrok-url.ngrok-free.dev

# WandB (disabled — using MLflow instead)
WANDB_DISABLED=true

# LLM
OPENAI_API_KEY=...
OPENAI_MODEL_ID=gpt-4o-mini
LLM_PROVIDER=openai
```

---

## How Training Is Triggered

You do NOT trigger training from RunPod. The GPU worker **listens** for tasks.

Training is triggered from your laptop:
```bash
# via FastAPI
curl http://localhost:8000/training/run?version=v1&model_name=ladybug

# or via Python
from ladybug.tasks.training_tasks import task_run_model_training
task_run_model_training.delay(version="v1", model_name="ladybug")
```

The task travels through CloudAMQP → RunPod GPU worker picks it up automatically.

---

## What Happens During Training

1. Worker receives `task_run_model_training` from the `gpu` queue
2. Downloads `train.jsonl`, `val.jsonl`, `dataset_info.json` from S3
3. Downloads `final_config.yaml` from S3
4. Runs `llamafactory-cli train` with Qwen2.5-1.5B-Instruct + LoRA
5. Logs params + metrics to MLflow
6. Uploads 78 adapter files to `s3://your-bucket/training/model_output/`

---

## Cost Saving

- **Stop** the pod when not training (preserves `/workspace`)
- **Terminate** only when you want a clean slate (deletes everything)
- Training takes ~6 minutes on RTX 4090 for 510 examples, 5 epochs
- Estimated cost: ~$0.50–1.00 per training run

---

## Verify GPU is Working

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output:
```
True NVIDIA GeForce RTX 4090
```