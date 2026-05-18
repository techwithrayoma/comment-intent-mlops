#!/bin/bash
set -e

echo "=== Installing core requirements ==="
pip install --no-cache-dir -r src/requirements.txt

echo "=== Installing GPU requirements ==="
pip install --no-cache-dir -r src/requirements.gpu.txt

if [ ! -d "LLaMA-Factory" ]; then
    echo "=== Installing LLaMA-Factory ==="
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory && pip install -e . && cd ..
else
    echo "=== LLaMA-Factory already installed — skipping ==="
fi

echo "=== Setting env ==="
set -a && source src/.env && set +a
export PYTHONPATH=/root/comment-intent-mlops  

echo "=== Starting GPU worker ==="
python -m celery -A src.app.celery_app:celery_app worker \
  --loglevel=info \
  --queues=gpu \
  --hostname=gpu-worker@%h \
  --concurrency=1 \
  --pool=solo