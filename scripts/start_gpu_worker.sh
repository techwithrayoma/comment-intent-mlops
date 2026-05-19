#!/bin/bash
set -e

echo "=== Installing core requirements ==="
pip install --no-cache-dir --ignore-installed blinker
pip install --no-cache-dir -r src/requirements.txt

if [ ! -d "LLaMA-Factory" ]; then
    echo "=== Cloning LLaMA-Factory ==="
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi

echo "=== Installing LLaMA-Factory ==="
cd LLaMA-Factory && pip install -e . && cd ..

echo "=== Installing GPU requirements ==="
pip install --no-cache-dir -r src/requirements.gpu.txt

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