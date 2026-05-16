#!/bin/bash
set -e

cd /app

echo "Starting fastapi • • •"

# If first argument is celery, run Celery
if [ "$1" = "celery" ]; then
    shift
    exec celery "$@"
else
    # default to CMD
    exec "$@"
fi
