#!/bin/bash
set -e

echo "--- [run.sh START] --- Preparing container environment..."

echo "[run.sh] Handling TensorRT Engines..."
SOURCE_ENGINE_DIR="/runpod-volume/engines"
LOCAL_ENGINE_DIR="/app/engines"

if [ -d "$SOURCE_ENGINE_DIR" ]; then
    if [ -d "$LOCAL_ENGINE_DIR" ]; then
        echo "[run.sh] Local engine directory already exists. Skipping."
    else
        echo "[run.sh] Copying engines from $SOURCE_ENGINE_DIR to $LOCAL_ENGINE_DIR via rsync..."
        mkdir -p "$LOCAL_ENGINE_DIR"
        rsync -ah --progress "$SOURCE_ENGINE_DIR/" "$LOCAL_ENGINE_DIR/"
        echo "[run.sh] Engine copy complete."
    fi
else
    echo "[run.sh] WARNING: Source engine directory $SOURCE_ENGINE_DIR not found."
fi

echo "--- [run.sh COMPLETE] --- Launching Python worker..."
exec python -u rp_handler.py