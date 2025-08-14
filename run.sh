#!/bin/bash
set -e

echo "--- [run.sh START] ---"

echo "[run.sh] Step 1: Handling TensorRT Engines..."
SOURCE_ENGINE_DIR="/runpod-volume/engines"
LOCAL_ENGINE_DIR="/app/engines"

if [ -d "$SOURCE_ENGINE_DIR" ]; then
    if [ -d "$LOCAL_ENGINE_DIR" ]; then
        echo "[run.sh] Local engine directory already exists. Skipping rsync."
    else
        echo "[run.sh] rsync'ing engines from $SOURCE_ENGINE_DIR to $LOCAL_ENGINE_DIR..."
        mkdir -p "$LOCAL_ENGINE_DIR"
        rsync -ah --progress "$SOURCE_ENGINE_DIR/" "$LOCAL_ENGINE_DIR/"
        echo "[run.sh] Engine rsync complete."
    fi
else
    echo "[run.sh] WARNING: Source engine directory $SOURCE_ENGINE_DIR not found."
fi

echo "[run.sh] Step 2: Handling Base Model files..."
SOURCE_MODEL_DIR="/runpod-volume/models/flux-dev-kontext"
LOCAL_MODEL_DIR="/app/models/flux-dev-kontext"

if [ -d "$SOURCE_MODEL_DIR" ]; then
    if [ -d "$LOCAL_MODEL_DIR" ]; then
        echo "[run.sh] Local model directory already exists. Skipping rsync."
    else
        echo "[run.sh] rsync'ing base model from $SOURCE_MODEL_DIR to $LOCAL_MODEL_DIR..."
        mkdir -p "$LOCAL_MODEL_DIR"
        rsync -ah --progress "$SOURCE_MODEL_DIR/" "$LOCAL_MODEL_DIR/"
        echo "[run.sh] Base model rsync complete."
    fi
else
    echo "[run.sh] Base model not found on volume. Python handler will download if needed."
fi

echo "[run.sh] Step 3: All setup complete. Launching Python worker..."
exec python -u rp_handler.py