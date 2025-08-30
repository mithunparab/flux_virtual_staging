#!/bin/bash
set -e

echo "--- [run.sh START] --- Preparing the container environment..."

echo "[run.sh] Models are already included in the image. Skipping copy."

echo "--- [run.sh COMPLETE] --- Launching Python worker..."
exec python -u rp_handler.py