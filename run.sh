#!/bin/bash
set -e

echo "--- [run.sh] ---"
echo "Container started. Skipping file copy."
echo "Models and engines will be loaded directly from the network volume to reduce cold start time."
echo "Launching Python worker..."
exec python -u rp_handler.py