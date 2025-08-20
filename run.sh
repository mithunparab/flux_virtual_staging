set -e

echo "--- [run.sh START] --- Preparing container with local cache for faster model loading..."

NETWORK_VOLUME="/runpod-volume"
LOCAL_CACHE="/app/local_model_cache"

SOURCE_ENGINES="${NETWORK_VOLUME}/engines"
DEST_ENGINES="${LOCAL_CACHE}/engines"

if [ -d "$SOURCE_ENGINES" ]; then
    if [ -d "$DEST_ENGINES" ]; then
        echo "[Cache] Local engine cache at $DEST_ENGINES already exists. Skipping copy."
    else
        echo "[Cache] Caching engines from $SOURCE_ENGINES to local disk at $DEST_ENGINES..."
        mkdir -p "$DEST_ENGINES"
        rsync -ah --info=progress2 "$SOURCE_ENGINES/" "$DEST_ENGINES/"
        echo "[Cache] Engine caching complete."
    fi
else
    echo "[Cache] WARNING: Source engine directory $SOURCE_ENGINES not found on network volume."
fi

SOURCE_MODEL="${NETWORK_VOLUME}/models"
DEST_MODEL="${LOCAL_CACHE}/models"

if [ -d "$SOURCE_MODEL" ]; then
    if [ -d "$DEST_MODEL" ]; then
        echo "[Cache] Local base model cache at $DEST_MODEL already exists. Skipping copy."
    else
        echo "[Cache] Caching base model from $SOURCE_MODEL to local disk at $DEST_MODEL..."
        mkdir -p "$DEST_MODEL"
        rsync -ah --info=progress2 "$SOURCE_MODEL/" "$DEST_MODEL/"
        echo "[Cache] Base model caching complete."
    fi
else
    echo "[Cache] WARNING: Source model directory $SOURCE_MODEL not found on network volume."
fi

echo "--- [run.sh COMPLETE] --- Caching finished. Launching Python worker..."
exec python -u rp_handler.py