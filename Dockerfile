FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV HOME=/app
ENV PATH=/root/.local/bin:/usr/local/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1-mesa-glx curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir uv

COPY requirements.lock .

RUN which uv || (echo "uv not found on PATH" && exit 1) \
    && uv --version \
    && uv pip sync --system requirements.lock --extra-index-url https://pypi.nvidia.com

COPY . .

ARG TARGET_GPU=H100
ENV GPU_TYPE=${TARGET_GPU}

CMD ["python", "-u", "rp_handler.py"]
