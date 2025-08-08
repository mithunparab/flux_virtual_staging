FROM nvcr.io/nvidia/pytorch:24.11-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV HOME=/app
ENV PATH=/opt/venv/bin:/usr/local/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git libgl1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install uv

COPY requirements.lock .

RUN uv pip sync requirements.lock --extra-index-url https://pypi.nvidia.com

COPY . .

ARG TARGET_GPU=H100
ENV GPU_TYPE=${TARGET_GPU}

CMD ["python", "-u", "rp_handler.py"]
