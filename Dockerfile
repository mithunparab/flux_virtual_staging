FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV HOME=/app
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1-mesa-glx curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://astral.sh/uv/install.sh -o /uv-installer.sh \
    && sh /uv-installer.sh \
    && rm /uv-installer.sh

COPY requirements.lock .

RUN uv pip sync requirements.lock --extra-index-url https://pypi.nvidia.com

COPY . .

ARG TARGET_GPU=H100
ENV GPU_TYPE=${TARGET_GPU}

CMD ["python", "-u", "rp_handler.py"]
