ARG CUDA_VERSION="12.1.1"
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3.10 \
    python3-pip \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --no-cache-dir --upgrade pip


WORKDIR /app

ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV HOME=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod

COPY . .

RUN huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

RUN python -c "from rp_handler import model"

CMD ["python", "-u", "rp_handler.py"]
