FROM nvcr.io/nvidia/pytorch:24.11-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV HOME=/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/venv/bin:/usr/local/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git libgl1 curl ca-certificates rsync \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install uv

RUN git clone https://github.com/black-forest-labs/flux && \
    cd flux && \
    pip install -e ".[tensorrt]" --extra-index-url https://pypi.nvidia.com

COPY requirements.lock .
RUN uv pip install -r requirements.lock --extra-index-url https://pypi.nvidia.com

COPY . .

RUN chmod +x run.sh

CMD ["./run.sh"]