FROM nvcr.io/nvidia/pytorch:24.05-py3 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/build/cache
ENV HUGGING_FACE_HUB_CACHE=/build/cache
ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1-mesa-glx curl ca-certificates && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

WORKDIR /app

RUN git clone https://github.com/black-forest-labs/flux ./flux
RUN cd flux && uv pip install -e ".[tensorrt]" --extra-index-url https://pypi.nvidia.com

COPY requirements.txt .
RUN uv pip install -r requirements.txt

COPY build_engines.py .
COPY config.py .

ARG TARGET_GPU=H100
ENV GPU_TYPE=${TARGET_GPU}

RUN --mount=type=secret,id=huggingface \
    mkdir -p ${HF_HOME} && echo -n $(cat /run/secrets/huggingface) > ${HF_HOME}/token && \
    python build_engines.py

FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV HOME=/app
ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1-mesa-glx curl ca-certificates && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

WORKDIR /app

RUN git clone https://github.com/black-forest-labs/flux ./flux
RUN cd flux && uv pip install -e ".[tensorrt]" --extra-index-url https://pypi.nvidia.com

COPY requirements.txt .
RUN uv pip install -r requirements.txt

COPY . .

ARG TARGET_GPU=H100
ENV GPU_TYPE=${TARGET_GPU}

COPY --from=builder /app/engines /app/engines

CMD ["python", "-u", "rp_handler.py"]