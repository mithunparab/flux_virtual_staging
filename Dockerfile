FROM nvcr.io/nvidia/pytorch:24.05-py3 AS builder


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/build/cache
ENV HUGGING_FACE_HUB_CACHE=/build/cache


RUN apt-get update && apt-get install -y --no-install-recommends git libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*


WORKDIR /build

RUN pip install "flux[tensorrt]@git+https://github.com/black-forest-labs/flux.git" --extra-index-url https://pypi.nvidia.com
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY build_engines.py .
COPY config.py .


ARG TARGET_GPU=H100
ENV GPU_TYPE=${TARGET_GPU}


RUN --mount=type=secret,id=huggingface \
    mkdir -p ${HF_HOME} && echo -n $(cat /run/secrets/huggingface) > ${HF_HOME}/token && \
    echo "--- Starting TensorRT Engine Compilation ---" && \
    python build_engines.py && \
    echo "--- Engine Compilation Complete ---"

FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install "flux[tensorrt]@git+https://github.com/black-forest-labs/flux.git" --extra-index-url https://pypi.nvidia.com
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG TARGET_GPU=H100
ENV GPU_TYPE=${TARGET_GPU}

COPY --from=builder /build/engines /app/engines

CMD ["python", "-u", "rp_handler.py"]