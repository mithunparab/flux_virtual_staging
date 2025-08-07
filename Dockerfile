FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV HUGGING_FACE_HUB_CACHE=/app/cache
ENV HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install opencv-python==4.8.0.76
RUN git clone https://github.com/black-forest-labs/flux ./flux
RUN cd flux && pip install -e ".[tensorrt]" --extra-index-url https://pypi.nvidia.com

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ARG TARGET_GPU=H100
ENV GPU_TYPE=${TARGET_GPU}

CMD ["python", "-u", "rp_handler.py"]