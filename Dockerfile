# CUDA 12.1 runtime with cuDNN
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils \
    ffmpeg git wget curl ca-certificates build-essential \
 && ln -s /usr/bin/python3.10 /usr/bin/python \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pinned stack (keeps torchvision.functional_tensor available)
RUN pip install --upgrade pip "numpy<2" && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu116 \
        torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 && \
    pip install --no-cache-dir \
        basicsr==1.4.2 \
        facexlib==0.2.5 \
        gfpgan==1.3.5 \
        numpy==1.24.4 \
        opencv-python==4.8.0.76 \
        Pillow==9.5.0 \
        tqdm==4.65.0 \
        ffmpeg-python==0.2.0 \
        pyyaml==6.0\
        streamlit

EXPOSE 8501
