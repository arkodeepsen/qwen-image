# RunPod Serverless Dockerfile for Qwen-Image
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install git+https://github.com/huggingface/diffusers
RUN python3 -m pip install transformers accelerate safetensors
RUN python3 -m pip install hf-transfer

RUN python3 -m pip install pillow

RUN python3 -m pip install runpod

RUN python3 -m pip cache purge

# Copy handler
COPY handler.py /workspace/handler.py

# Point HuggingFace cache to network volume (persistent 100GB storage)
# Model downloads ONCE to volume, then ALL workers share it
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

# RunPod will execute this
CMD ["python3", "-u", "handler.py"]
