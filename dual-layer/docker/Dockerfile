FROM nvidia/cuda:11.8-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    poppler-utils \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Verify PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Copy source code
COPY src /app/src
COPY config /app/config
COPY launcher.py /app/launcher.py
COPY docbench_example.py /app/docbench_example.py
COPY test_enhanced_pipeline.py /app/test_enhanced_pipeline.py

# Create data directories
RUN mkdir -p /app/data/pdfs /app/data/markdown /app/outputs

# Set environment variables for Hugging Face
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets
ENV CUDA_VISIBLE_DEVICES=0

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

