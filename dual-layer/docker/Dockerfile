FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including CUDA support for Hugging Face models
RUN apt-get update && apt-get install -y \
    poppler-utils \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Verify PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

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

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

