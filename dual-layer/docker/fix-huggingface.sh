#!/bin/bash
# Script to install missing Hugging Face dependencies in Docker container

echo "🔧 Installing missing Hugging Face dependencies..."

# Install PyTorch and Hugging Face dependencies
docker exec -it dual_app pip install --no-cache-dir \
    torch==2.1.2 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    datasets==2.16.1

echo "✅ Dependencies installed successfully!"

# Test the installation
echo "🧪 Testing Hugging Face installation..."
docker exec -it dual_app python -c "
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    
    from transformers import AutoTokenizer
    print('✅ Transformers library working')
    
    from src.extraction.huggingface_client import HuggingFaceCypherClient
    print('✅ Hugging Face client import successful')
    
except Exception as e:
    print(f'❌ Error: {e}')
"

echo "🎉 Setup complete! You can now use the enhanced pipeline."
