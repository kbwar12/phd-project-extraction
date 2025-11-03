# Docker PyTorch/Hugging Face Troubleshooting Guide

## 🚨 Common Issue: ModuleNotFoundError: No module named 'torch'

This error occurs when PyTorch and Hugging Face dependencies are not properly installed in the Docker container.

## 🔧 Quick Fix

### Option 1: Use the Fix Script (Recommended)

```bash
cd dual-layer/docker
chmod +x fix-huggingface.sh
./fix-huggingface.sh
```

### Option 2: Manual Installation

```bash
# Install missing dependencies in the running container
docker exec -it dual_app pip install --no-cache-dir \
    torch==2.1.2 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    datasets==2.16.1

# Verify installation
docker exec -it dual_app python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Option 3: Rebuild Container

```bash
# Stop containers
docker compose down

# Rebuild with updated Dockerfile
docker compose up -d --build

# Test installation
docker exec -it dual_app python -c "import torch; print('PyTorch working')"
```

## 🔍 Root Cause Analysis

The issue occurs because:

1. **Requirements.txt vs Dockerfile conflict**: The requirements.txt includes PyTorch, but the Dockerfile was trying to install it separately with a different index URL
2. **Installation order**: Dependencies need to be installed in the correct order
3. **Container caching**: Docker may be using cached layers that don't include the dependencies

## 🛠️ Prevention

### Updated Dockerfile (Fixed)

The Dockerfile now:
- Installs dependencies from requirements.txt only
- Verifies PyTorch installation
- Removes conflicting installation commands

### Graceful Fallbacks

The code now includes:
- Optional imports with graceful fallbacks
- Better error handling
- Clear error messages for missing dependencies

## 🧪 Testing the Fix

After applying the fix, test with:

```bash
# Test PyTorch
docker exec -it dual_app python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test Transformers
docker exec -it dual_app python -c "from transformers import AutoTokenizer; print('Transformers OK')"

# Test Hugging Face Client
docker exec -it dual_app python -c "
from src.extraction.huggingface_client import HuggingFaceCypherClient
client = HuggingFaceCypherClient()
print('Hugging Face client working')
"

# Test Knowledge Graph Extractor
docker exec -it dual_app python -c "
from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor
print('Knowledge graph extractor working')
"

# Test Enhanced Pipeline
docker exec -it dual_app python test_enhanced_pipeline.py
```

## 🔄 Alternative Solutions

### Use CPU-Only PyTorch

If you don't need GPU support:

```bash
docker exec -it dual_app pip install --no-cache-dir \
    torch==2.1.2+cpu \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    datasets==2.16.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

### Use GPU PyTorch

If you have GPU support:

```bash
docker exec -it dual_app pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    datasets==2.16.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

## 📋 Verification Checklist

- [ ] PyTorch imports successfully
- [ ] Transformers library imports successfully
- [ ] Hugging Face client initializes without errors
- [ ] Knowledge graph extractor imports successfully
- [ ] Enhanced pipeline test runs without errors
- [ ] GUI applications start without import errors

## 🚀 Next Steps

After fixing the dependencies:

1. **Test the enhanced pipeline**:
   ```bash
   docker exec -it dual_app python test_enhanced_pipeline.py
   ```

2. **Initialize the knowledge graph**:
   ```bash
   docker exec -it dual_app python src/cli/main.py init
   ```

3. **Start using the enhanced features**:
   - Text-to-Cypher conversion with Hugging Face models
   - Enhanced extraction with concepts and facts
   - GPU acceleration (if available)

## 📞 Support

If you continue to have issues:

1. Check Docker logs: `docker compose logs dual_app`
2. Verify container status: `docker compose ps`
3. Check Python environment: `docker exec -it dual_app pip list`
4. Test imports individually: `docker exec -it dual_app python -c "import torch"`

The enhanced pipeline should now work correctly with both Ollama (Qwen3:latest) and Hugging Face models!
