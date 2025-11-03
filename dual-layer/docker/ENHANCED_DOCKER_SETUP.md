# Enhanced Docker Setup Guide

This guide covers setting up the enhanced knowledge graph extraction pipeline with Docker, including Ollama (Qwen3:latest) and Hugging Face model support.

## 🚀 Quick Start

### Option 1: Use the Start Script (Recommended)

```bash
cd dual-layer/docker
bash start-gui.sh
```

This script will:
- Create the `.env` file with enhanced configuration
- Start all Docker containers (including GUIs)
- Pull the Ollama model (Qwen3:latest)
- Initialize the enhanced knowledge graph
- Test the enhanced pipeline

### Option 2: Manual Setup

```bash
cd dual-layer/docker

# 1. Create environment file
bash setup-env.sh

# 2. Start all services
docker compose up -d

# 3. Wait for services to be ready
sleep 30

# 4. Pull Ollama model
docker exec -it ollama ollama pull qwen3:latest

# 5. Initialize knowledge graph
docker exec -it dual_app python src/cli/main.py init

# 6. Test enhanced pipeline
docker exec -it dual_app python test_enhanced_pipeline.py
```

## 🔧 Configuration Options

### Standard Setup (CPU Only)

Use the standard `docker-compose.yml` for CPU-only operation:

```bash
docker compose up -d
```

### GPU-Enabled Setup (Recommended for Hugging Face)

For GPU acceleration with Hugging Face models:

```bash
docker compose -f docker-compose.gpu.yml up -d
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- Docker with GPU support enabled

## 📋 Environment Variables

The enhanced setup requires these environment variables in `.env`:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass

# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen3:latest

# Hugging Face Configuration
HUGGINGFACE_MODEL=dbands/Qwen2-5-Coder-0-5B-neo4j-text2cypher-2024v1-GGUF
HUGGINGFACE_DEVICE=auto
```

## 🌐 Access Points

Once running, access these services:

| Service | URL | Description |
|---------|-----|-------------|
| **DocBench GUI** | http://localhost:8501 | Main interface with enhanced extraction |
| **Troubleshooting** | http://localhost:8502 | System diagnostics and performance |
| **Configuration** | http://localhost:8503 | Settings and model configuration |
| **Neo4j Browser** | http://localhost:7474 | Direct Neo4j access (neo4j/neo4jpass) |
| **Ollama API** | http://localhost:11434 | Ollama model API |

## 🧪 Testing the Enhanced Pipeline

### Run the Test Suite

```bash
# Test the complete enhanced pipeline
docker exec -it dual_app python test_enhanced_pipeline.py
```

### Manual Testing

```bash
# Test Ollama integration
docker exec -it dual_app python -c "
from src.extraction.extractor import Extractor
import yaml
with open('config/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)
extractor = Extractor(config)
print('✅ Ollama integration working')
"

# Test Hugging Face integration
docker exec -it dual_app python -c "
from src.extraction.huggingface_client import HuggingFaceCypherClient
client = HuggingFaceCypherClient()
print('✅ Hugging Face integration working')
"

# Test knowledge graph integration
docker exec -it dual_app python -c "
from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor
kg = KnowledgeGraphExtractor('bolt://neo4j:7687', 'neo4j', 'neo4jpass')
print('✅ Knowledge graph integration working')
"
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Model Not Loading

```bash
# Check Ollama status
docker exec -it ollama ollama list

# Pull model manually
docker exec -it ollama ollama pull qwen3:latest

# Check Ollama logs
docker compose logs ollama
```

#### 2. Hugging Face Model Issues

```bash
# Check model loading
docker exec -it dual_app python -c "
from src.extraction.huggingface_client import HuggingFaceCypherClient
try:
    client = HuggingFaceCypherClient()
    print('✅ Model loaded successfully')
except Exception as e:
    print(f'❌ Error: {e}')
"

# Check GPU availability (for GPU setup)
docker exec -it dual_app python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# Clear Hugging Face cache
docker exec -it dual_app rm -rf /app/.cache/huggingface
```

#### 3. Neo4j Connection Issues

```bash
# Check Neo4j status
docker compose logs neo4j | grep "Started"

# Test Neo4j connection
docker exec -it dual_app python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', 'neo4jpass'))
with driver.session() as session:
    result = session.run('RETURN 1 as test')
    print('✅ Neo4j connection working')
driver.close()
"

# Initialize Neo4j constraints
docker exec -it dual_app python src/cli/main.py init
```

#### 4. GUI Not Accessible

```bash
# Check container status
docker compose ps

# Check GUI logs
docker compose logs gui-main

# Restart GUI services
docker compose restart gui-main gui-troubleshooting gui-config
```

### Performance Issues

#### Memory Usage

```bash
# Check container resource usage
docker stats

# Increase Neo4j memory if needed
# Edit docker-compose.yml:
# - NEO4J_dbms_memory_heap_max__size=4G
```

#### GPU Performance

```bash
# Check GPU usage
nvidia-smi

# Monitor GPU in container
docker exec -it dual_app nvidia-smi
```

## 📊 Monitoring and Logs

### View Logs

```bash
# All services
docker compose logs -f

# Specific services
docker compose logs -f neo4j
docker compose logs -f ollama
docker compose logs -f gui-main

# App container logs
docker exec -it dual_app tail -f /dev/null
```

### Monitor Performance

```bash
# Container resource usage
docker stats

# Service health checks
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:7474             # Neo4j
```

## 🔄 Updating and Maintenance

### Update Models

```bash
# Update Ollama model
docker exec -it ollama ollama pull qwen3:latest

# Update Hugging Face model (clear cache first)
docker exec -it dual_app rm -rf /app/.cache/huggingface
docker compose restart app gui-main gui-troubleshooting gui-config
```

### Rebuild Containers

```bash
# Rebuild after code changes
docker compose up -d --build

# Rebuild specific services
docker compose up -d --build gui-main
```

### Backup and Restore

```bash
# Backup Neo4j data
docker exec -it dual_neo4j neo4j-admin dump --database=neo4j --to=/tmp/backup.dump
docker cp dual_neo4j:/tmp/backup.dump ./neo4j_backup.dump

# Backup Hugging Face models
docker cp dual_app:/app/.cache/huggingface ./huggingface_backup

# Restore Neo4j data
docker cp ./neo4j_backup.dump dual_neo4j:/tmp/backup.dump
docker exec -it dual_neo4j neo4j-admin load --database=neo4j --from=/tmp/backup.dump
```

## 🛑 Stopping and Cleanup

### Stop Services

```bash
# Stop all services
docker compose down

# Stop specific services
docker compose stop gui-main
```

### Complete Cleanup

```bash
# Stop and remove containers
docker compose down

# Remove volumes (WARNING: deletes all data)
docker compose down -v

# Remove images
docker compose down --rmi all
```

## 📚 Additional Resources

- **Enhanced Pipeline Documentation**: [ENHANCED_PIPELINE_README.md](../ENHANCED_PIPELINE_README.md)
- **GUI Documentation**: [GUI_DOCKER_LAUNCH.md](GUI_DOCKER_LAUNCH.md)
- **Quick Start Guide**: [QUICK_START_GUI.md](QUICK_START_GUI.md)
- **Main Documentation**: [README.md](../README.md)

## 🎯 Example Workflow

```bash
# 1. Start enhanced pipeline
cd dual-layer/docker
bash start-gui.sh

# 2. Wait for all services to be ready
docker compose logs -f neo4j  # Wait for "Started"

# 3. Test the pipeline
docker exec -it dual_app python test_enhanced_pipeline.py

# 4. Process documents
docker exec -it dual_app python launcher.py process-pdf
docker exec -it dual_app python launcher.py extract

# 5. Query the knowledge graph
docker exec -it dual_app python launcher.py query --natural-language "What entities are mentioned?"

# 6. Access GUIs for monitoring
# Main: http://localhost:8501
# Troubleshooting: http://localhost:8502
# Config: http://localhost:8503

# 7. Monitor logs
docker compose logs -f

# 8. Stop when done
docker compose down
```

This enhanced Docker setup provides a complete, production-ready environment for the knowledge graph extraction pipeline with both Ollama and Hugging Face model support.
