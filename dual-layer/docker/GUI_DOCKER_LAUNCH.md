# Launching Enhanced Streamlit GUI from Docker

## Quick Start

### 1. Start All Services (Including Enhanced GUI)

Launch all services including the enhanced Streamlit GUI applications with Hugging Face support:

```bash
cd docker
docker compose up -d
```

This will start:
- **Neo4j** on port 7474 (Browser) and 7687 (Bolt)
- **Ollama** on port 11434 (with Qwen3:latest)
- **Main App** container for CLI operations (with Hugging Face support)
- **GUI Main** on http://localhost:8501
- **GUI Troubleshooting** on http://localhost:8502
- **GUI Config** on http://localhost:8503

### 2. GPU Support (Optional)

For GPU acceleration with Hugging Face models, use the GPU-enabled configuration:

```bash
# Use GPU-enabled Docker Compose
docker compose -f docker-compose.gpu.yml up -d
```

**Requirements for GPU support:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed
- Docker with GPU support enabled

### 3. Access the Enhanced GUIs

Once services are running, access the GUI applications:

- **Main Monitoring Interface**: http://localhost:8501
- **Troubleshooting Tools**: http://localhost:8502
- **Configuration Manager**: http://localhost:8503

## Enhanced Features

### New Capabilities

The enhanced Docker setup includes:

1. **Hugging Face Integration**: Text-to-Cypher conversion using specialized models
2. **GPU Support**: Optional CUDA acceleration for faster model inference
3. **Enhanced Extraction**: Richer knowledge graph population with concepts and facts
4. **Model Caching**: Persistent storage for Hugging Face models
5. **Improved Error Handling**: Better fallback mechanisms

### Environment Variables

The enhanced setup requires additional environment variables:

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

## Individual GUI Control

### Start Specific GUIs

```bash
# Start only the main monitoring interface
docker compose up -d gui-main

# Start only the troubleshooting tools
docker compose up -d gui-troubleshooting

# Start only the configuration manager
docker compose up -d gui-config
```

### Stop Specific GUIs

```bash
# Stop a specific GUI
docker compose stop gui-main

# Stop all GUIs
docker compose stop gui-main gui-troubleshooting gui-config

# Stop and remove containers
docker compose down gui-main gui-troubleshooting gui-config
```

### Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific GUI
docker compose restart gui-main
```

## Alternative: Launch GUI from Main App Container

If you prefer to launch GUIs manually from the main app container:

```bash
# Access the main app container
docker exec -it dual_app bash

# Launch GUI from within the container
streamlit run src/gui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Or use the launcher
python src/gui/launcher.py main
```

**Note**: When launching manually, you'll need to map ports when starting the container or use port forwarding.

## Port Mapping

The GUI services are configured with the following port mappings:

| Service | Container Port | Host Port | URL |
|---------|---------------|-----------|-----|
| gui-main | 8501 | 8501 | http://localhost:8501 |
| gui-troubleshooting | 8502 | 8502 | http://localhost:8502 |
| gui-config | 8503 | 8503 | http://localhost:8503 |

To change ports, edit `docker/docker-compose.yml`:

```yaml
gui-main:
  ports:
    - "YOUR_PORT:8501"  # Change YOUR_PORT to desired port
```

## Network Access

By default, GUIs are accessible only from localhost. To access from other machines on your network:

1. Update the docker-compose.yml to bind to `0.0.0.0` (already configured)
2. Use your machine's IP address: `http://YOUR_IP:8501`

## Troubleshooting

### GUI Not Accessible

1. **Check if containers are running**:
   ```bash
   docker compose ps
   ```

2. **Check logs for errors**:
   ```bash
   docker compose logs gui-main
   ```

3. **Verify ports are not in use**:
   ```bash
   # On Linux/Mac
   lsof -i :8501
   
   # On Windows
   netstat -ano | findstr :8501
   ```

### Services Not Detected in GUI

1. **Check service URLs in Configuration**:
   - Neo4j should be: `bolt://neo4j:7687` (within Docker network)
   - Ollama should be: `http://ollama:11434` (within Docker network)

2. **If running GUI outside Docker**, use:
   - Neo4j: `bolt://localhost:7687`
   - Ollama: `http://localhost:11434`

### GUI Shows Connection Errors

1. **Ensure all services are running**:
   ```bash
   docker compose ps
   ```

2. **Check Neo4j is ready**:
   ```bash
   docker compose logs neo4j | grep "Started"
   ```

3. **Check Ollama is ready**:
   ```bash
   docker compose logs ollama | tail -20
   ```

4. **Initialize Neo4j constraints**:
   ```bash
   docker exec -it dual_app python src/cli/main.py init
   ```

### Hugging Face Model Issues

1. **Check model loading**:
   ```bash
   docker exec -it dual_app python -c "
   from src.extraction.huggingface_client import HuggingFaceCypherClient
   client = HuggingFaceCypherClient()
   print('Model loaded successfully')
   "
   ```

2. **Check GPU availability**:
   ```bash
   docker exec -it dual_app python -c "
   import torch
   print(f'CUDA available: {torch.cuda.is_available()}')
   print(f'GPU count: {torch.cuda.device_count()}')
   "
   ```

3. **Clear Hugging Face cache**:
   ```bash
   docker exec -it dual_app rm -rf /app/.cache/huggingface
   ```

## Environment Variables

GUI services inherit environment variables from your `.env` file. Ensure it contains:

```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen3:latest
HUGGINGFACE_MODEL=dbands/Qwen2-5-Coder-0-5B-neo4j-text2cypher-2024v1-GGUF
HUGGINGFACE_DEVICE=auto
```

**Important**: When services run in Docker, use service names (e.g., `neo4j`, `ollama`) instead of `localhost` for connections.

## Rebuilding After Code Changes

If you modify the GUI code:

```bash
# Rebuild containers with GUI changes
docker compose up -d --build gui-main gui-troubleshooting gui-config

# Or rebuild everything
docker compose down
docker compose up -d --build
```

## Stopping Everything

```bash
# Stop all services including GUIs
docker compose down

# Stop and remove volumes (WARNING: deletes Neo4j data)
docker compose down -v
```

## Example Workflow

```bash
# 1. Start everything
cd docker
docker compose up -d

# 2. Wait for services to be ready (30-60 seconds)
docker compose logs -f neo4j  # Wait for "Started" message

# 3. Pull Ollama model if not already done
docker exec -it ollama ollama pull qwen3:latest

# 4. Initialize Neo4j
docker exec -it dual_app python src/cli/main.py init

# 5. Test enhanced pipeline
docker exec -it dual_app python test_enhanced_pipeline.py

# 6. Access GUIs
# Main: http://localhost:8501
# Troubleshooting: http://localhost:8502
# Config: http://localhost:8503

# 7. Monitor logs
docker compose logs -f gui-main

# 8. When done, stop everything
docker compose down
```

## Individual GUI Control

### Start Specific GUIs

```bash
# Start only the main monitoring interface
docker compose up -d gui-main

# Start only the troubleshooting tools
docker compose up -d gui-troubleshooting

# Start only the configuration manager
docker compose up -d gui-config
```

### Stop Specific GUIs

```bash
# Stop a specific GUI
docker compose stop gui-main

# Stop all GUIs
docker compose stop gui-main gui-troubleshooting gui-config

# Stop and remove containers
docker compose down gui-main gui-troubleshooting gui-config
```

### Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific GUI
docker compose restart gui-main
```

## Alternative: Launch GUI from Main App Container

If you prefer to launch GUIs manually from the main app container:

```bash
# Access the main app container
docker exec -it dual_app bash

# Launch GUI from within the container
streamlit run src/gui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Or use the launcher
python src/gui/launcher.py main
```

**Note**: When launching manually, you'll need to map ports when starting the container or use port forwarding.

## Port Mapping

The GUI services are configured with the following port mappings:

| Service | Container Port | Host Port | URL |
|---------|---------------|-----------|-----|
| gui-main | 8501 | 8501 | http://localhost:8501 |
| gui-troubleshooting | 8502 | 8502 | http://localhost:8502 |
| gui-config | 8503 | 8503 | http://localhost:8503 |

To change ports, edit `docker/docker-compose.yml`:

```yaml
gui-main:
  ports:
    - "YOUR_PORT:8501"  # Change YOUR_PORT to desired port
```

## Network Access

By default, GUIs are accessible only from localhost. To access from other machines on your network:

1. Update the docker-compose.yml to bind to `0.0.0.0` (already configured)
2. Use your machine's IP address: `http://YOUR_IP:8501`

## Troubleshooting

### GUI Not Accessible

1. **Check if containers are running**:
   ```bash
   docker compose ps
   ```

2. **Check logs for errors**:
   ```bash
   docker compose logs gui-main
   ```

3. **Verify ports are not in use**:
   ```bash
   # On Linux/Mac
   lsof -i :8501
   
   # On Windows
   netstat -ano | findstr :8501
   ```

### Services Not Detected in GUI

1. **Check service URLs in Configuration**:
   - Neo4j should be: `bolt://neo4j:7687` (within Docker network)
   - Ollama should be: `http://ollama:11434` (within Docker network)

2. **If running GUI outside Docker**, use:
   - Neo4j: `bolt://localhost:7687`
   - Ollama: `http://localhost:11434`

### GUI Shows Connection Errors

1. **Ensure all services are running**:
   ```bash
   docker compose ps
   ```

2. **Check Neo4j is ready**:
   ```bash
   docker compose logs neo4j | grep "Started"
   ```

3. **Check Ollama is ready**:
   ```bash
   docker compose logs ollama | tail -20
   ```

4. **Initialize Neo4j constraints**:
   ```bash
   docker exec -it dual_app python src/cli/main.py init
   ```

## Environment Variables

GUI services inherit environment variables from your `.env` file. Ensure it contains:

```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen3:latest
```

**Important**: When services run in Docker, use service names (e.g., `neo4j`, `ollama`) instead of `localhost` for connections.

## Rebuilding After Code Changes

If you modify the GUI code:

```bash
# Rebuild containers with GUI changes
docker compose up -d --build gui-main gui-troubleshooting gui-config

# Or rebuild everything
docker compose down
docker compose up -d --build
```

## Stopping Everything

```bash
# Stop all services including GUIs
docker compose down

# Stop and remove volumes (WARNING: deletes Neo4j data)
docker compose down -v
```

## Example Workflow

```bash
# 1. Start everything
cd docker
docker compose up -d

# 2. Wait for services to be ready (30-60 seconds)
docker compose logs -f neo4j  # Wait for "Started" message

# 3. Pull Ollama model if not already done
docker exec -it ollama ollama pull qwen3:latest

# 4. Initialize Neo4j
docker exec -it dual_app python src/cli/main.py init

# 5. Access GUIs
# Main: http://localhost:8501
# Troubleshooting: http://localhost:8502
# Config: http://localhost:8503

# 6. Monitor logs
docker compose logs interfaces -f

# 7. When done, stop everything
docker compose down
```
