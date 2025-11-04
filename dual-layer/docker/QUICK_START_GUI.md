# Quick Guide: Launching DocBench GUI from Docker

## 🚀 Fastest Way to Get Started with DocBench

### Prerequisites

**Important:** Ollama must be running in a separate container before starting the other services.

1. **Set up Ollama separately** (run this first):
   ```bash
   docker run -d --gpus all --name ollama -p 11434:11434 \
     -v ollama_models:/root/.ollama \
     -e OLLAMA_KEEP_ALIVE=24h \
     ollama/ollama:latest
   
   # Pull the model
   docker exec -it ollama ollama pull qwen3:latest
   ```

2. **GPU Requirements:**
   - NVIDIA GPU with CUDA support
   - NVIDIA Docker runtime installed
   - Docker with GPU support enabled

### Option 1: Use the Start Script (Recommended)

```bash
cd dual-layer/docker
bash start-gui.sh
```

This script will:
- Create the `.env` file if it doesn't exist
- Start all Docker containers (including GUIs)
- Note: Ollama must be running separately (see Prerequisites)
- Initialize DocBench knowledge graph

### Option 2: Manual Start

```bash
cd dual-layer/docker

# 1. Ensure Ollama is running separately (see Prerequisites above)

# 2. Create .env file if needed
bash setup-env.sh

# 3. Start all services
docker compose up -d

# 4. Wait for services to be ready
sleep 30

# 5. Initialize DocBench knowledge graph
docker exec -it dual_app python src/cli/main.py init
```

## 🌐 Access the GUIs

Once running, access these URLs in your browser:

| Application | URL | Description |
|------------|-----|-------------|
| **DocBench GUI** | http://localhost:8501 | Full DocBench interface with PDF processing, extraction, and evaluation |
| **Troubleshooting** | http://localhost:8502 | System diagnostics and performance analysis |
| **Configuration** | http://localhost:8503 | Settings and configuration management |
| **Neo4j Browser** | http://localhost:7474 | Direct Neo4j access (neo4j/neo4jpass) |

## 🚀 DocBench Quick Start Workflow

1. **Place PDF files** in `data/pdfs/` directory
2. **Process PDFs**: `docker exec -it dual_app python launcher.py process-pdf`
3. **Extract information**: `docker exec -it dual_app python launcher.py extract`
4. **Run evaluation**: `docker exec -it dual_app python launcher.py evaluate`
5. **Query knowledge graph**: `docker exec -it dual_app python launcher.py query --natural-language "What entities are mentioned?"`

## 🔧 Important Configuration for Docker

The `.env` file **must** use Docker service names, not `localhost`:

```bash
# ✅ Correct for Docker
NEO4J_URI=bolt://neo4j:7687
OLLAMA_BASE_URL=http://ollama:11434

# ❌ Wrong - this won't work in Docker
NEO4J_URI=bolt://localhost:7687
OLLAMA_BASE_URL=http://localhost:11434
```

## 📋 Managing Services

### Check Status
```bash
docker compose ps
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f gui-main
docker compose logs -f neo4j
```

### Restart Services
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart gui-main
```

### Stop Services
```bash
# Stop but keep data
docker compose stop

# Stop and remove containers
docker compose down

# Stop and remove everything including volumes
docker compose down -v  # WARNING: Deletes Neo4j data!
```

## 🐛 Troubleshooting

### GUI Not Loading

1. **Check if containers are running**:
   ```bash
   docker compose ps
   ```

2. **Check logs for errors**:
   ```bash
   docker compose logs gui-main
   ```

3. **Verify ports are available**:
   ```bash
   # Linux/Mac
   lsof -i :8501
   
   # Windows
   netstat -ano | findstr :8501
   ```

### Services Not Detected in GUI

The GUI will show "🔴 Offline" if services aren't reachable. Check:

1. **Environment variables** - Use service names, not localhost
2. **Container status** - All containers should be running
3. **Network connectivity** - Containers should be on same Docker network

### "Connection Refused" Errors

This usually means the environment variables are wrong. Update `.env`:

```bash
# Edit the env file
nano ../.env

# Restart containers
docker compose restart gui-main gui-troubleshooting gui-config
```

## 📚 More Information

- **Detailed GUI Documentation**: See [src/gui/README.md](../src/gui/README.md)
- **Complete Docker Guide**: See [docker/GUI_DOCKER_LAUNCH.md](GUI_DOCKER_LAUNCH.md)
- **CLI Usage**: See main [README.md](../README.md)

## 🎯 Quick Commands Reference

```bash
# Start everything
docker compose up -d

# Stop everything
docker compose down

# Rebuild after code changes
docker compose up -d --build

# View GUI logs
docker compose logs -f gui-main gui-troubleshooting gui-config

# Access Neo4j shell
docker exec -it dual_neo4j cypher-shell -u neo4j -p neo4jpass

# Run CLI commands
docker exec -it dual_app python src/cli/main.py [command]

# Check service health
curl http://localhost:11434/api/tags  # Ollama (runs separately)
curl http://localhost:7474             # Neo4j
```
