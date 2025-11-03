#!/bin/bash
# Quick start script for Enhanced DocBench: Knowledge Graph-Based Document Benchmarking

set -e

echo "🚀 Starting Enhanced DocBench: Knowledge Graph-Based Document Benchmarking..."
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from docker/ directory"
    exit 1
fi

# Check if .env exists
if [ ! -f "../.env" ]; then
    echo "⚠️  .env file not found. Creating it with Docker defaults..."
    cat > ../.env << 'EOF'
# Environment Variables for Enhanced DocBench: Knowledge Graph-Based Document Benchmarking
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen3:latest
HUGGINGFACE_MODEL=dbands/Qwen2-5-Coder-0-5B-neo4j-text2cypher-2024v1-GGUF
HUGGINGFACE_DEVICE=auto
EOF
    echo "✅ Created .env file"
fi

echo "📦 Starting Docker containers..."
docker compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 15

echo ""
echo "🔍 Checking service status..."
docker compose ps

echo ""
echo "📥 Setting up Ollama model (if needed)..."
docker exec -it ollama ollama pull qwen3:latest || echo "Model may already be installed"

echo ""
echo "🤗 Setting up Hugging Face model (if needed)..."
docker exec -it dual_app python -c "
try:
    from src.extraction.huggingface_client import HuggingFaceCypherClient
    client = HuggingFaceCypherClient()
    print('✅ Hugging Face model loaded successfully')
except ImportError as e:
    print(f'⚠️ Hugging Face dependencies not available: {e}')
    print('   Install with: pip install transformers torch accelerate datasets')
except Exception as e:
    print(f'⚠️ Hugging Face model setup issue: {e}')
" || echo "Hugging Face model may need manual setup"

echo ""
echo "🔧 Initializing Enhanced DocBench knowledge graph..."
docker exec -it dual_app python src/cli/main.py init || echo "Knowledge graph may already be initialized"

echo ""
echo "🧪 Running enhanced pipeline test..."
docker exec -it dual_app python test_enhanced_pipeline.py || echo "Test may have issues - check logs"

echo ""
echo "✅ Enhanced DocBench setup complete!"
echo ""
echo "🌐 Access the applications:"
echo "   - Neo4j Browser:    http://localhost:7474 (neo4j/neo4jpass)"
echo "   - Ollama API:       http://localhost:11434"
echo "   - DocBench GUI:     http://localhost:8501"
echo "   - GUI Troubleshooting: http://localhost:8502"
echo "   - GUI Config:       http://localhost:8503"
echo ""
echo "🚀 Enhanced DocBench Quick Start:"
echo "   1. Place PDF files in data/pdfs/"
echo "   2. Process PDFs: docker exec -it dual_app python launcher.py process-pdf"
echo "   3. Extract info: docker exec -it dual_app python launcher.py extract"
echo "   4. Run evaluation: docker exec -it dual_app python launcher.py evaluate"
echo "   5. Test enhanced pipeline: docker exec -it dual_app python test_enhanced_pipeline.py"
echo ""
echo "📊 View logs: docker compose logs -f"
echo "🛑 Stop all:  docker compose down"
echo ""
