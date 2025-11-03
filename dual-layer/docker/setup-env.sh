#!/bin/bash
# Create .env file for Docker deployment with proper service names

echo "Creating .env file for Docker deployment..."

cat > .env << 'EOF'
# Environment Variables for Enhanced Dual-Layer Extraction System
# Docker Configuration - uses service names for inter-container communication

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
EOF

echo "✅ Created .env file with Docker configuration"
echo ""
echo "Contents:"
cat .env
echo ""
echo "Note: For local development, you may need to change:"
echo "  NEO4J_URI=bolt://localhost:7687"
echo "  OLLAMA_BASE_URL=http://localhost:11434"
echo ""
echo "GPU Support: Ensure Docker has GPU support enabled for Hugging Face models"
