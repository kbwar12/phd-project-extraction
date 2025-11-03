# Enhanced Knowledge Graph Extraction Pipeline

This document describes the enhanced knowledge graph extraction pipeline that integrates **Ollama with Qwen3:latest** for information extraction and **Hugging Face models** for text-to-Cypher conversion.

## 🚀 New Features

### 1. Enhanced Ollama Integration with Qwen3:latest
- **Optimized prompts** specifically designed for Qwen3:latest model
- **Richer extraction** including entities, relations, concepts, and facts
- **Enhanced entity types**: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PERCENT, MISC
- **Detailed relation categories**: WORK, LOCATION, TEMPORAL, OWNERSHIP, CAUSAL, COMPARISON, OTHER
- **Properties support** for additional context and metadata

### 2. Hugging Face Integration for Text-to-Cypher Conversion
- **Smaller specialized models** for efficient text-to-Cypher conversion
- **Default model**: `google/flan-t5-base` (instruction-following model)
- **Fallback mechanisms** for robust operation
- **Context-aware conversion** with knowledge graph schema information

### 3. Enhanced Knowledge Graph Schema
- **Concepts**: Abstract ideas and topics extracted from documents
- **Facts**: Verifiable statements with source entity references
- **Properties**: Flexible metadata storage for all graph elements
- **Confidence scores**: Quality assessment for all extracted elements

## 📋 Architecture Overview

```
Document Text
     ↓
Qwen3:latest (Ollama) → Enhanced Extraction
     ↓
Entities + Relations + Concepts + Facts
     ↓
Neo4j Knowledge Graph Storage
     ↓
Natural Language Query
     ↓
Hugging Face Model → Cypher Query
     ↓
Neo4j Query Execution
```

## 🔧 Configuration

### Environment Variables

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:latest

# Hugging Face Configuration
HUGGINGFACE_MODEL=google/flan-t5-base
HUGGINGFACE_DEVICE=auto

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Configuration File (`config/settings.yaml`)

```yaml
ollama:
  base_url: ${OLLAMA_BASE_URL}
  model: ${OLLAMA_MODEL}
  temperature: 0.2
  top_p: 0.95
  max_tokens: 2048

huggingface:
  model: ${HUGGINGFACE_MODEL}
  device: ${HUGGINGFACE_DEVICE}
  max_length: 512
  temperature: 0.3
  top_p: 0.9

extraction:
  confidence_threshold: 0.5
  enable_concepts: true
  enable_facts: true
```

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Qwen3:latest model
ollama pull qwen3:latest

# Start Ollama service
ollama serve
```

### 3. Set Up Neo4j

```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest

# Or install locally
# Download from https://neo4j.com/download/
```

### 4. Set Up Environment

```bash
# Copy example environment file
cp env.example .env

# Edit .env with your configuration
nano .env
```

## 🧪 Testing

### Run the Test Suite

```bash
python test_enhanced_pipeline.py
```

This will test:
1. **Extraction Pipeline**: Entity, relation, concept, and fact extraction
2. **Text-to-Cypher Conversion**: Natural language to Cypher query conversion
3. **Knowledge Graph Integration**: Complete pipeline integration

### Sample Test Output

```
Testing enhanced extraction pipeline...
Initializing extractor with Qwen3:latest...
Extracting entities, relations, concepts, and facts...
Extraction completed for document: test_document_001
Entities found: 8
Relations found: 5
Concepts found: 3
Facts found: 4

Sample entities:
  - Apple Inc. (ORGANIZATION) - confidence: 0.95
  - Tim Cook (PERSON) - confidence: 0.92
  - Cupertino (LOCATION) - confidence: 0.88

Sample relations:
  - entity_1 -> entity_2 (works_for)
  - entity_3 -> entity_4 (located_in)
  - entity_5 -> entity_6 (founded_by)
```

## 📊 Usage Examples

### 1. Basic Extraction

```python
from src.extraction.extractor import Extractor
import yaml

# Load configuration
with open('config/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize extractor
extractor = Extractor(config)

# Extract from text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
result = extractor.extract(text, "doc_001")

print(f"Entities: {len(result.entities)}")
print(f"Relations: {len(result.relations)}")
print(f"Concepts: {len(result.concepts)}")
print(f"Facts: {len(result.facts)}")
```

### 2. Text-to-Cypher Conversion

```python
from src.extraction.huggingface_client import HuggingFaceCypherClient

# Initialize client
cypher_client = HuggingFaceCypherClient(model_name="google/flan-t5-base")

# Convert natural language to Cypher
query = "What entities are mentioned in the documents?"
cypher_query = cypher_client.convert_text_to_cypher(query)

print(f"Cypher: {cypher_query}")
# Output: MATCH (e:Entity) RETURN e.name, e.type LIMIT 100
```

### 3. Knowledge Graph Integration

```python
from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor

# Initialize extractor
kg_extractor = KnowledgeGraphExtractor(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    huggingface_model="google/flan-t5-base"
)

# Extract and store in knowledge graph
stats = kg_extractor.extract_and_store("doc_001", text, extraction_result)
print(f"Stored: {stats}")

# Convert query to Cypher
cypher_query = kg_extractor.convert_query_to_cypher(
    "What relationships exist between people and organizations?"
)
```

## 🔍 Advanced Features

### 1. Custom Prompts

You can customize the extraction prompts in `src/extraction/prompts.py`:

```python
CUSTOM_ENTITY_PROMPT = """Your custom prompt here..."""
```

### 2. Model Selection

Choose different Hugging Face models for text-to-Cypher conversion:

```python
# For better instruction following
cypher_client = HuggingFaceCypherClient(model_name="google/flan-t5-large")

# For conversational models
cypher_client = HuggingFaceCypherClient(model_name="microsoft/DialoGPT-medium")
```

### 3. Confidence Thresholds

Adjust extraction quality by modifying confidence thresholds:

```yaml
extraction:
  confidence_threshold: 0.7  # Higher threshold for better quality
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Hugging Face Model Loading Error**
   ```bash
   # Check available models
   python -c "from transformers import AutoTokenizer; print('Transformers working')"
   
   # Try a smaller model
   HUGGINGFACE_MODEL=google/flan-t5-small
   ```

3. **Neo4j Connection Error**
   ```bash
   # Check Neo4j status
   docker ps | grep neo4j
   
   # Test connection
   cypher-shell -u neo4j -p password
   ```

### Performance Optimization

1. **GPU Acceleration**
   ```bash
   # Enable CUDA for Hugging Face models
   HUGGINGFACE_DEVICE=cuda
   ```

2. **Model Caching**
   ```bash
   # Set Hugging Face cache directory
   export HF_HOME=/path/to/cache
   ```

3. **Batch Processing**
   ```python
   # Process multiple documents in batches
   for batch in document_batches:
       results = [extractor.extract(doc.text, doc.id) for doc in batch]
   ```

## 📈 Performance Metrics

### Expected Performance

- **Extraction Speed**: ~2-5 seconds per document (depending on text length)
- **Memory Usage**: ~2-4GB for Hugging Face models
- **Accuracy**: 85-95% for entity extraction, 70-85% for relation extraction

### Benchmarking

Run the benchmark suite to evaluate performance:

```bash
python src/benchmarks/runner.py --config config/benchmark_config.yaml
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** for providing easy access to large language models
- **Hugging Face** for the transformers library and model hub
- **Neo4j** for the graph database platform
- **Qwen** team for the excellent Qwen3:latest model
