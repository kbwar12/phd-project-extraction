# DocBench: Knowledge Graph-Based Document Benchmarking

This system implements a novel approach to document benchmarking using knowledge graphs and Cypher queries.

## Architecture Overview

The DocBench system follows this pipeline:

1. **PDF → Markdown**: Convert PDF documents to markdown for better text processing
2. **Text Extraction**: Extract entities, relations, concepts, and facts using the existing extraction pipeline
3. **Knowledge Graph Population**: Store extracted information in Neo4j
4. **Query Evaluation**: Execute natural language queries converted to Cypher against the knowledge graph
5. **Metrics Computation**: Evaluate extraction quality based on query results

## Key Components

### 1. PDF to Markdown Converter (`pdf_converter.py`)
- Converts PDF files to markdown format
- Preserves document structure and formatting
- Handles text cleaning and header detection

### 2. Knowledge Graph Extractor (`knowledge_graph_extractor.py`)
- Populates Neo4j with extracted information
- Creates nodes for: Documents, Entities, Relations, Concepts, Facts
- Establishes relationships between extracted elements

### 3. Text-to-Cypher Converter (`text_to_cypher.py`)
- Converts natural language queries to Cypher queries
- Uses pattern matching and rule-based conversion
- Supports various query types (entity extraction, relation queries, etc.)

### 4. DocBench Metrics (`metrics.py`)
- Evaluates extraction quality via knowledge graph queries
- Computes scores based on query result quality and completeness
- Provides comprehensive evaluation metrics

## Usage

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Neo4j (ensure it's running on bolt://localhost:7687)
# Default credentials: neo4j/password
```

### 2. Prepare Documents
```bash
# Place PDF files in data/pdfs/
cp your_documents.pdf data/pdfs/
```

### 3. Run DocBench
```bash
# Run the benchmark
python -m src.cli.main --config config/settings.yaml --benchmark src/benchmarks/configs/docbench.yaml
```

## Configuration

The DocBench configuration (`src/benchmarks/configs/docbench.yaml`) includes:

```yaml
dataset: docbench
data_dir: ./data
pdf_dir: ./data/pdfs
markdown_dir: ./data/markdown
neo4j_uri: bolt://localhost:7687
neo4j_user: neo4j
neo4j_password: password
max_samples: 10  # Limit for testing
```

## Query Types

DocBench supports various query types:

1. **Entity Extraction**: "What entities are mentioned in this document?"
2. **Relation Extraction**: "What relationships exist between entities?"
3. **Concept Extraction**: "What are the main concepts discussed?"
4. **Fact Extraction**: "What facts are stated in this document?"
5. **Type-specific Queries**: "What are the people mentioned?"
6. **Relationship Queries**: "Who works for which organization?"

## Advantages of This Approach

1. **Structured Evaluation**: Uses knowledge graph queries instead of simple text matching
2. **Flexible Queries**: Natural language queries can be easily extended
3. **Rich Representation**: Captures entities, relations, concepts, and facts
4. **Scalable**: Can handle large document collections efficiently
5. **Interpretable**: Query results provide clear insights into extraction quality

## Future Enhancements

1. **Advanced NLP**: Integrate more sophisticated NLP models for better extraction
2. **Query Learning**: Learn query patterns from examples
3. **Multi-modal**: Support for images, tables, and other document elements
4. **Domain Adaptation**: Specialized models for different document types
5. **Interactive Evaluation**: GUI for exploring extraction results

## Comparison with Traditional Benchmarks

| Aspect | Traditional (DocRED/SciERC) | DocBench |
|--------|----------------------------|----------|
| Evaluation | Text matching | Knowledge graph queries |
| Flexibility | Fixed schema | Dynamic queries |
| Interpretability | Limited | High (query results) |
| Scalability | Linear | Graph-based |
| Extensibility | Schema-dependent | Query-dependent |
