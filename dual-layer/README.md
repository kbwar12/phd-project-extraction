# DocBench: Knowledge Graph-Based Document Benchmarking

DocBench is an innovative document benchmarking system that uses knowledge graphs and Cypher queries to evaluate document extraction quality. Unlike traditional benchmarks that rely on simple text matching, DocBench leverages the power of structured knowledge representation for more meaningful evaluation.

## 🚀 Key Features

- **PDF to Markdown Pipeline**: Convert PDF documents to markdown for better text processing
- **Knowledge Graph Population**: Extract entities, relations, concepts, and facts into Neo4j
- **Natural Language to Cypher**: Convert natural language queries to Cypher queries
- **Graph-Based Evaluation**: Evaluate extraction quality using knowledge graph queries
- **Interactive GUI**: Streamlit-based interface for easy interaction
- **Comprehensive CLI**: Command-line interface for automation

## 🏗️ Architecture

```
PDF Documents → Markdown → Extraction → Knowledge Graph → Cypher Queries → Evaluation
```

1. **PDF Processing**: Convert PDFs to markdown format
2. **Information Extraction**: Extract entities, relations, concepts, and facts
3. **Knowledge Graph Storage**: Populate Neo4j with structured information
4. **Query Evaluation**: Execute natural language queries converted to Cypher
5. **Metrics Computation**: Assess extraction quality based on query results

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dual-layer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Neo4j**:
   - Install Neo4j Desktop or use Docker
   - Start Neo4j on `bolt://localhost:7687`
   - Default credentials: `neo4j/password`

4. **Set up Ollama** (optional):
   - Install Ollama
   - Pull a model: `ollama pull llama3.1`

## 🎯 Quick Start

### Using the Launcher (Recommended)

```bash
# Launch the GUI
python launcher.py gui

# Process PDFs
python launcher.py process-pdf

# Extract and populate knowledge graph
python launcher.py extract

# Run evaluation
python launcher.py evaluate

# Query the knowledge graph
python launcher.py query --natural-language "What entities are mentioned?"
```

### Using CLI Directly

```bash
# Process PDFs to markdown
python -m src.cli.main process-pdf

# Extract information and populate knowledge graph
python -m src.cli.main extract

# Run DocBench evaluation
python -m src.cli.main evaluate

# Query with natural language
python -m src.cli.main query --natural-language "What entities are mentioned?"

# Query with Cypher
python -m src.cli.main query --cypher "MATCH (e:Entity) RETURN e.name, e.type"
```

### Using the GUI

```bash
# Launch Streamlit GUI
python -m streamlit run src/gui/streamlit_app.py
```

## 📁 Project Structure

```
dual-layer/
├── src/
│   ├── benchmarks/
│   │   ├── configs/
│   │   │   └── docbench.yaml          # DocBench configuration
│   │   ├── datasets/
│   │   │   ├── pdf_converter.py        # PDF to markdown conversion
│   │   │   ├── docbench_loader.py      # DocBench data loading
│   │   │   ├── knowledge_graph_extractor.py  # Neo4j population
│   │   │   └── text_to_cypher.py       # Natural language to Cypher
│   │   ├── metrics.py                  # DocBench evaluation metrics
│   │   └── runner.py                   # DocBench runner
│   ├── cli/
│   │   └── main.py                     # CLI interface
│   ├── extraction/
│   │   ├── extractor.py               # Main extraction pipeline
│   │   ├── ollama_client.py            # Ollama integration
│   │   └── prompts.py                  # Extraction prompts
│   └── gui/
│       └── streamlit_app.py            # Streamlit GUI
├── config/
│   └── settings.yaml                   # Global configuration
├── data/
│   ├── pdfs/                           # PDF documents
│   ├── markdown/                       # Converted markdown files
│   └── docbench_queries.json          # Evaluation queries
├── outputs/                            # Evaluation results
├── launcher.py                         # Easy launcher script
└── docbench_example.py                 # Example script
```

## 🔧 Configuration

Edit `config/settings.yaml` to configure:

- **Neo4j connection**: URI, username, password
- **Ollama settings**: Base URL, model, temperature
- **DocBench paths**: PDF directory, markdown directory, data directory
- **Extraction settings**: Confidence threshold, max evidence per claim

## 📊 DocBench Workflow

### 1. PDF Processing
```bash
python launcher.py process-pdf
```
- Converts PDF files in `data/pdfs/` to markdown
- Outputs markdown files to `data/markdown/`

### 2. Information Extraction
```bash
python launcher.py extract
```
- Extracts entities, relations, concepts, and facts
- Populates Neo4j knowledge graph
- Stores structured information for evaluation

### 3. Evaluation
```bash
python launcher.py evaluate
```
- Loads evaluation queries from `data/docbench_queries.json`
- Executes queries against the knowledge graph
- Computes quality metrics based on query results

### 4. Querying
```bash
python launcher.py query --natural-language "What entities are mentioned?"
```
- Converts natural language to Cypher
- Executes queries against the knowledge graph
- Returns structured results

## 🎨 GUI Features

The Streamlit GUI provides:

- **📊 Dashboard**: System status and quick actions
- **📄 PDF Processing**: Convert PDFs to markdown
- **🔍 Document Extraction**: Extract and store information
- **📈 DocBench Evaluation**: Run comprehensive evaluation
- **🗂️ Knowledge Graph Explorer**: Browse the knowledge graph
- **🔍 Query Interface**: Natural language to Cypher conversion
- **⚙️ Configuration**: Manage system settings

## 🔍 Query Types

DocBench supports various query types:

- **Entity Extraction**: "What entities are mentioned in this document?"
- **Relation Extraction**: "What relationships exist between entities?"
- **Concept Extraction**: "What are the main concepts discussed?"
- **Fact Extraction**: "What facts are stated in this document?"
- **Type-specific Queries**: "What are the people mentioned?"
- **Relationship Queries**: "Who works for which organization?"

## 📈 Evaluation Metrics

DocBench provides:

- **Overall Score**: Composite score based on all queries
- **Query Success Rate**: Percentage of successful queries
- **Result Quality**: Quality of query results
- **Extraction Statistics**: Counts of entities, relations, concepts, facts
- **Per-Query Metrics**: Individual query performance

## 🚀 Advantages

1. **Structured Evaluation**: Uses knowledge graph queries instead of simple text matching
2. **Flexible Queries**: Natural language queries can be easily extended
3. **Rich Representation**: Captures entities, relations, concepts, and facts
4. **Scalable**: Graph-based evaluation scales better than linear text matching
5. **Interpretable**: Query results provide clear insights into extraction quality

## 🔮 Future Enhancements

- **Advanced NLP**: Integrate more sophisticated NLP models
- **Query Learning**: Learn query patterns from examples
- **Multi-modal**: Support for images, tables, and other document elements
- **Domain Adaptation**: Specialized models for different document types
- **Interactive Evaluation**: Enhanced GUI for exploring extraction results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Neo4j for the graph database
- Ollama for the LLM integration
- Streamlit for the GUI framework
- PyMuPDF for PDF processing