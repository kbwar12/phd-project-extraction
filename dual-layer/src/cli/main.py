import argparse
import yaml
import os
from pathlib import Path

def load_config():
    """Load global configuration"""
    config_path = Path("config/settings.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    for section in config.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    section[key] = os.getenv(env_var, value)
    
    return config

def cmd_process_pdf(args):
    """Process PDF files and convert to markdown"""
    from src.benchmarks.datasets.pdf_converter import PDFToMarkdownConverter
    
    pdf_dir = args.pdf_dir or "./data/pdfs"
    markdown_dir = args.markdown_dir or "./data/markdown"
    
    converter = PDFToMarkdownConverter(pdf_dir, markdown_dir)
    
    if args.file:
        # Convert single file
        result = converter.convert_pdf_to_markdown(args.file)
        print(f"✅ Converted {args.file} to {result}")
    else:
        # Convert all PDFs
        results = converter.convert_all_pdfs()
        print(f"✅ Converted {len(results)} PDF files to markdown")

def cmd_extract(args):
    """Extract entities and relations from documents and populate knowledge graph"""
    from src.extraction.extractor import Extractor
    from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor
    
    config = load_config()
    extractor = Extractor(config)
    
    # Initialize knowledge graph extractor
    kg_extractor = KnowledgeGraphExtractor(
        neo4j_uri=config["neo4j"]["uri"],
        neo4j_user=config["neo4j"]["user"],
        neo4j_password=config["neo4j"]["password"]
    )
    
    if args.clear:
        print("🧹 Clearing knowledge graph...")
        kg_extractor.clear_database()
    
    # Process documents
    if args.markdown_file:
        # Process single markdown file
        with open(args.markdown_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        doc_id = args.doc_id or Path(args.markdown_file).stem
        
        print(f"🔍 Extracting from: {doc_id}")
        result = extractor.extract(text, doc_id)
        
        print(f"📊 Extracted:")
        print(f"  - {len(result.entities)} entities")
        print(f"  - {len(result.relations)} relations")
        
        # Store in knowledge graph
        stats = kg_extractor.extract_and_store(doc_id, text, result)
        print(f"💾 Stored in knowledge graph:")
        print(f"  - {stats['entities_created']} entities")
        print(f"  - {stats['relations_created']} relations")
        print(f"  - {stats['concepts_created']} concepts")
        print(f"  - {stats['facts_created']} facts")
        
    else:
        # Process all markdown files
        from src.benchmarks.datasets.docbench_loader import load_docbench
        
        data_dir = args.data_dir or "./data"
        pdf_dir = args.pdf_dir or "./data/pdfs"
        markdown_dir = args.markdown_dir or "./data/markdown"
        
        documents = load_docbench(data_dir, pdf_dir, markdown_dir)
        
        print(f"📚 Processing {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            print(f"  Processing {i+1}/{len(documents)}: {doc['doc_id']}")
            
            try:
                result = extractor.extract(doc["text"], doc["doc_id"])
                stats = kg_extractor.extract_and_store(doc["doc_id"], doc["text"], result)
                print(f"    ✅ Extracted: {stats['entities_created']} entities, {stats['relations_created']} relations")
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    kg_extractor.close()

def cmd_query(args):
    """Query the knowledge graph using natural language or Cypher"""
    from src.benchmarks.datasets.text_to_cypher import TextToCypherConverter
    from neo4j import GraphDatabase
    
    config = load_config()
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config["neo4j"]["uri"],
        auth=(config["neo4j"]["user"], config["neo4j"]["password"])
    )
    
    try:
        with driver.session() as session:
            if args.natural_language:
                # Convert natural language to Cypher
                converter = TextToCypherConverter()
                cypher_query = converter.convert_query(args.natural_language)
                print(f"🔍 Natural Language Query: {args.natural_language}")
                print(f"🔧 Generated Cypher: {cypher_query}")
                
                # Execute query
                result = session.run(cypher_query)
                records = list(result)
                
                print(f"📊 Found {len(records)} results:")
                for i, record in enumerate(records[:args.limit]):
                    print(f"  {i+1}. {dict(record)}")
                    
            elif args.cypher:
                # Execute Cypher query directly
                print(f"🔧 Executing Cypher: {args.cypher}")
                result = session.run(args.cypher)
                records = list(result)
                
                print(f"📊 Found {len(records)} results:")
                for i, record in enumerate(records[:args.limit]):
                    print(f"  {i+1}. {dict(record)}")
                    
            else:
                # Show available queries
                converter = TextToCypherConverter()
                examples = converter.get_query_examples()
                
                print("📋 Available Query Examples:")
                for example in examples:
                    print(f"  Q: {example['query']}")
                    print(f"  C: {example['cypher']}")
                    print()
    
    finally:
        driver.close()

def cmd_evaluate(args):
    """Run DocBench evaluation"""
    from src.benchmarks.runner import BenchmarkRunner
    
    config = load_config()
    
    # Use default DocBench config if not specified
    if not args.config:
        args.config = "src/benchmarks/configs/docbench.yaml"
    
    runner = BenchmarkRunner(args.config, config)
    
    print("🚀 Running DocBench evaluation...")
    metrics = runner.run()
    
    print("\n" + "="*60)
    print("📊 DOCBENCH EVALUATION RESULTS")
    print("="*60)
    print(f"📈 Overall Score: {metrics.get('overall_score', 0):.3f}")
    print(f"✅ Successful Queries: {metrics.get('successful_queries', 0)}/{metrics.get('total_queries', 0)}")
    print(f"📚 Total Documents: {metrics.get('total_documents', 0)}")
    
    if 'query_results' in metrics:
        print("\n📋 Query Results:")
        for query_id, result in metrics['query_results'].items():
            if 'error' not in result:
                print(f"  {query_id}: {result['result_count']} results, score: {result['score']:.3f}")
            else:
                print(f"  {query_id}: ERROR - {result['error']}")

def cmd_init(args):
    """Initialize Neo4j database for DocBench"""
    from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor
    
    config = load_config()
    
    kg_extractor = KnowledgeGraphExtractor(
        neo4j_uri=config["neo4j"]["uri"],
        neo4j_user=config["neo4j"]["user"],
        neo4j_password=config["neo4j"]["password"]
    )
    
    print("🔧 Initializing DocBench knowledge graph...")
    kg_extractor.clear_database()  # Start fresh
    print("✅ Knowledge graph initialized and ready for DocBench")
    
    kg_extractor.close()

def cmd_clear(args):
    """Clear the knowledge graph"""
    from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor
    
    config = load_config()
    
    if not args.force:
        confirm = input("This will delete all data in the knowledge graph. Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("❌ Aborted")
            return
    
    kg_extractor = KnowledgeGraphExtractor(
        neo4j_uri=config["neo4j"]["uri"],
        neo4j_user=config["neo4j"]["user"],
        neo4j_password=config["neo4j"]["password"]
    )
    
    print("🧹 Clearing knowledge graph...")
    kg_extractor.clear_database()
    print("✅ Knowledge graph cleared")
    
    kg_extractor.close()

def main():
    parser = argparse.ArgumentParser(description="DocBench: Knowledge Graph-Based Document Benchmarking")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process PDF command
    pdf_parser = subparsers.add_parser("process-pdf", help="Convert PDF files to markdown")
    pdf_parser.add_argument("--file", help="Specific PDF file to convert")
    pdf_parser.add_argument("--pdf-dir", help="PDF directory (default: ./data/pdfs)")
    pdf_parser.add_argument("--markdown-dir", help="Markdown output directory (default: ./data/markdown)")
    pdf_parser.set_defaults(func=cmd_process_pdf)
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract information and populate knowledge graph")
    extract_parser.add_argument("--markdown-file", help="Specific markdown file to process")
    extract_parser.add_argument("--doc-id", help="Document ID (default: filename)")
    extract_parser.add_argument("--data-dir", help="Data directory (default: ./data)")
    extract_parser.add_argument("--pdf-dir", help="PDF directory (default: ./data/pdfs)")
    extract_parser.add_argument("--markdown-dir", help="Markdown directory (default: ./data/markdown)")
    extract_parser.add_argument("--clear", action="store_true", help="Clear knowledge graph before processing")
    extract_parser.set_defaults(func=cmd_extract)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_group = query_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--natural-language", help="Natural language query")
    query_group.add_argument("--cypher", help="Cypher query")
    query_parser.add_argument("--limit", type=int, default=10, help="Max results to show")
    query_parser.set_defaults(func=cmd_query)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run DocBench evaluation")
    eval_parser.add_argument("--config", help="DocBench config file (default: src/benchmarks/configs/docbench.yaml)")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize knowledge graph for DocBench")
    init_parser.set_defaults(func=cmd_init)
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the knowledge graph")
    clear_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    clear_parser.set_defaults(func=cmd_clear)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\n🚀 Quick Start:")
        print("  1. python -m src.cli.main process-pdf  # Convert PDFs to markdown")
        print("  2. python -m src.cli.main extract      # Extract and populate knowledge graph")
        print("  3. python -m src.cli.main evaluate     # Run DocBench evaluation")
        print("  4. python -m src.cli.main query --natural-language 'What entities are mentioned?'")
        return
    
    args.func(args)

if __name__ == "__main__":
    main()

