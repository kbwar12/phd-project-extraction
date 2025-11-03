#!/usr/bin/env python3
"""
DocBench Example Script

This script demonstrates how to use the DocBench system for document benchmarking.
It shows the complete pipeline from PDF to knowledge graph evaluation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.benchmarks.datasets.pdf_converter import PDFToMarkdownConverter
from src.benchmarks.datasets.docbench_loader import load_docbench, load_docbench_queries
from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor
from src.benchmarks.datasets.text_to_cypher import TextToCypherConverter
from src.benchmarks.metrics import compute_docbench_metrics
from src.extraction.extractor import Extractor

def create_sample_config():
    """Create a sample configuration for testing"""
    config = {
        "paths": {
            "outputs_dir": "./outputs"
        },
        "extraction": {
            "model": "ollama",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama3.1"
            }
        }
    }
    return config

def run_docbench_example():
    """Run a complete DocBench example"""
    print("🚀 DocBench Example - Knowledge Graph-Based Document Benchmarking")
    print("=" * 70)
    
    # Configuration
    data_dir = "./data"
    pdf_dir = "./data/pdfs"
    markdown_dir = "./data/markdown"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"
    
    # Check if Neo4j is available
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.close()
        print("✅ Neo4j connection successful")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("Please ensure Neo4j is running on bolt://localhost:7687")
        return
    
    # Check for PDF files
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        print("Please add some PDF files to test the system")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    # Step 1: Convert PDFs to Markdown
    print("\n📝 Step 1: Converting PDFs to Markdown")
    converter = PDFToMarkdownConverter(pdf_dir, markdown_dir)
    markdown_files = converter.convert_all_pdfs()
    print(f"✅ Converted {len(markdown_files)} PDFs to markdown")
    
    # Step 2: Load documents
    print("\n📚 Step 2: Loading documents")
    documents = load_docbench(data_dir, pdf_dir, markdown_dir)
    print(f"✅ Loaded {len(documents)} documents")
    
    # Step 3: Initialize extraction pipeline
    print("\n🔍 Step 3: Initializing extraction pipeline")
    config = create_sample_config()
    extractor = Extractor(config)
    
    # Step 4: Initialize knowledge graph extractor
    print("\n🕸️ Step 4: Setting up knowledge graph")
    kg_extractor = KnowledgeGraphExtractor(neo4j_uri, neo4j_user, neo4j_password)
    kg_extractor.clear_database()
    
    # Step 5: Extract and store information
    print("\n⚡ Step 5: Extracting information and populating knowledge graph")
    extraction_stats = []
    
    for i, doc in enumerate(documents[:2]):  # Limit to 2 documents for demo
        print(f"  Processing document {i+1}/{min(2, len(documents))}: {doc['doc_id']}")
        
        try:
            # Extract using the pipeline
            result = extractor.extract(doc["text"], doc["doc_id"])
            
            # Store in knowledge graph
            stats = kg_extractor.extract_and_store(doc["doc_id"], doc["text"], result)
            extraction_stats.append({
                "doc_id": doc["doc_id"],
                **stats
            })
            
            print(f"    ✅ Extracted: {stats['entities_created']} entities, {stats['relations_created']} relations")
            
        except Exception as e:
            print(f"    ❌ Error processing {doc['doc_id']}: {e}")
    
    # Step 6: Load and test queries
    print("\n🔍 Step 6: Testing knowledge graph queries")
    queries = load_docbench_queries(data_dir)
    
    # Test text-to-Cypher conversion
    text_converter = TextToCypherConverter()
    
    print("\n📝 Testing Text-to-Cypher conversion:")
    for query_template in queries[:3]:  # Test first 3 queries
        nl_query = query_template["natural_language"]
        cypher_query = text_converter.convert_query(nl_query)
        print(f"  Q: {nl_query}")
        print(f"  C: {cypher_query}")
        print()
    
    # Step 7: Compute DocBench metrics
    print("\n📊 Step 7: Computing DocBench metrics")
    try:
        metrics = compute_docbench_metrics(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            queries=queries
        )
        
        print(f"✅ DocBench Evaluation Complete!")
        print(f"📈 Overall Score: {metrics['overall_score']:.3f}")
        print(f"✅ Successful Queries: {metrics['successful_queries']}/{metrics['total_queries']}")
        
        print("\n📋 Query Results:")
        for query_id, result in metrics["query_results"].items():
            if "error" not in result:
                print(f"  {query_id}: {result['result_count']} results, score: {result['score']:.3f}")
            else:
                print(f"  {query_id}: ERROR - {result['error']}")
        
    except Exception as e:
        print(f"❌ Error computing metrics: {e}")
    
    # Cleanup
    kg_extractor.close()
    
    print("\n🎉 DocBench example completed!")
    print("\nNext steps:")
    print("1. Add more PDF files to data/pdfs/")
    print("2. Customize queries in data/docbench_queries.json")
    print("3. Run the full benchmark with: python -m src.cli.main --config config/settings.yaml --benchmark src/benchmarks/configs/docbench.yaml")

if __name__ == "__main__":
    run_docbench_example()
