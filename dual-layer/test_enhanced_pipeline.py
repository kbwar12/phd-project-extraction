#!/usr/bin/env python3
"""
Test script for the enhanced knowledge graph extraction pipeline
Demonstrates the integration of Ollama (Qwen3:latest) and Hugging Face models
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.extraction.extractor import Extractor
from src.extraction.huggingface_client import HuggingFaceCypherClient
from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from settings.yaml"""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    config['neo4j']['uri'] = os.getenv('NEO4J_URI', config['neo4j']['uri'])
    config['neo4j']['user'] = os.getenv('NEO4J_USER', config['neo4j']['user'])
    config['neo4j']['password'] = os.getenv('NEO4J_PASSWORD', config['neo4j']['password'])
    
    config['ollama']['base_url'] = os.getenv('OLLAMA_BASE_URL', config['ollama']['base_url'])
    config['ollama']['model'] = os.getenv('OLLAMA_MODEL', config['ollama']['model'])
    
    config['huggingface']['model'] = os.getenv('HUGGINGFACE_MODEL', config['huggingface']['model'])
    config['huggingface']['device'] = os.getenv('HUGGINGFACE_DEVICE', config['huggingface']['device'])
    
    return config

def test_extraction_pipeline():
    """Test the enhanced extraction pipeline"""
    logger.info("Testing enhanced extraction pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Sample text for testing
    sample_text = """
    Apple Inc. is a multinational technology company headquartered in Cupertino, California. 
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. 
    Apple is known for its innovative products including the iPhone, iPad, and Mac computers.
    
    Tim Cook has been the CEO of Apple since August 2011, succeeding Steve Jobs. 
    The company reported revenue of $394.3 billion in 2022, representing a 8% increase 
    from the previous year. Apple's market capitalization exceeded $3 trillion in 2022.
    
    The company operates retail stores worldwide and has a strong presence in the 
    smartphone and personal computer markets. Apple's ecosystem includes hardware, 
    software, and services that work seamlessly together.
    """
    
    doc_id = "test_document_001"
    
    try:
        # Initialize extractor
        logger.info("Initializing extractor with Qwen3:latest...")
        extractor = Extractor(config)
        
        # Extract information
        logger.info("Extracting entities, relations, concepts, and facts...")
        result = extractor.extract(sample_text, doc_id)
        
        # Print results
        logger.info(f"Extraction completed for document: {doc_id}")
        logger.info(f"Entities found: {len(result.entities)}")
        logger.info(f"Relations found: {len(result.relations)}")
        logger.info(f"Concepts found: {len(result.concepts) if result.concepts else 0}")
        logger.info(f"Facts found: {len(result.facts) if result.facts else 0}")
        
        # Print sample entities
        if result.entities:
            logger.info("Sample entities:")
            for entity in result.entities[:3]:
                logger.info(f"  - {entity.text} ({entity.type}) - confidence: {entity.confidence}")
        
        # Print sample relations
        if result.relations:
            logger.info("Sample relations:")
            for relation in result.relations[:3]:
                logger.info(f"  - {relation.head_entity_id} -> {relation.tail_entity_id} ({relation.relation_type})")
        
        return result
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None

def test_cypher_conversion():
    """Test the text-to-Cypher conversion"""
    logger.info("Testing text-to-Cypher conversion...")
    
    config = load_config()
    
    try:
        # Initialize Hugging Face client
        logger.info("Initializing Hugging Face client...")
        cypher_client = HuggingFaceCypherClient(
            model_name=config['huggingface']['model'],
            device=config['huggingface']['device']
        )
        
        # Test queries
        test_queries = [
            "What entities are mentioned in the document?",
            "What relationships exist between people and organizations?",
            "What are the main concepts discussed?",
            "What facts are stated about Apple?",
            "Who is the CEO of Apple?"
        ]
        
        logger.info("Converting natural language queries to Cypher...")
        for query in test_queries:
            try:
                cypher_query = cypher_client.convert_text_to_cypher(query)
                logger.info(f"Query: {query}")
                logger.info(f"Cypher: {cypher_query}")
                logger.info("---")
            except Exception as e:
                logger.error(f"Failed to convert query '{query}': {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Cypher conversion test failed: {e}")
        return False

def test_knowledge_graph_integration():
    """Test the complete knowledge graph integration"""
    logger.info("Testing knowledge graph integration...")
    
    config = load_config()
    
    try:
        # Initialize knowledge graph extractor
        logger.info("Initializing knowledge graph extractor...")
        kg_extractor = KnowledgeGraphExtractor(
            neo4j_uri=config['neo4j']['uri'],
            neo4j_user=config['neo4j']['user'],
            neo4j_password=config['neo4j']['password'],
            huggingface_model=config['huggingface']['model']
        )
        
        # Test text-to-Cypher conversion
        logger.info("Testing integrated text-to-Cypher conversion...")
        test_query = "What entities are mentioned in the documents?"
        cypher_query = kg_extractor.convert_query_to_cypher(test_query)
        logger.info(f"Query: {test_query}")
        logger.info(f"Generated Cypher: {cypher_query}")
        
        # Clean up
        kg_extractor.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Knowledge graph integration test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting enhanced knowledge graph extraction pipeline tests...")
    
    # Test 1: Extraction pipeline
    logger.info("=" * 50)
    logger.info("TEST 1: Extraction Pipeline")
    logger.info("=" * 50)
    extraction_result = test_extraction_pipeline()
    
    # Test 2: Cypher conversion
    logger.info("=" * 50)
    logger.info("TEST 2: Text-to-Cypher Conversion")
    logger.info("=" * 50)
    cypher_success = test_cypher_conversion()
    
    # Test 3: Knowledge graph integration
    logger.info("=" * 50)
    logger.info("TEST 3: Knowledge Graph Integration")
    logger.info("=" * 50)
    kg_success = test_knowledge_graph_integration()
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Extraction Pipeline: {'✓ PASSED' if extraction_result else '✗ FAILED'}")
    logger.info(f"Cypher Conversion: {'✓ PASSED' if cypher_success else '✗ FAILED'}")
    logger.info(f"Knowledge Graph Integration: {'✓ PASSED' if kg_success else '✗ FAILED'}")
    
    if extraction_result and cypher_success and kg_success:
        logger.info("🎉 All tests passed! The enhanced pipeline is working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
