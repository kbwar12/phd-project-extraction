import json
from pathlib import Path
from typing import List, Dict, Optional
from .pdf_converter import PDFToMarkdownConverter

def load_docbench(data_dir: str, pdf_dir: str, markdown_dir: str, split: str = "dev") -> List[Dict]:
    """
    Load DocBench dataset from PDF files converted to markdown
    
    Args:
        data_dir: Base data directory
        pdf_dir: Directory containing PDF files
        markdown_dir: Directory for markdown conversions
        split: Dataset split (dev/test/train)
        
    Returns:
        List of document dictionaries with markdown content
    """
    pdf_dir_path = Path(pdf_dir)
    markdown_dir_path = Path(markdown_dir)
    
    # Convert PDFs to markdown if needed
    converter = PDFToMarkdownConverter(pdf_dir, markdown_dir)
    
    # Check if markdown files already exist
    existing_markdown = list(markdown_dir_path.glob("*.md"))
    if not existing_markdown:
        print("No markdown files found, converting PDFs...")
        converter.convert_all_pdfs()
    
    # Load markdown files
    markdown_files = list(markdown_dir_path.glob("*.md"))
    documents = []
    
    for md_file in markdown_files:
        try:
            content = converter.get_markdown_content(md_file)
            
            doc = {
                "doc_id": md_file.stem,
                "text": content,
                "markdown_path": str(md_file),
                "pdf_path": str(pdf_dir_path / f"{md_file.stem}.pdf"),
                "entities": [],  # Will be populated by extraction
                "relations": []  # Will be populated by extraction
            }
            
            documents.append(doc)
            
        except Exception as e:
            print(f"Error loading {md_file}: {e}")
            continue
    
    print(f"Loaded {len(documents)} documents from DocBench")
    return documents

def load_docbench_queries(data_dir: str) -> List[Dict]:
    """
    Load DocBench query templates for evaluation
    
    Args:
        data_dir: Base data directory
        
    Returns:
        List of query templates
    """
    queries_file = Path(data_dir) / "docbench_queries.json"
    
    if not queries_file.exists():
        # Create default query templates if file doesn't exist
        default_queries = [
            {
                "id": "entity_extraction",
                "description": "Extract all entities mentioned in the document",
                "cypher_template": "MATCH (e:Entity) RETURN e.name, e.type",
                "natural_language": "What entities are mentioned in this document?"
            },
            {
                "id": "relation_extraction", 
                "description": "Extract all relations between entities",
                "cypher_template": "MATCH (e1:Entity)-[r:RELATION]->(e2:Entity) RETURN e1.name, r.type, e2.name",
                "natural_language": "What relationships exist between entities?"
            },
            {
                "id": "concept_extraction",
                "description": "Extract key concepts and their properties",
                "cypher_template": "MATCH (c:Concept) RETURN c.name, c.properties",
                "natural_language": "What are the main concepts discussed?"
            },
            {
                "id": "fact_extraction",
                "description": "Extract factual statements",
                "cypher_template": "MATCH (f:Fact) RETURN f.statement, f.confidence",
                "natural_language": "What facts are stated in this document?"
            }
        ]
        
        with open(queries_file, 'w') as f:
            json.dump(default_queries, f, indent=2)
        
        print(f"Created default DocBench queries at {queries_file}")
        return default_queries
    
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    
    return queries
