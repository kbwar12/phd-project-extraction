import yaml
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List

from src.extraction.extractor import Extractor
from src.benchmarks.metrics import compute_docbench_metrics
from src.benchmarks.datasets.docbench_loader import load_docbench, load_docbench_queries
from src.benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor
from src.benchmarks.datasets.text_to_cypher import TextToCypherConverter

class BenchmarkRunner:
    def __init__(self, config_path: str, global_config: Dict):
        with open(config_path, 'r') as f:
            self.bench_config = yaml.safe_load(f)
        
        self.global_config = global_config
        self.extractor = Extractor(global_config)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_dataset(self) -> List[Dict]:
        """Load dataset based on benchmark config"""
        dataset_name = self.bench_config["dataset"]
        data_dir = self.bench_config["data_dir"]
        split = self.bench_config.get("split", "dev")
        
        if dataset_name == "docbench":
            pdf_dir = self.bench_config["pdf_dir"]
            markdown_dir = self.bench_config["markdown_dir"]
            return load_docbench(data_dir, pdf_dir, markdown_dir, split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def run(self) -> Dict:
        """Run DocBench benchmark and return metrics"""
        print(f"Loading dataset: {self.bench_config['dataset']}")
        documents = self.load_dataset()
        
        # Limit samples for testing
        max_samples = self.bench_config.get("max_samples", len(documents))
        documents = documents[:max_samples]
        
        # Initialize knowledge graph extractor
        kg_extractor = KnowledgeGraphExtractor(
            neo4j_uri=self.bench_config["neo4j_uri"],
            neo4j_user=self.bench_config["neo4j_user"],
            neo4j_password=self.bench_config["neo4j_password"]
        )
        
        # Clear database for fresh run
        kg_extractor.clear_database()
        
        # Load DocBench queries
        queries = load_docbench_queries(self.bench_config["data_dir"])
        
        print(f"Processing {len(documents)} documents...")
        extraction_stats = []
        
        for doc in tqdm(documents):
            # Extract information using the existing pipeline
            result = self.extractor.extract(doc["text"], doc["doc_id"])
            
            # Store in knowledge graph
            stats = kg_extractor.extract_and_store(doc["doc_id"], doc["text"], result)
            extraction_stats.append({
                "doc_id": doc["doc_id"],
                **stats
            })
        
        # Compute DocBench metrics
        print("Computing DocBench metrics...")
        metrics = compute_docbench_metrics(
            neo4j_uri=self.bench_config["neo4j_uri"],
            neo4j_user=self.bench_config["neo4j_user"],
            neo4j_password=self.bench_config["neo4j_password"],
            queries=queries
        )
        
        # Add extraction statistics
        metrics["extraction_stats"] = extraction_stats
        metrics["total_documents"] = len(documents)
        
        # Save results
        self.save_results(metrics, documents, queries)
        
        # Close knowledge graph connection
        kg_extractor.close()
        
        return metrics
    
    def save_results(self, metrics: Dict, documents: List[Dict], queries: List[Dict]):
        """Save DocBench benchmark results"""
        output_dir = Path(self.global_config["paths"]["outputs_dir"]) / \
                     self.bench_config["dataset"] / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save documents info
        with open(output_dir / "documents.json", 'w') as f:
            json.dump(documents, f, indent=2)
        
        # Save queries
        with open(output_dir / "queries.json", 'w') as f:
            json.dump(queries, f, indent=2)
        
        # Save config snapshot
        with open(output_dir / "config.yaml", 'w') as f:
            yaml.dump({
                "benchmark": self.bench_config,
                "global": self.global_config,
                "run_id": self.run_id
            }, f)
        
        print(f"\nDocBench results saved to: {output_dir}")
