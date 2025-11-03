# Cursor AI Instructions: Build Dual-Layer Extraction System

## Project Overview

Build a modular LLM-driven information extraction system with a temporary knowledge graph (Gₜ) for benchmarking extraction capability on standard IE datasets (DocRED, SciERC).

---

## Step 1: Initialize Project Structure

Create the following directory structure:

```
dual-layer/
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
├── src/
│   ├── config/
│   │   └── settings.yaml
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── prompts.py
│   │   ├── ollama_client.py
│   │   └── extractor.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── gt_client.py
│   │   └── cypher/
│   │       └── init_gt.cypher
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── docred_loader.py
│   │   │   └── scierc_loader.py
│   │   ├── metrics.py
│   │   ├── runner.py
│   │   └── configs/
│   │       ├── docred.yaml
│   │       └── scierc.yaml
│   └── cli/
│       ├── __init__.py
│       └── main.py
├── data/
├── outputs/
├── config/
│   └── settings.yaml
├── .env.example
├── .env
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Step 2: Create Docker Configuration

### File: `docker/docker-compose.yml`

```yaml
version: "3.9"
services:
  neo4j:
    image: neo4j:5.25-community
    container_name: dual_neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/neo4jpass
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_security_auth__enabled=true
      - NEO4JLABS_PLUGINS=["apoc"]
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - OLLAMA_KEEP_ALIVE=24h
    restart: unless-stopped

  app:
    build:
      context: ..
      dockerfile: ./docker/Dockerfile
    container_name: dual_app
    depends_on:
      - neo4j
      - ollama
    env_file:
      - ../.env
    volumes:
      - ../src:/app/src
      - ../data:/app/data
      - ../outputs:/app/outputs
      - ../config:/app/config
    working_dir: /app
    stdin_open: true
    tty: true
    command: ["tail", "-f", "/dev/null"]

volumes:
  neo4j_data:
  neo4j_logs:
  ollama_models:
```

### File: `docker/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /app/src
COPY config /app/config

ENV PYTHONPATH=/app
```

---

## Step 3: Create Configuration Files

### File: `.env.example`

```bash
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpass

# Ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen3:latest

# Extraction settings
CONFIDENCE_THRESHOLD=0.5
MAX_TOKENS=2048
TEMPERATURE=0.2
```

### File: `config/settings.yaml`

```yaml
neo4j:
  uri: ${NEO4J_URI}
  user: ${NEO4J_USER}
  password: ${NEO4J_PASSWORD}

ollama:
  base_url: ${OLLAMA_BASE_URL}
  model: ${OLLAMA_MODEL}
  temperature: 0.2
  top_p: 0.95
  max_tokens: 2048

extraction:
  confidence_threshold: 0.5
  max_evidence_per_claim: 5

paths:
  data_dir: ./data
  outputs_dir: ./outputs
```

### File: `requirements.txt`

```
neo4j==5.23.0
pyyaml==6.0.1
pydantic==2.7.0
requests==2.31.0
tenacity==8.2.3
tqdm==4.66.2
numpy==1.26.4
scikit-learn==1.4.2
rapidfuzz==3.9.0
```

### File: `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Environment
.env

# Data
data/*
!data/.gitkeep
outputs/*
!outputs/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

---

## Step 4: Create Extraction Schema

### File: `src/extraction/schema.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Span(BaseModel):
    """Text span with character offsets"""
    start: int
    end: int
    text: str

class Entity(BaseModel):
    """Extracted entity"""
    id: str
    text: str
    type: str  # e.g., "PERSON", "ORG", "LOCATION", "MISC"
    span: Span
    confidence: float = Field(ge=0, le=1)

class Relation(BaseModel):
    """Extracted relation between entities"""
    id: str
    head_entity_id: str
    tail_entity_id: str
    relation_type: str  # e.g., "works_for", "located_in", "part_of"
    confidence: float = Field(ge=0, le=1)
    evidence_text: Optional[str] = None

class Evidence(BaseModel):
    """Supporting evidence for extraction"""
    doc_id: str
    sentence_id: Optional[str] = None
    text: str
    span: Optional[Span] = None

class ExtractionResult(BaseModel):
    """Complete extraction output for a document"""
    doc_id: str
    entities: List[Entity]
    relations: List[Relation]
    evidence: List[Evidence]
    metadata: dict = {}
```

---

## Step 5: Create Extraction Prompts

### File: `src/extraction/prompts.py`

```python
ENTITY_EXTRACTION_PROMPT = """You are an expert information extraction system.

Extract ALL entities from the following text. For each entity, provide:
- text: the exact entity mention
- type: one of [PERSON, ORGANIZATION, LOCATION, DATE, MISC]
- start: character offset where entity starts
- end: character offset where entity ends
- confidence: your confidence score (0.0 to 1.0)

Text:
---
{text}
---

Return ONLY valid JSON matching this schema:
{{
  "entities": [
    {{"id": "e1", "text": "...", "type": "...", "span": {{"start": 0, "end": 10, "text": "..."}}, "confidence": 0.95}}
  ]
}}
"""

RELATION_EXTRACTION_PROMPT = """You are an expert information extraction system.

Given these entities:
{entities_json}

And this text:
---
{text}
---

Extract ALL relations between entities. For each relation, provide:
- head_entity_id: ID of the source entity
- tail_entity_id: ID of the target entity
- relation_type: semantic relation (e.g., "works_for", "located_in", "part_of")
- confidence: your confidence score (0.0 to 1.0)
- evidence_text: the sentence/phrase supporting this relation

Return ONLY valid JSON matching this schema:
{{
  "relations": [
    {{"id": "r1", "head_entity_id": "e1", "tail_entity_id": "e2", "relation_type": "works_for", "confidence": 0.9, "evidence_text": "..."}}
  ]
}}
"""
```

---

## Step 6: Create Ollama Client

### File: `src/extraction/ollama_client.py`

```python
import os
import requests
import json
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Optional

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def generate(
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    model: Optional[str] = None
) -> str:
    """Call Ollama generate endpoint with retry logic"""
    url = f"{BASE_URL}/api/generate"
    payload = {
        "model": model or MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens
        },
        "stream": False
    }
    
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["response"]

def health_check() -> bool:
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False
```

---

## Step 7: Create Extraction Orchestrator

### File: `src/extraction/extractor.py`

```python
import json
import uuid
from typing import List
from src.extraction.ollama_client import generate
from src.extraction.prompts import ENTITY_EXTRACTION_PROMPT, RELATION_EXTRACTION_PROMPT
from src.extraction.schema import ExtractionResult, Entity, Relation, Evidence, Span

class Extractor:
    def __init__(self, config: dict):
        self.config = config
        self.confidence_threshold = config["extraction"]["confidence_threshold"]
    
    def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities from text"""
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        try:
            raw_response = generate(
                prompt,
                temperature=self.config["ollama"]["temperature"],
                max_tokens=self.config["ollama"]["max_tokens"]
            )
            
            # Parse JSON response
            parsed = json.loads(raw_response)
            entities = []
            
            for ent_data in parsed.get("entities", []):
                if ent_data.get("confidence", 0) >= self.confidence_threshold:
                    entities.append(Entity(**ent_data))
            
            return entities
        
        except Exception as e:
            print(f"Entity extraction failed for {doc_id}: {e}")
            return []
    
    def extract_relations(self, text: str, entities: List[Entity], doc_id: str) -> List[Relation]:
        """Extract relations between entities"""
        if not entities:
            return []
        
        entities_json = json.dumps([e.model_dump() for e in entities], indent=2)
        prompt = RELATION_EXTRACTION_PROMPT.format(
            entities_json=entities_json,
            text=text
        )
        
        try:
            raw_response = generate(
                prompt,
                temperature=self.config["ollama"]["temperature"],
                max_tokens=self.config["ollama"]["max_tokens"]
            )
            
            parsed = json.loads(raw_response)
            relations = []
            
            for rel_data in parsed.get("relations", []):
                if rel_data.get("confidence", 0) >= self.confidence_threshold:
                    relations.append(Relation(**rel_data))
            
            return relations
        
        except Exception as e:
            print(f"Relation extraction failed for {doc_id}: {e}")
            return []
    
    def extract(self, text: str, doc_id: str) -> ExtractionResult:
        """Full extraction pipeline"""
        # Step 1: Extract entities
        entities = self.extract_entities(text, doc_id)
        
        # Step 2: Extract relations
        relations = self.extract_relations(text, entities, doc_id)
        
        # Step 3: Package evidence
        evidence = [Evidence(doc_id=doc_id, text=text)]
        
        return ExtractionResult(
            doc_id=doc_id,
            entities=entities,
            relations=relations,
            evidence=evidence,
            metadata={"extractor_version": "v1.0"}
        )
```

---

## Step 8: Create Neo4j Client for Gₜ

### File: `src/graph/gt_client.py`

```python
from neo4j import GraphDatabase
import os
from typing import List, Dict
from src.extraction.schema import ExtractionResult, Entity, Relation

class GTClient:
    """Client for Temporary Knowledge Graph (Gₜ)"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
    
    def close(self):
        self.driver.close()
    
    def initialize_constraints(self):
        """Create indexes and constraints"""
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT relation_id IF NOT EXISTS FOR (r:Relation) REQUIRE r.id IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text)"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
    
    def clear_graph(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def store_extraction(self, result: ExtractionResult):
        """Store extraction result in Gₜ"""
        with self.driver.session() as session:
            # Store document
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.metadata = $metadata
            """, doc_id=result.doc_id, metadata=result.metadata)
            
            # Store entities
            for entity in result.entities:
                session.run("""
                    MERGE (e:Entity {id: $id})
                    SET e.text = $text,
                        e.type = $type,
                        e.confidence = $confidence,
                        e.span_start = $span_start,
                        e.span_end = $span_end
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:CONTAINS]->(e)
                """, 
                    id=entity.id,
                    text=entity.text,
                    type=entity.type,
                    confidence=entity.confidence,
                    span_start=entity.span.start,
                    span_end=entity.span.end,
                    doc_id=result.doc_id
                )
            
            # Store relations
            for relation in result.relations:
                session.run("""
                    MATCH (h:Entity {id: $head_id})
                    MATCH (t:Entity {id: $tail_id})
                    MERGE (h)-[r:RELATION {id: $rel_id}]->(t)
                    SET r.type = $rel_type,
                        r.confidence = $confidence,
                        r.evidence = $evidence
                    WITH r
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:SUPPORTS]->(r)
                """,
                    head_id=relation.head_entity_id,
                    tail_id=relation.tail_entity_id,
                    rel_id=relation.id,
                    rel_type=relation.relation_type,
                    confidence=relation.confidence,
                    evidence=relation.evidence_text,
                    doc_id=result.doc_id
                )
    
    def query_entities(self, entity_type: str = None, min_confidence: float = 0.0) -> List[Dict]:
        """Query entities from Gₜ"""
        query = """
            MATCH (e:Entity)
            WHERE ($entity_type IS NULL OR e.type = $entity_type)
              AND e.confidence >= $min_confidence
            RETURN e.id AS id, e.text AS text, e.type AS type, e.confidence AS confidence
        """
        
        with self.driver.session() as session:
            result = session.run(query, entity_type=entity_type, min_confidence=min_confidence)
            return [dict(record) for record in result]
    
    def query_relations(self, relation_type: str = None, min_confidence: float = 0.0) -> List[Dict]:
        """Query relations from Gₜ"""
        query = """
            MATCH (h:Entity)-[r:RELATION]->(t:Entity)
            WHERE ($relation_type IS NULL OR r.type = $relation_type)
              AND r.confidence >= $min_confidence
            RETURN r.id AS id, h.text AS head, t.text AS tail, 
                   r.type AS type, r.confidence AS confidence, r.evidence AS evidence
        """
        
        with self.driver.session() as session:
            result = session.run(query, relation_type=relation_type, min_confidence=min_confidence)
            return [dict(record) for record in result]
```

### File: `src/graph/cypher/init_gt.cypher`

```cypher
// Constraints for Gₜ
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT relation_id IF NOT EXISTS FOR (r:Relation) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Indexes
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text);
CREATE INDEX relation_type IF NOT EXISTS FOR (r:RELATION) ON (r.type);
```

---

## Step 9: Create Benchmark Infrastructure

### File: `src/benchmarks/datasets/docred_loader.py`

```python
import json
from pathlib import Path
from typing import List, Dict

def load_docred(data_dir: str, split: str = "dev") -> List[Dict]:
    """
    Load DocRED dataset
    Expected format: data/docred/dev.json
    """
    file_path = Path(data_dir) / "docred" / f"{split}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"DocRED {split} file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Transform to standard format
    documents = []
    for item in data:
        doc = {
            "doc_id": item.get("title", f"docred_{len(documents)}"),
            "text": " ".join([" ".join(sent) for sent in item["sents"]]),
            "entities": [],
            "relations": []
        }
        
        # Parse entities
        for ent in item.get("vertexSet", []):
            if ent:  # vertexSet is list of lists
                doc["entities"].append({
                    "text": ent[0]["name"],
                    "type": ent[0].get("type", "MISC"),
                    "mentions": len(ent)
                })
        
        # Parse relations (gold labels)
        for rel in item.get("labels", []):
            doc["relations"].append({
                "head": rel["h"],
                "tail": rel["t"],
                "relation": rel["r"]
            })
        
        documents.append(doc)
    
    return documents
```

### File: `src/benchmarks/datasets/scierc_loader.py`

```python
import json
from pathlib import Path
from typing import List, Dict

def load_scierc(data_dir: str, split: str = "dev") -> List[Dict]:
    """
    Load SciERC dataset
    Expected format: data/scierc/dev.json (one JSON object per line)
    """
    file_path = Path(data_dir) / "scierc" / f"{split}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"SciERC {split} file not found: {file_path}")
    
    documents = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            doc = {
                "doc_id": item.get("doc_key", f"scierc_{len(documents)}"),
                "text": " ".join([" ".join(sent) for sent in item["sentences"]]),
                "entities": [],
                "relations": []
            }
            
            # Parse entities (NER annotations)
            for sent_ner in item.get("ner", []):
                for ent in sent_ner:
                    start, end, ent_type = ent
                    doc["entities"].append({
                        "start": start,
                        "end": end,
                        "type": ent_type
                    })
            
            # Parse relations
            for sent_rels in item.get("relations", []):
                for rel in sent_rels:
                    head_start, head_end, tail_start, tail_end, rel_type = rel
                    doc["relations"].append({
                        "head": (head_start, head_end),
                        "tail": (tail_start, tail_end),
                        "relation": rel_type
                    })
            
            documents.append(doc)
    
    return documents
```

### File: `src/benchmarks/metrics.py`

```python
from typing import List, Dict, Set, Tuple
from collections import defaultdict

def compute_entity_metrics(predicted: List[Dict], gold: List[Dict]) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for entity extraction
    Matching criteria: exact text match + type match
    """
    pred_entities = set()
    gold_entities = set()
    
    for doc in predicted:
        for ent in doc.get("entities", []):
            pred_entities.add((doc["doc_id"], ent["text"].lower(), ent["type"]))
    
    for doc in gold:
        for ent in doc.get("entities", []):
            gold_entities.add((doc["doc_id"], ent["text"].lower(), ent["type"]))
    
    tp = len(pred_entities & gold_entities)
    fp = len(pred_entities - gold_entities)
    fn = len(gold_entities - pred_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "entity_tp": tp,
        "entity_fp": fp,
        "entity_fn": fn
    }

def compute_relation_metrics(predicted: List[Dict], gold: List[Dict]) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for relation extraction
    Matching criteria: (head_text, tail_text, relation_type) tuple match
    """
    pred_relations = set()
    gold_relations = set()
    
    # Build entity lookup for predicted
    for doc in predicted:
        ent_lookup = {e["id"]: e["text"].lower() for e in doc.get("entities", [])}
        for rel in doc.get("relations", []):
            head = ent_lookup.get(rel["head_entity_id"], "")
            tail = ent_lookup.get(rel["tail_entity_id"], "")
            pred_relations.add((doc["doc_id"], head, tail, rel["relation_type"]))
    
    # Build entity lookup for gold
    for doc in gold:
        for rel in doc.get("relations", []):
            # Assuming gold has direct text references or indices
            head = str(rel.get("head", "")).lower()
            tail = str(rel.get("tail", "")).lower()
            rel_type = rel.get("relation", "")
            gold_relations.add((doc["doc_id"], head, tail, rel_type))
    
    tp = len(pred_relations & gold_relations)
    fp = len(pred_relations - gold_relations)
    fn = len(gold_relations - pred_relations)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "relation_precision": precision,
        "relation_recall": recall,
        "relation_f1": f1,
        "relation_tp": tp,
        "relation_fp": fp,
        "relation_fn": fn
    }

def compute_all_metrics(predicted: List[Dict], gold: List[Dict]) -> Dict[str, float]:
    """Compute all extraction metrics"""
    entity_metrics = compute_entity_metrics(predicted, gold)
    relation_metrics = compute_relation_metrics(predicted, gold)
    
    return {**entity_metrics, **relation_metrics}
```

### File: `src/benchmarks/runner.py`

```python
import yaml
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List

from src.extraction.extractor import Extractor
from src.benchmarks.metrics import compute_all_metrics

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
        
        if dataset_name == "docred":
            from src.benchmarks.datasets.docred_loader import load_docred
            return load_docred(data_dir, split)
        elif dataset_name == "scierc":
            from src.benchmarks.datasets.scierc_loader import load_scierc
            return load_scierc(data_dir, split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def run(self) -> Dict:
        """Run benchmark and return metrics"""
        print(f"Loading dataset: {self.bench_config['dataset']}")
        gold_data = self.load_dataset()
        
        # Limit samples for testing
        max_samples = self.bench_config.get("max_samples", len(gold_data))
        gold_data = gold_data[:max_samples]
        
        print(f"Running extraction on {len(gold_data)} documents...")
        predicted_data = []
        
        for doc in tqdm(gold_data):
            result = self.extractor.extract(doc["text"], doc["doc_id"])
            
            predicted_data.append({
                "doc_id": result.doc_id,
                "entities": [
                    {
                        "id": e.id,
                        "text": e.text,
                        "type": e.type,
                        "confidence": e.confidence
                    }
                    for e in result.entities
                ],
                "relations": [
                    {
                        "id": r.id,
                        "head_entity_id": r.head_entity_id,
                        "tail_entity_id": r.tail_entity_id,
                        "relation_type": r.relation_type,
                        "confidence": r.confidence
                    }
                    for r in result.relations
                ]
            })
        
        # Compute metrics
        print("Computing metrics...")
        metrics = compute_all_metrics(predicted_data, gold_data)
        
        # Save results
        self.save_results(metrics, predicted_data, gold_data)
        
        return metrics
    
    def save_results(self, metrics: Dict, predicted: List[Dict], gold: List[Dict]):
        """Save benchmark results"""
        output_dir = Path(self.global_config["paths"]["outputs_dir"]) / \
                     self.bench_config["dataset"] / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        with open(output_dir / "predictions.json", 'w') as f:
            json.dump(predicted, f, indent=2)
        
        # Save config snapshot
        with open(output_dir / "config.yaml", 'w') as f:
            yaml.dump({
                "benchmark": self.bench_config,
                "global": self.global_config,
                "run_id": self.run_id
            }, f)
        
        print(f"\nResults saved to: {output_dir}")
```

### File: `src/benchmarks/configs/docred.yaml`

```yaml
dataset: docred
data_dir: ./data
split: dev
max_samples: 50  # Limit for testing; remove for full benchmark
```

### File: `src/benchmarks/configs/scierc.yaml`

```yaml
dataset: scierc
data_dir: ./data
split: dev
max_samples: 50
```

---

## Step 10: Create CLI Interface

### File: `src/cli/main.py`

```python
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

def cmd_extract(args):
    """Extract entities and relations from a document"""
    from src.extraction.extractor import Extractor
    from src.graph.gt_client import GTClient
    
    config = load_config()
    extractor = Extractor(config)
    gt_client = GTClient()
    
    # Read input
    with open(args.input, 'r') as f:
        text = f.read()
    
    doc_id = args.doc_id or Path(args.input).stem
    
    print(f"Extracting from: {doc_id}")
    result = extractor.extract(text, doc_id)
    
    print(f"\nExtracted:")
    print(f"  - {len(result.entities)} entities")
    print(f"  - {len(result.relations)} relations")
    
    if args.store:
        print("\nStoring in Gₜ...")
        gt_client.store_extraction(result)
        print("✓ Stored successfully")
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result.model_dump(), f, indent=2)
        print(f"\n✓ Saved to: {args.output}")
    
    gt_client.close()

def cmd_benchmark(args):
    """Run benchmark on a dataset"""
    from src.benchmarks.runner import BenchmarkRunner
    
    config = load_config()
    runner = BenchmarkRunner(args.config, config)
    
    metrics = runner.run()
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:25s}: {value:.4f}")

def cmd_query(args):
    """Query the temporary knowledge graph"""
    from src.graph.gt_client import GTClient
    
    gt_client = GTClient()
    
    if args.entities:
        entities = gt_client.query_entities(
            entity_type=args.type,
            min_confidence=args.min_confidence
        )
        print(f"\nFound {len(entities)} entities:")
        for ent in entities[:args.limit]:
            print(f"  - {ent['text']} ({ent['type']}) [conf: {ent['confidence']:.2f}]")
    
    if args.relations:
        relations = gt_client.query_relations(
            relation_type=args.type,
            min_confidence=args.min_confidence
        )
        print(f"\nFound {len(relations)} relations:")
        for rel in relations[:args.limit]:
            print(f"  - {rel['head']} --[{rel['type']}]--> {rel['tail']} [conf: {rel['confidence']:.2f}]")
    
    gt_client.close()

def cmd_init(args):
    """Initialize Neo4j constraints"""
    from src.graph.gt_client import GTClient
    
    gt_client = GTClient()
    print("Initializing Gₜ constraints...")
    gt_client.initialize_constraints()
    print("✓ Constraints created")
    gt_client.close()

def cmd_clear(args):
    """Clear the knowledge graph"""
    from src.graph.gt_client import GTClient
    
    if not args.force:
        confirm = input("This will delete all data in Gₜ. Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted")
            return
    
    gt_client = GTClient()
    print("Clearing Gₜ...")
    gt_client.clear_graph()
    print("✓ Graph cleared")
    gt_client.close()

def main():
    parser = argparse.ArgumentParser(description="Dual-Layer Extraction System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract from document")
    extract_parser.add_argument("--input", required=True, help="Input text file")
    extract_parser.add_argument("--doc-id", help="Document ID (default: filename)")
    extract_parser.add_argument("--store", action="store_true", help="Store in Gₜ")
    extract_parser.add_argument("--output", help="Save extraction to JSON file")
    extract_parser.set_defaults(func=cmd_extract)
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--config", required=True, help="Benchmark config file")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query Gₜ")
    query_parser.add_argument("--entities", action="store_true", help="Query entities")
    query_parser.add_argument("--relations", action="store_true", help="Query relations")
    query_parser.add_argument("--type", help="Filter by type")
    query_parser.add_argument("--min-confidence", type=float, default=0.0, help="Min confidence")
    query_parser.add_argument("--limit", type=int, default=10, help="Max results")
    query_parser.set_defaults(func=cmd_query)
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize Neo4j constraints")
    init_parser.set_defaults(func=cmd_init)
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear Gₜ")
    clear_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    clear_parser.set_defaults(func=cmd_clear)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)

if __name__ == "__main__":
    main()
```

---

## Step 11: Create Empty **init**.py Files

Create empty `__init__.py` files in all package directories:

* `src/__init__.py`

* `src/extraction/__init__.py`

* `src/graph/__init__.py`

* `src/benchmarks/__init__.py`

* `src/benchmarks/datasets/__init__.py`

* `src/cli/__init__.py`

---

## Step 12: Create README

### File: `README.md`

```markdown
# Dual-Layer Extraction System

Modular LLM-driven information extraction system with temporary knowledge graph (Gₜ) for benchmarking extraction capability.

## Quick Start

### 1. Setup
```bash
# Copy environment file
cp .env.example .env

# Start services
cd docker
docker compose up -d

# Pull Ollama model
docker exec -it ollama ollama pull qwen3:latest

# Initialize Neo4j constraints
docker exec -it dual_app python src/cli/main.py init
```

### 2\. Extract from Document

```bash
docker exec -it dual_app python src/cli/main.py extract \
  --input data/sample.txt \
  --store \
  --output outputs/sample_extraction.json
```

### 3\. Query Extracted Data

```bash
# Query entities
docker exec -it dual_app python src/cli/main.py query \
  --entities \
  --min-confidence 0.7

# Query relations
docker exec -it dual_app python src/cli/main.py query \
  --relations \
  --min-confidence 0.7
```

### 4\. Run Benchmarks

```bash
# Download DocRED/SciERC datasets to data/ directory first

# Run DocRED benchmark
docker exec -it dual_app python src/cli/main.py benchmark \
  --config src/benchmarks/configs/docred.yaml

# Results saved to outputs/docred/{run_id}/
```

## Architecture

* **Extraction Layer**: Qwen LLM extracts entities & relations

* **Temporary Graph (Gₜ)**: Neo4j stores raw extractions

* **Benchmarks**: DocRED, SciERC for validation

## Services

* Neo4j Browser: [http://localhost:7474](http://localhost:7474) (neo4j/neo4jpass)

* Ollama API: [http://localhost:11434](http://localhost:11434)

## CLI Commands

* `extract` - Extract from document

* `benchmark` - Run benchmark

* `query` - Query Gₜ

* `init` - Initialize constraints

* `clear` - Clear graph

```

---

## Step 13: Build and Run

Execute these commands in order:

```bash
# 1. Create project directory
mkdir dual-layer && cd dual-layer

# 2. Copy .env.example to .env
cp .env.example .env

# 3. Create data and outputs directories
mkdir -p data outputs

# 4. Start Docker services
cd docker
docker compose up -d

# 5. Wait for services to start (30 seconds)
sleep 30

# 6. Pull Ollama model
docker exec -it ollama ollama pull qwen3:latest

# 7. Initialize Neo4j constraints
docker exec -it dual_app python src/cli/main.py init

# 8. Test extraction (create a sample.txt first)
echo "John Smith works at Google in Mountain View." > data/sample.txt
docker exec -it dual_app python src/cli/main.py extract \
  --input data/sample.txt \
  --store \
  --output outputs/test.json

# 9. Query results
docker exec -it dual_app python src/cli/main.py query --entities
```

---

## Important Notes for Cursor

1. **Create ALL files exactly as specified** - do not skip any files

2. **Maintain exact indentation** - Python is whitespace-sensitive

3. **Create empty `__init__.py`** in all package directories

4. **Use forward slashes** in paths (cross-platform compatible)

5. **Test incrementally** - build Docker first, then test each component

6. **Check Neo4j connection** before running extraction

7. **Verify Ollama model** is pulled before extraction

---

## Testing Checklist

* \[ \] Docker services start successfully

* \[ \] Neo4j accessible at localhost:7474

* \[ \] Ollama model pulled successfully

* \[ \] Constraints initialized in Neo4j

* \[ \] Single document extraction works

* \[ \] Entities stored in Gₜ

* \[ \] Query commands return results

* \[ \] Benchmark config loads correctly

---

## Next Steps After Build

1. Download DocRED/SciERC datasets

2. Run benchmarks to validate extraction

3. Tune prompts and confidence thresholds

4. Add domain-specific extraction (KTH Innovation Index)

5. Implement promotion layer (Gₜ → Gₚ)