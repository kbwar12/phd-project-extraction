import json
import uuid
from typing import List
from src.extraction.ollama_client import generate
from src.extraction.prompts import (
    ENTITY_EXTRACTION_PROMPT, 
    RELATION_EXTRACTION_PROMPT,
    CONCEPT_EXTRACTION_PROMPT,
    FACT_EXTRACTION_PROMPT
)
from src.extraction.schema import ExtractionResult, Entity, Relation, Evidence, Span, Concept, Fact

class Extractor:
    def __init__(self, config: dict):
        self.config = config
        self.confidence_threshold = config["extraction"]["confidence_threshold"]
    
    def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities from text using Qwen3:latest"""
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
            
            for i, ent_data in enumerate(parsed.get("entities", [])):
                if ent_data.get("confidence", 0) >= self.confidence_threshold:
                    # Ensure entity has required fields
                    if "id" not in ent_data:
                        ent_data["id"] = f"{doc_id}_entity_{i}"
                    
                    # Handle properties field
                    if "properties" not in ent_data:
                        ent_data["properties"] = {}
                    
                    entities.append(Entity(**ent_data))
            
            return entities
        
        except Exception as e:
            print(f"Entity extraction failed for {doc_id}: {e}")
            return []
    
    def extract_relations(self, text: str, entities: List[Entity], doc_id: str) -> List[Relation]:
        """Extract relations between entities using Qwen3:latest"""
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
            
            for i, rel_data in enumerate(parsed.get("relations", [])):
                if rel_data.get("confidence", 0) >= self.confidence_threshold:
                    # Ensure relation has required fields
                    if "id" not in rel_data:
                        rel_data["id"] = f"{doc_id}_relation_{i}"
                    
                    # Handle properties field
                    if "properties" not in rel_data:
                        rel_data["properties"] = {}
                    
                    relations.append(Relation(**rel_data))
            
            return relations
        
        except Exception as e:
            print(f"Relation extraction failed for {doc_id}: {e}")
            return []
    
    def extract_concepts(self, text: str, doc_id: str) -> List[Concept]:
        """Extract concepts from text using Qwen3:latest"""
        prompt = CONCEPT_EXTRACTION_PROMPT.format(text=text)
        
        try:
            raw_response = generate(
                prompt,
                temperature=self.config["ollama"]["temperature"],
                max_tokens=self.config["ollama"]["max_tokens"]
            )
            
            parsed = json.loads(raw_response)
            concepts = []
            
            for i, concept_data in enumerate(parsed.get("concepts", [])):
                if concept_data.get("confidence", 0) >= self.confidence_threshold:
                    # Ensure concept has required fields
                    if "id" not in concept_data:
                        concept_data["id"] = f"{doc_id}_concept_{i}"
                    
                    # Handle properties field
                    if "properties" not in concept_data:
                        concept_data["properties"] = {}
                    
                    concepts.append(Concept(**concept_data))
            
            return concepts
        
        except Exception as e:
            print(f"Concept extraction failed for {doc_id}: {e}")
            return []
    
    def extract_facts(self, text: str, entities: List[Entity], doc_id: str) -> List[Fact]:
        """Extract facts from text using Qwen3:latest"""
        prompt = FACT_EXTRACTION_PROMPT.format(text=text)
        
        try:
            raw_response = generate(
                prompt,
                temperature=self.config["ollama"]["temperature"],
                max_tokens=self.config["ollama"]["max_tokens"]
            )
            
            parsed = json.loads(raw_response)
            facts = []
            
            for i, fact_data in enumerate(parsed.get("facts", [])):
                if fact_data.get("confidence", 0) >= self.confidence_threshold:
                    # Ensure fact has required fields
                    if "id" not in fact_data:
                        fact_data["id"] = f"{doc_id}_fact_{i}"
                    
                    # Handle properties field
                    if "properties" not in fact_data:
                        fact_data["properties"] = {}
                    
                    facts.append(Fact(**fact_data))
            
            return facts
        
        except Exception as e:
            print(f"Fact extraction failed for {doc_id}: {e}")
            return []
    
    def extract(self, text: str, doc_id: str) -> ExtractionResult:
        """Full extraction pipeline using Qwen3:latest"""
        # Step 1: Extract entities
        entities = self.extract_entities(text, doc_id)
        
        # Step 2: Extract relations
        relations = self.extract_relations(text, entities, doc_id)
        
        # Step 3: Extract concepts
        concepts = self.extract_concepts(text, doc_id)
        
        # Step 4: Extract facts
        facts = self.extract_facts(text, entities, doc_id)
        
        # Step 5: Package evidence
        evidence = [Evidence(doc_id=doc_id, text=text)]
        
        return ExtractionResult(
            doc_id=doc_id,
            entities=entities,
            relations=relations,
            concepts=concepts,
            facts=facts,
            evidence=evidence,
            metadata={"extractor_version": "v2.0", "model": "qwen3:latest"}
        )

