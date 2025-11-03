from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

# Optional import with graceful fallback
try:
    from src.extraction.huggingface_client import HuggingFaceCypherClient
    HUGGINGFACE_AVAILABLE = True
except ImportError as e:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceCypherClient = None

logger = logging.getLogger(__name__)

@dataclass
class GraphEntity:
    """Represents an entity in the knowledge graph"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = None
    confidence: float = 1.0

@dataclass
class GraphRelation:
    """Represents a relation in the knowledge graph"""
    id: str
    head_entity_id: str
    tail_entity_id: str
    relation_type: str
    properties: Dict[str, Any] = None
    confidence: float = 1.0

@dataclass
class GraphConcept:
    """Represents a concept in the knowledge graph"""
    id: str
    name: str
    description: str
    properties: Dict[str, Any] = None
    confidence: float = 1.0

@dataclass
class GraphFact:
    """Represents a factual statement in the knowledge graph"""
    id: str
    statement: str
    source_entities: List[str]
    confidence: float = 1.0
    properties: Dict[str, Any] = None

class KnowledgeGraphExtractor:
    """
    Extracts structured information from documents and populates Neo4j knowledge graph
    Uses Ollama with Qwen3:latest for extraction and Hugging Face models for text-to-Cypher conversion
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 huggingface_model: str = "google/flan-t5-base"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.session = self.driver.session()
        
        # Initialize Hugging Face client for text-to-Cypher conversion
        if HUGGINGFACE_AVAILABLE:
            try:
                self.cypher_client = HuggingFaceCypherClient(model_name=huggingface_model)
                logger.info(f"Initialized Hugging Face client with model: {huggingface_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Hugging Face client: {e}")
                self.cypher_client = None
        else:
            logger.warning("Hugging Face client not available - transformers library not installed")
            self.cypher_client = None
        
    def close(self):
        """Close the Neo4j connection"""
        self.session.close()
        self.driver.close()
    
    def clear_database(self):
        """Clear all data from the Neo4j database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared Neo4j database")
    
    def extract_and_store(self, doc_id: str, text: str, extraction_result: Any) -> Dict[str, Any]:
        """
        Extract entities, relations, concepts, and facts from text and store in Neo4j
        
        Args:
            doc_id: Document identifier
            text: Document text content
            extraction_result: Result from the extraction pipeline
            
        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            "entities_created": 0,
            "relations_created": 0,
            "concepts_created": 0,
            "facts_created": 0
        }
        
        try:
            # Create document node
            self._create_document_node(doc_id, text)
            
            # Extract and store entities
            entities = self._extract_entities(text, extraction_result)
            for entity in entities:
                self._create_entity_node(entity, doc_id)
                stats["entities_created"] += 1
            
            # Extract and store relations
            relations = self._extract_relations(text, extraction_result, entities)
            for relation in relations:
                self._create_relation_edge(relation, doc_id)
                stats["relations_created"] += 1
            
            # Extract and store concepts
            concepts = self._extract_concepts(text, extraction_result)
            for concept in concepts:
                self._create_concept_node(concept, doc_id)
                stats["concepts_created"] += 1
            
            # Extract and store facts
            facts = self._extract_facts(text, extraction_result, entities)
            for fact in facts:
                self._create_fact_node(fact, doc_id)
                stats["facts_created"] += 1
            
            logger.info(f"Extracted {stats} for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error extracting from document {doc_id}: {e}")
            raise
        
        return stats
    
    def convert_query_to_cypher(self, natural_language_query: str, context: Optional[str] = None) -> str:
        """
        Convert natural language query to Cypher using Hugging Face model
        
        Args:
            natural_language_query: The natural language query
            context: Optional context about the knowledge graph
            
        Returns:
            Cypher query string
        """
        if self.cypher_client:
            try:
                return self.cypher_client.convert_text_to_cypher(natural_language_query, context)
            except Exception as e:
                logger.error(f"Error converting query to Cypher: {e}")
                # Fallback to pattern-based conversion
                return self._fallback_cypher_conversion(natural_language_query)
        else:
            return self._fallback_cypher_conversion(natural_language_query)
    
    def _fallback_cypher_conversion(self, query: str) -> str:
        """Fallback method for text-to-Cypher conversion"""
        query_lower = query.lower()
        
        if 'entity' in query_lower or 'entities' in query_lower:
            return "MATCH (e:Entity) RETURN e.name, e.type LIMIT 100"
        elif 'relation' in query_lower or 'relationship' in query_lower:
            return "MATCH (e1:Entity)-[r:RELATION]->(e2:Entity) RETURN e1.name, r.type, e2.name LIMIT 100"
        elif 'concept' in query_lower or 'concepts' in query_lower:
            return "MATCH (c:Concept) RETURN c.name, c.description LIMIT 100"
        elif 'fact' in query_lower or 'facts' in query_lower:
            return "MATCH (f:Fact) RETURN f.statement, f.confidence LIMIT 100"
        else:
            return "MATCH (e:Entity) RETURN e.name, e.type LIMIT 100"
    
    def _create_document_node(self, doc_id: str, text: str):
        """Create a document node in Neo4j"""
        with self.driver.session() as session:
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.text = $text,
                    d.length = $length,
                    d.created_at = datetime()
            """, doc_id=doc_id, text=text, length=len(text))
    
    def _extract_entities(self, text: str, extraction_result: Any) -> List[GraphEntity]:
        """Extract entities from text using the extraction pipeline"""
        entities = []
        
        # Use the existing extraction pipeline
        if hasattr(extraction_result, 'entities'):
            for i, entity in enumerate(extraction_result.entities):
                graph_entity = GraphEntity(
                    id=f"{extraction_result.doc_id}_entity_{i}",
                    name=entity.text,
                    type=entity.type,
                    confidence=entity.confidence,
                    properties={
                        "start_pos": getattr(entity, 'start_pos', None),
                        "end_pos": getattr(entity, 'end_pos', None)
                    }
                )
                entities.append(graph_entity)
        
        return entities
    
    def _extract_relations(self, text: str, extraction_result: Any, entities: List[GraphEntity]) -> List[GraphRelation]:
        """Extract relations from text using the extraction pipeline"""
        relations = []
        
        if hasattr(extraction_result, 'relations'):
            for i, relation in enumerate(extraction_result.relations):
                graph_relation = GraphRelation(
                    id=f"{extraction_result.doc_id}_relation_{i}",
                    head_entity_id=relation.head_entity_id,
                    tail_entity_id=relation.tail_entity_id,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence
                )
                relations.append(graph_relation)
        
        return relations
    
    def _extract_concepts(self, text: str, extraction_result: Any) -> List[GraphConcept]:
        """Extract key concepts from text"""
        concepts = []
        
        # Simple concept extraction based on text analysis
        # This could be enhanced with more sophisticated NLP techniques
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('##') and len(line) > 3:  # Markdown headers
                concept_name = line.replace('##', '').strip()
                if len(concept_name) > 0:
                    concept = GraphConcept(
                        id=f"{extraction_result.doc_id}_concept_{i}",
                        name=concept_name,
                        description=f"Concept extracted from header: {concept_name}",
                        confidence=0.8
                    )
                    concepts.append(concept)
        
        return concepts
    
    def _extract_facts(self, text: str, extraction_result: Any, entities: List[GraphEntity]) -> List[GraphFact]:
        """Extract factual statements from text"""
        facts = []
        
        # Simple fact extraction - look for sentences with entities
        sentences = text.split('.')
        entity_names = [e.name.lower() for e in entities]
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 20:  # Reasonable sentence length
                # Check if sentence contains entities
                sentence_lower = sentence.lower()
                mentioned_entities = [name for name in entity_names if name in sentence_lower]
                
                if mentioned_entities:
                    fact = GraphFact(
                        id=f"{extraction_result.doc_id}_fact_{i}",
                        statement=sentence,
                        source_entities=mentioned_entities,
                        confidence=0.7
                    )
                    facts.append(fact)
        
        return facts
    
    def _create_entity_node(self, entity: GraphEntity, doc_id: str):
        """Create an entity node in Neo4j"""
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name,
                    e.type = $type,
                    e.confidence = $confidence,
                    e.properties = $properties
                
                MERGE (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(e)
            """, entity_id=entity.id, name=entity.name, type=entity.type,
                confidence=entity.confidence, properties=entity.properties or {},
                doc_id=doc_id)
    
    def _create_relation_edge(self, relation: GraphRelation, doc_id: str):
        """Create a relation edge in Neo4j"""
        with self.driver.session() as session:
            session.run("""
                MATCH (e1:Entity {id: $head_id})
                MATCH (e2:Entity {id: $tail_id})
                MERGE (e1)-[r:RELATION {id: $relation_id}]->(e2)
                SET r.type = $relation_type,
                    r.confidence = $confidence,
                    r.properties = $properties
                
                MERGE (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(r)
            """, head_id=relation.head_entity_id, tail_id=relation.tail_entity_id,
                relation_id=relation.id, relation_type=relation.relation_type,
                confidence=relation.confidence, properties=relation.properties or {},
                doc_id=doc_id)
    
    def _create_concept_node(self, concept: GraphConcept, doc_id: str):
        """Create a concept node in Neo4j"""
        with self.driver.session() as session:
            session.run("""
                MERGE (c:Concept {id: $concept_id})
                SET c.name = $name,
                    c.description = $description,
                    c.confidence = $confidence,
                    c.properties = $properties
                
                MERGE (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(c)
            """, concept_id=concept.id, name=concept.name, description=concept.description,
                confidence=concept.confidence, properties=concept.properties or {},
                doc_id=doc_id)
    
    def _create_fact_node(self, fact: GraphFact, doc_id: str):
        """Create a fact node in Neo4j"""
        with self.driver.session() as session:
            session.run("""
                MERGE (f:Fact {id: $fact_id})
                SET f.statement = $statement,
                    f.confidence = $confidence,
                    f.source_entities = $source_entities,
                    f.properties = $properties
                
                MERGE (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(f)
            """, fact_id=fact.id, statement=fact.statement, confidence=fact.confidence,
                source_entities=fact.source_entities, properties=fact.properties or {},
                doc_id=doc_id)
