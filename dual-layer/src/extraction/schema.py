from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

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
    properties: Optional[Dict[str, Any]] = None

class Relation(BaseModel):
    """Extracted relation between entities"""
    id: str
    head_entity_id: str
    tail_entity_id: str
    relation_type: str  # e.g., "works_for", "located_in", "part_of"
    confidence: float = Field(ge=0, le=1)
    evidence_text: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class Evidence(BaseModel):
    """Supporting evidence for extraction"""
    doc_id: str
    sentence_id: Optional[str] = None
    text: str
    span: Optional[Span] = None

class Concept(BaseModel):
    """Extracted concept"""
    id: str
    name: str
    description: str
    confidence: float = Field(ge=0, le=1)
    properties: Optional[Dict[str, Any]] = None

class Fact(BaseModel):
    """Extracted factual statement"""
    id: str
    statement: str
    source_entities: List[str]
    confidence: float = Field(ge=0, le=1)
    properties: Optional[Dict[str, Any]] = None

class ExtractionResult(BaseModel):
    """Complete extraction output for a document"""
    doc_id: str
    entities: List[Entity]
    relations: List[Relation]
    concepts: Optional[List[Concept]] = None
    facts: Optional[List[Fact]] = None
    evidence: List[Evidence]
    metadata: dict = {}

