ENTITY_EXTRACTION_PROMPT = """You are an expert information extraction system specialized in knowledge graph construction.

Extract ALL entities from the following text with high precision. For each entity, provide:
- text: the exact entity mention (preserve original capitalization)
- type: one of [PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PERCENT, MISC]
- start: character offset where entity starts
- end: character offset where entity ends
- confidence: your confidence score (0.0 to 1.0)
- properties: additional attributes (e.g., title, role, amount, etc.)

Focus on:
- Named entities (people, organizations, places)
- Temporal expressions (dates, times, periods)
- Quantities (money, percentages, measurements)
- Important concepts and terms

Text:
---
{text}
---

Return ONLY valid JSON matching this schema:
{{
  "entities": [
    {{"id": "e1", "text": "...", "type": "...", "span": {{"start": 0, "end": 10, "text": "..."}}, "confidence": 0.95, "properties": {{}}}}
  ]
}}
"""

RELATION_EXTRACTION_PROMPT = """You are an expert information extraction system specialized in knowledge graph construction.

Given these entities:
{entities_json}

And this text:
---
{text}
---

Extract ALL semantic relations between entities. For each relation, provide:
- head_entity_id: ID of the source entity
- tail_entity_id: ID of the target entity
- relation_type: semantic relation from these categories:
  * WORK: works_for, employed_by, manages, reports_to
  * LOCATION: located_in, situated_in, based_in, operates_in
  * TEMPORAL: occurs_at, happens_during, starts_at, ends_at
  * OWNERSHIP: owns, belongs_to, part_of, member_of
  * CAUSAL: causes, results_in, leads_to, prevents
  * COMPARISON: similar_to, different_from, better_than, worse_than
  * OTHER: related_to, associated_with, mentions, references
- confidence: your confidence score (0.0 to 1.0)
- evidence_text: the exact sentence/phrase supporting this relation
- properties: additional context (e.g., duration, amount, context)

Focus on:
- Explicit relationships mentioned in the text
- Implicit relationships that can be reasonably inferred
- Temporal and causal relationships
- Hierarchical and organizational relationships

Return ONLY valid JSON matching this schema:
{{
  "relations": [
    {{"id": "r1", "head_entity_id": "e1", "tail_entity_id": "e2", "relation_type": "works_for", "confidence": 0.9, "evidence_text": "...", "properties": {{}}}}
  ]
}}
"""

CONCEPT_EXTRACTION_PROMPT = """You are an expert information extraction system specialized in knowledge graph construction.

Extract key concepts and topics from the following text. For each concept, provide:
- name: the concept name or topic
- description: a brief description of what this concept represents
- confidence: your confidence score (0.0 to 1.0)
- properties: additional context (e.g., category, importance, context)

Focus on:
- Main topics and themes
- Technical concepts and terminology
- Abstract concepts and ideas
- Domain-specific knowledge
- Key themes that appear multiple times

Text:
---
{text}
---

Return ONLY valid JSON matching this schema:
{{
  "concepts": [
    {{"id": "c1", "name": "...", "description": "...", "confidence": 0.8, "properties": {{}}}}
  ]
}}
"""

FACT_EXTRACTION_PROMPT = """You are an expert information extraction system specialized in knowledge graph construction.

Extract factual statements from the following text. For each fact, provide:
- statement: the factual statement in natural language
- source_entities: list of entity names mentioned in this fact
- confidence: your confidence score (0.0 to 1.0)
- properties: additional context (e.g., source, context, type)

Focus on:
- Verifiable factual claims
- Statements about entities and their properties
- Relationships and events
- Quantitative information
- Causal relationships

Text:
---
{text}
---

Return ONLY valid JSON matching this schema:
{{
  "facts": [
    {{"id": "f1", "statement": "...", "source_entities": ["entity1", "entity2"], "confidence": 0.7, "properties": {{}}}}
  ]
}}
"""
