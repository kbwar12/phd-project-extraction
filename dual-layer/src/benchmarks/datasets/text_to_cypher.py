from typing import Dict, List, Optional, Any
import re
import logging

logger = logging.getLogger(__name__)

class TextToCypherConverter:
    """
    Converts natural language queries to Cypher queries for knowledge graph evaluation
    """
    
    def __init__(self):
        # Define common query patterns and their Cypher equivalents
        self.query_patterns = {
            # Entity extraction patterns
            r"what entities? (?:are )?(?:mentioned|found|present)": "MATCH (e:Entity) RETURN e.name, e.type",
            r"list all entities?": "MATCH (e:Entity) RETURN e.name, e.type",
            r"what (?:are )?the entities?": "MATCH (e:Entity) RETURN e.name, e.type",
            
            # Relation extraction patterns
            r"what relationships? (?:exist|are there)": "MATCH (e1:Entity)-[r:RELATION]->(e2:Entity) RETURN e1.name, r.type, e2.name",
            r"how (?:are|do) .* (?:related|connected)": "MATCH (e1:Entity)-[r:RELATION]->(e2:Entity) RETURN e1.name, r.type, e2.name",
            r"what (?:are )?the relations?": "MATCH (e1:Entity)-[r:RELATION]->(e2:Entity) RETURN e1.name, r.type, e2.name",
            
            # Concept extraction patterns
            r"what (?:are )?the (?:main )?concepts?": "MATCH (c:Concept) RETURN c.name, c.description",
            r"what topics? (?:are )?(?:discussed|covered)": "MATCH (c:Concept) RETURN c.name, c.description",
            r"list the concepts?": "MATCH (c:Concept) RETURN c.name, c.description",
            
            # Fact extraction patterns
            r"what facts? (?:are )?(?:stated|mentioned)": "MATCH (f:Fact) RETURN f.statement, f.confidence",
            r"what (?:are )?the facts?": "MATCH (f:Fact) RETURN f.statement, f.confidence",
            r"list the facts?": "MATCH (f:Fact) RETURN f.statement, f.confidence",
            
            # Specific entity type queries
            r"what (?:are )?the (?:people|persons?)": "MATCH (e:Entity {type: 'PERSON'}) RETURN e.name, e.properties",
            r"what (?:are )?the (?:organizations?|orgs?)": "MATCH (e:Entity {type: 'ORG'}) RETURN e.name, e.properties",
            r"what (?:are )?the (?:locations?|places?)": "MATCH (e:Entity {type: 'LOC'}) RETURN e.name, e.properties",
            r"what (?:are )?the (?:dates?|times?)": "MATCH (e:Entity {type: 'DATE'}) RETURN e.name, e.properties",
            
            # Relationship type queries
            r"who (?:works for|is employed by)": "MATCH (p:Entity {type: 'PERSON'})-[r:RELATION {type: 'WORKS_FOR'}]->(o:Entity {type: 'ORG'}) RETURN p.name, o.name",
            r"what (?:is|are) (?:located|situated) (?:in|at)": "MATCH (e1:Entity)-[r:RELATION {type: 'LOCATED_IN'}]->(e2:Entity {type: 'LOC'}) RETURN e1.name, e2.name",
            r"when (?:did|does) .* (?:happen|occur)": "MATCH (e:Entity)-[r:RELATION {type: 'OCCURS_AT'}]->(d:Entity {type: 'DATE'}) RETURN e.name, d.name",
        }
        
        # Entity type mappings
        self.entity_type_mappings = {
            'person': 'PERSON',
            'people': 'PERSON',
            'organization': 'ORG',
            'org': 'ORG',
            'location': 'LOC',
            'place': 'LOC',
            'date': 'DATE',
            'time': 'DATE',
            'money': 'MONEY',
            'percent': 'PERCENT'
        }
        
        # Relation type mappings
        self.relation_type_mappings = {
            'works for': 'WORKS_FOR',
            'employed by': 'WORKS_FOR',
            'located in': 'LOCATED_IN',
            'situated in': 'LOCATED_IN',
            'happens at': 'OCCURS_AT',
            'occurs at': 'OCCURS_AT',
            'born in': 'BORN_IN',
            'died in': 'DIED_IN',
            'founded by': 'FOUNDED_BY',
            'part of': 'PART_OF'
        }
    
    def convert_query(self, natural_language_query: str) -> str:
        """
        Convert a natural language query to a Cypher query
        
        Args:
            natural_language_query: The natural language query
            
        Returns:
            Cypher query string
        """
        query_lower = natural_language_query.lower().strip()
        
        # Try to match against predefined patterns
        for pattern, cypher_template in self.query_patterns.items():
            if re.search(pattern, query_lower):
                logger.info(f"Matched pattern '{pattern}' for query: {natural_language_query}")
                return cypher_template
        
        # If no pattern matches, try to construct a basic query
        return self._construct_basic_query(query_lower)
    
    def _construct_basic_query(self, query: str) -> str:
        """
        Construct a basic Cypher query when no pattern matches
        
        Args:
            query: Lowercase natural language query
            
        Returns:
            Basic Cypher query
        """
        # Look for entity types
        for entity_term, entity_type in self.entity_type_mappings.items():
            if entity_term in query:
                return f"MATCH (e:Entity {{type: '{entity_type}'}}) RETURN e.name, e.properties"
        
        # Look for relation types
        for relation_term, relation_type in self.relation_type_mappings.items():
            if relation_term in query:
                return f"MATCH (e1:Entity)-[r:RELATION {{type: '{relation_type}'}}]->(e2:Entity) RETURN e1.name, r.type, e2.name"
        
        # Default to general entity query
        logger.warning(f"No specific pattern found for query: {query}, using default")
        return "MATCH (e:Entity) RETURN e.name, e.type LIMIT 100"
    
    def convert_query_with_context(self, natural_language_query: str, document_context: str = None) -> str:
        """
        Convert a natural language query to Cypher with document context
        
        Args:
            natural_language_query: The natural language query
            document_context: Optional document context for better conversion
            
        Returns:
            Cypher query string
        """
        # For now, use the basic conversion
        # This could be enhanced with context-aware processing
        return self.convert_query(natural_language_query)
    
    def validate_cypher_query(self, cypher_query: str) -> bool:
        """
        Validate if a Cypher query is syntactically correct
        
        Args:
            cypher_query: The Cypher query to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - check for common Cypher keywords
            required_keywords = ['MATCH', 'RETURN']
            query_upper = cypher_query.upper()
            
            for keyword in required_keywords:
                if keyword not in query_upper:
                    return False
            
            # Check for balanced parentheses and brackets
            if not self._check_balanced_brackets(cypher_query):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating Cypher query: {e}")
            return False
    
    def _check_balanced_brackets(self, query: str) -> bool:
        """Check if brackets and parentheses are balanced in the query"""
        stack = []
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in query:
            if char in bracket_pairs:
                stack.append(char)
            elif char in bracket_pairs.values():
                if not stack:
                    return False
                last_open = stack.pop()
                if bracket_pairs[last_open] != char:
                    return False
        
        return len(stack) == 0
    
    def get_query_examples(self) -> List[Dict[str, str]]:
        """
        Get example natural language queries and their Cypher equivalents
        
        Returns:
            List of dictionaries with 'query' and 'cypher' keys
        """
        examples = [
            {
                "query": "What entities are mentioned in this document?",
                "cypher": "MATCH (e:Entity) RETURN e.name, e.type"
            },
            {
                "query": "What relationships exist between entities?",
                "cypher": "MATCH (e1:Entity)-[r:RELATION]->(e2:Entity) RETURN e1.name, r.type, e2.name"
            },
            {
                "query": "What are the main concepts discussed?",
                "cypher": "MATCH (c:Concept) RETURN c.name, c.description"
            },
            {
                "query": "What facts are stated in this document?",
                "cypher": "MATCH (f:Fact) RETURN f.statement, f.confidence"
            },
            {
                "query": "What are the people mentioned?",
                "cypher": "MATCH (e:Entity {type: 'PERSON'}) RETURN e.name, e.properties"
            },
            {
                "query": "Who works for which organization?",
                "cypher": "MATCH (p:Entity {type: 'PERSON'})-[r:RELATION {type: 'WORKS_FOR'}]->(o:Entity {type: 'ORG'}) RETURN p.name, o.name"
            }
        ]
        
        return examples
