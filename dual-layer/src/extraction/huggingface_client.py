import os
import logging
from typing import Optional, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt

# Optional imports with graceful fallback
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    pipeline = None

logger = logging.getLogger(__name__)

class HuggingFaceCypherClient:
    """
    Client for using Hugging Face models for text-to-Cypher conversion
    """
    
    def __init__(self, model_name: str = "dbands/Qwen2-5-Coder-0-5B-neo4j-text2cypher-2024v1-GGUF", device: str = "auto"):
        """
        Initialize the Hugging Face client
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ("auto", "cpu", "cuda")
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not available. Please install with: "
                "pip install transformers torch accelerate datasets"
            )
        
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer"""
        try:
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            
            # For text-to-Cypher, we'll use a text generation model
            # You can replace this with specialized models like:
            # - "microsoft/DialoGPT-medium" (general conversation)
            # - "facebook/blenderbot-400M-distill" (conversational)
            # - "google/flan-t5-base" (instruction following)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model.to(self.device)
            
            # Create pipeline for text generation
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.9
            )
            
            logger.info(f"Successfully loaded model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise
    
    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def convert_text_to_cypher(self, natural_language_query: str, context: Optional[str] = None) -> str:
        """
        Convert natural language query to Cypher using Hugging Face model
        
        Args:
            natural_language_query: The natural language query
            context: Optional context about the knowledge graph schema
            
        Returns:
            Cypher query string
        """
        try:
            # Create a prompt for text-to-Cypher conversion
            prompt = self._create_cypher_prompt(natural_language_query, context)
            
            # Generate Cypher query
            result = self.pipeline(
                prompt,
                max_length=256,
                num_return_sequences=1,
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=True
            )
            
            cypher_query = result[0]['generated_text'].strip()
            
            # Clean up the output
            cypher_query = self._clean_cypher_output(cypher_query)
            
            logger.info(f"Generated Cypher query: {cypher_query}")
            return cypher_query
            
        except Exception as e:
            logger.error(f"Error converting text to Cypher: {e}")
            # Fallback to basic pattern matching
            return self._fallback_cypher_conversion(natural_language_query)
    
    def _create_cypher_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Create a prompt for text-to-Cypher conversion"""
        
        schema_info = """
Knowledge Graph Schema:
- Nodes: Document, Entity, Concept, Fact
- Entity types: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PERCENT, MISC
- Relations: works_for, located_in, occurs_at, owns, causes, related_to, etc.
- Properties: name, type, confidence, properties (dict)
"""
        
        prompt = f"""Convert the following natural language query to a Cypher query for a knowledge graph.

{schema_info}

Natural Language Query: {query}

Cypher Query:"""
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        return prompt
    
    def _clean_cypher_output(self, output: str) -> str:
        """Clean and validate the generated Cypher query"""
        # Remove any extra text before the Cypher query
        lines = output.split('\n')
        cypher_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('MATCH') or line.startswith('RETURN') or line.startswith('WHERE') or line.startswith('WITH'):
                cypher_lines.append(line)
            elif line and not line.startswith('Natural Language') and not line.startswith('Cypher Query'):
                cypher_lines.append(line)
        
        return ' '.join(cypher_lines)
    
    def _fallback_cypher_conversion(self, query: str) -> str:
        """Fallback method using simple pattern matching"""
        query_lower = query.lower()
        
        # Simple pattern matching fallback
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
    
    def health_check(self) -> bool:
        """Check if the Hugging Face model is loaded and ready"""
        try:
            return self.pipeline is not None and self.model is not None
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.health_check(),
            "model_type": "text2text-generation"
        }

# Convenience function for easy integration
def create_cypher_client(model_name: str = "google/flan-t5-base", device: str = "auto") -> HuggingFaceCypherClient:
    """
    Create a Hugging Face Cypher client with default settings
    
    Args:
        model_name: Hugging Face model name
        device: Device to run on
        
    Returns:
        HuggingFaceCypherClient instance
    """
    return HuggingFaceCypherClient(model_name=model_name, device=device)
