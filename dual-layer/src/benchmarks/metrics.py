from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

def compute_docbench_metrics(neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                            queries: List[Dict], doc_id: str = None) -> Dict[str, Any]:
    """
    Compute DocBench metrics by executing queries against the knowledge graph
    
    Args:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        queries: List of query templates to evaluate
        doc_id: Optional document ID to filter results
        
    Returns:
        Dictionary with evaluation metrics
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    try:
        with driver.session() as session:
            metrics = {
                "total_queries": len(queries),
                "successful_queries": 0,
                "failed_queries": 0,
                "query_results": {},
                "overall_score": 0.0
            }
            
            total_score = 0.0
            
            for query_template in queries:
                query_id = query_template["id"]
                cypher_query = query_template["cypher_template"]
                
                try:
                    # Add document filter if specified
                    if doc_id:
                        # Modify query to filter by document
                        if "WHERE" in cypher_query.upper():
                            cypher_query = cypher_query.replace("WHERE", f"WHERE d.id = '{doc_id}' AND")
                        else:
                            # Add WHERE clause
                            cypher_query = cypher_query.replace("MATCH", f"MATCH (d:Document {{id: '{doc_id}'}})-[:CONTAINS]->")
                    
                    # Execute query
                    result = session.run(cypher_query)
                    records = list(result)
                    
                    # Calculate score based on result quality
                    score = _calculate_query_score(records, query_template)
                    
                    metrics["query_results"][query_id] = {
                        "cypher_query": cypher_query,
                        "natural_language": query_template.get("natural_language", ""),
                        "result_count": len(records),
                        "score": score,
                        "results": records[:10]  # Limit results for readability
                    }
                    
                    metrics["successful_queries"] += 1
                    total_score += score
                    
                except Exception as e:
                    logger.error(f"Error executing query {query_id}: {e}")
                    metrics["query_results"][query_id] = {
                        "cypher_query": cypher_query,
                        "natural_language": query_template.get("natural_language", ""),
                        "error": str(e),
                        "score": 0.0
                    }
                    metrics["failed_queries"] += 1
            
            # Calculate overall score
            if metrics["successful_queries"] > 0:
                metrics["overall_score"] = total_score / metrics["successful_queries"]
            
            return metrics
            
    finally:
        driver.close()

def _calculate_query_score(records: List, query_template: Dict) -> float:
    """
    Calculate a score for query results based on various factors
    
    Args:
        records: Query result records
        query_template: Original query template
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not records:
        return 0.0
    
    # Base score from result count (normalized)
    result_count_score = min(len(records) / 10.0, 1.0)  # Cap at 1.0 for 10+ results
    
    # Quality score based on result completeness
    quality_score = 0.0
    for record in records:
        # Check if record has meaningful data
        values = [v for v in record.values() if v is not None and str(v).strip()]
        if len(values) > 0:
            quality_score += 1.0
    
    if records:
        quality_score = quality_score / len(records)
    
    # Combine scores
    final_score = (result_count_score * 0.4) + (quality_score * 0.6)
    return min(final_score, 1.0)

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
        # Build entity lookup from gold entities
        ent_lookup = {e.get("id", -1): e["text"].lower() for e in doc.get("entities", [])}
        
        for rel in doc.get("relations", []):
            # Use head_text/tail_text if available, otherwise look up by index
            if "head_text" in rel and "tail_text" in rel:
                head = rel["head_text"]
                tail = rel["tail_text"]
            else:
                head = ent_lookup.get(rel.get("head", -1), "")
                tail = ent_lookup.get(rel.get("tail", -1), "")
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

