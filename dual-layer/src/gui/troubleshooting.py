import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import yaml
from datetime import datetime
import requests
from neo4j import GraphDatabase

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from extraction.extractor import Extractor
from graph.gt_client import GTClient
from benchmarks.runner import BenchmarkRunner

class MonitoringDashboard:
    """Real-time monitoring dashboard for the dual-layer extraction system"""
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration"""
        config_path = Path("config/settings.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def check_service_health(self):
        """Check health of all services"""
        health = {}
        
        # Check Neo4j
        try:
            driver = GraphDatabase.driver(
                self.config.get("neo4j", {}).get("uri", "bolt://localhost:7687"),
                auth=(self.config.get("neo4j", {}).get("user", "neo4j"), 
                      self.config.get("neo4j", {}).get("password", "neo4jpass"))
            )
            with driver.session() as session:
                session.run("RETURN 1")
            health["Neo4j"] = {"status": "healthy", "response_time": 0.1}
            driver.close()
        except Exception as e:
            health["Neo4j"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Ollama
        try:
            ollama_url = self.config.get("ollama", {}).get("base_url", "http://localhost:11434")
            start_time = time.time()
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health["Ollama"] = {"status": "healthy", "response_time": response_time}
            else:
                health["Ollama"] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            health["Ollama"] = {"status": "unhealthy", "error": str(e)}
        
        return health
    
    def get_graph_metrics(self):
        """Get real-time graph metrics"""
        try:
            gt_client = GTClient()
            with gt_client.driver.session() as session:
                metrics = {}
                
                # Basic counts
                metrics["total_entities"] = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
                metrics["total_relations"] = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()["count"]
                metrics["total_documents"] = session.run("MATCH (d:Document) RETURN count(d) as count").single()["count"]
                
                # Confidence distribution
                confidence_data = session.run("""
                    MATCH (e:Entity) 
                    RETURN e.confidence as confidence
                """).data()
                
                if confidence_data:
                    confidences = [d["confidence"] for d in confidence_data]
                    metrics["avg_entity_confidence"] = sum(confidences) / len(confidences)
                    metrics["min_entity_confidence"] = min(confidences)
                    metrics["max_entity_confidence"] = max(confidences)
                
                # Recent activity (last hour)
                recent_entities = session.run("""
                    MATCH (e:Entity)
                    WHERE e.created_at > datetime() - duration('PT1H')
                    RETURN count(e) as count
                """).single()
                metrics["recent_entities"] = recent_entities["count"] if recent_entities else 0
                
                gt_client.close()
                return metrics
        except Exception as e:
            return {"error": str(e)}
    
    def get_extraction_stats(self):
        """Get extraction performance statistics"""
        try:
            # This would typically come from a logging system
            # For now, we'll simulate some stats
            stats = {
                "total_extractions": 150,
                "successful_extractions": 142,
                "failed_extractions": 8,
                "avg_processing_time": 2.3,
                "entities_per_doc": 12.5,
                "relations_per_doc": 8.2
            }
            return stats
        except Exception as e:
            return {"error": str(e)}
    
    def render_real_time_metrics(self):
        """Render real-time metrics dashboard"""
        st.title("📊 Real-Time Monitoring")
        
        # Auto-refresh
        if st.checkbox("🔄 Auto-refresh (5s)", value=False):
            time.sleep(5)
            st.rerun()
        
        # Service health
        st.subheader("🏥 Service Health")
        health = self.check_service_health()
        
        col1, col2 = st.columns(2)
        
        with col1:
            for service, status in health.items():
                if status["status"] == "healthy":
                    st.success(f"✅ {service}: Healthy ({status.get('response_time', 0):.3f}s)")
                else:
                    st.error(f"❌ {service}: {status.get('error', 'Unknown error')}")
        
        with col2:
            # Health status chart
            health_df = pd.DataFrame([
                {"Service": service, "Status": 1 if status["status"] == "healthy" else 0}
                for service, status in health.items()
            ])
            
            fig = px.bar(health_df, x="Service", y="Status", 
                        title="Service Health Status",
                        color="Status", color_continuous_scale=["red", "green"])
            st.plotly_chart(fig, use_container_width=True)
        
        # Graph metrics
        st.subheader("🗂️ Knowledge Graph Metrics")
        metrics = self.get_graph_metrics()
        
        if "error" not in metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Entities", metrics.get("total_entities", 0))
            with col2:
                st.metric("Total Relations", metrics.get("total_relations", 0))
            with col3:
                st.metric("Total Documents", metrics.get("total_documents", 0))
            with col4:
                st.metric("Recent Entities", metrics.get("recent_entities", 0))
            
            # Confidence metrics
            if "avg_entity_confidence" in metrics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Confidence", f"{metrics['avg_entity_confidence']:.3f}")
                with col2:
                    st.metric("Min Confidence", f"{metrics['min_entity_confidence']:.3f}")
                with col3:
                    st.metric("Max Confidence", f"{metrics['max_entity_confidence']:.3f}")
        else:
            st.error(f"Failed to get graph metrics: {metrics['error']}")
        
        # Extraction performance
        st.subheader("🔍 Extraction Performance")
        stats = self.get_extraction_stats()
        
        if "error" not in stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                success_rate = (stats["successful_extractions"] / stats["total_extractions"]) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col2:
                st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.1f}s")
            with col3:
                st.metric("Entities per Doc", f"{stats['entities_per_doc']:.1f}")
            
            # Performance chart
            perf_data = {
                "Metric": ["Success Rate", "Avg Processing Time", "Entities per Doc"],
                "Value": [success_rate, stats['avg_processing_time'], stats['entities_per_doc']]
            }
            perf_df = pd.DataFrame(perf_data)
            
            fig = px.bar(perf_df, x="Metric", y="Value", title="Performance Metrics")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Failed to get extraction stats: {stats['error']}")

class TroubleshootingTools:
    """Tools for troubleshooting the system"""
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration"""
        config_path = Path("config/settings.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def test_neo4j_connection(self):
        """Test Neo4j connection and return detailed diagnostics"""
        diagnostics = {
            "connection": False,
            "auth": False,
            "constraints": False,
            "data_access": False,
            "error": None
        }
        
        try:
            driver = GraphDatabase.driver(
                self.config.get("neo4j", {}).get("uri", "bolt://localhost:7687"),
                auth=(self.config.get("neo4j", {}).get("user", "neo4j"), 
                      self.config.get("neo4j", {}).get("password", "neo4jpass"))
            )
            diagnostics["connection"] = True
            
            with driver.session() as session:
                # Test basic query
                session.run("RETURN 1")
                diagnostics["auth"] = True
                
                # Test constraints
                try:
                    session.run("SHOW CONSTRAINTS")
                    diagnostics["constraints"] = True
                except:
                    pass
                
                # Test data access
                try:
                    session.run("MATCH (n) RETURN count(n) LIMIT 1")
                    diagnostics["data_access"] = True
                except:
                    pass
            
            driver.close()
            
        except Exception as e:
            diagnostics["error"] = str(e)
        
        return diagnostics
    
    def test_ollama_connection(self):
        """Test Ollama connection and return detailed diagnostics"""
        diagnostics = {
            "connection": False,
            "api_accessible": False,
            "models_available": False,
            "model_loaded": False,
            "error": None
        }
        
        try:
            ollama_url = self.config.get("ollama", {}).get("base_url", "http://localhost:11434")
            
            # Test basic connection
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            diagnostics["connection"] = True
            
            if response.status_code == 200:
                diagnostics["api_accessible"] = True
                
                # Check available models
                models = response.json().get("models", [])
                diagnostics["models_available"] = len(models) > 0
                
                # Check if configured model is loaded
                configured_model = self.config.get("ollama", {}).get("model", "qwen3:latest")
                model_names = [m.get("name", "") for m in models]
                diagnostics["model_loaded"] = any(configured_model in name for name in model_names)
            
        except Exception as e:
            diagnostics["error"] = str(e)
        
        return diagnostics
    
    def run_system_diagnostics(self):
        """Run comprehensive system diagnostics"""
        st.subheader("🔧 System Diagnostics")
        
        # Neo4j diagnostics
        st.write("**Neo4j Diagnostics:**")
        neo4j_diag = self.test_neo4j_connection()
        
        for test, result in neo4j_diag.items():
            if test == "error":
                if result:
                    st.error(f"❌ Error: {result}")
            else:
                status = "✅" if result else "❌"
                st.write(f"{status} {test.replace('_', ' ').title()}: {'Pass' if result else 'Fail'}")
        
        # Ollama diagnostics
        st.write("**Ollama Diagnostics:**")
        ollama_diag = self.test_ollama_connection()
        
        for test, result in ollama_diag.items():
            if test == "error":
                if result:
                    st.error(f"❌ Error: {result}")
            else:
                status = "✅" if result else "❌"
                st.write(f"{status} {test.replace('_', ' ').title()}: {'Pass' if result else 'Fail'}")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if not neo4j_diag["connection"]:
            st.warning("🔧 Neo4j connection failed. Check if Neo4j is running and credentials are correct.")
        
        if not ollama_diag["connection"]:
            st.warning("🔧 Ollama connection failed. Check if Ollama is running and accessible.")
        
        if neo4j_diag["connection"] and not neo4j_diag["constraints"]:
            st.info("ℹ️ Neo4j constraints not initialized. Run 'Initialize Constraints' from the main dashboard.")
        
        if ollama_diag["connection"] and not ollama_diag["model_loaded"]:
            st.info("ℹ️ Configured model not loaded. Pull the model using: `docker exec -it ollama ollama pull qwen3:latest`")
    
    def render_performance_analysis(self):
        """Render performance analysis tools"""
        st.subheader("📈 Performance Analysis")
        
        # Query performance
        st.write("**Query Performance:**")
        
        queries = [
            ("Count all entities", "MATCH (e:Entity) RETURN count(e)"),
            ("Count all relations", "MATCH ()-[r:RELATION]->() RETURN count(r)"),
            ("Find high confidence entities", "MATCH (e:Entity) WHERE e.confidence > 0.8 RETURN count(e)"),
            ("Complex relation query", "MATCH (h:Entity)-[r:RELATION]->(t:Entity) WHERE r.confidence > 0.7 RETURN count(r)")
        ]
        
        results = []
        
        try:
            gt_client = GTClient()
            with gt_client.driver.session() as session:
                for query_name, query in queries:
                    start_time = time.time()
                    result = session.run(query)
                    execution_time = time.time() - start_time
                    count = result.single()[0] if result.single() else 0
                    
                    results.append({
                        "Query": query_name,
                        "Execution Time (s)": f"{execution_time:.3f}",
                        "Result Count": count
                    })
            
            gt_client.close()
            
            # Display results
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Performance chart
            fig = px.bar(results_df, x="Query", y="Execution Time (s)", 
                        title="Query Performance")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Performance analysis failed: {e}")
    
    def render_error_logs(self):
        """Render error logs and debugging information"""
        st.subheader("🐛 Error Logs & Debugging")
        
        # Simulated error logs (replace with actual logging system)
        error_logs = [
            {
                "timestamp": "2024-01-15 10:30:15",
                "level": "ERROR",
                "component": "Extractor",
                "message": "Failed to parse JSON response from Ollama",
                "details": "Invalid JSON format in entity extraction response"
            },
            {
                "timestamp": "2024-01-15 10:28:45",
                "level": "WARNING",
                "component": "GTClient",
                "message": "Low confidence relation detected",
                "details": "Relation confidence: 0.23 (below threshold: 0.5)"
            },
            {
                "timestamp": "2024-01-15 10:27:12",
                "level": "ERROR",
                "component": "OllamaClient",
                "message": "Connection timeout",
                "details": "Request to Ollama API timed out after 30 seconds"
            }
        ]
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            level_filter = st.selectbox("Filter by Level", ["ALL", "ERROR", "WARNING", "INFO"])
        with col2:
            component_filter = st.selectbox("Filter by Component", ["ALL", "Extractor", "GTClient", "OllamaClient"])
        
        # Filter logs
        filtered_logs = error_logs
        if level_filter != "ALL":
            filtered_logs = [log for log in filtered_logs if log["level"] == level_filter]
        if component_filter != "ALL":
            filtered_logs = [log for log in filtered_logs if log["component"] == component_filter]
        
        # Display logs
        for log in filtered_logs:
            with st.expander(f"{log['timestamp']} - {log['level']} - {log['component']}"):
                st.write(f"**Message:** {log['message']}")
                st.write(f"**Details:** {log['details']}")
        
        # Error statistics
        st.subheader("📊 Error Statistics")
        
        error_stats = {}
        for log in error_logs:
            level = log["level"]
            error_stats[level] = error_stats.get(level, 0) + 1
        
        if error_stats:
            stats_df = pd.DataFrame([
                {"Level": level, "Count": count}
                for level, count in error_stats.items()
            ])
            
            fig = px.pie(stats_df, values="Count", names="Level", title="Error Distribution")
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main troubleshooting interface"""
    st.title("🔧 Troubleshooting & Monitoring Tools")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Real-Time Monitor", "🔧 Diagnostics", "🐛 Error Analysis"])
    
    with tab1:
        monitor = MonitoringDashboard()
        monitor.render_real_time_metrics()
    
    with tab2:
        troubleshooter = TroubleshootingTools()
        troubleshooter.run_system_diagnostics()
        troubleshooter.render_performance_analysis()
    
    with tab3:
        troubleshooter = TroubleshootingTools()
        troubleshooter.render_error_logs()

if __name__ == "__main__":
    import time
    main()
