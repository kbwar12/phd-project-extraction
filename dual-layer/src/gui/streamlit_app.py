import streamlit as st
import yaml
import json
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests
from neo4j import GraphDatabase

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from extraction.extractor import Extractor
from benchmarks.datasets.knowledge_graph_extractor import KnowledgeGraphExtractor
from benchmarks.datasets.pdf_converter import PDFToMarkdownConverter
from benchmarks.datasets.text_to_cypher import TextToCypherConverter
from benchmarks.datasets.docbench_loader import load_docbench, load_docbench_queries
from benchmarks.runner import BenchmarkRunner
from benchmarks.metrics import compute_docbench_metrics

class StreamlitApp:
    def __init__(self):
        self.config = self.load_config()
        self.setup_page_config()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="DocBench: Knowledge Graph-Based Document Benchmarking",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_config(self) -> Dict:
        """Load configuration with environment variable expansion"""
        config_path = Path("config/settings.yaml")
        try:
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
        except Exception as e:
            st.error(f"Failed to load config: {e}")
            return {}
    
    def check_services(self) -> Dict[str, bool]:
        """Check if services are running"""
        services = {}
        
        # Check Neo4j
        try:
            driver = GraphDatabase.driver(
                self.config.get("neo4j", {}).get("uri", "bolt://localhost:7687"),
                auth=(self.config.get("neo4j", {}).get("user", "neo4j"), 
                      self.config.get("neo4j", {}).get("password", "neo4jpass"))
            )
            with driver.session() as session:
                session.run("RETURN 1")
            services["Neo4j"] = True
            driver.close()
        except:
            services["Neo4j"] = False
        
        # Check Ollama
        try:
            ollama_url = self.config.get("ollama", {}).get("base_url", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            services["Ollama"] = response.status_code == 200
        except:
            services["Ollama"] = False
        
        return services
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("📊 DocBench Monitor")
        
        # Service status
        st.sidebar.subheader("Service Status")
        services = self.check_services()
        for service, status in services.items():
            color = "🟢" if status else "🔴"
            st.sidebar.write(f"{color} {service}")
        
        # Navigation
        st.sidebar.subheader("Navigation")
        pages = {
            "📊 Dashboard": "dashboard",
            "📄 PDF Processing": "pdf_processing", 
            "🔍 Document Extraction": "extraction",
            "📈 DocBench Evaluation": "evaluation",
            "🗂️ Knowledge Graph": "knowledge_graph",
            "🔍 Query Interface": "query_interface",
            "⚙️ Configuration": "configuration",
            "📋 Logs": "logs"
        }
        
        selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
        return pages[selected_page]
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.title("📊 DocBench Dashboard")
        
        # Service status cards
        col1, col2, col3, col4 = st.columns(4)
        
        services = self.check_services()
        with col1:
            st.metric("Neo4j", "🟢 Online" if services.get("Neo4j") else "🔴 Offline")
        with col2:
            st.metric("Ollama", "🟢 Online" if services.get("Ollama") else "🔴 Offline")
        with col3:
            # Get DocBench graph stats
            try:
                kg_extractor = KnowledgeGraphExtractor(
                    neo4j_uri=self.config["neo4j"]["uri"],
                    neo4j_user=self.config["neo4j"]["user"],
                    neo4j_password=self.config["neo4j"]["password"]
                )
                with kg_extractor.driver.session() as session:
                    entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
                    relation_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()["count"]
                    doc_count = session.run("MATCH (d:Document) RETURN count(d) as count").single()["count"]
                    concept_count = session.run("MATCH (c:Concept) RETURN count(c) as count").single()["count"]
                    fact_count = session.run("MATCH (f:Fact) RETURN count(f) as count").single()["count"]
                kg_extractor.close()
                
                st.metric("Documents", doc_count)
                st.metric("Entities", entity_count)
                st.metric("Relations", relation_count)
                st.metric("Concepts", concept_count)
                st.metric("Facts", fact_count)
            except:
                st.metric("Knowledge Graph", "❌ Error")
        
        # DocBench workflow status
        st.subheader("📈 DocBench Workflow Status")
        
        # Check PDF files
        pdf_dir = Path("./data/pdfs")
        pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        
        # Check markdown files
        markdown_dir = Path("./data/markdown")
        markdown_files = list(markdown_dir.glob("*.md")) if markdown_dir.exists() else []
        
        # Check knowledge graph
        kg_status = "✅ Populated" if services.get("Neo4j") else "❌ Not Available"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("PDF Files", len(pdf_files))
        with col2:
            st.metric("Markdown Files", len(markdown_files))
        with col3:
            st.metric("Knowledge Graph", kg_status)
        with col4:
            st.metric("Ready for Evaluation", "✅" if len(markdown_files) > 0 and services.get("Neo4j") else "❌")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        with col2:
            if st.button("📄 Process PDFs"):
                try:
                    converter = PDFToMarkdownConverter("./data/pdfs", "./data/markdown")
                    results = converter.convert_all_pdfs()
                    st.success(f"✅ Converted {len(results)} PDF files!")
                    except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")
        
        with col3:
            if st.button("🔍 Extract & Store"):
                try:
                    documents = load_docbench("./data", "./data/pdfs", "./data/markdown")
                    if documents:
                        kg_extractor = KnowledgeGraphExtractor(
                            neo4j_uri=self.config["neo4j"]["uri"],
                            neo4j_user=self.config["neo4j"]["user"],
                            neo4j_password=self.config["neo4j"]["password"]
                        )
                        extractor = Extractor(self.config)
                        
                        for doc in documents[:5]:  # Process first 5 for demo
                            result = extractor.extract(doc["text"], doc["doc_id"])
                            kg_extractor.extract_and_store(doc["doc_id"], doc["text"], result)
                        
                        kg_extractor.close()
                        st.success("✅ Extraction completed!")
                else:
                        st.warning("No documents found to process")
                except Exception as e:
                    st.error(f"Failed to extract: {e}")
        
        with col4:
            if st.button("📊 Run Evaluation"):
                try:
                    queries = load_docbench_queries("./data")
                    metrics = compute_docbench_metrics(
                        neo4j_uri=self.config["neo4j"]["uri"],
                        neo4j_user=self.config["neo4j"]["user"],
                        neo4j_password=self.config["neo4j"]["password"],
                        queries=queries
                    )
                    st.success(f"✅ Evaluation completed! Score: {metrics.get('overall_score', 0):.3f}")
                except Exception as e:
                    st.error(f"Failed to evaluate: {e}")
        
        # Recent results
        st.subheader("📁 Recent Results")
        output_dir = Path("./outputs")
        if output_dir.exists():
            result_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            if result_dirs:
                # Show most recent result
                latest_result = max(result_dirs, key=lambda x: x.stat().st_mtime)
                metrics_file = latest_result / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Score", f"{metrics.get('overall_score', 0):.3f}")
                    with col2:
                        st.metric("Successful Queries", f"{metrics.get('successful_queries', 0)}/{metrics.get('total_queries', 0)}")
        with col3:
                        st.metric("Total Documents", metrics.get('total_documents', 0))
            else:
                st.info("No evaluation results found")
        else:
            st.info("No outputs directory found")
    
    def render_pdf_processing(self):
        """Render PDF processing interface"""
        st.title("📄 PDF Processing")
        
        st.markdown("Convert PDF documents to markdown format for DocBench processing.")
        
        # PDF directory status
        pdf_dir = Path("./data/pdfs")
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            st.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
            
            if pdf_files:
                st.subheader("📁 Available PDF Files")
                for pdf_file in pdf_files:
                    st.write(f"• {pdf_file.name}")
        else:
            st.warning(f"PDF directory {pdf_dir} does not exist. Please create it and add PDF files.")
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pdf_dir_input = st.text_input("PDF Directory", value="./data/pdfs")
            markdown_dir_input = st.text_input("Markdown Output Directory", value="./data/markdown")
        
        with col2:
            process_all = st.checkbox("Process All PDFs", value=True)
            overwrite_existing = st.checkbox("Overwrite Existing Markdown Files", value=False)
        
        # Process PDFs
        if st.button("🚀 Process PDFs"):
            try:
                converter = PDFToMarkdownConverter(pdf_dir_input, markdown_dir_input)
                
                if process_all:
                    results = converter.convert_all_pdfs()
                    st.success(f"✅ Successfully converted {len(results)} PDF files to markdown!")
                    
                    # Show results
                    st.subheader("📋 Conversion Results")
                    for result in results:
                        st.write(f"✅ {Path(result).name}")
                else:
                    st.info("Please select 'Process All PDFs' to convert files")
                
                except Exception as e:
                st.error(f"Failed to process PDFs: {e}")
                st.exception(e)
        
        # Show markdown files
        markdown_dir = Path(markdown_dir_input)
        if markdown_dir.exists():
            markdown_files = list(markdown_dir.glob("*.md"))
            if markdown_files:
                st.subheader("📝 Generated Markdown Files")
                for md_file in markdown_files:
                    st.write(f"• {md_file.name}")
    
    def render_extraction(self):
        """Render document extraction interface"""
        st.title("🔍 Document Extraction & Knowledge Graph Population")
        
        st.markdown("Extract entities, relations, concepts, and facts from documents and populate the knowledge graph.")
        
        # Input options
        input_method = st.radio("Input Method", ["Upload File", "Paste Text", "Select from Markdown Files"])
        
        text = ""
        doc_id = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'md', 'pdf'])
            if uploaded_file:
                if uploaded_file.name.endswith('.pdf'):
                    st.warning("PDF files should be processed through the PDF Processing page first.")
                else:
                text = uploaded_file.read().decode('utf-8')
                doc_id = uploaded_file.name
        
        elif input_method == "Paste Text":
            text = st.text_area("Paste your text here", height=200)
            doc_id = st.text_input("Document ID", value="manual_input")
        
        elif input_method == "Select from Markdown Files":
            markdown_dir = Path("./data/markdown")
            if markdown_dir.exists():
                markdown_files = list(markdown_dir.glob("*.md"))
                if markdown_files:
                    selected_file = st.selectbox("Select markdown file", markdown_files)
                if selected_file:
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                        doc_id = selected_file.stem
                else:
                    st.warning("No markdown files found. Please process PDFs first.")
            else:
                st.warning("Markdown directory not found. Please process PDFs first.")
        
        # Extraction settings
        st.subheader("⚙️ Extraction Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 
                                           self.config.get("extraction", {}).get("confidence_threshold", 0.5))
            temperature = st.slider("Temperature", 0.0, 1.0, 
                                   self.config.get("ollama", {}).get("temperature", 0.2))
        
        with col2:
            store_in_graph = st.checkbox("Store in Knowledge Graph", value=True)
            clear_graph_first = st.checkbox("Clear Graph Before Processing", value=False)
            save_output = st.checkbox("Save Output to File", value=False)
        
        # Run extraction
        if st.button("🚀 Extract & Store", disabled=not text):
            if not text:
                st.error("Please provide text to extract from")
                return
            
            # Update config for this extraction
            temp_config = self.config.copy()
            temp_config["extraction"]["confidence_threshold"] = confidence_threshold
            temp_config["ollama"]["temperature"] = temperature
            
            try:
                # Initialize components
                extractor = Extractor(temp_config)
                kg_extractor = KnowledgeGraphExtractor(
                    neo4j_uri=self.config["neo4j"]["uri"],
                    neo4j_user=self.config["neo4j"]["user"],
                    neo4j_password=self.config["neo4j"]["password"]
                )
                
                if clear_graph_first:
                    with st.spinner("Clearing knowledge graph..."):
                        kg_extractor.clear_database()
                
                with st.spinner("Extracting entities, relations, concepts, and facts..."):
                    result = extractor.extract(text, doc_id)
                
                # Display extraction results
                st.subheader("📋 Extraction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Entities Found", len(result.entities))
                    if result.entities:
                        entities_df = pd.DataFrame([
                            {
                                "Text": e.text,
                                "Type": e.type,
                                "Confidence": f"{e.confidence:.3f}",
                                "Span": f"{e.span.start}-{e.span.end}"
                            }
                            for e in result.entities
                        ])
                        st.dataframe(entities_df, use_container_width=True)
                
                with col2:
                    st.metric("Relations Found", len(result.relations))
                    if result.relations:
                        relations_df = pd.DataFrame([
                            {
                                "Head": r.head_entity_id,
                                "Relation": r.relation_type,
                                "Tail": r.tail_entity_id,
                                "Confidence": f"{r.confidence:.3f}"
                            }
                            for r in result.relations
                        ])
                        st.dataframe(relations_df, use_container_width=True)
                
                # Store in knowledge graph if requested
                if store_in_graph:
                    with st.spinner("Storing in knowledge graph..."):
                        stats = kg_extractor.extract_and_store(doc_id, text, result)
                    
                    st.success("✅ Results stored in knowledge graph!")
                    
                    # Show storage stats
                    st.subheader("💾 Knowledge Graph Storage Stats")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Entities Stored", stats['entities_created'])
                    with col2:
                        st.metric("Relations Stored", stats['relations_created'])
                    with col3:
                        st.metric("Concepts Stored", stats['concepts_created'])
                    with col4:
                        st.metric("Facts Stored", stats['facts_created'])
                
                # Save output if requested
                if save_output:
                    output_path = Path("outputs") / f"{doc_id}_extraction.json"
                    output_path.parent.mkdir(exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(result.model_dump(), f, indent=2)
                    st.success(f"✅ Results saved to {output_path}")
                
                kg_extractor.close()
                
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.exception(e)
    
        # Batch processing option
        st.subheader("📚 Batch Processing")
        if st.button("🔄 Process All Markdown Files"):
            try:
                documents = load_docbench("./data", "./data/pdfs", "./data/markdown")
                if documents:
                    kg_extractor = KnowledgeGraphExtractor(
                        neo4j_uri=self.config["neo4j"]["uri"],
                        neo4j_user=self.config["neo4j"]["user"],
                        neo4j_password=self.config["neo4j"]["password"]
                    )
                    extractor = Extractor(self.config)
                    
                    progress_bar = st.progress(0)
                    total_docs = len(documents)
                    
                    for i, doc in enumerate(documents):
                        st.write(f"Processing {i+1}/{total_docs}: {doc['doc_id']}")
                        result = extractor.extract(doc["text"], doc["doc_id"])
                        stats = kg_extractor.extract_and_store(doc["doc_id"], doc["text"], result)
                        progress_bar.progress((i + 1) / total_docs)
                    
                    kg_extractor.close()
                    st.success(f"✅ Successfully processed {total_docs} documents!")
                else:
                    st.warning("No documents found to process")
            except Exception as e:
                st.error(f"Batch processing failed: {e}")
                st.exception(e)
    
    def render_evaluation(self):
        """Render DocBench evaluation interface"""
        st.title("📈 DocBench Evaluation")
        
        st.markdown("Run DocBench evaluation using knowledge graph queries to assess extraction quality.")
        
        # Evaluation settings
        st.subheader("⚙️ Evaluation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_samples = st.number_input("Max Documents to Evaluate", min_value=1, max_value=100, value=10)
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
            clear_graph_first = st.checkbox("Clear Graph Before Evaluation", value=False)
            save_results = st.checkbox("Save Results", value=True)
        
        # Load queries
        try:
            queries = load_docbench_queries("./data")
            st.subheader("📋 Available Queries")
            
            query_df = pd.DataFrame([
                {
                    "ID": q["id"],
                    "Description": q["description"],
                    "Natural Language": q["natural_language"],
                    "Cypher Template": q["cypher_template"]
                }
                for q in queries
            ])
            st.dataframe(query_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to load queries: {e}")
            queries = []
        
        # Run evaluation
        if st.button("🚀 Run DocBench Evaluation"):
            if not queries:
                st.error("No queries available for evaluation")
                return
            
            try:
                # Update global config
                temp_config = self.config.copy()
                temp_config["extraction"]["confidence_threshold"] = confidence_threshold
                temp_config["ollama"]["temperature"] = temperature
                
                # Initialize components
                kg_extractor = KnowledgeGraphExtractor(
                    neo4j_uri=self.config["neo4j"]["uri"],
                    neo4j_user=self.config["neo4j"]["user"],
                    neo4j_password=self.config["neo4j"]["password"]
                )
                
                if clear_graph_first:
                    with st.spinner("Clearing knowledge graph..."):
                        kg_extractor.clear_database()
                
                # Load and process documents
                with st.spinner("Loading documents..."):
                    documents = load_docbench("./data", "./data/pdfs", "./data/markdown")
                    documents = documents[:max_samples]
                
                if not documents:
                    st.error("No documents found. Please process PDFs and extract information first.")
                    return
                
                # Extract and store information
                with st.spinner("Extracting information..."):
                    extractor = Extractor(temp_config)
                    extraction_stats = []
                    
                    progress_bar = st.progress(0)
                    total_docs = len(documents)
                    
                    for i, doc in enumerate(documents):
                        result = extractor.extract(doc["text"], doc["doc_id"])
                        stats = kg_extractor.extract_and_store(doc["doc_id"], doc["text"], result)
                        extraction_stats.append({
                            "doc_id": doc["doc_id"],
                            **stats
                        })
                        progress_bar.progress((i + 1) / total_docs)
                
                # Run evaluation
                with st.spinner("Running DocBench evaluation..."):
                    metrics = compute_docbench_metrics(
                        neo4j_uri=self.config["neo4j"]["uri"],
                        neo4j_user=self.config["neo4j"]["user"],
                        neo4j_password=self.config["neo4j"]["password"],
                        queries=queries
                    )
                
                # Add extraction statistics
                metrics["extraction_stats"] = extraction_stats
                metrics["total_documents"] = len(documents)
                
                # Display results
                st.subheader("📊 DocBench Evaluation Results")
                
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Score", f"{metrics.get('overall_score', 0):.3f}")
                with col2:
                    st.metric("Successful Queries", f"{metrics.get('successful_queries', 0)}/{metrics.get('total_queries', 0)}")
                with col3:
                    st.metric("Total Documents", metrics.get('total_documents', 0))
                with col4:
                    success_rate = (metrics.get('successful_queries', 0) / metrics.get('total_queries', 1)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Query results
                if 'query_results' in metrics:
                    st.subheader("📋 Query Results")
                    
                    query_results_df = pd.DataFrame([
                        {
                            "Query ID": query_id,
                            "Natural Language": result.get("natural_language", ""),
                            "Result Count": result.get("result_count", 0),
                            "Score": f"{result.get('score', 0):.3f}",
                            "Status": "✅ Success" if "error" not in result else "❌ Error",
                            "Error": result.get("error", "")
                        }
                        for query_id, result in metrics["query_results"].items()
                    ])
                    st.dataframe(query_results_df, use_container_width=True)
                
                # Extraction statistics
                if 'extraction_stats' in metrics:
                    st.subheader("📈 Extraction Statistics")
                    
                    total_entities = sum(stats.get('entities_created', 0) for stats in metrics['extraction_stats'])
                    total_relations = sum(stats.get('relations_created', 0) for stats in metrics['extraction_stats'])
                    total_concepts = sum(stats.get('concepts_created', 0) for stats in metrics['extraction_stats'])
                    total_facts = sum(stats.get('facts_created', 0) for stats in metrics['extraction_stats'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Entities", total_entities)
                    with col2:
                        st.metric("Total Relations", total_relations)
                    with col3:
                        st.metric("Total Concepts", total_concepts)
                    with col4:
                        st.metric("Total Facts", total_facts)
                
                # Save results if requested
                if save_results:
                    output_dir = Path("outputs") / "docbench" / datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_dir / "metrics.json", 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    with open(output_dir / "documents.json", 'w') as f:
                        json.dump(documents, f, indent=2)
                    
                    with open(output_dir / "queries.json", 'w') as f:
                        json.dump(queries, f, indent=2)
                    
                    st.success(f"✅ Results saved to {output_dir}")
                
                kg_extractor.close()
                
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)
        
        # Previous results
        st.subheader("📁 Previous Results")
        output_dir = Path("outputs") / "docbench"
        if output_dir.exists():
            result_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            if result_dirs:
                selected_result = st.selectbox("Select Result", result_dirs)
                if selected_result:
                    metrics_file = selected_result / "metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            prev_metrics = json.load(f)
                        
                        st.subheader("📊 Previous Evaluation Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Score", f"{prev_metrics.get('overall_score', 0):.3f}")
                        with col2:
                            st.metric("Successful Queries", f"{prev_metrics.get('successful_queries', 0)}/{prev_metrics.get('total_queries', 0)}")
                        with col3:
                            st.metric("Total Documents", prev_metrics.get('total_documents', 0))
            else:
                st.info("No evaluation results found")
        else:
            st.info("No outputs directory found")
    
    def render_query_interface(self):
        """Render query interface for natural language to Cypher conversion"""
        st.title("🔍 Query Interface")
        
        st.markdown("Convert natural language queries to Cypher and execute them against the knowledge graph.")
        
        # Query input
        st.subheader("📝 Query Input")
        
        query_type = st.radio("Query Type", ["Natural Language", "Cypher Direct"])
        
        if query_type == "Natural Language":
            natural_language_query = st.text_area(
                "Enter your question in natural language",
                placeholder="e.g., What entities are mentioned in this document?",
                height=100
            )
            
            if st.button("🔧 Convert to Cypher"):
                if natural_language_query:
                    try:
                        converter = TextToCypherConverter()
                        cypher_query = converter.convert_query(natural_language_query)
                        
                        st.subheader("🔧 Generated Cypher Query")
                        st.code(cypher_query, language="cypher")
                        
                        # Validate query
                        if converter.validate_cypher_query(cypher_query):
                            st.success("✅ Query is syntactically valid")
                        else:
                            st.warning("⚠️ Query may have syntax issues")
                        
                        # Execute query
                        if st.button("🚀 Execute Query"):
                            try:
                                kg_extractor = KnowledgeGraphExtractor(
                                    neo4j_uri=self.config["neo4j"]["uri"],
                                    neo4j_user=self.config["neo4j"]["user"],
                                    neo4j_password=self.config["neo4j"]["password"]
                                )
                                
                                with kg_extractor.driver.session() as session:
                                    result = session.run(cypher_query)
                                    records = list(result)
                                
                                st.subheader("📊 Query Results")
                                st.write(f"Found {len(records)} results:")
                                
                                if records:
                                    # Convert to DataFrame for better display
                                    if records and len(records) > 0:
                                        # Try to create a DataFrame
                                        try:
                                            result_df = pd.DataFrame([dict(record) for record in records])
                                            st.dataframe(result_df, use_container_width=True)
                                        except:
                                            # Fallback to JSON display
                                            for i, record in enumerate(records[:10]):  # Limit to 10 results
                                                st.json(dict(record))
                                else:
                                    st.info("No results found")
                                
                                kg_extractor.close()
                                
                            except Exception as e:
                                st.error(f"Query execution failed: {e}")
                    except Exception as e:
                        st.error(f"Conversion failed: {e}")
                else:
                    st.warning("Please enter a natural language query")
        
        else:  # Cypher Direct
            cypher_query = st.text_area(
                "Enter your Cypher query",
                placeholder="MATCH (e:Entity) RETURN e.name, e.type LIMIT 10",
                height=150
            )
            
            if st.button("🚀 Execute Cypher Query"):
                if cypher_query:
                    try:
                        kg_extractor = KnowledgeGraphExtractor(
                            neo4j_uri=self.config["neo4j"]["uri"],
                            neo4j_user=self.config["neo4j"]["user"],
                            neo4j_password=self.config["neo4j"]["password"]
                        )
                        
                        with kg_extractor.driver.session() as session:
                            result = session.run(cypher_query)
                            records = list(result)
                        
                        st.subheader("📊 Query Results")
                        st.write(f"Found {len(records)} results:")
                        
                        if records:
                            try:
                                result_df = pd.DataFrame([dict(record) for record in records])
                                st.dataframe(result_df, use_container_width=True)
                            except:
                                for i, record in enumerate(records[:10]):
                                    st.json(dict(record))
                        else:
                            st.info("No results found")
                        
                        kg_extractor.close()
                        
                    except Exception as e:
                        st.error(f"Query execution failed: {e}")
                else:
                    st.warning("Please enter a Cypher query")
        
        # Query examples
        st.subheader("📋 Query Examples")
        
        try:
            converter = TextToCypherConverter()
            examples = converter.get_query_examples()
            
            for i, example in enumerate(examples):
                with st.expander(f"Example {i+1}: {example['query']}"):
                    st.write("**Natural Language:**")
                    st.write(example['query'])
                    st.write("**Cypher:**")
                    st.code(example['cypher'], language="cypher")
                    
                    if st.button(f"Try Example {i+1}", key=f"example_{i}"):
                        st.session_state.natural_language_query = example['query']
                        st.rerun()
        except Exception as e:
            st.error(f"Failed to load examples: {e}")
    
    def render_knowledge_graph(self):
        """Render knowledge graph interface"""
        st.title("🗂️ Knowledge Graph Explorer")
        
        try:
            gt_client = GTClient()
            
            # Graph statistics
            st.subheader("📊 Graph Statistics")
            
            with gt_client.driver.session() as session:
                stats = {}
                stats["Entities"] = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
                stats["Relations"] = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()["count"]
                stats["Documents"] = session.run("MATCH (d:Document) RETURN count(d) as count").single()["count"]
                
                # Entity types
                entity_types = session.run("""
                    MATCH (e:Entity) 
                    RETURN e.type as type, count(e) as count 
                    ORDER BY count DESC
                """).data()
                
                # Relation types
                relation_types = session.run("""
                    MATCH ()-[r:RELATION]->() 
                    RETURN r.type as type, count(r) as count 
                    ORDER BY count DESC
                """).data()
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entities", stats["Entities"])
            with col2:
                st.metric("Total Relations", stats["Relations"])
            with col3:
                st.metric("Total Documents", stats["Documents"])
            
            # Entity types chart
            if entity_types:
                st.subheader("📈 Entity Types Distribution")
                entity_df = pd.DataFrame(entity_types)
                fig = px.pie(entity_df, values='count', names='type', title="Entity Types")
                st.plotly_chart(fig, use_container_width=True)
            
            # Relation types chart
            if relation_types:
                st.subheader("🔗 Relation Types Distribution")
                relation_df = pd.DataFrame(relation_types)
                fig = px.bar(relation_df, x='type', y='count', title="Relation Types")
                st.plotly_chart(fig, use_container_width=True)
            
            # Query interface
            st.subheader("🔍 Graph Query")
            
            query_type = st.selectbox("Query Type", ["Entities", "Relations", "Custom Cypher"])
            
            if query_type == "Entities":
                entity_type = st.selectbox("Entity Type", [None] + [et["type"] for et in entity_types])
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0)
                limit = st.number_input("Limit", min_value=1, max_value=100, value=20)
                
                if st.button("Query Entities"):
                    entities = gt_client.query_entities(entity_type, min_confidence)
                    entities = entities[:limit]
                    
                    if entities:
                        entities_df = pd.DataFrame(entities)
                        st.dataframe(entities_df, use_container_width=True)
                    else:
                        st.info("No entities found")
            
            elif query_type == "Relations":
                relation_type = st.selectbox("Relation Type", [None] + [rt["type"] for rt in relation_types])
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0)
                limit = st.number_input("Limit", min_value=1, max_value=100, value=20)
                
                if st.button("Query Relations"):
                    relations = gt_client.query_relations(relation_type, min_confidence)
                    relations = relations[:limit]
                    
                    if relations:
                        relations_df = pd.DataFrame(relations)
                        st.dataframe(relations_df, use_container_width=True)
                    else:
                        st.info("No relations found")
            
            elif query_type == "Custom Cypher":
                cypher_query = st.text_area("Cypher Query", height=100)
                
                if st.button("Execute Query"):
                    try:
                        with gt_client.driver.session() as session:
                            result = session.run(cypher_query)
                            records = [dict(record) for record in result]
                            
                            if records:
                                result_df = pd.DataFrame(records)
                                st.dataframe(result_df, use_container_width=True)
                            else:
                                st.info("Query executed successfully (no results)")
                    except Exception as e:
                        st.error(f"Query failed: {e}")
            
            gt_client.close()
            
        except Exception as e:
            st.error(f"Failed to connect to knowledge graph: {e}")
    
    def render_configuration(self):
        """Render configuration interface"""
        st.title("⚙️ Configuration Management")
        
        # Current configuration
        st.subheader("📋 Current Configuration")
        st.json(self.config)
        
        # Configuration editor
        st.subheader("✏️ Edit Configuration")
        
        # Neo4j settings
        st.subheader("🗄️ Neo4j Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            neo4j_uri = st.text_input("Neo4j URI", 
                                    value=self.config.get("neo4j", {}).get("uri", "bolt://localhost:7687"))
        with col2:
            neo4j_user = st.text_input("Neo4j User", 
                                      value=self.config.get("neo4j", {}).get("user", "neo4j"))
        with col3:
            neo4j_password = st.text_input("Neo4j Password", 
                                          value=self.config.get("neo4j", {}).get("password", "neo4jpass"),
                                          type="password")
        
        # Ollama settings
        st.subheader("🤖 Ollama Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ollama_url = st.text_input("Ollama Base URL", 
                                     value=self.config.get("ollama", {}).get("base_url", "http://localhost:11434"))
        with col2:
            ollama_model = st.text_input("Ollama Model", 
                                        value=self.config.get("ollama", {}).get("model", "qwen3:latest"))
        with col3:
            ollama_temp = st.slider("Temperature", 0.0, 1.0, 
                                   value=self.config.get("ollama", {}).get("temperature", 0.2))
        
        # Extraction settings
        st.subheader("🔍 Extraction Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 
                                      value=self.config.get("extraction", {}).get("confidence_threshold", 0.5))
        with col2:
            max_evidence = st.number_input("Max Evidence per Claim", 
                                          value=self.config.get("extraction", {}).get("max_evidence_per_claim", 5))
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            new_config = {
                "neo4j": {
                    "uri": neo4j_uri,
                    "user": neo4j_user,
                    "password": neo4j_password
                },
                "ollama": {
                    "base_url": ollama_url,
                    "model": ollama_model,
                    "temperature": ollama_temp,
                    "top_p": 0.95,
                    "max_tokens": 2048
                },
                "extraction": {
                    "confidence_threshold": conf_threshold,
                    "max_evidence_per_claim": max_evidence
                },
                "paths": {
                    "data_dir": "./data",
                    "outputs_dir": "./outputs"
                }
            }
            
            try:
                config_path = Path("config/settings.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(new_config, f, default_flow_style=False)
                st.success("✅ Configuration saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save configuration: {e}")
    
    def render_logs(self):
        """Render logs interface"""
        st.title("📋 System Logs")
        
        # Log level filter
        log_level = st.selectbox("Log Level", ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"])
        
        # Log content (placeholder - you can implement actual logging)
        st.subheader("📄 Recent Logs")
        
        # Simulated logs for demonstration
        logs = [
            {"timestamp": "2024-01-15 10:30:15", "level": "INFO", "message": "Extraction completed successfully"},
            {"timestamp": "2024-01-15 10:29:45", "level": "INFO", "message": "Starting entity extraction"},
            {"timestamp": "2024-01-15 10:28:12", "level": "WARNING", "message": "Low confidence relation detected"},
            {"timestamp": "2024-01-15 10:27:33", "level": "ERROR", "message": "Failed to connect to Ollama"},
            {"timestamp": "2024-01-15 10:26:55", "level": "INFO", "message": "Benchmark run completed"},
        ]
        
        # Filter logs
        if log_level != "ALL":
            logs = [log for log in logs if log["level"] == log_level]
        
        # Display logs
        for log in logs:
            color = {
                "ERROR": "🔴",
                "WARNING": "🟡", 
                "INFO": "🔵",
                "DEBUG": "⚪"
            }.get(log["level"], "⚪")
            
            st.write(f"{color} **{log['timestamp']}** [{log['level']}] {log['message']}")
        
        # Clear logs button
        if st.button("🗑️ Clear Logs"):
            st.info("Logs cleared (simulated)")
    
    def run(self):
        """Main application runner"""
        page = self.render_sidebar()
        
        if page == "dashboard":
            self.render_dashboard()
        elif page == "pdf_processing":
            self.render_pdf_processing()
        elif page == "extraction":
            self.render_extraction()
        elif page == "evaluation":
            self.render_evaluation()
        elif page == "knowledge_graph":
            self.render_knowledge_graph()
        elif page == "query_interface":
            self.render_query_interface()
        elif page == "configuration":
            self.render_configuration()
        elif page == "logs":
            self.render_logs()

def main():
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
