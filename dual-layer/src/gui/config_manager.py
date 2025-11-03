import streamlit as st
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import os

class ConfigManager:
    """Configuration management for the dual-layer extraction system"""
    
    def __init__(self):
        self.config_path = Path("config/settings.yaml")
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
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
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to YAML file"""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Failed to save config: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration template"""
        return {
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "neo4jpass"
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "qwen3:latest",
                "temperature": 0.2,
                "top_p": 0.95,
                "max_tokens": 2048
            },
            "extraction": {
                "confidence_threshold": 0.5,
                "max_evidence_per_claim": 5
            },
            "paths": {
                "data_dir": "./data",
                "outputs_dir": "./outputs"
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate configuration and return validation results"""
        errors = []
        
        # Validate Neo4j config
        neo4j_config = config.get("neo4j", {})
        if not neo4j_config.get("uri"):
            errors.append("Neo4j URI is required")
        if not neo4j_config.get("user"):
            errors.append("Neo4j username is required")
        if not neo4j_config.get("password"):
            errors.append("Neo4j password is required")
        
        # Validate Ollama config
        ollama_config = config.get("ollama", {})
        if not ollama_config.get("base_url"):
            errors.append("Ollama base URL is required")
        if not ollama_config.get("model"):
            errors.append("Ollama model is required")
        
        # Validate extraction config
        extraction_config = config.get("extraction", {})
        confidence = extraction_config.get("confidence_threshold", 0.5)
        if not (0.0 <= confidence <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        max_evidence = extraction_config.get("max_evidence_per_claim", 5)
        if not (1 <= max_evidence <= 100):
            errors.append("Max evidence per claim must be between 1 and 100")
        
        # Validate paths
        paths_config = config.get("paths", {})
        if not paths_config.get("data_dir"):
            errors.append("Data directory path is required")
        if not paths_config.get("outputs_dir"):
            errors.append("Outputs directory path is required")
        
        return len(errors) == 0, errors

class ConfigUI:
    """User interface for configuration management"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
    
    def render_config_editor(self):
        """Render configuration editor interface"""
        st.title("⚙️ Configuration Management")
        
        # Configuration tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🗄️ Neo4j", "🤖 Ollama", "🔍 Extraction", "📁 Paths"])
        
        with tab1:
            self.render_neo4j_config()
        
        with tab2:
            self.render_ollama_config()
        
        with tab3:
            self.render_extraction_config()
        
        with tab4:
            self.render_paths_config()
        
        # Action buttons
        st.subheader("💾 Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Configuration"):
                self.save_current_config()
        
        with col2:
            if st.button("🔄 Reset to Defaults"):
                self.reset_to_defaults()
        
        with col3:
            if st.button("📥 Load from File"):
                self.load_from_file()
        
        with col4:
            if st.button("📤 Export Configuration"):
                self.export_config()
    
    def render_neo4j_config(self):
        """Render Neo4j configuration section"""
        st.subheader("🗄️ Neo4j Database Configuration")
        
        neo4j_config = self.config_manager.config.get("neo4j", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            uri = st.text_input(
                "Neo4j URI",
                value=neo4j_config.get("uri", "bolt://localhost:7687"),
                help="Connection URI for Neo4j database"
            )
            
            user = st.text_input(
                "Username",
                value=neo4j_config.get("user", "neo4j"),
                help="Neo4j username"
            )
        
        with col2:
            password = st.text_input(
                "Password",
                value=neo4j_config.get("password", "neo4jpass"),
                type="password",
                help="Neo4j password"
            )
            
            # Test connection button
            if st.button("🔗 Test Connection"):
                self.test_neo4j_connection(uri, user, password)
        
        # Update config
        self.config_manager.config["neo4j"] = {
            "uri": uri,
            "user": user,
            "password": password
        }
    
    def render_ollama_config(self):
        """Render Ollama configuration section"""
        st.subheader("🤖 Ollama LLM Configuration")
        
        ollama_config = self.config_manager.config.get("ollama", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_url = st.text_input(
                "Base URL",
                value=ollama_config.get("base_url", "http://localhost:11434"),
                help="Ollama API base URL"
            )
            
            model = st.text_input(
                "Model Name",
                value=ollama_config.get("model", "qwen3:latest"),
                help="Ollama model to use for extraction"
            )
        
        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=ollama_config.get("temperature", 0.2),
                step=0.1,
                help="Controls randomness in LLM responses"
            )
            
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4096,
                value=ollama_config.get("max_tokens", 2048),
                help="Maximum tokens in LLM response"
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=ollama_config.get("top_p", 0.95),
                step=0.05,
                help="Nucleus sampling parameter"
            )
        
        # Test connection button
        if st.button("🔗 Test Ollama Connection"):
            self.test_ollama_connection(base_url, model)
        
        # Update config
        self.config_manager.config["ollama"] = {
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
    
    def render_extraction_config(self):
        """Render extraction configuration section"""
        st.subheader("🔍 Extraction Configuration")
        
        extraction_config = self.config_manager.config.get("extraction", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=extraction_config.get("confidence_threshold", 0.5),
                step=0.05,
                help="Minimum confidence for extracted entities/relations"
            )
            
            max_evidence = st.number_input(
                "Max Evidence per Claim",
                min_value=1,
                max_value=100,
                value=extraction_config.get("max_evidence_per_claim", 5),
                help="Maximum evidence pieces per claim"
            )
        
        with col2:
            # Additional extraction settings can be added here
            st.info("💡 Additional extraction parameters can be configured here")
        
        # Update config
        self.config_manager.config["extraction"] = {
            "confidence_threshold": confidence_threshold,
            "max_evidence_per_claim": max_evidence
        }
    
    def render_paths_config(self):
        """Render paths configuration section"""
        st.subheader("📁 Paths Configuration")
        
        paths_config = self.config_manager.config.get("paths", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_dir = st.text_input(
                "Data Directory",
                value=paths_config.get("data_dir", "./data"),
                help="Directory containing input datasets"
            )
            
            outputs_dir = st.text_input(
                "Outputs Directory",
                value=paths_config.get("outputs_dir", "./outputs"),
                help="Directory for saving extraction results"
            )
        
        with col2:
            # Path validation
            st.write("**Path Validation:**")
            
            data_path = Path(data_dir)
            if data_path.exists():
                st.success(f"✅ Data directory exists: {data_path.absolute()}")
            else:
                st.warning(f"⚠️ Data directory does not exist: {data_path.absolute()}")
            
            outputs_path = Path(outputs_dir)
            if outputs_path.exists():
                st.success(f"✅ Outputs directory exists: {outputs_path.absolute()}")
            else:
                st.warning(f"⚠️ Outputs directory does not exist: {outputs_path.absolute()}")
        
        # Update config
        self.config_manager.config["paths"] = {
            "data_dir": data_dir,
            "outputs_dir": outputs_dir
        }
    
    def test_neo4j_connection(self, uri: str, user: str, password: str):
        """Test Neo4j connection"""
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
            
            driver.close()
            
            if test_value == 1:
                st.success("✅ Neo4j connection successful!")
            else:
                st.error("❌ Neo4j connection test failed")
                
        except Exception as e:
            st.error(f"❌ Neo4j connection failed: {e}")
    
    def test_ollama_connection(self, base_url: str, model: str):
        """Test Ollama connection"""
        try:
            import requests
            
            # Test basic API connection
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                if any(model in name for name in model_names):
                    st.success(f"✅ Ollama connection successful! Model '{model}' is available.")
                else:
                    st.warning(f"⚠️ Ollama connection successful, but model '{model}' not found. Available models: {model_names}")
            else:
                st.error(f"❌ Ollama API returned status code: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Ollama connection failed: {e}")
    
    def save_current_config(self):
        """Save current configuration"""
        # Validate configuration
        is_valid, errors = self.config_manager.validate_config(self.config_manager.config)
        
        if not is_valid:
            st.error("❌ Configuration validation failed:")
            for error in errors:
                st.error(f"  - {error}")
            return
        
        # Save configuration
        if self.config_manager.save_config(self.config_manager.config):
            st.success("✅ Configuration saved successfully!")
        else:
            st.error("❌ Failed to save configuration")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        if st.button("⚠️ Confirm Reset", key="confirm_reset"):
            self.config_manager.config = self.config_manager.get_default_config()
            st.success("✅ Configuration reset to defaults!")
            st.rerun()
    
    def load_from_file(self):
        """Load configuration from uploaded file"""
        uploaded_file = st.file_uploader("Choose a YAML configuration file", type=['yaml', 'yml'])
        
        if uploaded_file:
            try:
                config_content = uploaded_file.read().decode('utf-8')
                config = yaml.safe_load(config_content)
                
                # Validate loaded configuration
                is_valid, errors = self.config_manager.validate_config(config)
                
                if is_valid:
                    self.config_manager.config = config
                    st.success("✅ Configuration loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid configuration file:")
                    for error in errors:
                        st.error(f"  - {error}")
                        
            except Exception as e:
                st.error(f"❌ Failed to load configuration file: {e}")
    
    def export_config(self):
        """Export current configuration"""
        config_yaml = yaml.dump(self.config_manager.config, default_flow_style=False, indent=2)
        
        st.download_button(
            label="📤 Download Configuration",
            data=config_yaml,
            file_name="settings.yaml",
            mime="text/yaml"
        )
    
    def render_config_preview(self):
        """Render configuration preview"""
        st.subheader("👁️ Configuration Preview")
        
        # Show current configuration as JSON
        config_json = json.dumps(self.config_manager.config, indent=2)
        st.code(config_json, language="json")
        
        # Show validation status
        is_valid, errors = self.config_manager.validate_config(self.config_manager.config)
        
        if is_valid:
            st.success("✅ Configuration is valid")
        else:
            st.error("❌ Configuration has errors:")
            for error in errors:
                st.error(f"  - {error}")

def main():
    """Main configuration interface"""
    config_ui = ConfigUI()
    
    # Sidebar navigation
    st.sidebar.title("⚙️ Configuration Manager")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["📝 Edit Configuration", "👁️ Preview Configuration", "🔧 Advanced Tools"]
    )
    
    if page == "📝 Edit Configuration":
        config_ui.render_config_editor()
    elif page == "👁️ Preview Configuration":
        config_ui.render_config_preview()
    elif page == "🔧 Advanced Tools":
        st.title("🔧 Advanced Configuration Tools")
        
        st.subheader("🔄 Environment Variables")
        st.write("Current environment variables affecting configuration:")
        
        env_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OLLAMA_BASE_URL", "OLLAMA_MODEL"]
        for var in env_vars:
            value = os.getenv(var, "Not set")
            st.write(f"**{var}**: `{value}`")
        
        st.subheader("📁 Configuration Files")
        config_files = list(Path("config").glob("*.yaml")) + list(Path("config").glob("*.yml"))
        
        if config_files:
            for file_path in config_files:
                st.write(f"📄 {file_path}")
        else:
            st.info("No configuration files found in config/ directory")

if __name__ == "__main__":
    main()
