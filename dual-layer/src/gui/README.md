# Streamlit GUI for Dual-Layer Extraction System

This directory contains Streamlit-based GUI applications for monitoring, troubleshooting, and managing the dual-layer extraction system.

## Available Applications

### 1. Main Monitoring Interface (`streamlit_app.py`)
- **Purpose**: Primary dashboard for system monitoring and control
- **Features**:
  - Real-time service status monitoring (Neo4j, Ollama)
  - Document extraction interface with live results
  - Benchmark runner with progress tracking
  - Knowledge graph explorer with interactive queries
  - Configuration management
  - System logs viewer

### 2. Troubleshooting Tools (`troubleshooting.py`)
- **Purpose**: Advanced diagnostics and troubleshooting
- **Features**:
  - Real-time system metrics and health monitoring
  - Comprehensive service diagnostics
  - Performance analysis and query profiling
  - Error logs and debugging information
  - Auto-refresh capabilities for live monitoring

### 3. Configuration Manager (`config_manager.py`)
- **Purpose**: GUI-based configuration management
- **Features**:
  - Visual configuration editor for all system settings
  - Service connection testing (Neo4j, Ollama)
  - Configuration validation and error checking
  - Import/export configuration files
  - Environment variable management

### 4. Application Launcher (`launcher.py`)
- **Purpose**: Convenient launcher for all GUI applications
- **Features**:
  - Single command to launch any GUI app
  - Automatic port management
  - App discovery and listing
  - Command-line interface for easy access

## Quick Start

### Prerequisites
Make sure you have installed the GUI dependencies:
```bash
pip install -r requirements.txt
```

### Launch Applications

1. **Main Monitoring Interface**:
   ```bash
   python src/gui/launcher.py main
   ```
   Access at: http://localhost:8501

2. **Troubleshooting Tools**:
   ```bash
   python src/gui/launcher.py troubleshooting
   ```
   Access at: http://localhost:8502

3. **Configuration Manager**:
   ```bash
   python src/gui/launcher.py config
   ```
   Access at: http://localhost:8503

4. **List Available Apps**:
   ```bash
   python src/gui/launcher.py --list
   ```

### Direct Launch (Alternative)
You can also launch apps directly with Streamlit:

```bash
# Main interface
streamlit run src/gui/streamlit_app.py --server.port 8501

# Troubleshooting
streamlit run src/gui/troubleshooting.py --server.port 8502

# Configuration manager
streamlit run src/gui/config_manager.py --server.port 8503
```

## Features Overview

### 📊 Dashboard
- **Service Status**: Real-time monitoring of Neo4j and Ollama services
- **Graph Statistics**: Live counts of entities, relations, and documents
- **Quick Actions**: One-click operations for common tasks
- **Recent Activity**: Timeline of system operations

### 🔍 Document Extraction
- **Multiple Input Methods**: File upload, text paste, or dataset selection
- **Live Extraction**: Real-time entity and relation extraction
- **Configurable Settings**: Adjustable confidence thresholds and LLM parameters
- **Results Visualization**: Interactive tables and charts
- **Graph Storage**: Direct integration with knowledge graph

### 📈 Benchmark Runner
- **Dataset Selection**: Choose from available benchmark configurations
- **Progress Tracking**: Real-time progress monitoring during benchmark runs
- **Results Visualization**: Interactive charts and metrics display
- **Historical Results**: Access to previous benchmark runs

### 🗂️ Knowledge Graph Explorer
- **Interactive Queries**: Built-in query interface for entities and relations
- **Visualization**: Charts showing entity/relation type distributions
- **Custom Cypher**: Execute custom Neo4j Cypher queries
- **Real-time Statistics**: Live graph metrics and statistics

### ⚙️ Configuration Management
- **Visual Editor**: User-friendly interface for all configuration settings
- **Service Testing**: Built-in connection testing for Neo4j and Ollama
- **Validation**: Real-time configuration validation with error reporting
- **Import/Export**: Easy configuration file management

### 🔧 Troubleshooting Tools
- **Health Monitoring**: Comprehensive service health checks
- **Performance Analysis**: Query performance profiling and analysis
- **Error Tracking**: Detailed error logs and debugging information
- **Diagnostics**: Automated system diagnostics with recommendations

## Configuration

The GUI applications automatically load configuration from `config/settings.yaml`. You can modify settings through the Configuration Manager interface or by editing the YAML file directly.

### Environment Variables
The following environment variables are supported:
- `NEO4J_URI`: Neo4j connection URI
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `OLLAMA_BASE_URL`: Ollama API base URL
- `OLLAMA_MODEL`: Ollama model name

## Troubleshooting

### Common Issues

1. **Services Not Detected**:
   - Ensure Neo4j and Ollama are running
   - Check connection settings in Configuration Manager
   - Verify environment variables are set correctly

2. **Import Errors**:
   - Make sure you're running from the project root directory
   - Check that all dependencies are installed
   - Verify Python path includes the project directory

3. **Port Conflicts**:
   - Use different ports with `--port` option
   - Check if ports are already in use
   - Use `--host 0.0.0.0` for network access

### Getting Help

- Check the service status indicators in the dashboard
- Use the built-in diagnostics tools
- Review error logs in the troubleshooting interface
- Validate configuration settings

## Development

### Adding New Features

1. **New GUI App**: Create a new Python file in this directory
2. **Update Launcher**: Add the new app to `launcher.py`
3. **Documentation**: Update this README with new features

### Code Structure

- `streamlit_app.py`: Main application with comprehensive monitoring
- `troubleshooting.py`: Specialized troubleshooting and diagnostics
- `config_manager.py`: Configuration management interface
- `launcher.py`: Application launcher and discovery
- `__init__.py`: Package initialization and documentation

Each application is designed to be self-contained while sharing common utilities and configuration management.
