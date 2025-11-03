#!/usr/bin/env python3
"""
DocBench Launcher

This script provides easy access to DocBench functionality.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main launcher function"""
    print("🚀 DocBench: Knowledge Graph-Based Document Benchmarking")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python launcher.py <command>")
        print("\nAvailable commands:")
        print("  gui          - Launch the Streamlit GUI")
        print("  cli          - Launch the CLI interface")
        print("  example      - Run the DocBench example")
        print("  process-pdf  - Process PDF files to markdown")
        print("  extract      - Extract information and populate knowledge graph")
        print("  evaluate     - Run DocBench evaluation")
        print("  query        - Query the knowledge graph")
        print("  init         - Initialize the knowledge graph")
        print("  clear        - Clear the knowledge graph")
        print("\nExamples:")
        print("  python launcher.py gui")
        print("  python launcher.py process-pdf")
        print("  python launcher.py extract")
        print("  python launcher.py evaluate")
        print("  python launcher.py query --natural-language 'What entities are mentioned?'")
        return
    
    command = sys.argv[1]
    
    if command == "gui":
        print("🌐 Launching DocBench GUI...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/gui/streamlit_app.py"])
    
    elif command == "cli":
        print("💻 Launching DocBench CLI...")
        subprocess.run([sys.executable, "-m", "src.cli.main"] + sys.argv[2:])
    
    elif command == "example":
        print("📚 Running DocBench example...")
        subprocess.run([sys.executable, "docbench_example.py"])
    
    elif command in ["process-pdf", "extract", "evaluate", "query", "init", "clear"]:
        print(f"🔧 Running DocBench CLI command: {command}")
        subprocess.run([sys.executable, "-m", "src.cli.main", command] + sys.argv[2:])
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python launcher.py' to see available commands")

if __name__ == "__main__":
    main()
