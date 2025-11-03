#!/usr/bin/env python3
"""
Streamlit GUI Launcher for DocBench: Knowledge Graph-Based Document Benchmarking

This script provides a convenient way to launch the Streamlit GUI applications
for DocBench document benchmarking system.

Usage:
    python src/gui/launcher.py [app_name]

Available apps:
    - main: Main DocBench interface with PDF processing, extraction, and evaluation
    - troubleshooting: Advanced troubleshooting tools
    - config: Configuration management interface
"""

import sys
import subprocess
import argparse
from pathlib import Path

def get_available_apps():
    """Get list of available Streamlit apps"""
    gui_dir = Path(__file__).parent
    apps = {}
    
    # Main monitoring app
    if (gui_dir / "streamlit_app.py").exists():
        apps["main"] = {
            "file": "streamlit_app.py",
            "description": "Main DocBench interface with PDF processing, extraction, and evaluation",
            "port": 8501
        }
    
    # Troubleshooting app
    if (gui_dir / "troubleshooting.py").exists():
        apps["troubleshooting"] = {
            "file": "troubleshooting.py", 
            "description": "Advanced troubleshooting and diagnostics",
            "port": 8502
        }
    
    # Configuration app
    if (gui_dir / "config_manager.py").exists():
        apps["config"] = {
            "file": "config_manager.py",
            "description": "Configuration management interface", 
            "port": 8503
        }
    
    return apps

def launch_app(app_name: str, port: int = None, host: str = "localhost"):
    """Launch a Streamlit app"""
    gui_dir = Path(__file__).parent
    apps = get_available_apps()
    
    if app_name not in apps:
        print(f"❌ App '{app_name}' not found!")
        print(f"Available apps: {', '.join(apps.keys())}")
        return False
    
    app_info = apps[app_name]
    app_file = gui_dir / app_info["file"]
    
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        return False
    
    # Use specified port or default from app info
    app_port = port or app_info["port"]
    
    # Build command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_file),
        "--server.port", str(app_port),
        "--server.address", host,
        "--server.headless", "true"
    ]
    
    print(f"🚀 Launching {app_name} app...")
    print(f"📄 File: {app_file}")
    print(f"🌐 URL: http://{host}:{app_port}")
    print(f"📝 Description: {app_info['description']}")
    print()
    
    try:
        # Launch the app
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch app: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
        return True

def list_apps():
    """List all available apps"""
    apps = get_available_apps()
    
    if not apps:
        print("❌ No Streamlit apps found!")
        return
    
    print("📱 Available Streamlit Apps:")
    print("=" * 50)
    
    for name, info in apps.items():
        print(f"🔹 {name}")
        print(f"   Description: {info['description']}")
        print(f"   Default Port: {info['port']}")
        print(f"   File: {info['file']}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Streamlit GUI Launcher for DocBench: Knowledge Graph-Based Document Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/gui/launcher.py main
    python src/gui/launcher.py troubleshooting --port 8502
    python src/gui/launcher.py config --host 0.0.0.0
    python src/gui/launcher.py --list
        """
    )
    
    parser.add_argument(
        "app",
        nargs="?",
        help="App name to launch (main, troubleshooting, config)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available apps"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Port to run the app on"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind the app to (default: localhost)"
    )
    
    args = parser.parse_args()
    
    # List apps if requested
    if args.list:
        list_apps()
        return
    
    # Check if app name provided
    if not args.app:
        print("❌ Please specify an app name or use --list to see available apps")
        print()
        list_apps()
        return
    
    # Launch the app
    success = launch_app(args.app, args.port, args.host)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
