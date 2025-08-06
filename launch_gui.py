#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Launcher for Telegram Bot Manager
Launches the Apple-style GUI interface
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment variables if .env file exists"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def main():
    """Main launcher"""
    print("🚀 Starting Telegram Bot Manager GUI...")
    
    # Setup environment
    setup_environment()
    
    # Import and run GUI
    try:
        from bot_gui import AppleStyleGUI
        
        app = AppleStyleGUI()
        app.root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        print("✅ GUI launched successfully!")
        app.run()
        
    except ImportError as e:
        print(f"❌ Failed to import GUI: {e}")
        print("Make sure all required packages are installed:")
        print("pip install tkinter")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()