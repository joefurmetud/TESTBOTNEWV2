#!/bin/bash

echo "🚀 Starting Telegram Bot Manager GUI..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 from https://python.org"
    exit 1
fi

# Check if the GUI file exists
if [ ! -f "bot_gui.py" ]; then
    echo "❌ Error: bot_gui.py not found in current directory"
    exit 1
fi

# Launch the GUI
python3 launch_gui.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ GUI failed to start. Check the error messages above."
    read -p "Press Enter to exit..."
fi