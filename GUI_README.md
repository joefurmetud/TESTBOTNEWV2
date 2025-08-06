# 🎨 Apple-Style Bot Manager GUI

A beautiful, clean, and modern graphical interface for managing your Telegram bot, inspired by Apple's design principles.

## ✨ Features

### 🎯 Clean Design
- **Apple-inspired interface** with rounded corners, subtle shadows, and premium colors
- **Elegant typography** using system fonts (SF Pro Display/Text on macOS)
- **Card-based layout** for organized content presentation
- **Responsive design** that adapts to different window sizes

### 🚀 Bot Management
- **One-click Start/Stop/Restart** bot controls
- **Real-time status monitoring** with visual indicators
- **Live log streaming** with syntax highlighting
- **Uptime tracking** and performance statistics

### 📊 Statistics Dashboard
- **Message count tracking** - Monitor bot activity
- **Active user statistics** - See user engagement
- **Error monitoring** - Track issues in real-time
- **Visual progress indicators** for key metrics

### ⚙️ Smart Settings
- **Environment validation** - Automatic check for required variables
- **Quick actions** - Easy access to logs and maintenance
- **Log management** - Clear logs and open log directories
- **Configuration status** - Visual confirmation of setup

## 🚀 Quick Start

### Option 1: Simple Launch (Recommended)

**Windows:**
```bash
# Double-click start_gui.bat
# OR run in terminal:
start_gui.bat
```

**macOS/Linux:**
```bash
# Make executable (first time only)
chmod +x start_gui.sh

# Run
./start_gui.sh
```

### Option 2: Direct Python Launch

```bash
# Launch with Python
python launch_gui.py

# Or directly
python bot_gui.py
```

## 📋 Requirements

### System Requirements
- **Python 3.7+** (Python 3.8+ recommended)
- **tkinter** (usually included with Python)
- **Operating System:** Windows 10+, macOS 10.14+, or Linux

### Environment Setup
Make sure your bot environment variables are configured:

```env
TELEGRAM_TOKEN=your_bot_token_here
ADMIN_CHAT_ID=your_admin_chat_id
GROUP_CHAT_ID=your_group_chat_id
VOTING_GROUP_CHAT_ID=your_voting_group_chat_id
VOTING_GROUP_LINK=your_voting_group_link
```

## 🎨 Interface Overview

### Header Section
- **Bot Manager Title** - Clean, prominent branding
- **Status Indicator** - Live status with color coding
  - 🟢 Green: Bot is running
  - 🔴 Red: Bot is stopped
- **Uptime Counter** - Real-time uptime tracking

### Control Panel
- **Start Bot** - Green button to launch the bot
- **Stop Bot** - Red button to terminate the bot
- **Restart Bot** - Gray button for quick restart

### Statistics Dashboard
Three key metrics displayed in elegant cards:
- **Messages** - Total messages processed
- **Active Users** - Number of active users
- **Errors** - Error count for monitoring

### Quick Settings
- **Environment Status** - Visual check of configuration
- **View Logs** - Open logs folder in file manager
- **Clear Logs** - Remove old log files

### Live Logs Panel
- **Real-time streaming** - See bot output as it happens
- **Syntax highlighting** - Color-coded log levels
  - 🟢 Green: INFO messages
  - 🟡 Yellow: WARNING messages
  - 🔴 Red: ERROR messages
  - ⚪ Gray: DEBUG messages
- **Auto-scroll** - Always shows latest messages
- **Clear function** - Clean log display

## 🔧 Customization

### Color Scheme
The GUI uses Apple's design system colors:
- **Primary Background:** Pure white (#ffffff)
- **Secondary Background:** Light gray (#f5f5f7)
- **Accent Blue:** System blue (#007aff)
- **Success Green:** System green (#34c759)
- **Error Red:** System red (#ff3b30)
- **Text Primary:** Near black (#1d1d1f)
- **Text Secondary:** Medium gray (#86868b)

### Typography
- **Headlines:** SF Pro Display (Bold, 24pt)
- **Subheadings:** SF Pro Display (Bold, 16pt)
- **Body Text:** SF Pro Text (Regular, 13pt)
- **Code/Logs:** SF Mono (Regular, 11pt)

## 🛠️ Troubleshooting

### Common Issues

**"Required environment variables are missing"**
- Check your `.env` file exists
- Verify all required variables are set
- Restart the GUI after making changes

**"Failed to start bot"**
- Ensure `OGbotas.py` exists in the same directory
- Check Python path and permissions
- Verify bot token is valid

**GUI doesn't look right**
- Update to Python 3.8+ for better tkinter support
- On Linux, install `python3-tk` package
- Try different system themes

### Log Locations
- **Windows:** `%DATA_DIR%\logs\` or `C:\opt\render\data\logs\`
- **macOS/Linux:** `$DATA_DIR/logs/` or `/opt/render/data/logs/`

## 🎯 Performance Tips

- **Memory Usage:** GUI uses minimal resources (~10-20MB)
- **Log Rotation:** Logs auto-rotate to prevent disk filling
- **Responsive Updates:** Statistics update in real-time without lag
- **Clean Shutdown:** Always use Stop button before closing

## 🔄 Updates

To update the GUI:
1. Download new `bot_gui.py` file
2. Replace existing file
3. Restart the GUI application
4. Check this README for new features

## 🆘 Support

If you encounter issues:
1. Check the Live Logs panel for error messages
2. Verify environment configuration
3. Ensure bot file (`OGbotas.py`) is present
4. Check Python and tkinter installation

## 🎨 Design Philosophy

This GUI follows Apple's Human Interface Guidelines:
- **Clarity** - Clean, readable interface
- **Deference** - Content over chrome
- **Depth** - Subtle visual layers and hierarchy
- **Consistency** - Familiar patterns and behaviors
- **Accessibility** - Works across different systems

The result is a professional, premium-feeling interface that makes bot management a pleasure rather than a chore.

---

**Enjoy your beautiful new bot management experience! 🚀**