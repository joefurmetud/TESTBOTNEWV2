#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo version of the GUI for testing and screenshots
Shows the interface with sample data
"""

import tkinter as tk
from bot_gui import AppleStyleGUI
import time

class DemoGUI(AppleStyleGUI):
    def __init__(self):
        super().__init__()
        self.setup_demo_data()
    
    def setup_demo_data(self):
        """Setup demo data for showcase"""
        # Set demo status
        self.status_var.set("● Running")
        self.status_label.configure(foreground=self.colors['accent_green'])
        
        # Set demo uptime
        self.uptime_var.set("02:34:17")
        
        # Set demo statistics
        self.messages_var.set("1,247")
        self.users_var.set("89")
        self.errors_var.set("3")
        
        # Update button states
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.restart_btn.configure(state='normal')
        
        # Add some demo logs
        self.root.after(1000, self.add_demo_logs)
    
    def add_demo_logs(self):
        """Add demo log entries"""
        demo_logs = [
            ("Bot started successfully", "INFO"),
            ("Connected to Telegram API", "INFO"),
            ("Loading trusted sellers from database", "INFO"),
            ("Scheduler configured and started", "INFO"),
            ("Webhook configured for production", "INFO"),
            ("Processing message from user @testuser", "INFO"),
            ("Vote recorded for @seller1 (+15 points)", "INFO"),
            ("Daily points awarded to 23 users", "INFO"),
            ("Weekly leaderboard updated", "INFO"),
            ("Connection timeout, retrying...", "WARNING"),
            ("Reconnected successfully", "INFO"),
            ("Database backup completed", "INFO"),
            ("Memory cleanup performed", "DEBUG"),
            ("Processing coinflip challenge", "INFO"),
            ("User @user123 earned 5 points", "INFO")
        ]
        
        for i, (message, level) in enumerate(demo_logs):
            self.root.after(i * 200, lambda msg=message, lvl=level: self.add_log(msg, lvl))
    
    def check_environment(self):
        """Override to always return True for demo"""
        return True

def main():
    """Run the demo GUI"""
    print("🎨 Starting Demo GUI...")
    print("This shows the interface with sample data for demonstration")
    print("Close the window when done viewing")
    
    app = DemoGUI()
    app.root.title("Bot Manager - Demo Mode")
    app.root.protocol("WM_DELETE_WINDOW", app.root.destroy)
    app.run()

if __name__ == "__main__":
    main()