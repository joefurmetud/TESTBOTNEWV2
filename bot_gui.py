# -*- coding: utf-8 -*-
"""
Apple-Style GUI for Telegram Bot Management
Clean, modern interface with premium aesthetics
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

class AppleStyleGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_styles()
        self.setup_variables()
        self.create_interface()
        self.bot_process = None
        self.log_thread = None
        self.is_running = False
        
    def setup_window(self):
        """Configure main window with Apple-style properties"""
        self.root.title("Bot Manager")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")
        
        # Configure for macOS-like appearance
        self.root.configure(bg='#f5f5f7')
        
        # Try to set window icon if available
        try:
            if sys.platform == "darwin":  # macOS
                self.root.call('wm', 'iconbitmap', self.root._w, '')
        except:
            pass
    
    def setup_styles(self):
        """Configure ttk styles for Apple-like appearance"""
        self.style = ttk.Style()
        
        # Configure theme
        if sys.platform == "darwin":
            self.style.theme_use('aqua')
        else:
            self.style.theme_use('clam')
        
        # Apple color palette
        self.colors = {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f5f5f7',
            'bg_tertiary': '#f2f2f7',
            'accent_blue': '#007aff',
            'accent_green': '#34c759',
            'accent_red': '#ff3b30',
            'accent_orange': '#ff9500',
            'text_primary': '#1d1d1f',
            'text_secondary': '#86868b',
            'border': '#d2d2d7',
            'shadow': '#00000010'
        }
        
        # Custom button styles
        self.style.configure('Primary.TButton',
                           background=self.colors['accent_blue'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           padding=(20, 12))
        
        self.style.configure('Success.TButton',
                           background=self.colors['accent_green'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           padding=(20, 12))
        
        self.style.configure('Danger.TButton',
                           background=self.colors['accent_red'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           padding=(20, 12))
        
        self.style.configure('Secondary.TButton',
                           background=self.colors['bg_tertiary'],
                           foreground=self.colors['text_primary'],
                           borderwidth=1,
                           bordercolor=self.colors['border'],
                           focuscolor='none',
                           padding=(20, 12))
        
        # Custom frame styles
        self.style.configure('Card.TFrame',
                           background=self.colors['bg_primary'],
                           borderwidth=1,
                           relief='solid',
                           bordercolor=self.colors['border'])
        
        # Custom label styles
        self.style.configure('Title.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_primary'],
                           font=('SF Pro Display', 24, 'bold'))
        
        self.style.configure('Heading.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_primary'],
                           font=('SF Pro Display', 16, 'bold'))
        
        self.style.configure('Body.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_secondary'],
                           font=('SF Pro Text', 13))
        
        self.style.configure('Status.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['accent_green'],
                           font=('SF Pro Text', 13, 'bold'))
    
    def setup_variables(self):
        """Initialize tkinter variables"""
        self.status_var = tk.StringVar(value="● Stopped")
        self.uptime_var = tk.StringVar(value="00:00:00")
        self.messages_var = tk.StringVar(value="0")
        self.users_var = tk.StringVar(value="0")
        self.errors_var = tk.StringVar(value="0")
        
    def create_interface(self):
        """Create the main interface"""
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['bg_secondary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header(main_container)
        
        # Content area with two columns
        content_frame = tk.Frame(main_container, bg=self.colors['bg_secondary'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left column - Controls
        left_column = tk.Frame(content_frame, bg=self.colors['bg_secondary'])
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right column - Logs
        right_column = tk.Frame(content_frame, bg=self.colors['bg_secondary'])
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Create sections
        self.create_control_panel(left_column)
        self.create_statistics_panel(left_column)
        self.create_settings_panel(left_column)
        self.create_log_panel(right_column)
        
    def create_header(self, parent):
        """Create the header section"""
        header_frame = self.create_card(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Header content
        header_content = tk.Frame(header_frame, bg=self.colors['bg_primary'])
        header_content.pack(fill=tk.X, padx=30, pady=25)
        
        # Title and status
        title_frame = tk.Frame(header_content, bg=self.colors['bg_primary'])
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(title_frame, text="Telegram Bot Manager", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Status indicator
        status_frame = tk.Frame(title_frame, bg=self.colors['bg_primary'])
        status_frame.pack(side=tk.RIGHT)
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        uptime_label = ttk.Label(status_frame, textvariable=self.uptime_var, style='Body.TLabel')
        uptime_label.pack(side=tk.RIGHT)
        
    def create_control_panel(self, parent):
        """Create the control panel"""
        card = self.create_card(parent)
        card.pack(fill=tk.X, pady=(0, 20))
        
        # Card header
        header = tk.Frame(card, bg=self.colors['bg_primary'])
        header.pack(fill=tk.X, padx=30, pady=(25, 15))
        
        ttk.Label(header, text="Bot Control", style='Heading.TLabel').pack(side=tk.LEFT)
        
        # Control buttons
        controls = tk.Frame(card, bg=self.colors['bg_primary'])
        controls.pack(fill=tk.X, padx=30, pady=(0, 25))
        
        self.start_btn = ttk.Button(controls, text="Start Bot", 
                                   style='Success.TButton',
                                   command=self.start_bot)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        self.stop_btn = ttk.Button(controls, text="Stop Bot", 
                                  style='Danger.TButton',
                                  command=self.stop_bot,
                                  state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        self.restart_btn = ttk.Button(controls, text="Restart Bot", 
                                     style='Secondary.TButton',
                                     command=self.restart_bot,
                                     state='disabled')
        self.restart_btn.pack(side=tk.LEFT)
        
    def create_statistics_panel(self, parent):
        """Create the statistics panel"""
        card = self.create_card(parent)
        card.pack(fill=tk.X, pady=(0, 20))
        
        # Card header
        header = tk.Frame(card, bg=self.colors['bg_primary'])
        header.pack(fill=tk.X, padx=30, pady=(25, 15))
        
        ttk.Label(header, text="Statistics", style='Heading.TLabel').pack(side=tk.LEFT)
        
        # Stats grid
        stats_frame = tk.Frame(card, bg=self.colors['bg_primary'])
        stats_frame.pack(fill=tk.X, padx=30, pady=(0, 25))
        
        # Configure grid
        for i in range(3):
            stats_frame.grid_columnconfigure(i, weight=1)
        
        # Messages stat
        self.create_stat_item(stats_frame, "Messages", self.messages_var, 0, 0)
        
        # Users stat
        self.create_stat_item(stats_frame, "Active Users", self.users_var, 0, 1)
        
        # Errors stat
        self.create_stat_item(stats_frame, "Errors", self.errors_var, 0, 2)
        
    def create_stat_item(self, parent, label, variable, row, col):
        """Create a single statistic item"""
        stat_frame = tk.Frame(parent, bg=self.colors['bg_tertiary'])
        stat_frame.grid(row=row, column=col, padx=(0, 15) if col < 2 else (0, 0), 
                       pady=5, sticky='ew')
        
        # Add padding
        stat_content = tk.Frame(stat_frame, bg=self.colors['bg_tertiary'])
        stat_content.pack(padx=20, pady=15)
        
        value_label = tk.Label(stat_content, textvariable=variable,
                              bg=self.colors['bg_tertiary'],
                              fg=self.colors['text_primary'],
                              font=('SF Pro Display', 20, 'bold'))
        value_label.pack()
        
        label_label = tk.Label(stat_content, text=label,
                              bg=self.colors['bg_tertiary'],
                              fg=self.colors['text_secondary'],
                              font=('SF Pro Text', 12))
        label_label.pack()
        
    def create_settings_panel(self, parent):
        """Create the settings panel"""
        card = self.create_card(parent)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        header = tk.Frame(card, bg=self.colors['bg_primary'])
        header.pack(fill=tk.X, padx=30, pady=(25, 15))
        
        ttk.Label(header, text="Quick Settings", style='Heading.TLabel').pack(side=tk.LEFT)
        
        # Settings content
        settings_frame = tk.Frame(card, bg=self.colors['bg_primary'])
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 25))
        
        # Environment status
        env_frame = tk.Frame(settings_frame, bg=self.colors['bg_primary'])
        env_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(env_frame, text="Environment:", style='Body.TLabel').pack(side=tk.LEFT)
        
        env_status = "✅ Configured" if self.check_environment() else "❌ Missing Variables"
        env_color = self.colors['accent_green'] if self.check_environment() else self.colors['accent_red']
        
        env_label = tk.Label(env_frame, text=env_status,
                            bg=self.colors['bg_primary'],
                            fg=env_color,
                            font=('SF Pro Text', 13, 'bold'))
        env_label.pack(side=tk.RIGHT)
        
        # Action buttons
        actions_frame = tk.Frame(settings_frame, bg=self.colors['bg_primary'])
        actions_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(actions_frame, text="View Logs", 
                  style='Secondary.TButton',
                  command=self.open_logs_folder).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Button(actions_frame, text="Clear Logs", 
                  style='Secondary.TButton',
                  command=self.clear_logs).pack(side=tk.LEFT)
        
    def create_log_panel(self, parent):
        """Create the log panel"""
        card = self.create_card(parent)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        header = tk.Frame(card, bg=self.colors['bg_primary'])
        header.pack(fill=tk.X, padx=30, pady=(25, 15))
        
        ttk.Label(header, text="Live Logs", style='Heading.TLabel').pack(side=tk.LEFT)
        
        # Clear logs button
        ttk.Button(header, text="Clear", 
                  style='Secondary.TButton',
                  command=self.clear_log_display).pack(side=tk.RIGHT)
        
        # Log display
        log_frame = tk.Frame(card, bg=self.colors['bg_primary'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 25))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('SF Mono', 11),
            wrap=tk.WORD,
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=0
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for colored output
        self.log_text.tag_configure('INFO', foreground='#00ff00')
        self.log_text.tag_configure('WARNING', foreground='#ffff00')
        self.log_text.tag_configure('ERROR', foreground='#ff0000')
        self.log_text.tag_configure('DEBUG', foreground='#888888')
        
    def create_card(self, parent):
        """Create a card-style container"""
        card = tk.Frame(parent, bg=self.colors['bg_primary'], 
                       relief='solid', bd=1, 
                       highlightbackground=self.colors['border'],
                       highlightthickness=1)
        return card
        
    def check_environment(self):
        """Check if required environment variables are set"""
        required_vars = ['TELEGRAM_TOKEN', 'ADMIN_CHAT_ID', 'GROUP_CHAT_ID', 
                        'VOTING_GROUP_CHAT_ID', 'VOTING_GROUP_LINK']
        
        for var in required_vars:
            if not os.getenv(var):
                return False
        return True
        
    def start_bot(self):
        """Start the bot process"""
        if not self.check_environment():
            messagebox.showerror("Configuration Error", 
                               "Required environment variables are missing.\n"
                               "Please check your .env file or system environment.")
            return
            
        try:
            # Start bot process
            self.bot_process = subprocess.Popen(
                [sys.executable, 'OGbotas.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.is_running = True
            self.start_time = time.time()
            
            # Update UI
            self.status_var.set("● Running")
            self.status_label.configure(foreground=self.colors['accent_green'])
            self.start_btn.configure(state='disabled')
            self.stop_btn.configure(state='normal')
            self.restart_btn.configure(state='normal')
            
            # Start log monitoring
            self.start_log_monitoring()
            
            # Start uptime counter
            self.update_uptime()
            
            self.add_log("Bot started successfully", "INFO")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start bot: {str(e)}")
            self.add_log(f"Failed to start bot: {str(e)}", "ERROR")
            
    def stop_bot(self):
        """Stop the bot process"""
        try:
            if self.bot_process:
                self.bot_process.terminate()
                self.bot_process.wait(timeout=5)
                self.bot_process = None
                
            self.is_running = False
            
            # Update UI
            self.status_var.set("● Stopped")
            self.status_label.configure(foreground=self.colors['accent_red'])
            self.start_btn.configure(state='normal')
            self.stop_btn.configure(state='disabled')
            self.restart_btn.configure(state='disabled')
            
            self.add_log("Bot stopped", "INFO")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop bot: {str(e)}")
            self.add_log(f"Failed to stop bot: {str(e)}", "ERROR")
            
    def restart_bot(self):
        """Restart the bot"""
        self.add_log("Restarting bot...", "INFO")
        self.stop_bot()
        time.sleep(2)  # Wait a bit before restarting
        self.start_bot()
        
    def start_log_monitoring(self):
        """Start monitoring bot logs"""
        if self.log_thread and self.log_thread.is_alive():
            return
            
        self.log_thread = threading.Thread(target=self.monitor_logs, daemon=True)
        self.log_thread.start()
        
    def monitor_logs(self):
        """Monitor bot output and update log display"""
        if not self.bot_process:
            return
            
        while self.bot_process and self.bot_process.poll() is None:
            try:
                line = self.bot_process.stdout.readline()
                if line:
                    self.add_log(line.strip())
                    self.update_statistics(line)
            except Exception as e:
                self.add_log(f"Log monitoring error: {str(e)}", "ERROR")
                break
                
    def add_log(self, message, level="INFO"):
        """Add a log message to the display"""
        if not message:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        def update_display():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, formatted_message, level)
            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)
            
        self.root.after(0, update_display)
        
    def update_statistics(self, log_line):
        """Update statistics based on log content"""
        # Simple pattern matching for statistics
        if "message" in log_line.lower():
            current = int(self.messages_var.get())
            self.messages_var.set(str(current + 1))
            
        if "user" in log_line.lower() and "new" in log_line.lower():
            current = int(self.users_var.get())
            self.users_var.set(str(current + 1))
            
        if "error" in log_line.lower() or "failed" in log_line.lower():
            current = int(self.errors_var.get())
            self.errors_var.set(str(current + 1))
            
    def update_uptime(self):
        """Update uptime display"""
        if self.is_running:
            uptime_seconds = int(time.time() - self.start_time)
            hours = uptime_seconds // 3600
            minutes = (uptime_seconds % 3600) // 60
            seconds = uptime_seconds % 60
            
            self.uptime_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
        # Schedule next update
        self.root.after(1000, self.update_uptime)
        
    def clear_log_display(self):
        """Clear the log display"""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
    def clear_logs(self):
        """Clear log files"""
        try:
            data_dir = os.getenv('DATA_DIR', '/opt/render/data')
            log_dir = Path(data_dir) / 'logs'
            
            if log_dir.exists():
                for log_file in log_dir.glob('*.log*'):
                    log_file.unlink()
                    
            messagebox.showinfo("Success", "Log files cleared successfully")
            self.add_log("Log files cleared", "INFO")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear logs: {str(e)}")
            self.add_log(f"Failed to clear logs: {str(e)}", "ERROR")
            
    def open_logs_folder(self):
        """Open logs folder in file manager"""
        try:
            data_dir = os.getenv('DATA_DIR', '/opt/render/data')
            log_dir = Path(data_dir) / 'logs'
            
            if sys.platform == "win32":
                os.startfile(log_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", log_dir])
            else:
                subprocess.run(["xdg-open", log_dir])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open logs folder: {str(e)}")
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()
        
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Bot is still running. Stop it before closing?"):
                self.stop_bot()
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    """Main entry point"""
    app = AppleStyleGUI()
    app.root.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.run()

if __name__ == "__main__":
    main()