"""
NLP Command Parser GUI (Tkinter)
===============================
A simple offline GUI for parsing temperature/fan commands using your NLP parser.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
try:
    from nlp.command_parser import parse_command
except ImportError:
    from command_parser import parse_command

# Main window
root = tk.Tk()
root.title("NLP Command Parser")
root.geometry("500x300")
root.configure(bg="#f0f4f8")

# Title
title = tk.Label(root, text="NLP Command Parser", font=("Arial", 18, "bold"), bg="#f0f4f8", fg="#2a4d69")
title.pack(pady=10)

# Input frame
frame = tk.Frame(root, bg="#f0f4f8")
frame.pack(pady=10)

cmd_label = tk.Label(frame, text="Enter command:", font=("Arial", 12), bg="#f0f4f8")
cmd_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
cmd_entry = tk.Entry(frame, font=("Arial", 12), width=40)
cmd_entry.grid(row=0, column=1, padx=5, pady=5)

# Output area
output_label = tk.Label(root, text="Parsed result:", font=("Arial", 12), bg="#f0f4f8")
output_label.pack(pady=(10,0))
output_text = tk.Text(root, font=("Consolas", 12), height=16, width=80, bg="#eaf6fb", fg="#222")
output_text.pack(pady=5)

# Parse function
def parse():
    text = cmd_entry.get().strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter a command to parse.")
        return
    try:
        result = parse_command(text)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Error: {str(e)}")

# Parse button
parse_btn = tk.Button(root, text="Parse", font=("Arial", 14, "bold"), bg="#4f8a8b", fg="white", command=parse)
parse_btn.pack(pady=10)

# Footer
footer = tk.Label(root, text="Made with Tkinter (NLP parser)", font=("Arial", 10), bg="#f0f4f8", fg="#888")
footer.pack(side="bottom", pady=10)

root.mainloop()
