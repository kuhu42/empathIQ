import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Sample CSV data as a string (you can save it as a file if needed)
csv_data = """Session,Date,RULA,REBA
1,2025-01-01,5,6
2,2025-01-03,4,5
3,2025-01-05,3,4
4,2025-01-07,3,4
5,2025-01-09,2,3
6,2025-01-11,3,3
7,2025-01-13,2,2
8,2025-01-15,2,2
9,2025-01-17,1,1
10,2025-01-19,1,1"""

# Function to load the CSV data from a file (optional) or from the sample string
def load_data():
    try:
        # Uncomment the next line if you want to load from a file
        # data = pd.read_csv(filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")]))
        
        # For demonstration, load data from the sample CSV string
        from io import StringIO
        data = pd.read_csv(StringIO(csv_data), parse_dates=['Date'])
        data.sort_values("Session", inplace=True)
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Could not load data: {e}")
        return None

def plot_progress(data):
    # Extract RULA and REBA arrays
    sessions = data['Session']
    rula_scores = data['RULA']
    reba_scores = data['REBA']
    
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sessions, rula_scores, marker='o', color='blue', label='RULA Score')
    ax.plot(sessions, reba_scores, marker='o', color='green', label='REBA Score')
    ax.set_title("Posture Improvement Over Sessions")
    ax.set_xlabel("Session")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)
    return fig

# Main UI
class PostureDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Posture Progress Report")
        self.geometry("900x700")
        self.create_widgets()
        
    def create_widgets(self):
        # Frame for the plot
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Frame for control buttons
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.btn_load = ttk.Button(self.control_frame, text="Load Data", command=self.load_and_plot)
        self.btn_load.pack(side=tk.LEFT, padx=10)
        
        self.btn_quit = ttk.Button(self.control_frame, text="Exit", command=self.destroy)
        self.btn_quit.pack(side=tk.RIGHT, padx=10)
        
        # Label for progress summary
        self.report_text = tk.StringVar()
        self.report_label = ttk.Label(self.control_frame, textvariable=self.report_text, font=('Helvetica', 12))
        self.report_label.pack(side=tk.LEFT, padx=10)
        
    def load_and_plot(self):
        data = load_data()
        if data is not None:
            fig = plot_progress(data)
            # Clear the previous canvas if it