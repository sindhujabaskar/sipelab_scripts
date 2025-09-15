"""
Pupil GUI Viewer - A self-contained GUI for visualizing pupil videos and diameter traces

This module provides a simple GUI interface to:
1. Browse and select directories containing pupil pickle files
2. Load and display corresponding MP4 videos
3. Visualize pupil diameter traces in real-time
4. Synchronize video playback with trace visualization

Dependencies: tkinter, cv2, matplotlib, pandas, numpy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import pandas as pd
import numpy as np
import math
import statistics as st
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import time


class PupilGUIViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Pupil Video & Diameter Trace Viewer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.current_directory = None
        self.pickle_files = []
        self.current_file_index = 0
        self.video_cap = None
        self.pupil_data = None
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.fps = 30  # Default FPS
        
        # Threading
        self.video_thread = None
        self.stop_thread = False
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Directory selection
        dir_frame = ttk.Frame(control_frame)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(dir_frame, text="Select Directory", 
                  command=self.select_directory).pack(side=tk.LEFT)
        
        self.dir_label = ttk.Label(dir_frame, text="No directory selected")
        self.dir_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="Pickle Files:").pack(side=tk.LEFT)
        
        self.file_combo = ttk.Combobox(file_frame, state="readonly", width=50)
        self.file_combo.pack(side=tk.LEFT, padx=(10, 0))
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_selected)
        
        # Playback controls
        playback_frame = ttk.Frame(control_frame)
        playback_frame.pack(fill=tk.X)
        
        self.play_button = ttk.Button(playback_frame, text="Play", 
                                     command=self.toggle_playback, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT)
        
        self.frame_scale = tk.Scale(playback_frame, from_=0, to=100, 
                                   orient=tk.HORIZONTAL, command=self.on_frame_change)
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        self.frame_label = ttk.Label(playback_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # Content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display
        video_frame = ttk.LabelFrame(content_frame, text="Video", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(video_frame, text="No video loaded")
        self.video_label.pack(expand=True)
        
        # Plot area
        plot_frame = ttk.LabelFrame(content_frame, text="Pupil Diameter Trace", padding=10)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plot
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Pupil Diameter (mm)")
        self.ax.set_title("Pupil Diameter Over Time")
        self.line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.current_frame_line = self.ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
    def select_directory(self):
        """Select directory containing pupil pickle files"""
        directory = filedialog.askdirectory(title="Select Directory with Pupil Pickle Files")
        if directory:
            self.current_directory = Path(directory)
            self.dir_label.config(text=f"Directory: {directory}")
            self.scan_pickle_files()
            
    def scan_pickle_files(self):
        """Scan directory for pickle files and populate combo box"""
        if not self.current_directory:
            return
            
        # Find all pickle files
        pickle_pattern = "*full.pickle"
        self.pickle_files = list(self.current_directory.glob(pickle_pattern))
        
        if not self.pickle_files:
            messagebox.showwarning("No Files", "No pickle files found in selected directory")
            return
            
        # Populate combo box
        file_names = [f.name for f in self.pickle_files]
        self.file_combo['values'] = file_names
        
        if file_names:
            self.file_combo.current(0)
            self.on_file_selected(None)
            
    def on_file_selected(self, event):
        """Handle file selection from combo box"""
        if not self.file_combo.get():
            return
            
        self.current_file_index = self.file_combo.current()
        selected_file = self.pickle_files[self.current_file_index]
        
        # Load pickle data and corresponding video
        self.load_pupil_data(selected_file)
        self.load_video(selected_file)
        
    def load_pupil_data(self, pickle_file):
        """Load and analyze pupil data from pickle file"""
        try:
            # Load raw pickle data using existing function
            raw_data = self.deeplabcut_pickle(pickle_file)
            
            # Analyze pupil data using existing function
            self.pupil_data = self.analyze_pupil_data(raw_data)
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load pupil data: {str(e)}")
            
    def load_video(self, pickle_file):
        """Load corresponding video file"""
        # Look for video file with same base name
        video_extensions = ['.mp4', '.avi', '.mov']
        base_name = pickle_file.stem
        
        video_file = None
        for ext in video_extensions:
            potential_video = pickle_file.parent / (base_name + ext)
            if potential_video.exists():
                video_file = potential_video
                break
                
        if not video_file:
            messagebox.showwarning("Video Not Found", 
                                 f"No video file found for {pickle_file.name}")
            return
            
        try:
            # Release previous video if any
            if self.video_cap:
                self.video_cap.release()
                
            # Load new video
            self.video_cap = cv2.VideoCapture(str(video_file))
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Update controls
            self.frame_scale.config(to=self.total_frames - 1)
            self.play_button.config(state=tk.NORMAL)
            self.current_frame = 0
            self.frame_scale.set(0)
            
            # Display first frame
            self.display_frame(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            
    def display_frame(self, frame_number):
        """Display specific video frame"""
        if not self.video_cap:
            return
            
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to fit display
            height, width = frame_rgb.shape[:2]
            max_width, max_height = 400, 300
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PhotoImage for tkinter
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep reference
            
            # Update frame info
            self.frame_label.config(text=f"Frame: {frame_number}/{self.total_frames}")
            
            # Update plot cursor
            if self.pupil_data is not None:
                self.current_frame_line.set_xdata([frame_number, frame_number])
                self.canvas.draw_idle()
                
    def update_plot(self):
        """Update the pupil diameter plot"""
        if self.pupil_data is None:
            return
            
        # Get data
        frames = range(len(self.pupil_data))
        diameters = self.pupil_data['pupil_diameter_mm'].values
        
        # Update plot
        self.line.set_data(frames, diameters)
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Update current frame line
        self.current_frame_line.set_xdata([self.current_frame, self.current_frame])
        
        self.canvas.draw()
        
    def on_frame_change(self, value):
        """Handle frame slider change"""
        frame_number = int(float(value))
        self.current_frame = frame_number
        
        if not self.is_playing:  # Only update if not playing
            self.display_frame(frame_number)
            
    def toggle_playback(self):
        """Toggle video playback"""
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
            
    def start_playback(self):
        """Start video playback"""
        self.is_playing = True
        self.play_button.config(text="Pause")
        self.stop_thread = False
        
        self.video_thread = threading.Thread(target=self.playback_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
        
    def stop_playback(self):
        """Stop video playback"""
        self.is_playing = False
        self.play_button.config(text="Play")
        self.stop_thread = True
        
    def playback_loop(self):
        """Video playback loop (runs in separate thread)"""
        frame_delay = 1.0 / self.fps
        
        while self.is_playing and not self.stop_thread and self.current_frame < self.total_frames:
            start_time = time.time()
            
            # Update GUI in main thread
            self.root.after(0, self.display_frame, self.current_frame)
            self.root.after(0, lambda: self.frame_scale.set(self.current_frame))
            
            self.current_frame += 1
            
            # Control playback speed
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
            
        # Stop playback when done
        self.root.after(0, self.stop_playback)
        
    def deeplabcut_pickle(self, filepath: Path) -> pd.DataFrame:
        """Load DeepLabCut pickle data (copied from original script)"""
        data = pd.read_pickle(filepath)

        coordinates_dict = {}
        confidence_dict = {}
        for frame_key, frame_data in data.items():
            coordinates_dict[frame_key] = frame_data.get('coordinates')
            confidence_dict[frame_key] = frame_data.get('confidence')

        coords_series = pd.Series(coordinates_dict).iloc[1:]
        conf_series = pd.Series(confidence_dict).iloc[1:]

        df = pd.DataFrame({
            'coordinates': coords_series,
            'confidence': conf_series,
        })
        df.index.name = 'frame'
        df = df.drop(columns=['metadata'], errors='ignore')

        return df

    def euclidean_distance(self, coord1, coord2):
        """Calculate Euclidean distance between two points"""
        return math.dist(coord1, coord2)

    def analyze_pupil_data(self, pickle_data: pd.DataFrame,
                          confidence_threshold: float = 0.95,
                          pixel_to_mm: float = 53.6,
                          dpi: int = 300) -> pd.DataFrame:
        """Analyze pupil data from DeepLabCut output (copied from original script)"""
        coords_list = pickle_data['coordinates'].tolist()[1:]
        conf_list = pickle_data['confidence'].tolist()[1:]
        
        if not any(np.any(np.array(c) >= confidence_threshold) for c in conf_list):
            print(f"[WARNING] No confidence values above threshold {confidence_threshold}.")
            
        coords_arrs = [np.array(c) for c in coords_list]
        conf_arrs = [np.array(c) for c in conf_list]

        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        diameters = []
        
        for i, (coords, conf) in enumerate(zip(coords_arrs, conf_arrs)):
            pts = np.squeeze(coords)
            cvals = np.squeeze(conf)
            
            if pts.ndim != 2 or cvals.ndim != 1:
                print(f"[WARNING] frame {i} unexpected pts.shape={pts.shape}, conf.shape={cvals.shape}")
                diameters.append(np.nan)
                continue
                
            valid = cvals >= confidence_threshold
            ds = [
                self.euclidean_distance(pts[a], pts[b])
                for a, b in pairs
                if a < pts.shape[0] and b < pts.shape[0] and valid[a] and valid[b]
            ]
            diameters.append(st.mean(ds) if ds else np.nan)

        pupil_series = (
            pd.Series(diameters, index=pickle_data.index[1:])
              .interpolate()
              .divide(pixel_to_mm)
        )
        pupil_full = pupil_series.reindex(pickle_data.index)

        return pd.DataFrame({'pupil_diameter_mm': pupil_full.iloc[1:]})
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.video_cap:
            self.video_cap.release()


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = PupilGUIViewer(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        # Cleanup
        if hasattr(app, 'video_cap') and app.video_cap:
            app.video_cap.release()


if __name__ == "__main__":
    main()
