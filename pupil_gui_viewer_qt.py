"""
Pupil GUI Viewer (PyQt6) - A self-contained GUI for visualizing pupil videos and diameter traces

This module provides a modern PyQt6 interface to:
1. Browse and select directories containing pupil pickle files
2. Load and display corresponding MP4 videos
3. Visualize pupil diameter traces in real-time
4. Synchronize video playback with trace visualization

Dependencies: PyQt6, cv2, matplotlib, pandas, numpy
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QSlider, QFileDialog, QMessageBox, QGroupBox,
                             QSplitter, QFrame)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import pandas as pd
import numpy as np
import math
import statistics as st
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time


class VideoThread(QThread):
    """Separate thread for video playback to prevent GUI freezing"""
    frame_changed = pyqtSignal(int)
    
    def __init__(self, fps=30):
        super().__init__()
        self.fps = fps
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.frame_delay = 1.0 / fps
        
    def set_video_params(self, total_frames, fps):
        self.total_frames = total_frames
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
    def play(self):
        self.is_playing = True
        
    def pause(self):
        self.is_playing = False
        
    def set_frame(self, frame):
        self.current_frame = frame
        
    def run(self):
        while self.is_playing and self.current_frame < self.total_frames:
            start_time = time.time()
            
            self.frame_changed.emit(self.current_frame)
            self.current_frame += 1
            
            # Control playback speed
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_delay - elapsed)
            self.msleep(int(sleep_time * 1000))
            
        self.is_playing = False


class MatplotlibWidget(QWidget):
    """Custom widget to embed matplotlib plot"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Create plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Pupil Diameter (mm)")
        self.ax.set_title("Pupil Diameter Over Time")
        self.line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.current_frame_line = self.ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
    def update_plot(self, pupil_data):
        """Update the plot with new data"""
        if pupil_data is not None:
            frames = range(len(pupil_data))
            diameters = pupil_data['pupil_diameter_mm'].values
            
            self.line.set_data(frames, diameters)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            
    def update_frame_indicator(self, frame_number):
        """Update the red line showing current frame"""
        self.current_frame_line.set_xdata([frame_number, frame_number])
        self.canvas.draw_idle()


class PupilGUIViewerQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pupil Video & Diameter Trace Viewer (PyQt6)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.current_directory = None
        self.pickle_files = []
        self.current_file_index = 0
        self.video_cap = None
        self.pupil_data = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30  # Default FPS
        
        # Video thread
        self.video_thread = VideoThread(self.fps)
        self.video_thread.frame_changed.connect(self.on_frame_update)
        
        # Timer for video playback (alternative to thread)
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_frame)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        self.setup_control_panel(main_layout)
        
        # Content area with splitter
        self.setup_content_area(main_layout)
        
    def setup_control_panel(self, parent_layout):
        """Setup the control panel with buttons and sliders"""
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_button = QPushButton("Select Directory")
        self.dir_button.clicked.connect(self.select_directory)
        self.dir_label = QLabel("No directory selected")
        
        dir_layout.addWidget(self.dir_button)
        dir_layout.addWidget(self.dir_label)
        dir_layout.addStretch()
        
        # File selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Pickle Files:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_selected)
        file_layout.addWidget(self.file_combo)
        file_layout.addStretch()
        
        # Playback controls
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        
        self.frame_label = QLabel("Frame: 0/0")
        
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.frame_slider)
        playback_layout.addWidget(self.frame_label)
        
        # Add to control layout
        control_layout.addLayout(dir_layout)
        control_layout.addLayout(file_layout)
        control_layout.addLayout(playback_layout)
        
        parent_layout.addWidget(control_group)
        
    def setup_content_area(self, parent_layout):
        """Setup the main content area with video and plot"""
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Video panel
        video_group = QGroupBox("Video")
        video_layout = QVBoxLayout(video_group)
        
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid gray;")
        video_layout.addWidget(self.video_label)
        
        # Plot panel
        plot_group = QGroupBox("Pupil Diameter Trace")
        plot_layout = QVBoxLayout(plot_group)
        
        self.plot_widget = MatplotlibWidget()
        plot_layout.addWidget(self.plot_widget)
        
        # Add to splitter
        splitter.addWidget(video_group)
        splitter.addWidget(plot_group)
        splitter.setSizes([600, 600])  # Equal sizes initially
        
        parent_layout.addWidget(splitter)
        
    def select_directory(self):
        """Select directory containing pupil pickle files"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory with Pupil Pickle Files"
        )
        if directory:
            self.current_directory = Path(directory)
            self.dir_label.setText(f"Directory: {directory}")
            self.scan_pickle_files()
            
    def scan_pickle_files(self):
        """Scan directory for pickle files and populate combo box"""
        if not self.current_directory:
            return
            
        # Find all pickle files
        self.pickle_files = list(self.current_directory.glob("*.pkl"))
        
        if not self.pickle_files:
            QMessageBox.warning(self, "No Files", 
                              "No pickle files found in selected directory")
            return
            
        # Populate combo box
        self.file_combo.clear()
        file_names = [f.name for f in self.pickle_files]
        self.file_combo.addItems(file_names)
        
        if file_names:
            self.on_file_selected(0)
            
    def on_file_selected(self, index):
        """Handle file selection from combo box"""
        if index < 0 or index >= len(self.pickle_files):
            return
            
        self.current_file_index = index
        selected_file = self.pickle_files[index]
        
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
            self.plot_widget.update_plot(self.pupil_data)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load pupil data: {str(e)}")
            
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
            QMessageBox.warning(self, "Video Not Found", 
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
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.play_button.setEnabled(True)
            self.current_frame = 0
            self.frame_slider.setValue(0)
            
            # Update video thread parameters
            self.video_thread.set_video_params(self.total_frames, self.fps)
            self.playback_timer.setInterval(int(1000 / self.fps))
            
            # Display first frame
            self.display_frame(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
            
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
            height, width, channel = frame_rgb.shape
            max_width, max_height = 640, 480
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                height, width = new_height, new_width
            
            # Convert to QImage and then QPixmap
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Update label
            self.video_label.setPixmap(pixmap)
            
            # Update frame info
            self.frame_label.setText(f"Frame: {frame_number}/{self.total_frames}")
            
            # Update plot cursor
            self.plot_widget.update_frame_indicator(frame_number)
            
    def on_frame_change(self, value):
        """Handle frame slider change"""
        self.current_frame = value
        
        if not self.video_thread.is_playing:  # Only update if not playing
            self.display_frame(value)
            
    def toggle_playback(self):
        """Toggle video playback"""
        if self.video_thread.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
            
    def start_playback(self):
        """Start video playback"""
        self.play_button.setText("Pause")
        self.video_thread.set_frame(self.current_frame)
        self.video_thread.play()
        self.video_thread.start()
        
    def stop_playback(self):
        """Stop video playback"""
        self.play_button.setText("Play")
        self.video_thread.pause()
        if self.video_thread.isRunning():
            self.video_thread.wait()
            
    def on_frame_update(self, frame_number):
        """Handle frame update from video thread"""
        if frame_number < self.total_frames:
            self.current_frame = frame_number
            self.frame_slider.setValue(frame_number)
            self.display_frame(frame_number)
        else:
            self.stop_playback()
            
    def advance_frame(self):
        """Advance one frame (for timer-based playback)"""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.display_frame(self.current_frame)
        else:
            self.playback_timer.stop()
            self.play_button.setText("Play")
            
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
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.video_cap:
            self.video_cap.release()
        if self.video_thread.isRunning():
            self.video_thread.pause()
            self.video_thread.wait()
        event.accept()


def main():
    """Main function to run the PyQt6 GUI"""
    app = QApplication(sys.argv)
    app.setApplicationName("Pupil GUI Viewer")
    app.setApplicationVersion("2.0")
    
    window = PupilGUIViewerQt()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
