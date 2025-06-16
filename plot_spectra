"""
compare_avi_spatial_power_spectra.py

Compute and plot the radial spatial power spectrum of each .avi file
in the specified movie directory for texture/edge comparison.
"""

import glob
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

VIDEO_DIR = r"C:\2P\Experiment Types\Natural Movies Experiment\movies"

def radial_profile(data: np.ndarray) -> np.ndarray:
    """
    Compute the radial average of a 2D array around its center.
    """
    y, x = np.indices(data.shape)
    center = np.array([(x.max() - x.min()) / 2.0,
                       (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])
    r_int = r.astype(np.int32)
    tbin = np.bincount(r_int.ravel(), data.ravel())
    nr = np.bincount(r_int.ravel())
    return tbin / np.maximum(nr, 1)


def process_video_spatial(path: str):
    """
    Load a video, compute its average 2D power spectrum, then radial profile.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path!r}")

    sum_power = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fshift = np.fft.fftshift(np.fft.fft2(gray))
        power = np.abs(fshift) ** 2

        if sum_power is None:
            sum_power = np.zeros_like(power)
        sum_power += power
        frame_count += 1

    cap.release()
    if frame_count == 0:
        raise ValueError(f"No frames found in {path!r}")

    avg_power = sum_power / frame_count
    radial_prof = radial_profile(avg_power)

    # Spatial frequency bins (cycles/pixel), max approx 0.5 at Nyquist
    h, w = avg_power.shape
    N = min(h, w)
    freqs = np.arange(len(radial_prof)) / N

    return freqs, radial_prof


def main():
    # Build full list of .avi paths
    pattern = os.path.join(VIDEO_DIR, "*.avi")
    video_files = sorted(glob.glob(pattern))
    if not video_files:
        raise FileNotFoundError(f"No .avi files found in {VIDEO_DIR!r}")

    plt.figure(figsize=(8, 6))
    for vid_path in video_files:
        freqs, prof = process_video_spatial(vid_path)
        plt.loglog(freqs[1:], prof[1:], label=os.path.basename(vid_path), linewidth=1)

    plt.xlabel("Spatial frequency (cycles/pixel)", fontsize=12)
    plt.ylabel("Power", fontsize=12)
    plt.title("Radial Spatial Power Spectra Comparison", fontsize=14)
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
