"""
video_power_spectra.py

Compute and plot BOTH power spectra for every .avi file in a directory:

  * Spatial  : FFT each frame over (x, y) -> |.|^2 -> average over time t,
               then azimuthally (radially) average -> P(spatial frequency)

  * Temporal : FFT each pixel's intensity over time t -> |.|^2,
               then average over all pixels -> P(temporal frequency)

Both are 1-D *marginals* of the full 3-D power spectrum P(kx, ky, ft):
the spatial curve sums it over temporal frequency, the temporal curve
sums it over spatial frequency. (You do NOT hold one axis at a fixed
value -- you average/collapse over it.)

This script ALSO builds the full 2-D spatiotemporal power spectrum
R(ws, wt) and plots 1-D *slices* of it -- holding the opposite axis
fixed at several values rather than averaging it away. This reproduces
the style of Fig. 16.4 (Hyvarinen et al., "Natural Image Statistics"):

  * Panel a : ws on the x-axis, one curve per fixed temporal freq wt
  * Panel b : wt on the x-axis, one curve per fixed spatial  freq ws

Each video gets its own figure with the two slice panels side by side.
"""

import argparse
import glob
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VIDEO_DIR = r"D:\Projects\ACUTEVIS\movies"
# Figures are written here (next to this script) regardless of the cwd.
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def radial_profile(data: np.ndarray) -> np.ndarray:
    """Azimuthal (radial) average of a 2-D array about its center."""
    y, x = np.indices(data.shape)
    cx = (x.max() + x.min()) / 2.0
    cy = (y.max() + y.min()) / 2.0
    r = np.hypot(x - cx, y - cy).astype(np.int32)
    tbin = np.bincount(r.ravel(), weights=data.ravel())  # sum of power per radius
    nr = np.bincount(r.ravel())                           # pixel count per radius
    return tbin / np.maximum(nr, 1)                       # mean power per radius


# ---------------------------------------------------------------------------
# Spatial power spectrum  ->  marginalizes out temporal frequency
# ---------------------------------------------------------------------------
def spatial_power_spectrum(path: str, max_frames=None, apply_window=True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path!r}")

    sum_power = None
    win = None
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if apply_window:                       # Hann window kills the edge-wrap "+" artifact
            if win is None:
                win = np.outer(np.hanning(gray.shape[0]),
                               np.hanning(gray.shape[1])).astype(np.float32)
            gray = gray * win

        f = np.fft.fftshift(np.fft.fft2(gray))
        power = np.abs(f) ** 2

        if sum_power is None:
            sum_power = np.zeros_like(power)
        sum_power += power
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    cap.release()

    if n == 0:
        raise ValueError(f"No frames found in {path!r}")

    avg_power = sum_power / n                   # divide by the REAL frame count
    prof = radial_profile(avg_power)
    N = min(avg_power.shape)
    freqs = np.arange(len(prof)) / N           # cycles/pixel, Nyquist = 0.5
    return freqs, prof


# ---------------------------------------------------------------------------
# Temporal power spectrum  ->  marginalizes out spatial frequency
# ---------------------------------------------------------------------------
def temporal_power_spectrum(path: str, max_frames=256, spatial_downsample=4,
                            apply_window=True, remove_mean=False):
    """
    spatial_downsample : take every Nth pixel in x and y to keep the frame
                         stack in memory. Harmless for the temporal axis --
                         temporal resolution depends only on (#frames, fps).
    remove_mean        : subtract each pixel's time-average so the static
                         background does not dominate the f_t = 0 bin.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if spatial_downsample > 1:
            gray = gray[::spatial_downsample, ::spatial_downsample]
        frames.append(gray)
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    cap.release()

    if n < 2:
        raise ValueError("Need at least 2 frames for a temporal spectrum.")

    stack = np.stack(frames, axis=0)           # shape (T, H, W)
    T = stack.shape[0]

    if remove_mean:
        stack -= stack.mean(axis=0, keepdims=True)
    if apply_window:                           # Hann along time reduces leakage
        stack *= np.hanning(T).astype(np.float32)[:, None, None]

    F = np.fft.rfft(stack, axis=0)             # FFT along time only (real signal -> rfft)
    power = np.abs(F) ** 2                      # (T//2 + 1, H, W)
    psd = power.mean(axis=(1, 2))              # average over all pixels -> P(f_t)
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)    # Hz  (use d=1.0 for cycles/frame)
    return freqs, psd


# ---------------------------------------------------------------------------
# Full 2-D spatiotemporal power spectrum R(ws, wt)  ->  slice, don't average
# ---------------------------------------------------------------------------
def spatiotemporal_power_spectrum(path: str, max_frames=256,
                                  spatial_downsample=4, apply_window=True,
                                  remove_mean=True):
    """
    Build the average 2-D spatiotemporal power spectrum R(ws, wt).

    Pipeline:
      1. stack frames                          -> (T, H, W)
      2. FFT over (y, x) and over t            -> P(kx, ky, ft)
      3. for EACH temporal-frequency bin ft,
         radially (azimuthally) average the
         spatial plane                         -> R(ws, ft)

    Returns
    -------
    s_freq : (Nr,)  spatial frequency  (cycles / analyzed-pixel, Nyquist 0.5)
    t_freq : (Nt,)  temporal frequency (Hz)
    R      : (Nt, Nr)  R[t_index, s_index] -- the 2-D spectrum to slice

    Notes
    -----
    spatial_downsample rescales the pixel unit: ws is in cycles per
    *downsampled* pixel, so its Nyquist is still 0.5 c/p but one "pixel"
    now spans `spatial_downsample` original pixels.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if spatial_downsample > 1:
            gray = gray[::spatial_downsample, ::spatial_downsample]
        frames.append(gray)
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    cap.release()

    if n < 2:
        raise ValueError("Need at least 2 frames for a spatiotemporal spectrum.")

    stack = np.stack(frames, axis=0)           # (T, H, W)
    T, H, W = stack.shape

    if remove_mean:                            # drop the static background (ws=wt=0)
        stack -= stack.mean(axis=0, keepdims=True)
    if apply_window:                           # separable Hann over t, y, x
        stack *= np.hanning(T).astype(np.float32)[:, None, None]
        spatial_win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
        stack *= spatial_win[None, :, :]

    # temporal FFT first (real signal -> rfft), then spatial FFT (complex ok).
    # rfft requires REAL input, so it must precede the complex-valued fft2.
    temporal_f = np.fft.rfft(stack, axis=0)    # (Nt, H, W) complex
    full_f = np.fft.fftshift(np.fft.fft2(temporal_f, axes=(1, 2)), axes=(1, 2))
    power = np.abs(full_f) ** 2                 # P(ws-plane, ft)

    Nt = power.shape[0]
    R = np.stack([radial_profile(power[k]) for k in range(Nt)], axis=0)  # (Nt, Nr)

    Nr = R.shape[1]
    N = min(H, W)
    s_freq = np.arange(Nr) / N                  # cycles/pixel, Nyquist = 0.5
    t_freq = np.fft.rfftfreq(T, d=1.0 / fps)    # Hz
    return s_freq, t_freq, R


def pick_slice_indices(n_total: int, n_slices: int) -> np.ndarray:
    """Evenly spaced (and unique) indices spanning [0, n_total - 1]."""
    n_slices = max(1, min(n_slices, n_total))
    return np.unique(np.linspace(0, n_total - 1, n_slices).astype(int))


def best_separable_approx(R: np.ndarray) -> np.ndarray:
    """
    Best space-time *separable* approximation of R(ws, wt), i.e. the
    R_s(ws) * R_t(wt) that minimizes the least-mean-square distance to R.

    The minimum-Frobenius-distance rank-1 factorization is the leading term
    of the SVD:  R ~= s1 * u1 (x) v1.  R is a non-negative power matrix, so
    by Perron-Frobenius the leading singular vectors are sign-definite; we
    flip their sign to make the returned spectrum positive.

    (This is exactly the construction behind Fig. 16.5 -- "best" is defined
    by minimal least mean square distance.)
    """
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    u, v = U[:, 0], Vt[0, :]
    if u.sum() < 0:                            # make both factors positive
        u, v = -u, -v
    return S[0] * np.outer(u, v)               # same shape as R: (Nt, Nr)


# ---------------------------------------------------------------------------
def find_avi_files(directory: str) -> list:
    """Return a sorted list of .avi files in `directory` (case-insensitive)."""
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory!r}")
    files = glob.glob(os.path.join(directory, "*.avi"))
    files += glob.glob(os.path.join(directory, "*.AVI"))
    return sorted(set(files))


def _panel_letter(ax, letter: str):
    """Bold panel label (a / b) at the top-left, as in Figs. 16.4-16.5."""
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="right")


def plot_slices_figure(s_freq, t_freq, R, title, n_slices=5):
    """Fig. 16.4 style: 1-D slices of R(ws, wt), one colored curve per fixed
    value of the held axis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel a: hold wt fixed at several values, sweep ws
    for ti in pick_slice_indices(len(t_freq), n_slices):
        ax1.loglog(s_freq[1:], R[ti, 1:], linewidth=1,
                   label=f"$\\omega_t$ = {t_freq[ti]:.2g} Hz")
    ax1.set_xlabel("$\\omega_s$ (c/p)")
    ax1.set_ylabel("$R(\\omega_s, \\omega_t)$")
    ax1.set_title("Spatial slices ($\\omega_t$ held constant)")
    ax1.legend(fontsize=8)
    _panel_letter(ax1, "a")

    # Panel b: hold ws fixed at several values, sweep wt
    for si in pick_slice_indices(len(s_freq), n_slices):
        ax2.loglog(t_freq[1:], R[1:, si], linewidth=1,
                   label=f"$\\omega_s$ = {s_freq[si]:.2g} c/p")
    ax2.set_xlabel("$\\omega_t$ (Hz)")
    ax2.set_ylabel("$R(\\omega_s, \\omega_t)$")
    ax2.set_title("Temporal slices ($\\omega_s$ held constant)")
    ax2.legend(fontsize=8)
    _panel_letter(ax2, "b")

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_separable_figure(s_freq, t_freq, R, approx, title, n_slices=5):
    """
    Fig. 16.5 style: the same slices, overlaying the observed R(ws, wt)
    (open circles) with the best separable R_s(ws) R_t(wt) (filled stars).
    All curves share one of two styles, so the legend has just two entries.
    """
    obs_kw = dict(color="k", linewidth=0.7, marker="o", markersize=3,
                  markerfacecolor="none", markeredgewidth=0.6)
    sep_kw = dict(color="k", linewidth=0.7, marker="*", markersize=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel a: ws on x-axis, one observed + one separable curve per fixed wt
    for ti in pick_slice_indices(len(t_freq), n_slices):
        ax1.loglog(s_freq[1:], R[ti, 1:], **obs_kw)
        ax1.loglog(s_freq[1:], approx[ti, 1:], **sep_kw)
    ax1.set_xlabel("$\\omega_s$ (c/p)")
    ax1.set_ylabel("$R(\\omega_s, \\omega_t)$")
    _panel_letter(ax1, "a")

    # Panel b: wt on x-axis, one observed + one separable curve per fixed ws
    for si in pick_slice_indices(len(s_freq), n_slices):
        ax2.loglog(t_freq[1:], R[1:, si], **obs_kw)
        ax2.loglog(t_freq[1:], approx[1:, si], **sep_kw)
    ax2.set_xlabel("$\\omega_t$ (Hz)")
    ax2.set_ylabel("$R(\\omega_s, \\omega_t)$")
    _panel_letter(ax2, "b")

    legend_handles = [
        Line2D([], [], **obs_kw, label="observed $R(\\omega_s, \\omega_t)$"),
        Line2D([], [], **sep_kw,
               label="best separable $R_s(\\omega_s) R_t(\\omega_t)$"),
    ]
    for ax in (ax1, ax2):
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_video_spectra(path: str, n_slices: int = 5):
    """
    Build the 2-D spatiotemporal spectrum once, then return two figures:

      fig_slices    -- Fig. 16.4 style (colored 1-D slices)
      fig_separable -- Fig. 16.5 style (observed vs. best-separable overlay)
    """
    s_freq, t_freq, R = spatiotemporal_power_spectrum(path)

    # restrict the spatial axis to the valid band (DC .. Nyquist = 0.5 c/p);
    # radial bins beyond Nyquist are sparse, noisy frame-corner frequencies.
    s_valid = s_freq <= 0.5
    s_freq = s_freq[s_valid]
    R = R[:, s_valid]

    approx = best_separable_approx(R)

    title = os.path.basename(path)
    fig_slices = plot_slices_figure(s_freq, t_freq, R, title, n_slices)
    fig_separable = plot_separable_figure(s_freq, t_freq, R, approx, title,
                                          n_slices)
    return fig_slices, fig_separable


def main():
    parser = argparse.ArgumentParser(
        description="Plot spatial and temporal power spectra for every "
                    ".avi file in a directory.")
    parser.add_argument("directory", nargs="?", default=VIDEO_DIR,
                        help="Directory containing .avi files "
                             f"(default: {VIDEO_DIR!r})")
    parser.add_argument("--figdir", default=FIG_DIR,
                        help=f"Where to save PNGs (default: {FIG_DIR!r})")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without opening interactive windows.")
    args = parser.parse_args()

    avi_files = find_avi_files(args.directory)
    if not avi_files:
        raise FileNotFoundError(f"No .avi files found in: {args.directory!r}")

    os.makedirs(args.figdir, exist_ok=True)

    for path in avi_files:
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing {os.path.basename(path)} ...")
        try:
            fig_slices, fig_separable = plot_video_spectra(path)
        except (IOError, ValueError) as exc:
            print(f"  Skipped {os.path.basename(path)}: {exc}")
            continue

        for suffix, fig in (("slices", fig_slices), ("separable", fig_separable)):
            out = os.path.join(args.figdir, f"{base}_{suffix}.png")
            fig.savefig(out, dpi=150)
            print(f"  saved {out}")
            if args.no_show:
                plt.close(fig)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()