from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
# ─── Detection parameters ────────────────────────────────────────────────

# Savitzky–Golay smoothing
SG_WINDOW = 11
SG_POLYORDER = 3

# Hysteresis thresholds (percentiles of the smoothed trace)
HIGH_PERCENTILE = 80.0
LOW_PERCENTILE = 60.0

# Event filtering
MAX_GAP_S = 1      # fill gaps shorter than this (seconds)
MIN_DURATION_S = 2  # discard events shorter than this (seconds)


# ─── Artifact rejection ──────────────────────────────────────────────────


def reject_artifacts(
    trace: np.ndarray,
    t_s: np.ndarray,
    k: float = 8.0,
    pad: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect and interpolate across physiologically implausible transients.

    Uses the median absolute deviation (MAD) of the frame-to-frame
    derivative to set an adaptive threshold.  Frames where |Δx| > k·MAD
    are flagged, expanded by ``pad`` frames on each side, and linearly
    interpolated over.

    Why MAD instead of std?  The outliers we want to catch (blinks,
    tracking dropouts) are themselves heavy-tailed — they inflate σ,
    making a std-based cutoff too permissive.  MAD ignores them.

    Parameters
    ----------
    trace : array
        Raw signal values.
    t_s : array
        Timestamps (same length as trace).
    k : float
        Multiplier on the MAD-derived scale (σ̂ = 1.4826 · MAD).
        Higher = more permissive.  6–8 is a reasonable starting range
        for pupil; 8–10 for smoother signals like widefield ΔF/F.
    pad : int
        Frames to expand each flagged region on both sides.
        Catches the ramps into/out of an artifact that individually
        may not exceed the derivative threshold.

    Returns
    -------
    cleaned : array
        Trace with artifacts replaced by linear interpolation.
    artifact_mask : boolean array
        True where artifacts were detected (before interpolation).
    """
    if len(trace) < 3:
        return trace.copy(), np.zeros(len(trace), dtype=bool)

    # ── Frame-to-frame derivative ──
    dx = np.diff(trace, prepend=trace[0])

    # ── Adaptive threshold via MAD ──
    med_dx = np.median(dx)
    mad = np.median(np.abs(dx - med_dx))

    # 1.4826 is the consistency constant: MAD → σ under normality.
    # Not that the derivative is necessarily Gaussian, but it gives
    # a principled scale factor without assuming it.
    sigma_est = 1.4826 * mad if mad > 0 else np.std(dx)

    threshold = k * sigma_est
    if threshold <= 0:
        return trace.copy(), np.zeros(len(trace), dtype=bool)

    # ── Flag outlier frames ──
    flagged = np.abs(dx - med_dx) > threshold

    # ── Expand flagged regions by `pad` frames ──
    if pad > 0 and flagged.any():
        expanded = flagged.copy()
        for shift in range(1, pad + 1):
            expanded[shift:] |= flagged[:-shift]
            expanded[:-shift] |= flagged[shift:]
        flagged = expanded

    artifact_mask = flagged.copy()

    # ── Interpolate across flagged regions ──
    cleaned = trace.copy()
    if flagged.any():
        good = np.where(~flagged)[0]
        if len(good) >= 2:
            bad = np.where(flagged)[0]
            cleaned[bad] = np.interp(bad, good, cleaned[good])
        # If nearly everything is flagged, fall back to original —
        # better noisy data than a flat interpolated line.

    return cleaned, artifact_mask


# ─── Detrending ───────────────────────────────────────────────────────────


def detrend_rolling_baseline(
    trace: np.ndarray,
    t_s: np.ndarray,
    window_s: float = 120.0,
    quantile: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove slow drift via rolling-quantile subtraction.

    Parameters
    ----------
    trace : array
        Signal to detrend (ideally already artifact-cleaned).
    t_s : array
        Timestamps (used to compute window size in frames).
    window_s : float
        Window duration in seconds.  Should be several times longer
        than the longest event you expect.
    quantile : float
        Quantile for the rolling window.  0.5 = median (tracks the
        center).  Lower values (e.g. 0.1) track the floor of the
        signal, preventing transient peaks from inflating the
        baseline and producing artificial negative dips.

    Returns
    -------
    detrended : array
        trace − baseline.
    baseline : array
        The rolling-quantile estimate (for plotting / diagnostics).
    """
    if len(trace) < 3:
        return trace.copy(), np.zeros_like(trace)

    dt = np.median(np.diff(t_s))
    if not np.isfinite(dt) or dt <= 0:
        return trace.copy(), np.zeros_like(trace)

    window_frames = int(window_s / dt)
    if window_frames % 2 == 0:
        window_frames += 1
    window_frames = max(window_frames, 3)

    # pandas rolling median wraps a C-level implementation —
    # fast enough for traces up to ~10⁶ samples.
    baseline = pd.Series(trace).rolling(
        window=window_frames, center=True, min_periods=1
    ).quantile(quantile).values

    detrended = trace - baseline

    return detrended, baseline


# ─── Event detection functions ────────────────────────────────────────────


def smooth_trace(trace: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky–Golay filter; handles NaN/Inf and short traces."""
    clean = trace.copy()
    bad = ~np.isfinite(clean)
    if bad.all():
        return clean
    if bad.any():
        good_idx = np.where(~bad)[0]
        clean[bad] = np.interp(np.where(bad)[0], good_idx, clean[good_idx])

    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(clean):
        window_length = len(clean) if len(clean) % 2 == 1 else len(clean) - 1
    if window_length < polyorder + 2:
        return clean
    return savgol_filter(clean, window_length, polyorder)


def compute_hysteresis_thresholds(
    trace: np.ndarray, high_percentile: float, low_percentile: float
) -> tuple[float, float]:
    """Percentile-based high/low thresholds for hysteresis gating."""
    return float(np.percentile(trace, high_percentile)), float(np.percentile(trace, low_percentile))


def apply_hysteresis_mask(trace: np.ndarray, high_th: float, low_th: float) -> np.ndarray:
    """Boolean mask: enters event when trace ≥ high_th, exits when ≤ low_th."""
    mask = np.zeros(len(trace), dtype=bool)
    in_event = False
    start = 0
    for i, v in enumerate(trace):
        if not in_event and v >= high_th:
            in_event = True
            start = i
        elif in_event and v <= low_th:
            mask[start:i] = True
            in_event = False
    if in_event:
        mask[start:] = True
    return mask


def fill_short_gaps(mask: np.ndarray, t_s: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Close gaps in the boolean mask shorter than max_gap_s."""
    if len(t_s) < 2:
        return mask
    dt = np.median(np.diff(t_s))
    if not np.isfinite(dt) or dt <= 0:
        return mask
    max_gap_frames = int(max_gap_s / dt)
    inv = ~mask
    edges = np.diff(inv.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    filled = mask.copy()
    for s, e in zip(starts, ends):
        if (e - s) <= max_gap_frames:
            filled[s:e] = True
    return filled


def extract_events(
    mask: np.ndarray, t_s: np.ndarray, min_duration_s: float
) -> tuple[list[tuple[int, int]], list[float]]:
    """Extract (start_idx, end_idx) pairs and durations from the final mask."""
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    events: list[tuple[int, int]] = []
    durations_s: list[float] = []
    for s, e in zip(starts, ends):
        dur = t_s[min(e, len(t_s) - 1)] - t_s[s]
        if dur >= min_duration_s:
            events.append((s, e))
            durations_s.append(float(dur))
    return events, durations_s


def detect_events(
    trace: np.ndarray,
    t_s: np.ndarray,
    *,
    sg_window: int = SG_WINDOW,
    sg_poly: int = SG_POLYORDER,
    high_pct: float = HIGH_PERCENTILE,
    low_pct: float = LOW_PERCENTILE,
    max_gap_s: float = MAX_GAP_S,
    min_dur_s: float = MIN_DURATION_S,
    # ── Gaussian pre-smoothing ──
    gauss_sigma: float = 75,
    # ── Artifact rejection ──
    artifact_k: float = 8.0,
    artifact_pad: int = 3,
    # ── Detrending ──
    do_detrend: bool = False,
    detrend_window_s: float = 120.0,
    detrend_quantile: float = 0.5,
) -> dict:
    """
    Full pipeline:
        artifact reject → detrend → smooth → threshold →
        hysteresis → gap-fill → extract.

    Returns a dict with all intermediate results for diagnostics:
      artifact_mask   – boolean, True at rejected frames
      cleaned         – trace after artifact interpolation
      baseline        – rolling-median baseline (zeros if detrend=False)
      detrended       – trace after baseline subtraction
      working_trace   – the trace that was actually thresholded
      smoothed, high_th, low_th, mask, events, durations_s – as before
    """
    # ── 1. Artifact rejection ──
    cleaned, artifact_mask = reject_artifacts(
        trace, t_s, k=artifact_k, pad=artifact_pad
    )

    # ── 2. Detrending ──
    if do_detrend:
        detrended, baseline = detrend_rolling_baseline(
            cleaned, t_s, window_s=detrend_window_s, quantile=detrend_quantile
        )
    else:
        detrended = cleaned
        baseline = np.zeros_like(cleaned)

    # ── 3. Gaussian smoothing ──
    # Apply to both the detrended trace (for detection) and the cleaned
    # trace (for amplitude measurement) so smoothing is consistent.
    if gauss_sigma > 0:
        detrended = gaussian_filter1d(detrended, sigma=gauss_sigma)
        cleaned_smooth = gaussian_filter1d(cleaned, sigma=gauss_sigma)
    else:
        cleaned_smooth = cleaned.copy()

    # ── 4. Smooth → threshold → hysteresis → filter ──
    smoothed = smooth_trace(detrended, sg_window, sg_poly)
    high_th, low_th = compute_hysteresis_thresholds(smoothed, high_pct, low_pct)
    raw_mask = apply_hysteresis_mask(smoothed, high_th, low_th)
    filled_mask = fill_short_gaps(raw_mask, t_s, max_gap_s)
    events, durations_s = extract_events(filled_mask, t_s, min_dur_s)

    # ── 5. Peak detection: timestamp of maximum within each event (smoothed trace) ──
    peak_times_s: list[float] = []
    for event_start, event_end in events:
        e_idx = min(event_end, len(smoothed) - 1)
        segment = smoothed[event_start:e_idx]
        if len(segment) > 0:
            peak_offset = int(np.argmax(segment))
            peak_times_s.append(float(t_s[event_start + peak_offset]))
        else:
            peak_times_s.append(float(t_s[event_start]))

    return {
        # ── Diagnostic intermediates ──
        "artifact_mask": artifact_mask,
        "cleaned": cleaned,
        "cleaned_smooth": cleaned_smooth,  # non-detrended, Gaussian-smoothed
        "baseline": baseline,
        "detrended": detrended,
        "working_trace": detrended,
        # ── Detection outputs ──
        "smoothed": smoothed,
        "high_th": high_th,
        "low_th": low_th,
        "mask": filled_mask,
        "events": events,
        "durations_s": durations_s,
        "peak_times_s": peak_times_s,
    }


def plot_single_session(
    t_s: np.ndarray,
    raw: np.ndarray,
    det: dict,
    title: str,
    spec: SignalSpec,
) -> plt.Figure:
    """
    Diagnostic plot.

    When artifact rejection or detrending is active, a top panel shows
    the original trace with artifact highlights and the rolling-median
    baseline overlay.  The middle panel shows the working (detrended)
    trace with smoothed line, thresholds, and event shading.  Bottom
    panel is the binary event mask.

    When neither feature is active, falls back to the original
    two-panel layout.
    """
    has_artifacts = det["artifact_mask"].any()
    has_detrend = spec.detrend
    show_top = has_artifacts or has_detrend

    if show_top:
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, figsize=(16, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 4, 1], "hspace": 0.08},
            facecolor="white", layout="constrained",
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 5.5), sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
            facecolor="white", layout="constrained",
        )
        ax0 = None

    # ── Top panel: original signal + artifacts + baseline ──
    if ax0 is not None:
        ax0.plot(t_s, raw, color="#AAAAAA", linewidth=0.4,
                 label=f"raw {spec.label}", rasterized=True)
        ax0.plot(t_s, det["cleaned"], color="#1A1A2E", linewidth=0.6,
                 label="artifact-cleaned")

        if has_detrend:
            ax0.plot(t_s, det["baseline"], color="#E76F51", linewidth=1.4,
                     alpha=0.85, label="rolling-median baseline")

        if has_artifacts:
            art = det["artifact_mask"]
            ylo, yhi = raw[np.isfinite(raw)].min(), raw[np.isfinite(raw)].max()
            ax0.fill_between(t_s, ylo, yhi, where=art,
                             color="#FF6B6B", alpha=0.25, lw=0, label="artifacts")
            n_art_events = (np.diff(art.astype(int), prepend=0) == 1).sum()
            ax0.set_title(
                f"Original signal — {n_art_events} artifact region(s) rejected",
                fontsize=9, fontstyle="italic", loc="left", pad=4,
            )

        ax0.set_ylabel(spec.ylabel, fontsize=10)
        ax0.legend(loc="upper right", fontsize=7, framealpha=0.9, edgecolor="0.8")

    # ── Middle panel: working trace + smoothed + thresholds + events ──
    for s, e in det["events"]:
        e_idx = min(e, len(t_s) - 1)
        ax1.axvspan(t_s[s], t_s[e_idx], color=spec.color_event, alpha=0.45, lw=0)

    working = det["working_trace"]
    ylabel_mid = f"{spec.ylabel} (detrended)" if has_detrend else spec.ylabel
    trace_label = f"{'detrended' if has_detrend else 'raw'} {spec.label}"

    ax1.plot(t_s, working, color="#AAAAAA", linewidth=0.35,
             label=trace_label, rasterized=True)
    ax1.plot(t_s, det["smoothed"], color="#1A1A2E", linewidth=1.0, label="smoothed")
    ax1.axhline(det["high_th"], color="#E63946", ls="--", lw=1.0, alpha=0.85,
                label=f"high threshold ({HIGH_PERCENTILE:.0f}th %ile)")
    ax1.axhline(det["low_th"], color="#F4A261", ls="--", lw=1.0, alpha=0.85,
                label=f"low threshold ({LOW_PERCENTILE:.0f}th %ile)")

    # ── Peak markers ──
    if det["peak_times_s"]:
        peak_values = np.interp(det["peak_times_s"], t_s, det["smoothed"])
        ax1.scatter(
            det["peak_times_s"], peak_values,
            marker="v", color="#E63946", s=28, zorder=5,
            label="peaks", edgecolors="white", linewidths=0.5,
        )

    ax1.set_ylabel(ylabel_mid, fontsize=10)
    ax1.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="0.8")

    # ── Bottom panel: event mask ──
    ax2.fill_between(t_s, det["mask"].astype(float), step="mid",
                     color=spec.color_mask, alpha=0.6, lw=0)
    ax2.set_ylabel("Events", fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["off", "on"], fontsize=8)

    all_axes = [ax for ax in [ax0, ax1, ax2] if ax is not None]
    fig.align_ylabels(all_axes)
    return fig

@dataclass
class SignalSpec:
    """Configuration for one signal to run event detection on."""
    source: str             # data source name in dataset
    signal: str             # signal column name within the source
    label: str              # human-readable label for plots / filenames
    ylabel: str             # y-axis label
    use_dff: bool           # True → compute ΔF/F; False → use raw values
    color_event: str        # event shading color
    color_mask: str         # mask fill color
    palette: tuple[str, ...]  # bar chart palette

    # ── Gaussian pre-smoothing ──
    gauss_sigma: float = 75        # Gaussian σ in samples (~1.5 s at 50 Hz)

    # ── Artifact rejection ──
    artifact_k: float = 8.0        # MAD multiplier for derivative outlier detection
    artifact_pad: int = 3          # frames to expand around each flagged artifact

    # ── Detrending ──
    detrend: bool = False          # whether to subtract rolling baseline
    detrend_window_s: float = 120.0  # rolling window in seconds
    detrend_quantile: float = 0.5  # 0.5 = median; lower (e.g. 0.1) tracks the floor

    # ── Normalization ──
    normalize: str = "none"        # "none", "zscore", or "dff"


SIGNALS: list[SignalSpec] = [
    # ── Pupil diameter ──
    SignalSpec(
        source="pupil",
        signal="pupil_diameter_mm",
        label="pupil",
        ylabel="Pupil (z-score)",
        use_dff=False,
        color_event="#E8D5F5",
        color_mask="#7B2D8E",
        palette=("#7B2D8E", "#4A0E5C", "#C39BD3", "#884EA0",
                 "#6C3483", "#D2B4DE", "#A569BD", "#8E44AD"),
        # Pupil drifts substantially — detrend by default
        detrend=True,
        gauss_sigma=50,
        detrend_window_s=120.0,
        artifact_k=5.0,      # pupil blinks are sharp; be more aggressive
        artifact_pad=5,
        normalize="zscore",
    ),]

dataset = pd.read_pickle(r"C:\Projects\ACUTEVIS\260503_ACUTEVIS_dataset.pkl")
pup = dataset.pupil.pupil_diameter_mm.loc['SB18', 'ses-04', 'task-grayscreen']
time = dataset.pupil.time_elapsed_s.loc['SB18', 'ses-04', 'task-grayscreen']
results = detect_events(pup, time)


fig = plot_single_session(time, pup, results, title="SB18 ses-04 pupil detection", spec=SIGNALS[0])
fig.savefig(r"C:\dev\sipelab_scripts\pupil_detection_example.pdf", dpi=300)