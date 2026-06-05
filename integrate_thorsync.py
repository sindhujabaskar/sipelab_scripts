"""
Faster load_sync_episode — drop-in replacement for the one in thorsync.py.

Two changes from the original port, both targeting I/O:
  1. `channels=` reads ONLY the named DI/AI channels (the /Global counter is
     always read because `time` depends on it). Skips the analog channels you
     never use.
  2. Windowing slices the dataset OBJECT (`dset[s:stop]`) so h5py reads only
     the requested bytes, instead of reading everything and slicing after.
     The stride (`interval`) is applied in-memory afterward — a contiguous
     read + numpy subsample is usually faster wall-clock than a strided HDF5
     read (see note in the answer). Pass stride_on_disk=True to flip that
     trade if you are memory-bound rather than time-bound.
"""

from __future__ import annotations

import numpy as np
import h5py
import pandas as pd

CLOCK_RATE_HZ = 20_000_000


def _unique_name(name, existing):
    if name not in existing:
        return name
    i = 1
    while f"{name}{i:03d}" in existing:
        i += 1
    return f"{name}{i:03d}"


def load_sync_episode(h5_path, sample_rate=None, start=None, length=None,
                      interval=None, channels=None, stride_on_disk=False,
                      clock_rate_hz=CLOCK_RATE_HZ):
    out: dict[str, np.ndarray] = {}
    want = {c.replace(" ", "_") for c in channels} if channels else None

    windowed = (start is not None) or (length is not None)
    if (windowed or interval) and sample_rate is None:
        raise ValueError("sample_rate required when using start/length/interval")

    s0 = int(round((start or 0) * sample_rate)) if sample_rate else 0
    n = (int(round(length * sample_rate))
         if (length is not None and np.isfinite(length)) else None)
    step = (max(1, int(round(interval * sample_rate)))
            if interval else 1)
    stop = (s0 + n) if n is not None else None

    with h5py.File(h5_path, "r") as f:
        for gname, group in f.items():
            if not isinstance(group, h5py.Group):
                continue
            g = gname.strip("/")
            is_global = g.lower() == "global"

            for dname, dset in group.items():
                if not isinstance(dset, h5py.Dataset):
                    continue

                key = dname.replace(" ", "_")
                # Channel filter: /Global is always needed for `time`.
                if want is not None and not is_global and key not in want:
                    continue

                # Read only what we need. ThorSync stores (1,N) or (N,1).
                if windowed and stride_on_disk:
                    arr = dset[..., s0:stop:step] if dset.ndim > 1 else dset[s0:stop:step]
                    arr = np.asarray(arr).reshape(-1)
                elif windowed:
                    arr = dset[..., s0:stop] if dset.ndim > 1 else dset[s0:stop]
                    arr = np.asarray(arr).reshape(-1)[::step]
                else:
                    arr = np.asarray(dset[()]).reshape(-1)[::step]

                if g.upper() == "AI":
                    arr = (arr > 0).astype(np.uint8)
                if is_global:
                    arr = arr.astype(np.float64) / clock_rate_hz
                    key = "time"

                out[_unique_name(key, out)] = arr

    return out


if __name__ == "__main__":
    import sys, time
    from pathlib import Path
    import matplotlib.pyplot as plt

    h5 = r"G:\Resources\ThorSync\Episode_0000.h5"
    t0 = time.perf_counter()
    full = load_sync_episode(h5, sample_rate=30000)
    t1 = time.perf_counter()
    subset = load_sync_episode(h5, sample_rate=30000,
                               channels=["FrameOut", "Treadmill_encoder_ai2"])
    t2 = time.perf_counter()
    print(f"all channels : {t1 - t0:5.2f}s  ({len(full)} arrays)")
    print(f"3 channels   : {t2 - t1:5.2f}s  ({len(subset)} arrays)")
    df = pd.DataFrame(subset)

    plt.subplot(211)
    plt.plot(df["time"], df["FrameOut"], label="FrameOut")

    plt.subplot(212)
    plt.plot(df["time"], df["Treadmill_encoder_ai2"], label="Treadmill_encoder_ai2")


    plt.xlim(0, 100)
    plt.xlabel("time (s)")
    plt.show()