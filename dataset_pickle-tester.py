"""
Created on Mon Nov 24 14:11:01 2025

@author: sindhuja s. baskar
"""
#%%
import numpy as np
import pandas as pd

dataset = pd.read_pickle(r"E:\Projects\ACUTEVIS\dataset.pickle")

print(dataset.head())

keys = pd.DataFrame(dataset.keys())

# data = pd.read_hdf(r"E:\Projects\ACUTEVIS\dataset.h5")
# %%



#%%
def find_missing_injection_columns(dataset, injection_key: str = "injection") -> pd.DataFrame:
    """Inspect session configuration payloads for missing injection metadata."""
    session_config = getattr(dataset, "session_config", None)
    if session_config is None:
        raise AttributeError("Dataset is missing 'session_config' attribute")

    raw_output = getattr(session_config, "raw_output", None)
    if raw_output is None:
        raise AttributeError("Dataset session_config is missing 'raw_output'")

    missing: list[tuple[str, str, str]] = []

    def _iter_subjects(obj) -> list[tuple[str, object]]:
        if isinstance(obj, pd.DataFrame):
            return [(str(col), obj[col]) for col in obj.columns]
        if isinstance(obj, pd.Series):
            label = str(obj.name) if obj.name is not None else "unknown_subject"
            return [(label, obj)]
        if isinstance(obj, Mapping):
            return [(str(k), v) for k, v in obj.items()]
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            return [(str(k), v) for k, v in to_dict().items()]
        return []

    def _iter_items(obj):
        if isinstance(obj, pd.DataFrame):
            return list(obj.items())
        if isinstance(obj, pd.Series):
            return list(obj.items())
        if isinstance(obj, Mapping):
            return list(obj.items())
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            return list(to_dict().items())
        return []

    def _is_missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, float):
            return bool(np.isnan(value))
        if isinstance(value, (np.floating, np.integer)):
            return bool(np.isnan(value))  # type: ignore[arg-type]
        return False

    for subject, subject_payload in _iter_subjects(raw_output):
        for session, task_payload in _iter_items(subject_payload):
            if _is_missing(task_payload):
                continue

            for task, payload in _iter_items(task_payload):
                if _is_missing(payload):
                    missing.append((subject, str(session), str(task)))
                    continue

                has_injection = False
                if isinstance(payload, pd.Series):
                    if injection_key in payload:
                        value = payload[injection_key]
                        has_injection = not _is_missing(value)
                # elif isinstance(payload, Mapping):
                #     value = payload.get(injection_key)
                #     has_injection = not _is_missing(value)
                else:
                    to_dict_payload = getattr(payload, "to_dict", None)
                    if callable(to_dict_payload):
                        value = to_dict_payload().get(injection_key)
                        has_injection = not _is_missing(value)
                    else:
                        try:
                            has_injection = injection_key in payload  # type: ignore[operator]
                        except TypeError:
                            has_injection = False

                if not has_injection:
                    missing.append((subject, str(session), str(task)))

    return pd.DataFrame(missing, columns=["subject", "session", "task"])


missing_injection = find_missing_injection_columns(dataset)

# %% PLOTTING FUNCTIONS
import numpy as np
import matplotlib.pyplot as plt

def compute_peak_session_stats(database, session_labels=None):
    """
    Compute per-session peak count statistics (mean, std, count, sem, label) and plot
    mean bars with individual subject trajectories.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # Get peak counts with MultiIndex
    peak_counts = database[('suite2p', 'num_peaks_prominence')]
    # Convert to DataFrame for easier grouping
    peak_df = peak_counts.reset_index()
    peak_df.columns = ['Subject', 'Session', 'task','num_peaks']
    # Group by session and compute mean and SEM
    session_stats = peak_df.groupby('Session')['num_peaks'].agg(['mean', 'std', 'count']).reset_index()
    session_stats['sem'] = session_stats['std'] / np.sqrt(session_stats['count'])
    # Apply default session labels if none provided
    if session_labels is None:
        session_labels = {
            'ses-01': 'Baseline',
            'ses-02': 'Saline',
            'ses-03': 'High EtOH',
            'ses-04': 'Low EtOH'
        }
    session_stats['label'] = session_stats['Session'].map(session_labels)
    # Set up high-quality plotting parameters
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.size': 6,
        'ytick.major.width': 1.5,
        'legend.frameon': False
    })
    # Render mean peak counts per session and overlay per-subject trends
    labels = session_stats['label'].fillna(session_stats['Session'])
    x_positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x_positions, session_stats['mean'], color="tab:blue", alpha=0.6, zorder=1)

    # Overlay individual subject trajectories across sessions
    session_order = session_stats['Session'].tolist()
    for subject, subject_df in peak_df.groupby('Subject'):
        # Collapse multiple tasks within a session to a single mean value per session
        subject_session_means = subject_df.groupby('Session')['num_peaks'].mean()
        subject_values = subject_session_means.reindex(session_order).to_numpy(dtype=np.float64)
        if np.isfinite(subject_values).any():
            ax.plot(
                x_positions,
                subject_values,
                marker='o',
                linewidth=1.5,
                alpha=0.7,
                label=str(subject),
                zorder=2,
            )

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles,
            legend_labels,
            title='Subject ID',
            loc='upper right',
            borderaxespad=0.5,
            frameon=True,
        )
        legend.get_title().set_fontsize(8)
        for text in legend.get_texts():
            text.set_fontsize(7)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Session')
    ax.set_ylabel('Mean Peaks')
    ax.set_title('Mean Peak Counts by Session')
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    return session_stats, fig

test_stats, test_fig = compute_peak_session_stats(dataset)



