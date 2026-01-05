"""
Created on Mon Nov 24 14:11:01 2025

@author: sindhuja s. baskar
"""
#%%
import numpy as np
import pandas as pd

dataset = pd.read_pickle(r'/Users/sindhubaskar/Documents/work/Sipe_Lab/dataset.pickle')

print(dataset.head())

keys = pd.DataFrame(dataset.keys())

# %% CALCULATE MEAN OSI DSI ACROSS SUBJECTS

# create time vector (15Hz) for dff
dataset[('meta','time_vector')] = dataset[('suite2p','deltaf_f')].map(lambda arr : pd.Series(np.arange(arr.shape[1]) / 15.0))

# CREATE AN ARTIFICIAL TIME INDEX FOR EVERY TRIAL
N_TRIALS   = 130
TRIAL_S    = 5.0
GRAY_S     = 3.0               # within each trial
GRATING_S  = 2.0               # (informational; TRIAL_S should equal GRAY_S + GRATING_S)

# 0-based artificial trial indices (length = 130) stored as a Series in the MultiIndex column
dataset[('psychopy', 'trial_index')] = dataset[('suite2p','deltaf_f')].map(
    lambda _ : pd.Series(np.arange(N_TRIALS), dtype="Int64")
)

# Per-row DataFrame of trial stamps (same columns as your psychopy-derived structure)
dataset[('psychopy', 'trial_stamps')] = dataset[('suite2p','deltaf_f')].map(
    lambda _ : pd.DataFrame({
        "trial_num":      np.arange(N_TRIALS, dtype=np.int64),
        "trial_start":    np.arange(N_TRIALS) * TRIAL_S,
        "gray_start":     np.arange(N_TRIALS) * TRIAL_S,
        "gratings_start": np.arange(N_TRIALS) * TRIAL_S + GRAY_S,
        "trial_end":      np.arange(N_TRIALS) * TRIAL_S + TRIAL_S,
    })
)

# create trials dataframe
def trials(row):

    directions = [0, 45, 90, 135, 180, 225, 270, 315]  # the 8 grating angles you cycle through
    orientations = [0, 45, 90, 135]
    block_size   = len(directions)

    # define inputs
    deltaf_f   = np.asarray(row[('suite2p', 'deltaf_f')])  
    timestamps = np.asarray(row[('meta', 'time_vector')])      
    trial_df   = row[('psychopy', 'trial_stamps')]          

    all_trials = []
    for trial_id, (start, grating, stop) in enumerate(zip(trial_df['trial_start'], trial_df['gratings_start'], trial_df['trial_end'])):
        trial_mask    = (timestamps >= start) & (timestamps < stop)
        off_mask      = (timestamps >= start) & (timestamps < grating)
        on_mask       = (timestamps >= grating) & (timestamps < stop)
        time_slice    = timestamps[trial_mask]           # (t,)
        trial_data    = deltaf_f[:, trial_mask]          # (n_rois, t)
        time_off      = timestamps[off_mask]            # (t_off,)
        dff_off       = deltaf_f[:, off_mask]           # (n_rois, t_off)
        time_on       = timestamps[on_mask]             # (t_on,)
        dff_on        = deltaf_f[:, on_mask]            # (n_rois, t_on)
        direction     = directions[trial_id % len(directions)]
        orientation   = orientations[trial_id % len(orientations)]
        trial_block   = trial_id // block_size

        all_trials.append({
            'trial':       trial_id,
            'block':       trial_block,
            'direction':   direction,
            'orientation': orientation,
            'time':        time_slice,
            'dff':         trial_data,
            'time_off':    time_off,
            'dff_off':     dff_off,
            'time_on':     time_on,
            'dff_on':      dff_on
        })

    return pd.DataFrame(all_trials)

dataset[('meta','trials')] = dataset.apply(trials, axis=1)


#%%
## TUNING, OSI/DSI VECTOR
subject = 'ACUTEVIS06'
session = 'ses-01'
task =  'task-gratings'
roi_id = 20  # specify ROI index

trials = dataset.meta.trials.loc[subject, session, task]
dff_on = trials['dff_on'].to_numpy()  # list-like of length n_trials; each: (n_rois, n_time)
dff_off = trials['dff_off'].to_numpy()  # list-like of length n_trials; each: (n_rois, n_time)
dff = trials['dff'].to_numpy()

# calculate per-trial mean dF/F for a ROI
def mean_dff_per_trial(dff_trials, roi_id):
    trial_means = np.array([np.mean(trial[roi_id, :]) for trial in dff_trials])
    orientations = trials['orientation'].to_numpy()
    directions = trials['direction'].to_numpy()
    return pd.DataFrame({'trial_means': trial_means , 'orientations': orientations, 'directions': directions})

on_means = mean_dff_per_trial(dff_on, roi_id)
off_means = mean_dff_per_trial(dff_off, roi_id)

# identify tuned stimuli
def preferred_stimulus(on_means):
    ori = on_means.groupby('orientations')['trial_means']
    dir = on_means.groupby('directions')['trial_means']
    ori_mean = ori.agg(lambda x: x.mean())
    dir_mean = dir.agg(lambda x: x.mean())
    pref_ori = ori_mean.idxmax()
    pref_dir = dir_mean.idxmax()
    return {'ori': {'pref_ori': pref_ori, 'ori_mean': ori_mean}, 'dir': {'pref_dir': pref_dir, 'dir_mean': dir_mean}}

pref_on = pd.DataFrame(preferred_stimulus(on_means))
print(f"Preferred Orientation: {pref_on.ori.pref_ori} Degrees" , pref_on.ori.ori_mean)
print(f"Preferred Direction: {pref_on.dir.pref_dir} Degrees" , pref_on.dir.dir_mean)
# plt.plot(pref_on.dir.dir_mean)
# plt.xticks(pref_on.dir.dir_mean.index)
# plt.show()

def normalize_tuning(tuning_dict):
    ori_mean = tuning_dict['ori']['ori_mean']
    dir_mean = tuning_dict['dir']['dir_mean']
    pref_ori = tuning_dict['ori']['pref_ori']
    pref_dir = tuning_dict['dir']['pref_dir']
    
    # find orthogonal orientation
    orth_ori = (pref_ori + 90) % 180
    norm_ori = (ori_mean[pref_ori] - ori_mean[orth_ori]) / ori_mean[pref_ori] #change to ori_mean[i]/sum(ori_mean)
    osi = np.abs(np.sum(ori_mean*np.exp(2j* pref_ori) / np.sum(ori_mean)))

    # find orthogonal direction
    orth_dir = (pref_dir + 180) % 360
    norm_dir = (dir_mean[pref_dir] - dir_mean[orth_dir]) / dir_mean[pref_dir]
    dsi = np.abs(np.sum(dir_mean*np.exp(1j* pref_dir) / np.sum(dir_mean)))
    
    return {'norm_ori': norm_ori, 'norm_dir': norm_dir, 'osi': osi, 'dsi': dsi}
norm_tuning = normalize_tuning(pref_on)

print(f"Normalized Orientation Tuning: {norm_tuning['norm_ori']}")
print(f"Normalized Direction Tuning: {norm_tuning['norm_dir']}")

def osi_dsi(tuning_dict):
    ori_mean = tuning_dict['ori']['ori_mean']
    dir_mean = tuning_dict['dir']['dir_mean']

    # OSI calculation
    angle_ori = np.deg2rad(ori_mean.index)
    tc_ori = ori_mean.values
    osi = (np.abs(np.sum(tc_ori * np.exp(2j * angle_ori)) / np.sum(tc_ori)))

    # DSI calculation
    angle_dir = np.deg2rad(dir_mean.index)
    tc_dir = dir_mean.values
    dsi = (np.abs(np.sum(tc_dir * np.exp(1j * angle_dir)) / np.sum(tc_dir)))

    return {'osi': osi, 'dsi': dsi}
osi_dsi_values = osi_dsi(pref_on)
print(f"Orientation Selectivity Index (OSI): {osi_dsi_values['osi']}")
print(f"Direction Selectivity Index (DSI): {osi_dsi_values['dsi']}")

#%% create dataframe of osi dsi across subject, session, task, roi


#%%

def plot_tuning_curve(angles, responses):
    """
    Plot tuning curve with polar plot.
    
    Args:
      angles    : array-like of angles in degrees
      responses : array-like of responses (same length as angles)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    angles = np.asarray(angles)
    responses = np.asarray(responses)
    angles_rad = np.deg2rad(angles)

    if angles_rad.size:
        closed_angles = np.append(angles_rad, angles_rad[0])
        closed_responses = np.append(responses, responses[0])
    else:
        closed_angles = angles_rad
        closed_responses = responses
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(closed_angles, closed_responses, marker='o', linestyle='-')
    
    ax.set_theta_zero_location('E')  # 0° on the right
    ax.set_theta_direction(1)         # increase counter-clockwise
    
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels([f'{i}°' for i in range(0, 360, 45)])
    
    ax.set_title(f"ROI #{roi_id}", loc= 'left')
    
    plt.tight_layout()
    plt.show()

plot_tuning_curve(pref_on.dir.dir_mean.index, pref_on.dir.dir_mean.values)
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



