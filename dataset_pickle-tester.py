"""
Created on Mon Nov 24 14:11:01 2025

@author: sindhuja s. baskar
"""
#%%
import numpy as np
import pandas as pd
from collections import defaultdict
from collections.abc import Mapping, Sequence

dataset = pd.read_pickle(r"C:\Projects\260122_ACUTEVIS.pkl")

print(dataset.head())

keys = pd.DataFrame(dataset.keys())
#%% CREATE TRIALS DATAFRAME

# create time vector (15Hz) for dff
dataset[('meta','time_vector')] = dataset[('suite2p','deltaf_f')].map(lambda arr : pd.Series(np.arange(arr.shape[1]) / 15.0))

# CREATE AN ARTIFICIAL TIME INDEX FOR EVERY TRIAL
N_TRIALS   = 120
TRIAL_S    = 5.0
GRAY_S     = 3.0               # within each trial
GRATING_S  = 2.0               # (informational; TRIAL_S should equal GRAY_S + GRATING_S)

# 0-based artificial trial indices (length = 130) stored as a Series in the MultiIndex column
dataset[('psychopy', 'trial_index')] = dataset[('suite2p','deltaf_f')].map(
    lambda _ : pd.Series(np.arange(N_TRIALS), dtype="Int64"))

# Per-row DataFrame of trial stamps (same columns as your psychopy-derived structure)
dataset[('psychopy', 'trial_stamps')] = dataset[('suite2p','deltaf_f')].map(
    lambda _ : pd.DataFrame({
        "trial_num":      np.arange(N_TRIALS, dtype=np.int64),
        "trial_start":    np.arange(N_TRIALS) * TRIAL_S,
        "gray_start":     np.arange(N_TRIALS) * TRIAL_S,
        "gratings_start": np.arange(N_TRIALS) * TRIAL_S + GRAY_S,
        "trial_end":      np.arange(N_TRIALS) * TRIAL_S + TRIAL_S,
    }))

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
#%% Mean ROI response to ON/OFF across sessions (cohort) 

import matplotlib.pyplot as plt

def _roi_mean_response(trials_df: pd.DataFrame):
    if trials_df is None or trials_df.empty:
        return None
    dff_on_trials = trials_df['dff_on'].to_numpy()
    dff_off_trials = trials_df['dff_off'].to_numpy()
    if len(dff_on_trials) == 0 or len(dff_off_trials) == 0:
        return None
    on_stack = np.stack(dff_on_trials, axis=0)
    off_stack = np.stack(dff_off_trials, axis=0)
    roi_on = np.nanmean(on_stack, axis=(0, 2))
    roi_off = np.nanmean(off_stack, axis=(0, 2))
    return roi_on, roi_off

roi_by_injection = defaultdict(lambda: {'on': [], 'off': []})

for _, row in dataset.iterrows():
    injection = row[('session_config', 'injection')]
    if pd.isna(injection):
        continue
    trials_df = row[('meta', 'trials')]
    roi_result = _roi_mean_response(trials_df)
    if roi_result is None:
        continue
    roi_on, roi_off = roi_result
    roi_by_injection[injection]['on'].extend(roi_on.tolist())
    roi_by_injection[injection]['off'].extend(roi_off.tolist())

preferred_order = ['Baseline', 'Saline', 'Low', 'High']
injection_ids = [inj for inj in preferred_order if inj in roi_by_injection]
injection_ids.extend([inj for inj in roi_by_injection.keys() if inj not in injection_ids])

means_on, means_off, errs_on, errs_off = [], [], [], []
for inj in injection_ids:
    on_vals = np.asarray(roi_by_injection[inj]['on'], dtype=float)
    off_vals = np.asarray(roi_by_injection[inj]['off'], dtype=float)
    means_on.append(np.nanmean(on_vals))
    means_off.append(np.nanmean(off_vals))
    errs_on.append(np.nanstd(on_vals))
    errs_off.append(np.nanstd(off_vals))

x_pos = np.arange(len(injection_ids))
bar_w = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x_pos - bar_w / 2, means_on, width=bar_w, yerr=errs_on, label='ON', color='#4C78A8', alpha=0.9)
ax.bar(x_pos + bar_w / 2, means_off, width=bar_w, yerr=errs_off, label='OFF', color='#F58518', alpha=0.9)
ax.set_xticks(x_pos)
ax.set_xticklabels(injection_ids)
ax.set_ylabel('Mean dF/F')
ax.set_xlabel('Injection')
ax.set_title('Mean dF/F ON/OFF by Condition')
ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)
ax.legend()
plt.tight_layout()

#%% TUNING, OSI/DSI VECTOR
subject = 'ACUTEVIS16'
session = 'ses-01'
task =  'task-gratings'
roi_id = 0  # specify ROI index

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
def compute_task_grating_selectivity(dataset: pd.DataFrame, task_name: str = 'task-gratings') -> pd.DataFrame:
    """Return trial-mean orientation/direction tuning vectors per subject/session."""

    def _empty_summary() -> pd.DataFrame:
        empty_cols = pd.MultiIndex.from_tuples([
            ('meta', 'injection'),
            ('roi', 'ids'),
            ('roi', 'count'),
            ('metrics', 'osi'),
            ('metrics', 'dsi'),
            ('metrics', 'norm_osi'),
            ('metrics', 'norm_dsi'),
            ('metrics', 'preferred_orientation'),
            ('metrics', 'preferred_direction'),
        ])
        empty_index = pd.MultiIndex.from_tuples([], names=['subject', 'session', 'task'])
        return pd.DataFrame(columns=empty_cols, index=empty_index)

    if ('meta', 'trials') not in dataset.columns:
        raise KeyError("dataset is missing ('meta', 'trials') column")

    if dataset.index.nlevels < 2:
        raise ValueError('dataset expected to be indexed by subject and session')

    task_level_name = dataset.index.names[-1]
    task_level = dataset.index.nlevels - 1 if task_level_name is None else task_level_name
    task_mask = dataset.index.get_level_values(task_level) == task_name
    task_df = dataset.loc[task_mask]
    if task_df.empty:
        return _empty_summary()

    trial_entries = task_df[('meta', 'trials')]

    def _collect_unique_angles(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        orientation_values: set[float] = set()
        direction_values: set[float] = set()
        for trials in series:
            if trials.empty:
                continue
            orientation_values.update(pd.unique(trials['orientation']))
            direction_values.update(pd.unique(trials['direction']))
        if not orientation_values or not direction_values:
            raise ValueError('No orientation/direction values found for task-gratings trials')
        orientations = np.array(sorted(float(angle) for angle in orientation_values), dtype=float)
        directions = np.array(sorted(float(angle) for angle in direction_values), dtype=float)
        return orientations, directions

    def _mean_dff_on(trials: pd.Series) -> np.ndarray | None:
        if trials.empty:
            return None
        dff_arrays = trials.to_numpy()
        if len(dff_arrays) == 0:
            return None
        trial_stack = np.stack(dff_arrays, axis=0)
        return np.nanmean(trial_stack, axis=2)

    def _mean_map(angles: np.ndarray, labels: np.ndarray, trial_means: np.ndarray) -> dict[float, np.ndarray]:
        n_rois = trial_means.shape[1]
        mean_map: dict[float, np.ndarray] = {}
        for angle in angles:
            mask = labels == angle
            mean_map[angle] = np.nanmean(trial_means[mask], axis=0) if mask.any() else np.full(n_rois, np.nan)
        return mean_map

    orientation_list, direction_list = _collect_unique_angles(trial_entries)
    injection_available = ('session_config', 'injection') in dataset.columns

    records: list[dict] = []

    for (subject, session, task), trials in trial_entries.items():
        if trials.empty:
            continue

        trial_means = _mean_dff_on(trials['dff_on'])
        if trial_means is None:
            continue

        orientations = trials['orientation'].to_numpy(dtype=float)
        directions = trials['direction'].to_numpy(dtype=float)
        n_rois = trial_means.shape[1]

        ori_mean_map = _mean_map(orientation_list, orientations, trial_means)
        dir_mean_map = _mean_map(direction_list, directions, trial_means)

        ori_array = np.vstack([ori_mean_map[angle] for angle in orientation_list])
        dir_array = np.vstack([dir_mean_map[angle] for angle in direction_list])

        ori_weights = np.exp(2j * np.deg2rad(orientation_list))[:, None]
        dir_weights = np.exp(1j * np.deg2rad(direction_list))[:, None]

        ori_valid = np.where(np.isfinite(ori_array), ori_array, 0.0)
        dir_valid = np.where(np.isfinite(dir_array), dir_array, 0.0)

        ori_totals = np.sum(ori_valid, axis=0)
        dir_totals = np.sum(dir_valid, axis=0)

        with np.errstate(invalid='ignore', divide='ignore'):
            osi = np.abs(np.sum(ori_valid * ori_weights, axis=0) / ori_totals)
            dsi = np.abs(np.sum(dir_valid * dir_weights, axis=0) / dir_totals)

        osi[ori_totals == 0] = np.nan
        dsi[dir_totals == 0] = np.nan

        finite_ori = np.where(np.isfinite(ori_array), ori_array, -np.inf)
        finite_dir = np.where(np.isfinite(dir_array), dir_array, -np.inf)
        ori_choice = np.argmax(finite_ori, axis=0)
        dir_choice = np.argmax(finite_dir, axis=0)

        has_ori = np.isfinite(ori_array).any(axis=0)
        has_dir = np.isfinite(dir_array).any(axis=0)

        preferred_ori = np.full(n_rois, np.nan)
        preferred_dir = np.full(n_rois, np.nan)
        preferred_ori[has_ori] = orientation_list[ori_choice[has_ori]]
        preferred_dir[has_dir] = direction_list[dir_choice[has_dir]]

        norm_osi = np.full(n_rois, np.nan, dtype=float)
        norm_dsi = np.full(n_rois, np.nan, dtype=float)

        for roi_idx in range(n_rois):
            pref_ori_value = preferred_ori[roi_idx]
            pref_dir_value = preferred_dir[roi_idx]
            if not (np.isfinite(pref_ori_value) and np.isfinite(pref_dir_value)):
                continue

            ori_series = pd.Series(ori_array[:, roi_idx], index=orientation_list, dtype=float).dropna()
            dir_series = pd.Series(dir_array[:, roi_idx], index=direction_list, dtype=float).dropna()

            orth_ori_value = (pref_ori_value + 90) % 180
            orth_dir_value = (pref_dir_value + 180) % 360

            if pref_ori_value not in ori_series.index or orth_ori_value not in ori_series.index:
                continue
            if pref_dir_value not in dir_series.index or orth_dir_value not in dir_series.index:
                continue
            if ori_series[pref_ori_value] == 0 or dir_series[pref_dir_value] == 0:
                continue

            tuning_payload = {
                'ori': {'pref_ori': pref_ori_value, 'ori_mean': ori_series},
                'dir': {'pref_dir': pref_dir_value, 'dir_mean': dir_series},
            }

            try:
                norm_payload = normalize_tuning(tuning_payload)
            except Exception:
                continue

            norm_osi[roi_idx] = norm_payload.get('norm_ori', np.nan)
            norm_dsi[roi_idx] = norm_payload.get('norm_dir', np.nan)

        if injection_available:
            injection_value = dataset.loc[(subject, session, task), ('session_config', 'injection')]
        else:
            injection_value = np.nan

        row: dict = {
            'subject': subject,
            'session': session,
            'task': task,
            ('meta', 'injection'): injection_value,
            ('roi', 'ids'): np.arange(n_rois, dtype=int),
            ('roi', 'count'): n_rois,
            ('metrics', 'osi'): osi,
            ('metrics', 'dsi'): dsi,
            ('metrics', 'norm_osi'): norm_osi,
            ('metrics', 'norm_dsi'): norm_dsi,
            ('metrics', 'preferred_orientation'): preferred_ori,
            ('metrics', 'preferred_direction'): preferred_dir,
        }

        for angle in orientation_list:
            row[('orientation_mean', int(angle))] = ori_mean_map[angle]
        for angle in direction_list:
            row[('direction_mean', int(angle))] = dir_mean_map[angle]

        records.append(row)

    if not records:
        return _empty_summary()

    summary_df = pd.DataFrame.from_records(records)
    summary_df = summary_df.set_index(['subject', 'session', 'task']).sort_index()

    column_tuples = [col if isinstance(col, tuple) else ('meta', str(col)) for col in summary_df.columns]
    summary_df.columns = pd.MultiIndex.from_tuples(column_tuples)
    return summary_df

grating_selectivity_summary = compute_task_grating_selectivity(dataset)

#%% COMBINE OSI/DSI ACROSS ALL SUBJECTS
# Make percent-of-baseline OSI distribution plot per injection condition

def plot_osi_percent_of_baseline(summary_df, metric=('metrics', 'norm_osi'), figsize=(7, 4)):
    """Plot OSI values expressed as percent of each subject's baseline session."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if summary_df.empty:
        raise ValueError("summary_df is empty; run compute_task_grating_selectivity first")

    metric_key = tuple(metric)
    if metric_key not in summary_df.columns:
        raise KeyError(f"summary_df is missing column {metric_key}")

    summary_reset = summary_df.reset_index()
    injection_col = ('meta', 'injection')
    if injection_col not in summary_reset.columns:
        raise KeyError("summary_df is missing ('meta', 'injection') column")

    def _resolve_scalar(value: object) -> object | None:
        while True:
            if isinstance(value, pd.DataFrame):
                flattened = value.stack(dropna=False)
                if flattened.empty:
                    return None
                value = flattened.iloc[0]
                continue
            if isinstance(value, Mapping):
                for candidate in value.values():
                    if pd.isna(candidate):
                        continue
                    value = candidate
                    break
                else:
                    return None
                continue
            if isinstance(value, pd.Series):
                value = value.dropna()
                if value.empty:
                    return None
                value = value.iloc[0]
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                filtered = [item for item in value if not pd.isna(item)]
                if not filtered:
                    return None
                value = filtered[0]
                continue
            if isinstance(value, np.generic):
                value = value.item()
                continue
            break
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    def _as_label(raw: object) -> str | None:
        scalar = _resolve_scalar(raw)
        if scalar is None:
            return None
        return str(scalar)

    def _extract_injection_label(raw: object) -> str | None:
        return _as_label(raw)

    records: list[dict] = []
    for _, row in summary_reset.iterrows():
        injection_label = _extract_injection_label(row[injection_col])
        if injection_label is None:
            continue

        subject_label = _as_label(row['subject'])
        session_label = _as_label(row['session'])
        task_label = _as_label(row['task'])
        if subject_label is None or session_label is None or task_label is None:
            continue

        metric_values = np.asarray(row[metric_key], dtype=float).ravel()
        if metric_values.size == 0:
            continue

        valid_values = metric_values[np.isfinite(metric_values)]
        if valid_values.size == 0:
            continue

        for value in valid_values:
            records.append({
                'subject': subject_label,
                'session': session_label,
                'task': task_label,
                'injection': injection_label,
                'value': float(value),
            })

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        raise ValueError('No OSI values available for plotting')

    def _session_order(label: object) -> float:
        if isinstance(label, (int, float, np.number)):
            return float(label)
        if isinstance(label, str):
            digits = ''.join(ch for ch in label if ch.isdigit())
            if digits:
                return float(digits)
        return float('inf')

    long_df['session_order'] = long_df['session'].apply(_session_order)
    baseline_mask = long_df.groupby('subject')['session_order'].transform('min') == long_df['session_order']
    baseline_lookup = (
        long_df.loc[baseline_mask]
        .groupby('subject')['value']
        .median()
    )
    baseline_series = long_df['subject'].map(baseline_lookup)

    with np.errstate(divide='ignore', invalid='ignore'):
        percent_baseline = np.where(
            (~np.isfinite(baseline_series)) | np.isclose(baseline_series, 0.0),
            np.nan,
            (long_df['value'] / baseline_series) * 100.0,
        )

    long_df['percent_baseline'] = percent_baseline
    long_df = long_df[np.isfinite(long_df['percent_baseline'])]
    if long_df.empty:
        raise ValueError('Unable to compute percent-baseline values (baseline missing or zero)')

    long_df['subject'] = long_df['subject'].astype(str)
    long_df['session'] = long_df['session'].astype(str)
    long_df['task'] = long_df['task'].astype(str)

    injections = [label for label in pd.unique(long_df['injection']) if isinstance(label, str) and label]
    if not injections:
        raise ValueError('No injection labels available for plotting')

    values_by_injection = {
        label: long_df.loc[long_df['injection'] == label, 'percent_baseline'].to_numpy(dtype=float)
        for label in injections
    }

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(injections) + 1, dtype=float)

    for xpos, label in zip(positions, injections):
        values = values_by_injection[label]
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.25, 0.25, size=values.size)
        ax.scatter(
            np.full(values.size, xpos) + jitter,
            values,
            color='#666666',
            s=14,
            alpha=0.4,
            linewidths=0,
            zorder=1,
        )

    violin_values = [values_by_injection[label] for label in injections if values_by_injection[label].size]
    violin_positions = [positions[idx] for idx, label in enumerate(injections) if values_by_injection[label].size]
    if violin_values:
        violin = ax.violinplot(
            violin_values,
            positions=violin_positions,
            widths=0.6,
            showmeans=True,
            showextrema=False,
        )
        for body in violin['bodies']:
            body.set_facecolor('#4C72B0')
            body.set_edgecolor('black')
            body.set_alpha(0.5)
        cmeans = violin.get('cmeans')
        if cmeans is not None:
            cmeans.set_color('black')
            cmeans.set_linewidth(1.0)

    ax.axhline(100.0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(injections, rotation=30, ha='right')
    ax.set_ylabel('Normalized OSI (% of baseline)')

    max_percent = max((np.nanmax(values) for values in values_by_injection.values() if values.size), default=100.0)
    upper_bound = max(120.0, max_percent * 1.1)
    ax.set_ylim(bottom=0.0, top=upper_bound)
    ax.set_title('OSI Distribution Relative to Baseline')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig


osi_percent_baseline_fig = plot_osi_percent_of_baseline(grating_selectivity_summary)
#%% Plot distribution of selectivity across ROIs per subject, per injection

def plot_subject_selectivity_distribution(summary_df, metric="osi", figsize=None):
    """Plot per-subject ROI selectivity distributions grouped by injection."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if summary_df.empty:
        raise ValueError("summary_df is empty; run compute_task_grating_selectivity first")

    metric = str(metric).lower()
    allowed_metrics = {"osi", "dsi"}
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of {sorted(allowed_metrics)}")

    metric_map = {"osi": ("metrics", "norm_osi"), "dsi": ("metrics", "norm_dsi")}
    metric_key = metric_map[metric]
    needed = [("meta", "injection"), metric_key]
    missing = [col for col in needed if col not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df is missing columns: {missing}")

    summary_reset = summary_df.reset_index()
    subjects = list(pd.unique(summary_reset["subject"]))
    if not subjects:
        raise ValueError("No subjects found in summary_df")

    injection_column = summary_reset[("meta", "injection")].dropna()
    injections = list(pd.unique(injection_column))
    if not injections:
        raise ValueError("No injection labels found in summary_df")

    n_subjects = len(subjects)
    n_cols = min(3, max(1, n_subjects))
    n_rows = int(np.ceil(n_subjects / n_cols))
    if figsize is None:
        figsize = (n_cols * 4.0, n_rows * 3.0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    rng = np.random.default_rng(0)

    for ax in axes_flat[n_subjects:]:
        ax.set_visible(False)

    for idx, subject in enumerate(subjects):
        ax = axes_flat[idx]
        subject_df = summary_reset.loc[summary_reset["subject"] == subject]

        values_by_injection = {inj: [] for inj in injections}
        for _, row in subject_df.iterrows():
            injection_value = row[("meta", "injection")]
            if pd.isna(injection_value):
                continue
            metric_values = np.asarray(row[metric_key], dtype=float).ravel()
            metric_values = metric_values[np.isfinite(metric_values)]
            if metric_values.size:
                values_by_injection[injection_value].append(metric_values)

        aggregated = {
            injection_label: np.concatenate(chunks) if chunks else np.array([], dtype=float)
            for injection_label, chunks in values_by_injection.items()
        }

        positions = np.arange(1, len(injections) + 1, dtype=float)

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(len(injections))]

        violin_values: list[np.ndarray] = []
        violin_positions: list[float] = []
        violin_colors: list[tuple[float, float, float, float]] = []

        for inj_idx, injection_label in enumerate(injections):
            values = aggregated[injection_label]
            xpos = positions[inj_idx]
            if values.size:
                jitter = rng.uniform(-0.2, 0.2, size=values.size)
                ax.scatter(
                    np.full(values.size, xpos) + jitter,
                    values,
                    color="#666666",
                    s=12,
                    alpha=0.35,
                    linewidths=0,
                    zorder=1,
                )
                violin_values.append(values)
                violin_positions.append(float(xpos))
                violin_colors.append(colors[inj_idx])

        if violin_values:
            violin = ax.violinplot(
                violin_values,
                positions=violin_positions,
                widths=0.6,
                showmeans=True,
                showextrema=False,
            )
            for body, color in zip(violin["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.55)
                body.set_zorder(2)
            cmeans = violin.get("cmeans")
            if cmeans is not None:
                cmeans.set_color("black")
                cmeans.set_linewidth(1.0)
                cmeans.set_zorder(3)

        ax.set_xticks(positions)
        ax.set_xticklabels(injections, rotation=30, ha="right")
        ax.set_ylabel(f"Normalized {metric.upper()}")
        valid_max = max((float(vals.max()) for vals in aggregated.values() if vals.size), default=1.0)
        upper_bound = valid_max * 1.1 if valid_max > 0 else 1.0
        ax.set_ylim(0.0, max(1.0, upper_bound))
        ax.set_title(f"{subject}")
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.suptitle(f"{metric.upper()} Distribution", fontsize=16)
    fig.tight_layout()
    return fig


osi_distribution_fig = plot_subject_selectivity_distribution(grating_selectivity_summary, metric="osi")
dsi_distribution_fig = plot_subject_selectivity_distribution(grating_selectivity_summary, metric="dsi")

#%% Plot preferred orientation tuning curves

def plot_preferred_orientation_tuning(summary_df, task_name='task-gratings', figsize=(10, 8)):
    """Plot mean orientation tuning curves grouped by preferred orientation across subjects."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if summary_df.empty:
        raise ValueError("summary_df is empty; run compute_task_grating_selectivity first")

    subset = summary_df
    if 'task' in summary_df.index.names and task_name is not None:
        task_mask = summary_df.index.get_level_values('task') == task_name
        subset = summary_df.loc[task_mask]
        if subset.empty:
            raise ValueError(f"No records found for task {task_name}")

    orientation_cols = [col for col in subset.columns if isinstance(col, tuple) and col[0] == 'orientation_mean']
    if not orientation_cols:
        raise KeyError("summary_df lacks orientation_mean columns")

    orientation_angles = sorted(int(col[1]) for col in orientation_cols)
    if not orientation_angles:
        raise ValueError("No orientation angles available for plotting")

    orientation_angles_array = np.asarray(orientation_angles, dtype=float)

    injection_series = subset[('meta', 'injection')].dropna()
    if injection_series.empty:
        raise ValueError("No injection metadata available for plotting")
    injection_labels = sorted(pd.unique(injection_series))

    width, height = figsize
    fig, axes = plt.subplots(2, 2, figsize=(width, height), squeeze=False)
    axes_flat = axes.flatten()

    cmap = plt.get_cmap('tab10')
    color_map = {label: cmap(idx % cmap.N) for idx, label in enumerate(injection_labels)}
    legend_handles: dict[str, object] = {}

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= len(orientation_angles):
            ax.set_visible(False)
            continue

        pref_angle = orientation_angles[ax_idx]
        has_data = False

        for injection_label in injection_labels:
            inj_rows = subset.loc[subset[('meta', 'injection')] == injection_label]
            if inj_rows.empty:
                continue

            collected_responses: list[np.ndarray] = []

            for _, row in inj_rows.iterrows():
                pref_orientation = np.asarray(row[('metrics', 'preferred_orientation')], dtype=float).ravel()
                roi_mask = np.isfinite(pref_orientation) & np.isclose(pref_orientation, pref_angle)
                if not np.any(roi_mask):
                    continue

                orientation_stack = np.vstack([
                    np.asarray(row[('orientation_mean', angle)], dtype=float).ravel()
                    for angle in orientation_angles
                ])

                roi_responses = orientation_stack[:, roi_mask]
                if roi_responses.size:
                    collected_responses.append(roi_responses)

            if not collected_responses:
                continue

            all_responses = np.hstack(collected_responses)
            has_data = True
            with np.errstate(invalid='ignore'):
                mean_response = np.nanmean(all_responses, axis=1)
                std_response = np.nanstd(all_responses, axis=1, ddof=1)
            counts = np.sum(np.isfinite(all_responses), axis=1)
            sem_response = np.divide(
                std_response,
                np.sqrt(np.maximum(counts, 1)),
                out=np.zeros_like(std_response),
                where=counts > 1,
            )

            line, = ax.plot(
                orientation_angles_array,
                mean_response,
                marker='o',
                linewidth=2,
                color=color_map[injection_label],
            )
            ax.fill_between(
                orientation_angles_array,
                mean_response - sem_response,
                mean_response + sem_response,
                color=color_map[injection_label],
                alpha=0.2,
            )

            if injection_label not in legend_handles:
                legend_handles[injection_label] = line

        if not has_data:
            ax.text(0.5, 0.5, 'No ROIs', ha='center', va='center', transform=ax.transAxes, fontsize=10)

        ax.set_title(f'{pref_angle}°')
        ax.set_xlabel('Orientation (°)')
        ax.set_ylabel('Response')
        ax.set_xticks(orientation_angles_array)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    legend_order = [label for label in injection_labels if label in legend_handles]
    if legend_order:
        fig.legend(
            [legend_handles[label] for label in legend_order],
            legend_order,
            loc='upper right',
            frameon=False,
        )

    title_suffix = f" ({task_name})" if task_name is not None else ""
    fig.suptitle(f"Mean Response of ROIs Grouped by Preferred Orientation")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


orientation_tuning_fig = plot_preferred_orientation_tuning(grating_selectivity_summary)
#%% Plot preferred direction tuning curves

def plot_preferred_direction_tuning(summary_df, task_name='task-gratings', figsize=(14, 8)):
    """Plot mean direction tuning curves grouped by preferred direction across subjects."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if summary_df.empty:
        raise ValueError("summary_df is empty; run compute_task_grating_selectivity first")

    subset = summary_df
    if 'task' in summary_df.index.names and task_name is not None:
        task_mask = summary_df.index.get_level_values('task') == task_name
        subset = summary_df.loc[task_mask]
        if subset.empty:
            raise ValueError(f"No records found for task {task_name}")

    direction_cols = [col for col in subset.columns if isinstance(col, tuple) and col[0] == 'direction_mean']
    if not direction_cols:
        raise KeyError("summary_df lacks direction_mean columns")

    direction_angles = sorted(int(col[1]) for col in direction_cols)
    if not direction_angles:
        raise ValueError("No direction angles available for plotting")

    direction_angles_array = np.asarray(direction_angles, dtype=float)

    injection_series = subset[('meta', 'injection')].dropna()
    if injection_series.empty:
        raise ValueError("No injection metadata available for plotting")
    injection_labels = sorted(pd.unique(injection_series))

    fig, axes = plt.subplots(2, 4, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    cmap = plt.get_cmap('tab10')
    color_map = {label: cmap(idx % cmap.N) for idx, label in enumerate(injection_labels)}
    legend_handles: dict[str, object] = {}

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= len(direction_angles):
            ax.set_visible(False)
            continue

        pref_angle = direction_angles[ax_idx]
        has_data = False

        for injection_label in injection_labels:
            inj_rows = subset.loc[subset[('meta', 'injection')] == injection_label]
            if inj_rows.empty:
                continue

            collected_responses: list[np.ndarray] = []

            for _, row in inj_rows.iterrows():
                pref_direction = np.asarray(row[('metrics', 'preferred_direction')], dtype=float).ravel()
                roi_mask = np.isfinite(pref_direction) & np.isclose(pref_direction, pref_angle)
                if not np.any(roi_mask):
                    continue

                direction_stack = np.vstack([
                    np.asarray(row[('direction_mean', angle)], dtype=float).ravel()
                    for angle in direction_angles
                ])

                roi_responses = direction_stack[:, roi_mask]
                if roi_responses.size:
                    collected_responses.append(roi_responses)

            if not collected_responses:
                continue

            all_responses = np.hstack(collected_responses)
            has_data = True
            with np.errstate(invalid='ignore'):
                mean_response = np.nanmean(all_responses, axis=1)
                std_response = np.nanstd(all_responses, axis=1, ddof=1)
            counts = np.sum(np.isfinite(all_responses), axis=1)
            sem_response = np.divide(
                std_response,
                np.sqrt(np.maximum(counts, 1)),
                out=np.zeros_like(std_response),
                where=counts > 1,
            )

            line, = ax.plot(
                direction_angles_array,
                mean_response,
                marker='o',
                linewidth=2,
                color=color_map[injection_label],
            )
            ax.fill_between(
                direction_angles_array,
                mean_response - sem_response,
                mean_response + sem_response,
                color=color_map[injection_label],
                alpha=0.2,
            )

            if injection_label not in legend_handles:
                legend_handles[injection_label] = line

        if not has_data:
            ax.text(0.5, 0.5, 'No ROIs', ha='center', va='center', transform=ax.transAxes, fontsize=10)

        ax.set_title(f'{pref_angle}°')
        ax.set_xlabel('Direction (°)')
        ax.set_ylabel('Response')
        ax.set_xticks(direction_angles_array)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    legend_order = [label for label in injection_labels if label in legend_handles]
    if legend_order:
        fig.legend(
            [legend_handles[label] for label in legend_order],
            legend_order,
            loc='upper right',
            frameon=False,
        )

    title_suffix = f" ({task_name})" if task_name is not None else ""
    fig.suptitle(f"Mean Response of ROIs Grouped by Preferred Direction")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


direction_tuning_fig = plot_preferred_direction_tuning(grating_selectivity_summary)
#%% Plot population tuning curves

def _collect_direction_tuning_entries(summary_df: pd.DataFrame, task_name: str | None):
    """Gather per-ROI normalized direction tuning entries for downstream plots."""
    if summary_df.empty:
        raise ValueError("summary_df is empty; run compute_task_grating_selectivity first")

    subset = summary_df
    if 'task' in summary_df.index.names and task_name is not None:
        task_mask = summary_df.index.get_level_values('task') == task_name
        subset = summary_df.loc[task_mask]
        if subset.empty:
            raise ValueError(f"No records found for task {task_name}")

    direction_cols = [col for col in subset.columns if isinstance(col, tuple) and col[0] == 'direction_mean']
    if not direction_cols:
        raise KeyError("summary_df lacks direction_mean columns")

    direction_angles = sorted(int(col[1]) for col in direction_cols)
    if not direction_angles:
        raise ValueError("No direction angles available for plotting")

    direction_angles_array = np.asarray(direction_angles, dtype=float)

    injection_series = subset[('meta', 'injection')].dropna()
    if injection_series.empty:
        raise ValueError("No injection metadata available for plotting")

    injection_labels = sorted(pd.unique(injection_series))
    relative_angle_set: set[float] = set()
    roi_entries: list[dict] = []

    for _, row in subset.iterrows():
        injection_label = row[('meta', 'injection')]
        if pd.isna(injection_label):
            continue

        direction_stack = np.vstack([
            np.asarray(row[('direction_mean', angle)], dtype=float).ravel()
            for angle in direction_angles
        ])
        if direction_stack.size == 0:
            continue

        pref_direction = np.asarray(row[('metrics', 'preferred_direction')], dtype=float).ravel()
        dsi_values = np.asarray(row.get(('metrics', 'dsi'), np.nan), dtype=float).ravel()
        n_rois = direction_stack.shape[1]

        for roi_idx in range(n_rois):
            if roi_idx >= pref_direction.size:
                break
            pref_angle = pref_direction[roi_idx]
            if not np.isfinite(pref_angle):
                continue

            dsi_value = np.nan
            if roi_idx < dsi_values.size:
                dsi_value = float(dsi_values[roi_idx])

            angle_match = np.isclose(direction_angles_array, pref_angle, atol=1e-6)
            if not angle_match.any():
                continue
            pref_pos = int(np.flatnonzero(angle_match)[0])

            responses = direction_stack[:, roi_idx]
            finite_mask = np.isfinite(responses)
            if np.sum(finite_mask) < 2 or not finite_mask[pref_pos]:
                continue

            finite_responses = responses[finite_mask]
            min_resp = np.nanmin(finite_responses)
            max_resp = np.nanmax(finite_responses)
            if not np.isfinite(min_resp) or not np.isfinite(max_resp) or np.isclose(max_resp, min_resp):
                continue

            normalized = (responses - min_resp) / (max_resp - min_resp)
            finite_norm = normalized[np.isfinite(normalized)]
            if finite_norm.size == 0:
                continue

            relative_angles = ((direction_angles_array - pref_angle + 180.0) % 360.0) - 180.0
            relative_angles = np.asarray(np.round(relative_angles, 6), dtype=float)

            relative_angle_set.update(relative_angles[np.isfinite(normalized)])
            roi_entries.append({
                'injection': injection_label,
                'relative_angles': relative_angles,
                'normalized': normalized,
                'dsi': dsi_value,
            })

    if not roi_entries:
        raise ValueError("No normalized tuning data available for plotting")

    sorted_relative_angles = np.asarray(sorted(relative_angle_set), dtype=float)
    if sorted_relative_angles.size == 0:
        raise ValueError("No relative angles available for plotting")

    return sorted_relative_angles, injection_labels, roi_entries


def _render_direction_tuning_plot(
    roi_entries: list[dict],
    injection_labels: list[str],
    relative_angles: np.ndarray,
    title: str,
    figsize: tuple[float, float],
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    if relative_angles.size == 0 or not roi_entries:
        ax.text(0.5, 0.5, 'No ROIs', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    rounded_angles = np.asarray(np.round(relative_angles, 6), dtype=float)
    label_order = [label for label in injection_labels if any(entry['injection'] == label for entry in roi_entries)]

    if not label_order:
        ax.text(0.5, 0.5, 'No ROIs', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    injection_data: dict[str, defaultdict[float, list[float]]] = {
        label: defaultdict(list) for label in label_order
    }

    for entry in roi_entries:
        label = entry['injection']
        if label not in injection_data:
            continue
        rel_angles = np.asarray(np.round(entry['relative_angles'], 6), dtype=float)
        for angle, value in zip(rel_angles, entry['normalized']):
            if np.isfinite(value):
                injection_data[label][float(angle)].append(float(value))

    cmap = plt.get_cmap('tab10')
    color_map = {label: cmap(idx % cmap.N) for idx, label in enumerate(label_order)}
    legend_handles: dict[str, object] = {}

    for label in label_order:
        angle_means: list[float] = []
        angle_sems: list[float] = []
        for angle in rounded_angles:
            values = injection_data[label].get(float(angle), [])
            if not values:
                angle_means.append(np.nan)
                angle_sems.append(np.nan)
                continue
            values_arr = np.asarray(values, dtype=float)
            valid_mask = np.isfinite(values_arr)
            if not valid_mask.any():
                angle_means.append(np.nan)
                angle_sems.append(np.nan)
                continue
            valid_values = values_arr[valid_mask]
            mean_val = float(np.nanmean(valid_values))
            if valid_values.size > 1:
                std_val = float(np.nanstd(valid_values, ddof=1))
                sem_val = std_val / np.sqrt(valid_values.size)
            else:
                sem_val = 0.0
            angle_means.append(mean_val)
            angle_sems.append(sem_val)

        means_arr = np.asarray(angle_means, dtype=float)
        sems_arr = np.asarray(angle_sems, dtype=float)
        valid_mean_mask = np.isfinite(means_arr)
        if not valid_mean_mask.any():
            continue

        color = color_map[label]
        line, = ax.plot(
            rounded_angles[valid_mean_mask],
            means_arr[valid_mean_mask],
            marker='o',
            linewidth=2,
            color=color,
            label=label,
        )

        sem_lower = np.clip(means_arr[valid_mean_mask] - sems_arr[valid_mean_mask], 0.0, 1.0)
        sem_upper = np.clip(means_arr[valid_mean_mask] + sems_arr[valid_mean_mask], 0.0, 1.0)
        ax.fill_between(
            rounded_angles[valid_mean_mask],
            sem_lower,
            sem_upper,
            color=color,
            alpha=0.2,
        )

        legend_handles[label] = line

    if legend_handles:
        ax.legend(legend_handles.values(), legend_handles.keys(), loc='upper right', frameon=False)

    ax.axvline(0, color='0.3', linestyle='--', linewidth=1)
    ax.set_xlabel('Degrees from preferred direction')
    ax.set_ylabel('Normalized response')
    ax.set_xticks(rounded_angles)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_combined_preferred_direction_tuning(summary_df, task_name='task-gratings', figsize=(10, 6)):
    """Plot normalized direction tuning curves aligned to each ROI's preferred direction."""
    relative_angles, injection_labels, roi_entries = _collect_direction_tuning_entries(summary_df, task_name)
    title_suffix = f" ({task_name})" if task_name is not None else ""
    title = f'Normalized Direction Tuning Relative to Preferred Direction{title_suffix}'
    return _render_direction_tuning_plot(roi_entries, injection_labels, relative_angles, title, figsize)


def plot_direction_tuning_percentile_panels(
    summary_df: pd.DataFrame,
    task_name: str = 'task-gratings',
    dsi_threshold: float = 0.3,
    figsize: tuple[float, float] = (10, 6),
):
    """Plot separate direction tuning summaries split by a DSI threshold."""
    relative_angles, injection_labels, roi_entries = _collect_direction_tuning_entries(summary_df, task_name)

    dsi_values = np.asarray([entry.get('dsi', np.nan) for entry in roi_entries], dtype=float)
    if dsi_values.size == 0:
        raise ValueError('No DSI values available for threshold split')

    finite_mask = np.isfinite(dsi_values)
    if not finite_mask.any():
        raise ValueError('No finite DSI values available for threshold split')

    threshold = float(dsi_threshold)
    if not np.isfinite(threshold):
        threshold = 0.3

    high_entries = [entry for entry, value in zip(roi_entries, dsi_values) if np.isfinite(value) and value > threshold]
    low_entries = [entry for entry, value in zip(roi_entries, dsi_values) if np.isfinite(value) and value <= threshold]

    finite_values = dsi_values[finite_mask]

    if not high_entries:
        best_idx = int(np.nanargmax(finite_values))
        high_entries = [roi_entries[np.flatnonzero(finite_mask)[best_idx]]]

    if not low_entries:
        worst_idx = int(np.nanargmin(finite_values))
        low_entries = [roi_entries[np.flatnonzero(finite_mask)[worst_idx]]]

    title_suffix = f" ({task_name})" if task_name is not None else ""
    high_title = f'High-DSI ROIs (> {threshold:.2f} DSI){title_suffix}'
    low_title = f'Low-DSI ROIs (<= {threshold:.2f} DSI){title_suffix}'

    high_fig = _render_direction_tuning_plot(high_entries, injection_labels, relative_angles, high_title, figsize)
    low_fig = _render_direction_tuning_plot(low_entries, injection_labels, relative_angles, low_title, figsize)

    return high_fig, low_fig


combined_direction_tuning_fig = plot_combined_preferred_direction_tuning(grating_selectivity_summary)
top_direction_fig, bottom_direction_fig = plot_direction_tuning_percentile_panels(
    grating_selectivity_summary,
    dsi_threshold=0.4,
)
#%% PLOT SINGLE ROI TUNING CURVE
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




#%% PLOTTING FUNCTIONS
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



