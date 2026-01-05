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

# %% CALCULATE MEAN OSI DSI ACROSS SUBJECTS

# create time vector (15Hz) for dff
dataset[('meta','time_vector')] = dataset[('suite2p','deltaf_f')].map(lambda arr : pd.Series(np.arange(arr.shape[1]) / 15.0))

# CREATE AN ARTIFICIAL TIME INDEX FOR EVERY TRIAL
N_TRIALS   = 120
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
    if ('meta', 'trials') not in dataset.columns:
        raise KeyError("dataset is missing ('meta', 'trials') column")

    if dataset.index.nlevels < 2:
        raise ValueError('dataset expected to be indexed by subject and session')

    task_level = dataset.index.names[-1] if dataset.index.names[-1] is not None else -1
    task_mask = dataset.index.get_level_values(task_level) == task_name
    task_df = dataset.loc[task_mask]

    if task_df.empty:
        empty_cols = pd.MultiIndex.from_tuples([
            ('meta', 'injection'),
            ('roi', 'ids'),
            ('roi', 'count'),
            ('metrics', 'osi'),
            ('metrics', 'dsi'),
            ('metrics', 'norm_osi'),
            ('metrics', 'norm_dsi'),
            ('metrics', 'preferred_orientation'),
            ('metrics', 'preferred_direction')
        ])
        return pd.DataFrame(columns=empty_cols, index=pd.MultiIndex.from_tuples([], names=['subject', 'session', 'task']))

    trial_entries = task_df[('meta', 'trials')]
    orientation_values: set[float] = set()
    direction_values: set[float] = set()
    for trials in trial_entries:
        if trials.empty:
            continue
        orientation_values.update(trials['orientation'].unique())
        direction_values.update(trials['direction'].unique())

    if not orientation_values or not direction_values:
        raise ValueError('No orientation/direction values found for task-gratings trials')

    orientation_list = np.array(sorted(orientation_values), dtype=float)
    direction_list = np.array(sorted(direction_values), dtype=float)

    injection_available = ('session_config', 'injection') in dataset.columns

    records: list[dict] = []

    for (subject, session, task), trials in trial_entries.items():
        if trials.empty:
            continue

        dff_on_trials = trials['dff_on'].to_numpy()
        if len(dff_on_trials) == 0:
            continue

        trial_stack = np.stack(dff_on_trials, axis=0)
        trial_means = np.nanmean(trial_stack, axis=2)

        orientations = trials['orientation'].to_numpy()
        directions = trials['direction'].to_numpy()

        n_rois = trial_means.shape[1]

        ori_mean_map: dict[float, np.ndarray] = {}
        for angle in orientation_list:
            mask = orientations == angle
            if mask.any():
                ori_mean_map[angle] = np.nanmean(trial_means[mask], axis=0)
            else:
                ori_mean_map[angle] = np.full(n_rois, np.nan)

        dir_mean_map: dict[float, np.ndarray] = {}
        for angle in direction_list:
            mask = directions == angle
            if mask.any():
                dir_mean_map[angle] = np.nanmean(trial_means[mask], axis=0)
            else:
                dir_mean_map[angle] = np.full(n_rois, np.nan)

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

        ori_choice = np.argmax(np.where(np.isfinite(ori_array), ori_array, -np.inf), axis=0)
        dir_choice = np.argmax(np.where(np.isfinite(dir_array), dir_array, -np.inf), axis=0)

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

        injection_value = dataset.loc[(subject, session, task), ('session_config', 'injection')] if injection_available else np.nan

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
        empty_cols = pd.MultiIndex.from_tuples([
            ('meta', 'injection'),
            ('roi', 'ids'),
            ('roi', 'count'),
            ('metrics', 'osi'),
            ('metrics', 'dsi'),
            ('metrics', 'norm_osi'),
            ('metrics', 'norm_dsi'),
            ('metrics', 'preferred_orientation'),
            ('metrics', 'preferred_direction')
        ])
        return pd.DataFrame(columns=empty_cols, index=pd.MultiIndex.from_tuples([], names=['subject', 'session', 'task']))

    summary_df = pd.DataFrame.from_records(records)
    summary_df = summary_df.set_index(['subject', 'session', 'task']).sort_index()

    column_tuples = []
    for col in summary_df.columns:
        if isinstance(col, tuple):
            column_tuples.append(col)
        else:
            column_tuples.append(('meta', str(col)))

    summary_df.columns = pd.MultiIndex.from_tuples(column_tuples)
    return summary_df


grating_selectivity_summary = compute_task_grating_selectivity(dataset)

#%%
# Plot distribution of selectivity across ROIs per subject, per injection

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

#%%

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
#%%

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


#%%


#%%
# PLOT SINGLE ROI TUNING CURVE
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



