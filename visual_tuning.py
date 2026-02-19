#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

dataset = pd.read_pickle(r"D:\Projects\ACUTEVIS\picklejar\260217_ACUTEVIS_long_dataset.pkl")

#%% CALCULATE OSI DSI
subject = 'ACUTEVIS06'
session = 'ses-01'
task = 'task-gratings'
roi_id = 0


def _roi_column_name(df, roi_id):
    candidate = f"deltaf_f_roi{roi_id}"
    if candidate in df.columns:
        return candidate
    if "deltaf_f" in df.columns:
        return "deltaf_f"
    raise KeyError(f"No ROI column found for roi_id={roi_id}. Expected {candidate!r} or 'deltaf_f'.")


def mean_dff_per_trial(long_df, subject, session, task, roi_id, epoch_col="is_gratings"):
    roi_col = _roi_column_name(long_df, roi_id)
    mask = (
        (long_df["Subject"] == subject)
        & (long_df["Session"] == session)
        & (long_df["Task"] == task)
    )
    task_df = long_df.loc[mask].sort_values("time_elapsed_s").copy()
    if task_df.empty:
        return pd.DataFrame(columns=["trial_means", "orientations", "directions"])

    in_epoch = task_df[epoch_col].fillna(False).astype(bool).to_numpy()
    prev_in_epoch = np.r_[False, in_epoch[:-1]]
    epoch_start = in_epoch & (~prev_in_epoch)
    epoch_id = np.cumsum(epoch_start)
    task_df["trial_id"] = np.where(in_epoch, epoch_id, 0)

    trial_df = task_df[task_df["trial_id"] > 0]
    if trial_df.empty:
        return pd.DataFrame(columns=["trial_means", "orientations", "directions"])

    out = (
        trial_df.groupby("trial_id", as_index=False)
        .agg(
            trial_means=(roi_col, "mean"),
            orientations=("orientations", "first"),
            directions=("directions", "first"),
        )
        [["trial_means", "orientations", "directions"]]
    )
    return out


on_means = mean_dff_per_trial(dataset, subject, session, task, roi_id, epoch_col="is_gratings")
off_means = mean_dff_per_trial(dataset, subject, session, task, roi_id, epoch_col="is_gray")


def _count_epochs(long_df, subject, session, task, epoch_col):
    mask = (
        (long_df["Subject"] == subject)
        & (long_df["Session"] == session)
        & (long_df["Task"] == task)
    )
    task_df = long_df.loc[mask].sort_values("time_elapsed_s")
    in_epoch = task_df[epoch_col].fillna(False).astype(bool).to_numpy()
    if len(in_epoch) == 0:
        return 0
    prev_in_epoch = np.r_[False, in_epoch[:-1]]
    starts = in_epoch & (~prev_in_epoch)
    return int(starts.sum())


gratings_epoch_count = _count_epochs(dataset, subject, session, task, "is_gratings")
gray_epoch_count = _count_epochs(dataset, subject, session, task, "is_gray")

print(f"Grating epochs detected: {gratings_epoch_count}")
print(f"Gray epochs detected: {gray_epoch_count}")

# identify tuned stimuli
def preferred_stimulus(on_means):
    if on_means.empty:
        raise ValueError("on_means is empty. Check subject/session/task filters and is_gratings epochs.")

    on_means = on_means.dropna(subset=["orientations", "directions"]).copy()
    if on_means.empty:
        raise ValueError("No orientation/direction labels found in on_means.")

    ori = on_means.groupby('orientations', observed=False)['trial_means']
    dir = on_means.groupby('directions', observed=False)['trial_means']
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
    pref_ori_resp = ori_mean.get(pref_ori, np.nan)
    orth_ori_resp = ori_mean.get(orth_ori, np.nan)
    norm_ori = (pref_ori_resp - orth_ori_resp) / pref_ori_resp

    ori_angles = np.deg2rad(ori_mean.index.to_numpy(dtype=float))
    ori_tc = ori_mean.to_numpy(dtype=float)
    osi = np.abs(np.sum(ori_tc * np.exp(2j * ori_angles)) / np.sum(ori_tc))

    # find orthogonal direction
    orth_dir = (pref_dir + 180) % 360
    pref_dir_resp = dir_mean.get(pref_dir, np.nan)
    orth_dir_resp = dir_mean.get(orth_dir, np.nan)
    norm_dir = (pref_dir_resp - orth_dir_resp) / pref_dir_resp

    dir_angles = np.deg2rad(dir_mean.index.to_numpy(dtype=float))
    dir_tc = dir_mean.to_numpy(dtype=float)
    dsi = np.abs(np.sum(dir_tc * np.exp(1j * dir_angles)) / np.sum(dir_tc))
    
    return {'norm_ori': norm_ori, 'norm_dir': norm_dir, 'osi': osi, 'dsi': dsi}
norm_tuning = normalize_tuning(pref_on)

print(f"Normalized Orientation Tuning: {norm_tuning['norm_ori']}")
print(f"Normalized Direction Tuning: {norm_tuning['norm_dir']}")

def osi_dsi(tuning_dict):
    ori_mean = tuning_dict['ori']['ori_mean']
    dir_mean = tuning_dict['dir']['dir_mean']

    # OSI calculation
    angle_ori = np.deg2rad(ori_mean.index.to_numpy(dtype=float))
    tc_ori = ori_mean.to_numpy(dtype=float)
    osi = (np.abs(np.sum(tc_ori * np.exp(2j * angle_ori)) / np.sum(tc_ori)))

    # DSI calculation
    angle_dir = np.deg2rad(dir_mean.index.to_numpy(dtype=float))
    tc_dir = dir_mean.to_numpy(dtype=float)
    dsi = (np.abs(np.sum(tc_dir * np.exp(1j * angle_dir)) / np.sum(tc_dir)))

    return {'osi': osi, 'dsi': dsi}
osi_dsi_values = osi_dsi(pref_on)
print(f"Orientation Selectivity Index (OSI): {osi_dsi_values['osi']}")
print(f"Direction Selectivity Index (DSI): {osi_dsi_values['dsi']}")

#%% CREATE SELECTIVITY SUMMARY DF
def compute_task_grating_selectivity(dataset: pd.DataFrame, task_name: str = 'task-gratings') -> pd.DataFrame:
    """Return per-ROI orientation/direction tuning metrics for all subject/session task-gratings groups."""

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

    def _pick_column(df: pd.DataFrame, names: list[str], label: str) -> str:
        for name in names:
            if name in df.columns:
                return name
        raise KeyError(f"Missing required column for {label}: expected one of {names}")

    subject_col = _pick_column(dataset, ['Subject', 'subject'], 'subject')
    session_col = _pick_column(dataset, ['Session', 'session'], 'session')
    task_col = _pick_column(dataset, ['Task', 'task'], 'task')
    time_col = _pick_column(dataset, ['time_elapsed_s', 'time', 'timestamp_s'], 'time')
    grating_col = _pick_column(dataset, ['is_gratings'], 'is_gratings')
    orientation_col = _pick_column(dataset, ['orientations', 'orientation'], 'orientations')
    direction_col = _pick_column(dataset, ['directions', 'direction'], 'directions')
    injection_col = _pick_column(dataset, ['injection'], 'injection')

    roi_cols = [col for col in dataset.columns if isinstance(col, str) and col.startswith('deltaf_f_roi')]
    if roi_cols:
        roi_cols = sorted(roi_cols, key=lambda name: int(name.replace('deltaf_f_roi', '')))
    elif 'deltaf_f' in dataset.columns:
        roi_cols = ['deltaf_f']
    else:
        raise KeyError("No ROI dF/F columns found. Expected 'deltaf_f_roiN' or 'deltaf_f'.")

    task_df = dataset[dataset[task_col] == task_name].copy()
    if task_df.empty:
        return _empty_summary()

    angle_rows = task_df[task_df[grating_col].fillna(False).astype(bool)]
    orientation_list = np.array(sorted(pd.to_numeric(angle_rows[orientation_col], errors='coerce').dropna().unique()), dtype=float)
    direction_list = np.array(sorted(pd.to_numeric(angle_rows[direction_col], errors='coerce').dropna().unique()), dtype=float)
    if len(orientation_list) == 0 or len(direction_list) == 0:
        raise ValueError('No orientation/direction values found for task-gratings rows.')

    records: list[dict] = []

    grouped = task_df.groupby([subject_col, session_col, task_col], sort=True)
    for (subject, session, task), group in grouped:
        group = group.sort_values(time_col).copy()
        in_gratings = group[grating_col].fillna(False).astype(bool).to_numpy()
        if len(in_gratings) == 0 or not in_gratings.any():
            continue

        prev_in_gratings = np.r_[False, in_gratings[:-1]]
        trial_start = in_gratings & (~prev_in_gratings)
        trial_id = np.cumsum(trial_start)
        group['trial_id'] = np.where(in_gratings, trial_id, 0)

        grating_rows = group[group['trial_id'] > 0].copy()
        if grating_rows.empty:
            continue

        n_rois = len(roi_cols)
        osi = np.full(n_rois, np.nan, dtype=float)
        dsi = np.full(n_rois, np.nan, dtype=float)
        norm_osi = np.full(n_rois, np.nan, dtype=float)
        norm_dsi = np.full(n_rois, np.nan, dtype=float)
        preferred_ori = np.full(n_rois, np.nan, dtype=float)
        preferred_dir = np.full(n_rois, np.nan, dtype=float)

        ori_mean_map: dict[float, np.ndarray] = {
            angle: np.full(n_rois, np.nan, dtype=float) for angle in orientation_list
        }
        dir_mean_map: dict[float, np.ndarray] = {
            angle: np.full(n_rois, np.nan, dtype=float) for angle in direction_list
        }

        for roi_idx, roi_col in enumerate(roi_cols):
            trial_means = (
                grating_rows.groupby('trial_id', as_index=False)
                .agg(
                    trial_means=(roi_col, 'mean'),
                    orientations=(orientation_col, 'first'),
                    directions=(direction_col, 'first'),
                )
                [['trial_means', 'orientations', 'directions']]
            )

            trial_means['orientations'] = pd.to_numeric(trial_means['orientations'], errors='coerce')
            trial_means['directions'] = pd.to_numeric(trial_means['directions'], errors='coerce')
            trial_means = trial_means.dropna(subset=['orientations', 'directions'])
            if trial_means.empty:
                continue

            tuning = preferred_stimulus(trial_means)
            ori_series = tuning['ori']['ori_mean']
            dir_series = tuning['dir']['dir_mean']
            pref_ori_value = float(tuning['ori']['pref_ori'])
            pref_dir_value = float(tuning['dir']['pref_dir'])

            preferred_ori[roi_idx] = pref_ori_value
            preferred_dir[roi_idx] = pref_dir_value

            for angle in orientation_list:
                ori_mean_map[angle][roi_idx] = float(ori_series.get(angle, np.nan))
            for angle in direction_list:
                dir_mean_map[angle][roi_idx] = float(dir_series.get(angle, np.nan))

            vector_metrics = osi_dsi(tuning)
            osi[roi_idx] = vector_metrics.get('osi', np.nan)
            dsi[roi_idx] = vector_metrics.get('dsi', np.nan)

            norm_metrics = normalize_tuning(tuning)
            norm_osi[roi_idx] = norm_metrics.get('norm_ori', np.nan)
            norm_dsi[roi_idx] = norm_metrics.get('norm_dir', np.nan)

        injection_value = group[injection_col].iloc[0]

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
    fig.suptitle(f"Average Response to Preferred Orientation")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


orientation_tuning_fig = plot_preferred_orientation_tuning(grating_selectivity_summary)
#%% Plot preferred direction tuning curves

def plot_preferred_direction_tuning(data_df, task_name='task-gratings', figsize=(14, 8)):
    """Plot mean direction tuning curves grouped by preferred direction across subjects.

    Accepts either:
    - long_df (columns like Subject/Session/Task + direction/orientation metrics source), or
    - precomputed grating_selectivity_summary.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Accept long_df directly and compute summary in this function.
    if isinstance(data_df.columns, pd.MultiIndex):
        summary_df = data_df
    else:
        summary_df = compute_task_grating_selectivity(data_df, task_name=task_name)

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
    fig.suptitle(f"Average Response to Preferred Direction")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


direction_tuning_fig = plot_preferred_direction_tuning(dataset, task_name='task-gratings')



#%% VISUALIZE MEAN TRIAL (GRAY + GRATING)
def plot_mean_trial_gray_grating(
    long_df: pd.DataFrame,
    subject: str,
    session: str,
    roi_id: int = 1,
    task_name: str = "task-gratings",
):
    """Plot mean dF/F across all gray+grating trials, aligned to grating onset (t=0)."""

    # Select one recording block and sort by time.
    task_df = long_df[
        (long_df["Subject"] == subject)
        & (long_df["Session"] == session)
        & (long_df["Task"] == task_name)
    ].sort_values("time_elapsed_s").copy()

    # Pull time, signal, and state vectors.
    roi_col = _roi_column_name(task_df, roi_id)
    time_vals = task_df["time_elapsed_s"].to_numpy(dtype=float)
    signal_vals = pd.to_numeric(task_df[roi_col], errors="coerce").to_numpy(dtype=float)
    is_gratings = task_df["is_gratings"].fillna(False).astype(bool).to_numpy()
    is_gray = task_df["is_gray"].fillna(False).astype(bool).to_numpy()

    # Find every grating onset (False -> True).
    grating_starts = np.where(is_gratings & (~np.r_[False, is_gratings[:-1]]))[0]

    # Build one aligned segment per trial: preceding gray + grating.
    aligned_time_segments = []
    aligned_signal_segments = []
    pre_durations = []
    post_durations = []

    for onset_idx in grating_starts:
        gray_start_idx = onset_idx
        while gray_start_idx > 0 and is_gray[gray_start_idx - 1]:
            gray_start_idx -= 1

        grating_end_idx = onset_idx
        while grating_end_idx < len(is_gratings) and is_gratings[grating_end_idx]:
            grating_end_idx += 1

        t_seg = time_vals[gray_start_idx:grating_end_idx] - time_vals[onset_idx]
        y_seg = signal_vals[gray_start_idx:grating_end_idx]

        aligned_time_segments.append(t_seg)
        aligned_signal_segments.append(y_seg)
        pre_durations.append(-t_seg[0])
        post_durations.append(t_seg[-1])

    # Use the time window shared by all trials.
    common_pre = float(np.min(pre_durations))
    common_post = float(np.min(post_durations))
    dt = float(np.median(np.diff(time_vals)))
    time_grid = np.arange(-common_pre, common_post + 0.5 * dt, dt)

    # Interpolate each trial to the shared grid, then average.
    trial_matrix = np.full((len(aligned_time_segments), len(time_grid)), np.nan, dtype=float)
    for i, (t_seg, y_seg) in enumerate(zip(aligned_time_segments, aligned_signal_segments)):
        valid = (time_grid >= t_seg[0]) & (time_grid <= t_seg[-1])
        trial_matrix[i, valid] = np.interp(time_grid[valid], t_seg, y_seg)
    mean_trace = np.nanmean(trial_matrix, axis=0)

    # Plot mean trace with gray/grating spans and stim-on marker.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_grid, mean_trace, linewidth=1.8, label=f"ROI {roi_id} mean (n={len(aligned_time_segments)} trials)")
    ax.axvspan(-common_pre, 0.0, alpha=0.15, label="Gray")
    ax.axvspan(0.0, common_post, alpha=0.10, label="Grating")
    ax.axvline(0.0, linestyle="--", linewidth=1.2, label="Stim ON (is_gratings)")
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("dF/F")
    ax.set_title(f"{subject} | {session} | mean across trials (gray + grating)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig, ax


_fig, _ax = plot_mean_trial_gray_grating(
    dataset,
    subject='ACUTEVIS09',
    session=session,
    roi_id=2,
    task_name=task,
)


#%% PLOT MEAN TRIAL ACROSS ALL ROIS BY SUBJECT (LINES = INJECTION/SESSION)
def mean_trial_trace_all_rois(
    long_df: pd.DataFrame,
    subject: str,
    session: str,
    task_name: str = "task-gratings",
    max_trials: int = 120,
):
    """Compute mean trial trace aligned to stimulus onset using all ROIs for one subject/session."""

    task_df = long_df[
        (long_df["Subject"] == subject)
        & (long_df["Session"] == session)
        & (long_df["Task"] == task_name)
    ].sort_values("time_elapsed_s").copy()

    roi_cols = [col for col in task_df.columns if isinstance(col, str) and col.startswith("deltaf_f_roi")]
    if not roi_cols and "deltaf_f" in task_df.columns:
        roi_cols = ["deltaf_f"]

    # Keep only ROI columns that are complete (no missing values) and non-zero for this subject/session.
    roi_df = task_df[roi_cols].apply(pd.to_numeric, errors="coerce")
    valid_roi_mask = roi_df.notna().all(axis=0) & (roi_df.abs().sum(axis=0) > 0)
    roi_cols = roi_df.columns[valid_roi_mask].tolist()

    time_vals = task_df["time_elapsed_s"].to_numpy(dtype=float)
    is_gratings = task_df["is_gratings"].fillna(False).astype(bool).to_numpy()
    is_gray = task_df["is_gray"].fillna(False).astype(bool).to_numpy()
    roi_matrix = task_df[roi_cols].to_numpy(dtype=float)

    # Trial onsets are rising edges of is_gratings.
    grating_starts = np.where(is_gratings & (~np.r_[False, is_gratings[:-1]]))[0]
    grating_starts = grating_starts[:max_trials]

    aligned_time_segments = []
    aligned_signal_segments = []
    pre_durations = []
    post_durations = []

    for onset_idx in grating_starts:
        # Include contiguous gray period immediately before grating onset.
        gray_start_idx = onset_idx
        while gray_start_idx > 0 and is_gray[gray_start_idx - 1]:
            gray_start_idx -= 1

        # Include contiguous grating period after onset.
        grating_end_idx = onset_idx
        while grating_end_idx < len(is_gratings) and is_gratings[grating_end_idx]:
            grating_end_idx += 1

        # Average across ROIs first, then align this trial in time.
        y_seg = np.nanmean(roi_matrix[gray_start_idx:grating_end_idx, :], axis=1)
        t_seg = time_vals[gray_start_idx:grating_end_idx] - time_vals[onset_idx]

        aligned_time_segments.append(t_seg)
        aligned_signal_segments.append(y_seg)
        pre_durations.append(-t_seg[0])
        post_durations.append(t_seg[-1])

    # Fixed shared time window.
    common_pre = 2.0
    common_post = 2.0
    dt = float(np.median(np.diff(time_vals)))
    time_grid = np.arange(-common_pre, common_post + 0.5 * dt, dt)

    # Interpolate each trial to shared grid and average across trials.
    trial_matrix = np.full((len(aligned_time_segments), len(time_grid)), np.nan, dtype=float)
    for i, (t_seg, y_seg) in enumerate(zip(aligned_time_segments, aligned_signal_segments)):
        valid = (time_grid >= t_seg[0]) & (time_grid <= t_seg[-1])
        trial_matrix[i, valid] = np.interp(time_grid[valid], t_seg, y_seg)
    mean_trace = np.nanmean(trial_matrix, axis=0)

    injection_label = "unknown"
    if "injection" in task_df.columns and task_df["injection"].notna().any():
        injection_label = str(task_df.loc[task_df["injection"].notna(), "injection"].iloc[0])

    return {
        "time_grid": time_grid,
        "mean_trace": mean_trace,
        "injection": injection_label,
        "n_trials": len(aligned_time_segments),
    }


def plot_subject_mean_trials_by_injection(
    long_df: pd.DataFrame,
    task_name: str = "task-gratings",
    max_trials: int = 120,
):
    """Create one plot per subject with one mean-trial line per session/injection."""

    task_df = long_df[long_df["Task"] == task_name].copy()
    subjects = sorted(task_df["Subject"].dropna().unique())

    # Consistent color per injection label across subjects.
    if "injection" in task_df.columns and task_df["injection"].notna().any():
        injections = sorted(task_df["injection"].dropna().astype(str).unique())
    else:
        injections = []
    cmap = plt.get_cmap("tab10")
    color_map = {inj: cmap(i % cmap.N) for i, inj in enumerate(injections)}

    # Build all subject-session traces first, then enforce one common overlap grid.
    trace_cache = {}
    all_traces = []
    for subject in subjects:
        subj_df = task_df[task_df["Subject"] == subject]
        sessions = sorted(subj_df["Session"].dropna().unique())
        for session in sessions:
            trace_info = mean_trial_trace_all_rois(
                long_df,
                subject,
                session,
                task_name=task_name,
                max_trials=max_trials,
            )
            trace_cache[(subject, session)] = trace_info
            all_traces.append((trace_info["time_grid"], trace_info["mean_trace"]))

    dt_values = np.concatenate([np.diff(t) for t, _ in all_traces])
    dt = float(np.median(dt_values[dt_values > 0]))
    common_time_grid = np.arange(-2.0, 2.0 + 0.5 * dt, dt)

    figures = {}

    for subject in subjects:
        subj_df = task_df[task_df["Subject"] == subject]
        sessions = sorted(subj_df["Session"].dropna().unique())

        fig, ax = plt.subplots(figsize=(9, 4))

        for session in sessions:
            trace_info = trace_cache[(subject, session)]
            t = trace_info["time_grid"]
            y = trace_info["mean_trace"]
            inj = trace_info["injection"]
            n_trials = trace_info["n_trials"]

            y_common = np.interp(common_time_grid, t, y, left=np.nan, right=np.nan)

            color = color_map.get(inj, None)
            ax.plot(common_time_grid, y_common, linewidth=1.8, color=color, label=f"{inj} ({session}, n={n_trials})")

        ax.axvspan(float(np.min(common_time_grid)), 0.0, alpha=0.12)
        ax.axvspan(0.0, float(np.max(common_time_grid)), alpha=0.08)

        ax.axvline(0.0, linestyle="--", linewidth=1.2, color="k", label="Stim ON")
        ax.set_xlabel("Time from stimulus onset (s)")
        ax.set_ylabel("Mean dF/F across ROIs")
        ax.set_title(f"{subject} | mean trial response by injection/session")
        ax.grid(alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)
        plt.tight_layout()

        figures[str(subject)] = fig

    return figures


subject_mean_trial_figs = plot_subject_mean_trials_by_injection(dataset, task_name="task-gratings", max_trials=120)

#%% MEAN TRIAL RESPONSE ACROSS ALL SUBJECTS (LINES = INJECTION)
def plot_global_mean_trial_by_injection(
    long_df: pd.DataFrame,
    task_name: str = "task-gratings",
    max_trials: int = 120,
):
    """Single figure: mean trial response per injection, averaged across all subject/session traces."""

    task_df = long_df[long_df["Task"] == task_name].copy()
    pairs = (
        task_df[["Subject", "Session"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["Subject", "Session"])
        .to_records(index=False)
    )

    traces_by_injection: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
    traces_by_injection = {}
    all_traces: List[Tuple[np.ndarray, np.ndarray]]
    all_traces = []
    for subject, session in pairs:
        trace_info = mean_trial_trace_all_rois(
            long_df,
            subject=str(subject),
            session=str(session),
            task_name=task_name,
            max_trials=max_trials,
        )
        inj = str(trace_info["injection"])
        traces_by_injection.setdefault(inj, []).append((trace_info["time_grid"], trace_info["mean_trace"]))
        all_traces.append((trace_info["time_grid"], trace_info["mean_trace"]))

    # One common overlap grid across all subject/session traces.
    dt_values = np.concatenate([np.diff(t) for t, _ in all_traces])
    dt = float(np.median(dt_values[dt_values > 0]))
    common_time_grid = np.arange(-2.0, 2.0 + 0.5 * dt, dt)
    off_mask = common_time_grid < 0

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, injection in enumerate(sorted(traces_by_injection.keys())):
        traces = traces_by_injection[injection]

        # Interpolate each subject/session trace and average.
        stacked = np.full((len(traces), len(common_time_grid)), np.nan, dtype=float)
        for row_idx, (t, y) in enumerate(traces):
            y_interp = np.interp(common_time_grid, t, y, left=np.nan, right=np.nan)
            off_baseline = np.nanmean(y_interp[off_mask])
            stacked[row_idx, :] = y_interp - off_baseline

        mean_trace = np.nanmean(stacked, axis=0)
        std_trace = np.nanstd(stacked, axis=0, ddof=1)
        counts = np.sum(np.isfinite(stacked), axis=0)
        sem_trace = np.divide(
            std_trace,
            np.sqrt(np.maximum(counts, 1)),
            out=np.zeros_like(std_trace),
            where=counts > 1,
        )

        color = cmap(idx % cmap.N)
        ax.plot(common_time_grid, mean_trace, linewidth=2.0, color=color, label=f"{injection} (n={len(traces)})")
        ax.fill_between(
            common_time_grid,
            mean_trace - sem_trace,
            mean_trace + sem_trace,
            color=color,
            alpha=0.2,
        )

    ax.axvspan(float(np.min(common_time_grid)), 0.0, alpha=0.12)
    ax.axvspan(0.0, float(np.max(common_time_grid)), alpha=0.08)

    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="k", label="Stim ON")
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Baseline-normalized mean dF/F")
    ax.set_title("Event Averaged Response")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax


global_injection_fig, global_injection_ax = plot_global_mean_trial_by_injection(
    dataset,
    task_name="task-gratings",
    max_trials=120,
)


#%% EXPORT SELECTED PLOTS TO PDF
def export_plots_to_pdf(
    selected_plots,
    save_dir: str,
    file_name: str,
    dpi: int = 300,
    orientation: str = "landscape",
):
    """Export selected matplotlib figure(s) into one PDF.

    Parameters
    ----------
    selected_plots : Figure | list[Figure] | dict[Any, Figure]
        The plots you want to export. If dict, all values are exported in insertion order.
    save_dir : str
        Output directory.
    file_name : str
        PDF file name (with or without .pdf).
    dpi : int
        Export resolution used for page rendering (default: 300).
    orientation : str
        PDF page orientation for multi-plot pages: "landscape" or "portrait".
    """
    import os
    from matplotlib.backends.backend_pdf import PdfPages

    if not file_name.lower().endswith(".pdf"):
        file_name = f"{file_name}.pdf"

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, file_name)

    if isinstance(selected_plots, dict):
        figures = list(selected_plots.values())
    elif isinstance(selected_plots, list):
        figures = selected_plots
    else:
        figures = [selected_plots]

    if len(figures) == 0:
        raise ValueError("No plots were provided for export.")

    orientation = orientation.lower()
    if orientation not in {"landscape", "portrait"}:
        raise ValueError("orientation must be 'landscape' or 'portrait'")

    with PdfPages(out_path) as pdf:
        if len(figures) == 1:
            pdf.savefig(figures[0], bbox_inches="tight", dpi=dpi)
        else:
            import io

            n_plots = len(figures)
            n_cols = int(np.ceil(np.sqrt(n_plots)))
            n_rows = int(np.ceil(n_plots / n_cols))

            # A4 page size in inches.
            if orientation == "portrait":
                page_width, page_height = 8.27, 11.69
                if n_cols > n_rows:
                    n_rows, n_cols = n_cols, n_rows
            else:
                page_width, page_height = 11.69, 8.27
                if n_rows > n_cols:
                    n_rows, n_cols = n_cols, n_rows

            composite_fig, axes = plt.subplots(n_rows, n_cols, figsize=(page_width, page_height))
            axes = np.atleast_1d(axes).ravel()

            for i, fig in enumerate(figures):
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
                buffer.seek(0)
                image = plt.imread(buffer)
                buffer.close()

                axes[i].imshow(image)
                axes[i].axis("off")

            for j in range(n_plots, len(axes)):
                axes[j].axis("off")

            composite_fig.tight_layout()
            pdf.savefig(composite_fig, bbox_inches="tight", dpi=dpi)
            plt.close(composite_fig)

    return out_path


def export_selected_plots_to_pdf(
    selected_plots,
    save_dir: str,
    file_name: str,
    dpi: int = 300,
    orientation: str = "landscape",
):
    """Backward-compatible wrapper for export_plots_to_pdf."""
    return export_plots_to_pdf(
        selected_plots=selected_plots,
        save_dir=save_dir,
        file_name=file_name,
        dpi=dpi,
        orientation=orientation,
    )






# %% 
