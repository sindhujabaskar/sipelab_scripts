import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
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
    off_mask = common_time_grid < 0

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
            stacked[row_idx, :] = np.interp(common_time_grid, t, y, left=np.nan, right=np.nan)

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
    ax.set_ylabel("Mean dF/F")
    ax.set_title("Mean Trial Response by Injection Across All Subjects")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax


global_injection_fig, global_injection_ax = plot_global_mean_trial_by_injection(
    dataset,
    task_name="task-gratings",
    max_trials=120,
)

#%% PER ROI T-TESTS: GRATING VS GRAY
def per_roi_ttests_grating_vs_gray(
    long_df: pd.DataFrame,
    task_name: str = "task-gratings",
    max_trials: int = 120,
    alpha: float = 0.05,
):
    """Run cycle-averaged per-ROI paired t-tests between grating and gray means.

    For each trial onset, uses fixed windows relative to onset time:
    - gray window: [-2, 0) seconds
    - grating window: [0, 2] seconds
    Trials are then grouped into consecutive 8-trial direction cycles
    [0, 45, 90, 135, 180, 225, 270, 315] in presentation order.
    A paired t-test is performed per ROI on cycle-mean grating vs gray values.
    """
    from scipy.stats import ttest_rel

    task_df = long_df[long_df["Task"] == task_name].copy()
    pairs = (
        task_df[["Subject", "Session"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["Subject", "Session"])
        .to_records(index=False)
    )

    results = []

    for subject, session in pairs:
        session_df = task_df[
            (task_df["Subject"] == subject)
            & (task_df["Session"] == session)
        ].sort_values("time_elapsed_s").copy()

        roi_cols = [col for col in session_df.columns if isinstance(col, str) and col.startswith("deltaf_f_roi")]
        if not roi_cols and "deltaf_f" in session_df.columns:
            roi_cols = ["deltaf_f"]
        if len(roi_cols) == 0:
            continue

        roi_df = session_df[roi_cols].apply(pd.to_numeric, errors="coerce")
        valid_roi_mask = roi_df.notna().all(axis=0) & (roi_df.abs().sum(axis=0) > 0)
        roi_cols = roi_df.columns[valid_roi_mask].tolist()
        if len(roi_cols) == 0:
            continue

        is_gratings = session_df["is_gratings"].fillna(False).astype(bool).to_numpy()
        is_gray = session_df["is_gray"].fillna(False).astype(bool).to_numpy()
        time_vals = session_df["time_elapsed_s"].to_numpy(dtype=float)
        orientation_vals = pd.to_numeric(session_df["orientations"], errors="coerce").to_numpy(dtype=float)
        roi_matrix = session_df[roi_cols].to_numpy(dtype=float)

        grating_starts = np.where(is_gratings & (~np.r_[False, is_gratings[:-1]]))[0][:max_trials]
        if len(grating_starts) == 0:
            continue

        # Collect trial-wise grating and gray means in onset order.
        grating_trials_list: list[np.ndarray] = []
        gray_trials_list: list[np.ndarray] = []

        for onset_idx in grating_starts:
            onset_time = time_vals[onset_idx]

            gray_mask = (
                (time_vals >= (onset_time - 2.0))
                & (time_vals < onset_time)
                & is_gray
            )
            grating_mask = (
                (time_vals >= onset_time)
                & (time_vals <= (onset_time + 2.0))
                & is_gratings
            )

            if not gray_mask.any() or not grating_mask.any():
                continue

            gray_mean = np.nanmean(roi_matrix[gray_mask, :], axis=0)
            grating_mean = np.nanmean(roi_matrix[grating_mask, :], axis=0)

            gray_trials_list.append(gray_mean)
            grating_trials_list.append(grating_mean)

        if len(grating_trials_list) == 0:
            continue

        gray_trials = np.vstack(gray_trials_list)
        grating_trials = np.vstack(grating_trials_list)

        # Group by 8-trial direction cycles and average within each cycle.
        n_complete_cycles = min(len(gray_trials), len(grating_trials)) // 8
        if n_complete_cycles < 2:
            continue

        usable_trials = n_complete_cycles * 8
        gray_cycles = gray_trials[:usable_trials].reshape(n_complete_cycles, 8, -1).mean(axis=1)
        grating_cycles = grating_trials[:usable_trials].reshape(n_complete_cycles, 8, -1).mean(axis=1)

        injection_label = "unknown"
        if "injection" in session_df.columns and session_df["injection"].notna().any():
            injection_label = str(session_df.loc[session_df["injection"].notna(), "injection"].iloc[0])

        for roi_idx, roi_col in enumerate(roi_cols):
            grating_vals = grating_cycles[:, roi_idx]
            gray_vals = gray_cycles[:, roi_idx]

            pair_mask = np.isfinite(grating_vals) & np.isfinite(gray_vals)
            if np.sum(pair_mask) < 2:
                continue

            grating_vals = grating_vals[pair_mask]
            gray_vals = gray_vals[pair_mask]
            test = ttest_rel(grating_vals, gray_vals)
            mean_diff = np.nanmean(grating_vals - gray_vals)

            results.append(
                {
                    "Subject": str(subject),
                    "Session": str(session),
                    "Task": task_name,
                    "injection": injection_label,
                    "roi_col": roi_col,
                    "roi_id": int(roi_col.replace("deltaf_f_roi", "")) if roi_col.startswith("deltaf_f_roi") else 0,
                    "n_cycles": int(np.sum(pair_mask)),
                    "mean_diff": float(mean_diff),
                    "t_stat": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "is_significant": bool(np.isfinite(test.pvalue) and (test.pvalue < alpha)),
                }
            )

    return pd.DataFrame(results)


def plot_global_injection_significant_rois(
    long_df: pd.DataFrame,
    ttest_df: pd.DataFrame,
    task_name: str = "task-gratings",
    max_trials: int = 120,
    alpha: float = 0.05,
):
    """Global injection plot using only ROIs significant in grating-vs-gray t-tests."""

    sig_df = ttest_df[np.isfinite(ttest_df["p_value"]) & (ttest_df["p_value"] < alpha)].copy()
    if sig_df.empty:
        raise ValueError(f"No significant ROIs found at alpha={alpha}.")

    task_df = long_df[long_df["Task"] == task_name].copy()
    pairs = (
        task_df[["Subject", "Session"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["Subject", "Session"])
        .to_records(index=False)
    )

    traces_by_injection: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    all_traces: List[Tuple[np.ndarray, np.ndarray]] = []

    for subject, session in pairs:
        pair_sig = sig_df[(sig_df["Subject"] == str(subject)) & (sig_df["Session"] == str(session))]
        roi_cols = pair_sig["roi_col"].dropna().unique().tolist()
        if len(roi_cols) == 0:
            continue

        session_df = task_df[
            (task_df["Subject"] == subject)
            & (task_df["Session"] == session)
        ].sort_values("time_elapsed_s").copy()

        roi_cols = [col for col in roi_cols if col in session_df.columns]
        if len(roi_cols) == 0:
            continue

        time_vals = session_df["time_elapsed_s"].to_numpy(dtype=float)
        is_gratings = session_df["is_gratings"].fillna(False).astype(bool).to_numpy()
        is_gray = session_df["is_gray"].fillna(False).astype(bool).to_numpy()
        roi_matrix = session_df[roi_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        grating_starts = np.where(is_gratings & (~np.r_[False, is_gratings[:-1]]))[0][:max_trials]
        if len(grating_starts) == 0:
            continue

        aligned_time_segments = []
        aligned_signal_segments = []
        for onset_idx in grating_starts:
            gray_start_idx = onset_idx
            while gray_start_idx > 0 and is_gray[gray_start_idx - 1]:
                gray_start_idx -= 1

            grating_end_idx = onset_idx
            while grating_end_idx < len(is_gratings) and is_gratings[grating_end_idx]:
                grating_end_idx += 1

            t_seg = time_vals[gray_start_idx:grating_end_idx] - time_vals[onset_idx]
            y_seg = np.nanmean(roi_matrix[gray_start_idx:grating_end_idx, :], axis=1)
            aligned_time_segments.append(t_seg)
            aligned_signal_segments.append(y_seg)

        if len(aligned_time_segments) == 0:
            continue

        dt = float(np.median(np.diff(time_vals)))
        time_grid = np.arange(-2.0, 2.0 + 0.5 * dt, dt)
        trial_matrix = np.full((len(aligned_time_segments), len(time_grid)), np.nan, dtype=float)
        for i, (t_seg, y_seg) in enumerate(zip(aligned_time_segments, aligned_signal_segments)):
            valid = (time_grid >= t_seg[0]) & (time_grid <= t_seg[-1])
            trial_matrix[i, valid] = np.interp(time_grid[valid], t_seg, y_seg)

        mean_trace = np.nanmean(trial_matrix, axis=0)
        injection_label = "unknown"
        if "injection" in session_df.columns and session_df["injection"].notna().any():
            injection_label = str(session_df.loc[session_df["injection"].notna(), "injection"].iloc[0])

        traces_by_injection.setdefault(injection_label, []).append((time_grid, mean_trace))
        all_traces.append((time_grid, mean_trace))

    if len(all_traces) == 0:
        raise ValueError("No significant ROI traces available to plot.")

    dt_values = np.concatenate([np.diff(t) for t, _ in all_traces])
    dt = float(np.median(dt_values[dt_values > 0]))
    common_time_grid = np.arange(-2.0, 2.0 + 0.5 * dt, dt)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, injection in enumerate(sorted(traces_by_injection.keys())):
        traces = traces_by_injection[injection]
        stacked = np.full((len(traces), len(common_time_grid)), np.nan, dtype=float)
        for row_idx, (t, y) in enumerate(traces):
            stacked[row_idx, :] = np.interp(common_time_grid, t, y, left=np.nan, right=np.nan)

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
        ax.fill_between(common_time_grid, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.2)

    ax.axvspan(float(np.min(common_time_grid)), 0.0, alpha=0.12)
    ax.axvspan(0.0, float(np.max(common_time_grid)), alpha=0.08)
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="k", label="Stim ON")
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Mean dF/F (significant ROIs only)")
    ax.set_title(f"Mean Trial Response by Injection (15 trials, alpha={alpha})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig, ax


def summarize_roi_filtration(ttest_df: pd.DataFrame, alpha: float = 0.05):
    """Summarize ROI filtration counts for a given alpha threshold."""

    if ttest_df.empty:
        raise ValueError("ttest_df is empty.")

    required_cols = {"Subject", "Session", "injection", "roi_col", "p_value"}
    missing = required_cols.difference(ttest_df.columns)
    if missing:
        raise KeyError(f"ttest_df missing required columns: {sorted(missing)}")

    unique_roi_rows = ttest_df[["Subject", "Session", "injection", "roi_col"]].drop_duplicates()
    sig_unique_roi_rows = (
        ttest_df[np.isfinite(ttest_df["p_value"]) & (ttest_df["p_value"] < alpha)]
        [["Subject", "Session", "injection", "roi_col"]]
        .drop_duplicates()
    )

    total_rois = int(len(unique_roi_rows))
    kept_rois = int(len(sig_unique_roi_rows))
    dropped_rois = int(total_rois - kept_rois)
    kept_pct = (100.0 * kept_rois / total_rois) if total_rois > 0 else np.nan

    by_session_total = (
        unique_roi_rows.groupby(["Subject", "Session"], as_index=False)
        .size()
        .rename(columns={"size": "n_total_rois"})
    )
    by_session_kept = (
        sig_unique_roi_rows.groupby(["Subject", "Session"], as_index=False)
        .size()
        .rename(columns={"size": "n_kept_rois"})
    )
    by_session = by_session_total.merge(by_session_kept, on=["Subject", "Session"], how="left")
    by_session["n_kept_rois"] = by_session["n_kept_rois"].fillna(0).astype(int)
    by_session["n_dropped_rois"] = by_session["n_total_rois"] - by_session["n_kept_rois"]
    by_session["kept_pct"] = np.where(
        by_session["n_total_rois"] > 0,
        100.0 * by_session["n_kept_rois"] / by_session["n_total_rois"],
        np.nan,
    )

    by_injection_total = (
        unique_roi_rows.groupby(["injection"], as_index=False)
        .size()
        .rename(columns={"size": "n_total_rois"})
    )
    by_injection_kept = (
        sig_unique_roi_rows.groupby(["injection"], as_index=False)
        .size()
        .rename(columns={"size": "n_kept_rois"})
    )
    by_injection = by_injection_total.merge(by_injection_kept, on=["injection"], how="left")
    by_injection["n_kept_rois"] = by_injection["n_kept_rois"].fillna(0).astype(int)
    by_injection["n_dropped_rois"] = by_injection["n_total_rois"] - by_injection["n_kept_rois"]
    by_injection["kept_pct"] = np.where(
        by_injection["n_total_rois"] > 0,
        100.0 * by_injection["n_kept_rois"] / by_injection["n_total_rois"],
        np.nan,
    )

    overall = pd.DataFrame(
        {
            "alpha": [alpha],
            "n_total_rois": [total_rois],
            "n_kept_rois": [kept_rois],
            "n_dropped_rois": [dropped_rois],
            "kept_pct": [kept_pct],
        }
    )

    return {
        "overall": overall,
        "by_session": by_session.sort_values(["Subject", "Session"]).reset_index(drop=True),
        "by_injection": by_injection.sort_values(["injection"]).reset_index(drop=True),
    }


def plot_global_injection_significant_rois_by_alpha(
    long_df: pd.DataFrame,
    ttest_df: pd.DataFrame,
    alpha_values: list[float],
    task_name: str = "task-gratings",
    max_trials: int = 120,
):
    """Generate significant-ROI global injection plots for multiple alpha thresholds."""

    fig_by_alpha: dict[float, tuple] = {}
    for alpha in alpha_values:
        fig, ax = plot_global_injection_significant_rois(
            long_df,
            ttest_df,
            task_name=task_name,
            max_trials=max_trials,
            alpha=float(alpha),
        )
        fig_by_alpha[float(alpha)] = (fig, ax)
    return fig_by_alpha


roi_ttest_results = per_roi_ttests_grating_vs_gray(dataset, task_name="task-gratings", max_trials=120, alpha=0.05)

roi_filtration_summary = summarize_roi_filtration(roi_ttest_results, alpha=0.05)

global_injection_sigroi_fig, global_injection_sigroi_ax = plot_global_injection_significant_rois(
    dataset,
    roi_ttest_results,
    task_name="task-gratings",
    max_trials=120,
    alpha=0.05,
)

#%% COMBINED DIRECTION TUNING, ALIGNED TO PREFERRED
from collections import defaultdict

def _collect_direction_tuning_entries(data_df: pd.DataFrame, task_name: str | None):
    """Gather per-ROI normalized direction tuning entries for downstream plots.

    Accepts either:
    - long_df and computes selectivity summary internally, or
    - precomputed selectivity summary with MultiIndex columns.
    """
    if isinstance(data_df.columns, pd.MultiIndex):
        summary_df = data_df
    else:
        summary_df = compute_task_grating_selectivity(data_df, task_name=task_name or 'task-gratings')

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


def plot_combined_preferred_direction_tuning(data_df, task_name='task-gratings', figsize=(10, 6)):
    """Plot normalized direction tuning curves aligned to each ROI's preferred direction.

    Accepts either:
    - long_df, or
    - precomputed grating_selectivity_summary.
    """
    relative_angles, injection_labels, roi_entries = _collect_direction_tuning_entries(data_df, task_name)
    title = f'Tuning relative to preferred direction - POST ALIGNMENT'
    return _render_direction_tuning_plot(roi_entries, injection_labels, relative_angles, title, figsize)


combined_direction_tuning_fig = plot_combined_preferred_direction_tuning(dataset, task_name='task-gratings')



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
