#%%
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_pickle(r"G:\Projects\ACUTEVIS\picklejar\260426_ACUTEVIS_dataset.pkl")

from scipy.stats import zscore

# this is the directory that contains a /track2p folder that is output by running the track2p algorithm
t2p_save_path = r'G:\Projects\ACUTEVIS\processed\sub-ACUTEVIS06' # (change this based on your data)
plane = 'plane0' # which plane to process (the example dataset is single-plane)

# np.load() the match matrix (plane0_match_mat.npy)
t2p_match_mat = np.load(os.path.join(t2p_save_path, 'track2p', f'{plane}_match_mat.npy'), allow_pickle=True)

# np.load() settings (this contains suite2p paths etc.) (track_ops.npy)
track_ops_dict = np.load(os.path.join(t2p_save_path, 'track2p', 'track_ops.npy'), allow_pickle=True).item()
track_ops = SimpleNamespace(**track_ops_dict) # create dummy object from the track_ops dictionary

# get the rows that do not contain any Nones (if track2p doesnt find a match for a cell across two consecutive days it will append a None) -> cells with no Nones are cells matched across all days
t2p_match_mat_allday = t2p_match_mat[~np.any(t2p_match_mat==None, axis=1), :]

print(f'Shape of match matrix for cells present on all days: {t2p_match_mat_allday.shape} (cells, days)')

print('Datasets used for t2p:\n')
for ds_path in track_ops.all_ds_path:
    print(ds_path)

# lets take the last dataset
last_ds_path = track_ops.all_ds_path[-1]
print(f'We will look at the dataset saved at: {last_ds_path}')

# load the three files
last_ops = np.load(os.path.join(last_ds_path, 'suite2p', plane, 'ops.npy'), allow_pickle=True).item()
last_f = np.load(os.path.join(last_ds_path, 'suite2p', plane, 'F.npy'), allow_pickle=True)
iscell = np.load(os.path.join(last_ds_path, 'suite2p', plane, 'iscell.npy'), allow_pickle=True)

# we filter the traces based on suite2p's iscell probability (note: it is crucial to use the same probability as in the track2p settings to keep the correct indexing of matches)
iscell_thr = track_ops.iscell_thr

print(f'The iscell threshold used when running track2p was: {iscell_thr}')

if track_ops.iscell_thr==None:
    last_f_iscell = last_f[iscell[:, 0] == 1, :]

else:
    last_f_iscell = last_f[iscell[:, 1] > iscell_thr, :]

# now first plot the mean image of the movie (it is saved in ops.npy, for more info see the suite2p outputs documentation)
plt.imshow(last_ops['meanImg'], cmap='gray')
plt.axis('off')
plt.title('Mean image')
plt.show()

plt.figure(figsize=(10, 1))
nonmatch_nrn_idx = 0
plt.plot(last_f[nonmatch_nrn_idx, :])
plt.xlabel('Frame')
plt.ylabel('F')
plt.title(f'Example trace (nrn_idx: {nonmatch_nrn_idx})')
plt.show()

plt.figure(figsize=(10, 3))
plt.imshow(zscore(last_f_iscell, axis=1), aspect='auto', cmap='Greys', vmin=0, vmax=1.96)
plt.xlabel('Frame')
plt.ylabel('ROI')
plt.title('Raster plot')
plt.show()

iscell_thr = track_ops.iscell_thr # use the same threshold as when running the algo (to be consistent with indexing)

all_stat_t2p = []
all_f_t2p = []
all_ops = [] # ops dont change

for (i, ds_path) in enumerate(track_ops.all_ds_path):
    ops = np.load(os.path.join(ds_path, 'suite2p', plane, 'ops.npy'), allow_pickle=True).item()
    stat = np.load(os.path.join(ds_path, 'suite2p', plane, 'stat.npy'), allow_pickle=True)
    f = np.load(os.path.join(ds_path, 'suite2p', plane, 'F.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(ds_path, 'suite2p', plane, 'iscell.npy'), allow_pickle=True)
    
    
    if track_ops.iscell_thr==None:
        stat_iscell = stat[iscell[:, 0] == 1]
        f_iscell = f[iscell[:, 0] == 1, :]

    else:
        stat_iscell = stat[iscell[:, 1] > iscell_thr]
        f_iscell = f[iscell[:, 1] > iscell_thr, :]
    
    stat_t2p = stat_iscell[t2p_match_mat_allday[:,i].astype(int)]
    f_t2p = f_iscell[t2p_match_mat_allday[:,i].astype(int), :]

    all_stat_t2p.append(stat_t2p)
    all_f_t2p.append(f_t2p)
    all_ops.append(ops)

wind = 24
nrn_idx = 0

for i in range(len(track_ops.all_ds_path)):
    mean_img = all_ops[i]['meanImg']
    stat_t2p = all_stat_t2p[i]
    median_coord = stat_t2p[nrn_idx]['med']

    plt.figure(figsize=(1.5,1.5))
    plt.imshow(mean_img[int(median_coord[0])-wind:int(median_coord[0])+wind, int(median_coord[1])-wind:int(median_coord[1])+wind], cmap='gray') # plot a short window around the ROI centroid
    plt.scatter(wind, wind)
    plt.axis('off')
    plt.show()
#%%
# first plot the trace of cell c for all days
nrn_idx = 0 # the activity of the ROI visualised above on all days

for i in range(len(track_ops.all_ds_path)):
    plt.figure(figsize=(10, 1)) # make a wide figure
    plt.plot(all_f_t2p[i][nrn_idx, :])
    plt.xlabel('Frame')
    plt.ylabel('F')
    plt.show()

for i in range(len(track_ops.all_ds_path)):
    plt.figure(figsize=(10, 3)) # make a wide figure
    f_plot = zscore(all_f_t2p[i], axis=1)
    plt.imshow(f_plot, aspect='auto', cmap='Greys', vmin=0, vmax=1.96)
    plt.xlabel('Frame') 

#%%
def compute_dff(f_trace, percentile=8):
    """Compute df/f using a percentile-based baseline."""
    f0 = np.percentile(f_trace, percentile)
    return (f_trace - f0) / f0

def plot_dff_overlay(all_f_t2p, nrn_idx=0):
    """Plot overlaid df/f traces for a single neuron across the first 3 sessions."""
    labels = [f'Session {i}' for i in range(3)]
    fig, ax = plt.subplots(figsize=(12, 3))
    for i in range(3):
        trace = all_f_t2p[i][nrn_idx, :]
        dff = compute_dff(trace)
        ax.plot(dff, alpha=0.7, label=labels[i])
    ax.set_xlabel('Frame')
    ax.set_ylabel('dF/F')
    ax.set_title(f'dF/F overlay — neuron index {nrn_idx}')
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_dff_overlay(all_f_t2p, nrn_idx=30)

# %%
def plot_mean_dff_by_condition(all_f_t2p, track_ops, dataset, subject='ACUTEVIS06', task='task-gratings'):
    """Bar graph of mean dF/F for tracked cells across conditions for a given subject.

    Parameters
    ----------
    all_f_t2p   : list of ndarray, shape (n_cells, n_frames) per session
    track_ops   : SimpleNamespace from track_ops.npy (contains all_ds_path)
    dataset     : pd.DataFrame loaded from the project pickle
    subject     : subject ID string used to look up condition labels (default 'ACUTEVIS06')
    task        : task name string used to look up condition labels (default 'task-movies')
    """
    import re

    session_means = []
    session_sems = []
    condition_labels = []
    all_cell_means = []

    for i, ds_path in enumerate(track_ops.all_ds_path):
        # extract session ID (e.g. 'ses-01') from the dataset path
        ses_match = re.search(r'ses-\d+', ds_path)
        session_id = ses_match.group(0) if ses_match else f'ses-{i+1:02d}'

        # look up the condition / injection label from the dataset
        key = (subject, session_id, task)
        try:
            injection = dataset.loc[key, ('session_config', 'injection')]
            label = str(injection) if pd.notna(injection) else session_id
        except KeyError:
            label = session_id

        # compute dF/F for every matched cell, then average over frames
        f_session = all_f_t2p[i]  # (n_cells, n_frames)
        dff_per_cell = np.vstack([compute_dff(f_session[c, :]) for c in range(f_session.shape[0])])
        mean_per_cell = dff_per_cell.mean(axis=1)  # scalar per cell

        session_means.append(mean_per_cell.mean())
        session_sems.append(mean_per_cell.std() / np.sqrt(len(mean_per_cell)))
        condition_labels.append(label)
        all_cell_means.append(mean_per_cell)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(condition_labels))
    ax.bar(x, session_means, yerr=session_sems, capsize=5, color='steelblue', alpha=0.6)

    n_cells = len(all_cell_means[0])
    jitter = np.random.uniform(-0.15, 0.15, size=(n_cells, len(all_cell_means)))
    for i, cell_vals in enumerate(all_cell_means):
        ax.scatter(i + jitter[:, i], cell_vals, color='black', s=6, alpha=0.4, zorder=3)
    for c in range(n_cells):
        ys = [all_cell_means[i][c] for i in range(len(all_cell_means))]
        xs = [i + jitter[c, i] for i in range(len(all_cell_means))]
        ax.plot(xs, ys, color='black', lw=0.5, alpha=0.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, rotation=15, ha='right')
    ax.set_ylabel('Mean dF/F')
    ax.set_title(f'Mean dF/F across conditions — {subject}')
    plt.tight_layout()
    plt.show()

plot_mean_dff_by_condition(all_f_t2p, track_ops, dataset, subject='ACUTEVIS06')

# %%
def _bar_pair(ax, vals_a, vals_b, label_a, label_b, title):
    """Draw a paired bar + scatter + line plot for two conditions on a given axis."""
    means = [vals_a.mean(), vals_b.mean()]
    sems  = [vals_a.std() / np.sqrt(len(vals_a)), vals_b.std() / np.sqrt(len(vals_b))]
    n_cells = len(vals_a)
    jitter = np.random.uniform(-0.15, 0.15, size=(n_cells, 2))

    ax.bar([0, 1], means, yerr=sems, capsize=5, color='steelblue', alpha=0.6)
    ax.scatter(0 + jitter[:, 0], vals_a, color='black', s=6, alpha=0.4, zorder=3)
    ax.scatter(1 + jitter[:, 1], vals_b, color='black', s=6, alpha=0.4, zorder=3)
    for c in range(n_cells):
        ax.plot([jitter[c, 0], 1 + jitter[c, 1]], [vals_a[c], vals_b[c]],
                color='black', lw=0.5, alpha=0.2, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel('Mean dF/F')
    ax.set_title(title)


def plot_pairwise_dff(all_f_t2p, track_ops, dataset, subject='ACUTEVIS06', task='task-gratings'):
    """Three paired bar graphs: Saline-High, Saline-Low, Low-High.

    Condition matching is case-insensitive substring search on the injection label.
    """
    import re

    condition_data = {}  # label -> mean dF/F array (one value per cell)

    for i, ds_path in enumerate(track_ops.all_ds_path):
        ses_match = re.search(r'ses-\d+', ds_path)
        session_id = ses_match.group(0) if ses_match else f'ses-{i+1:02d}'

        key = (subject, session_id, task)
        try:
            injection = dataset.loc[key, ('session_config', 'injection')]
            label = str(injection) if pd.notna(injection) else session_id
        except KeyError:
            label = session_id

        f_session = all_f_t2p[i]
        dff_per_cell = np.vstack([compute_dff(f_session[c, :]) for c in range(f_session.shape[0])])
        condition_data[label] = dff_per_cell.mean(axis=1)

    # find labels that match saline / high / low (case-insensitive)
    def _find(keyword):
        for lbl in condition_data:
            if keyword.lower() in lbl.lower():
                return lbl
        return None

    saline_lbl = _find('saline')
    high_lbl   = _find('high')
    low_lbl    = _find('low')

    pairs = [
        (saline_lbl, high_lbl,  'Saline vs High'),
        (saline_lbl, low_lbl,   'Saline vs Low'),
        (low_lbl,    high_lbl,  'Low vs High'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (lbl_a, lbl_b, title) in zip(axes, pairs):
        if lbl_a is None or lbl_b is None:
            ax.set_title(f'{title}\n(data not found)')
            continue
        _bar_pair(ax, condition_data[lbl_a], condition_data[lbl_b], lbl_a, lbl_b, title)

    fig.suptitle(f'Pairwise mean dF/F comparisons — {subject}', y=1.02)
    plt.tight_layout()
    plt.show()


plot_pairwise_dff(all_f_t2p, track_ops, dataset, subject='ACUTEVIS06')

# %%
def plot_directional_dff(all_f_t2p, track_ops, dataset, subject='ACUTEVIS06', task='task-gratings'):
    """Six bar graphs splitting ROIs by direction of change (increase vs decrease).

    For each of the 3 pairwise comparisons (Saline→High, Saline→Low, Low→High),
    produces two subplots: one for cells that increased, one for cells that decreased.
    Layout: 2 rows × 3 columns (row 0 = increasing, row 1 = decreasing).
    """
    import re

    condition_data = {}

    for i, ds_path in enumerate(track_ops.all_ds_path):
        ses_match = re.search(r'ses-\d+', ds_path)
        session_id = ses_match.group(0) if ses_match else f'ses-{i+1:02d}'

        key = (subject, session_id, task)
        try:
            injection = dataset.loc[key, ('session_config', 'injection')]
            label = str(injection) if pd.notna(injection) else session_id
        except KeyError:
            label = session_id

        f_session = all_f_t2p[i]
        dff_per_cell = np.vstack([compute_dff(f_session[c, :]) for c in range(f_session.shape[0])])
        condition_data[label] = dff_per_cell.mean(axis=1)

    def _find(keyword):
        for lbl in condition_data:
            if keyword.lower() in lbl.lower():
                return lbl
        return None

    saline_lbl = _find('saline')
    high_lbl   = _find('high')
    low_lbl    = _find('low')

    pairs = [
        (saline_lbl, high_lbl, 'Saline → High'),
        (saline_lbl, low_lbl,  'Saline → Low'),
        (low_lbl,    high_lbl, 'Low → High'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    row_titles = ['Increasing ROIs', 'Decreasing ROIs']

    for col, (lbl_a, lbl_b, pair_title) in enumerate(pairs):
        for row in range(2):
            ax = axes[row, col]
            ax.set_title(f'{pair_title}\n({row_titles[row]})')
            ax.set_ylabel('Mean dF/F')

            if lbl_a is None or lbl_b is None:
                ax.text(0.5, 0.5, 'data not found', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            vals_a = condition_data[lbl_a]
            vals_b = condition_data[lbl_b]

            if row == 0:
                mask = vals_b > vals_a   # increased from A to B
                color = 'tomato'
            else:
                mask = vals_b <= vals_a  # decreased (or flat) from A to B
                color = 'cornflowerblue'

            sub_a = vals_a[mask]
            sub_b = vals_b[mask]
            n = len(sub_a)

            if n == 0:
                ax.text(0.5, 0.5, 'no cells', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([lbl_a, lbl_b])
                continue

            means = [sub_a.mean(), sub_b.mean()]
            sems  = [sub_a.std() / np.sqrt(n), sub_b.std() / np.sqrt(n)]
            jitter = np.random.uniform(-0.15, 0.15, size=(n, 2))

            ax.bar([0, 1], means, yerr=sems, capsize=5, color=color, alpha=0.6)
            ax.scatter(0 + jitter[:, 0], sub_a, color='black', s=6, alpha=0.4, zorder=3)
            ax.scatter(1 + jitter[:, 1], sub_b, color='black', s=6, alpha=0.4, zorder=3)
            for c in range(n):
                ax.plot([jitter[c, 0], 1 + jitter[c, 1]], [sub_a[c], sub_b[c]],
                        color='black', lw=0.5, alpha=0.2, zorder=2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([lbl_a, lbl_b])
            ax.set_xlabel(f'n = {n} cells')

    fig.suptitle(f'Directional dF/F changes — {subject}', y=1.01)
    plt.tight_layout()
    plt.show()


plot_directional_dff(all_f_t2p, track_ops, dataset, subject='ACUTEVIS06')

# %%
