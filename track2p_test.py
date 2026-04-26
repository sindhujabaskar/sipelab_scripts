#%%
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_pickle(r"G:\Projects\ACUTEVIS\picklejar\260212_ACUTEVIS_dataset.pkl")

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
