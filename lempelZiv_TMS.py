# Import packages
import numpy as np
import pandas as pd
import mne
import re
import eeglabio
import scipy.stats
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation 
import pickle
from pathlib import Path
import seaborn as sns
mne.viz.use_browser_backend('qt')

# Define functions
def LZ(x):
    #run length encoding
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    runs = list(zip(lengths, values))
    #count unique runs
    no_dupes = set(runs)
    uniques = len(no_dupes)
    return uniques

def LZtimeseries(labels_t, window_size, type): #calculates running LZ complexity using provided increment
    timepoints = int(len(labels_t)/window_size)
    LZ_t = np.zeros(timepoints)
    if type == 'incremented':
        for i in range(timepoints):
            LZ_t[i] = LZ(labels_t[:window_size + window_size*i])
    if type == 'sliding':
        interval = int(window_size / 2)
        indexes = np.arange(interval, labels_t.shape[0] - interval)
        LZ_t = np.zeros(indexes.shape[0] + interval)
        for i in indexes:
            i = int(i)
            start = int(i - interval)
            end = int(i + interval)
            LZ_t[i] = LZ(labels_t[start:end])
    if type == 'binned':
        bins = int(labels_t.shape[0] / window_size)
        LZ_t = np.zeros(bins)
        for i in range(bins):
            LZ_t[i] = LZ(labels_t[i*window_size: i*window_size + window_size])
    return LZ_t

# Import matrices containing microstate results (produced in MATLAB using "microstates_TMS.mlx")
tep_directory = 'Datasets/TMS_EEG/derivatives/variables/tep_microstates/alpha'
tep_pathlist = list(Path(tep_directory).glob('*.mat'))

rest_directory = 'Datasets/TMS_EEG/derivatives/variables/resting_microstates/alpha'
rest_pathlist = list(Path(rest_directory).glob('*.mat'))

#match names
tep_names = [str(path).split('/')[-1].split('_')[0] for path in tep_pathlist]
rest_names = [str(path).split('/')[-1].split('_')[0] for path in rest_pathlist]
rest_final = []
tep_final = []
for i, name in enumerate(rest_names):
    if name in tep_names and name not in ['768','779']:
        rest_final.append(rest_pathlist[i])
        tep_final.append(tep_pathlist[i])
tep_final.sort()
rest_final.sort()

#load matrices in dict
tep_microstates = {}
resting_microstates = {}
for i, path in enumerate(tep_final):
    tep_microstates[i] = scipy.io.loadmat(path, simplify_cells=True)
    resting_microstates[i] = scipy.io.loadmat(rest_final[i], simplify_cells=True)

# Plot binned Lempel Ziv Complexity
bin_size = 100    
gr_tep_LZ_t = np.zeros((len(tep_microstates), int(tep_microstates[0]['fit']['labels'][:,1500:3000].shape[1]/bin_size)))
gr_rest_LZ_t = np.zeros_like(gr_tep_LZ_t)
for i in tep_microstates:
    if i not in [0,13,15]:
        tep_labels_ext = tep_microstates[i]['fit']['labels'][:,1500:3000]
        rest_labels_t = resting_microstates[i]['fit']['labels']
        tep_LZ_t = np.zeros((tep_labels_ext.shape[0], int(tep_labels_ext.shape[1]/bin_size)))
        rest_LZ_t = np.zeros_like(tep_LZ_t)
        for ii in range(tep_labels_ext.shape[0]):
            tep_LZ_t[ii,:] = LZtimeseries(tep_labels_ext[ii,:], bin_size, type='binned')
            r_start = np.random.choice((resting_microstates[i]['fit']['labels'].shape[0]-tep_labels_ext.shape[1]))
            r_rest = rest_labels_t[r_start:r_start+tep_labels_ext.shape[1]]
            rest_LZ_t[ii,:] = LZtimeseries(r_rest, bin_size, type='binned')
        gr_tep_LZ_t[i] = tep_LZ_t.mean(axis=0)
        gr_rest_LZ_t[i] = rest_LZ_t.mean(axis=0)
gr_tep_baselined = (gr_tep_LZ_t.mean(axis=0) / np.mean(gr_tep_LZ_t.mean(axis=0)[0:3])) * 100
gr_rest_baselined = (gr_rest_LZ_t.mean(axis=0) / np.mean(gr_rest_LZ_t.mean(axis=0)[0:3])) * 100
fig, axs = plt.subplots()
pulse = axs.axvline(500, label='Pulse', alpha=0.6, color=pal['English Violet'])
resting = axs.plot(np.arange(0,1500,100), gr_rest_baselined, alpha=0.8, color=pal['Tiffany Blue'], label='Resting')
tep = axs.plot(np.arange(0,1500,100), gr_tep_baselined, alpha=0.8, color=pal["Persimmon"], label='TMS evoked')
axs.set_xlabel('Time (ms)')
axs.set_ylabel('Lempel-Ziv Complexity (Baselined)')
blue_line = mlines.Line2D([],[],color=pal['Tiffany Blue'])
orange_line = mlines.Line2D([],[],color=pal['Persimmon'])
axs.legend([pulse, blue_line, orange_line], ['Pulse','Resting','TMS Evoked'])
fig.show()
fig.savefig(f'Derivatives/Mtsb_TMS/figures/LZ_gr.png')

# Plot microstate transition probabilities
path = 'Derivatives/Mtsb_TMS/figures/ms_heatmaps'
for i in tep_microstates:
    fig, axs = plt.subplots(1,3,figsize=[16,8],gridspec_kw={'width_ratios': [1, 1, 0.1]})
    stat1 = resting_microstates[i]['stats']['TP']
    stat2 = tep_microstates[i]['stats']['avgs']['TP']
    sns.heatmap(stat1, ax=axs[0], vmin=0, vmax=stat2.max(), cmap='mako', cbar=False)
    sns.heatmap(stat2, ax=axs[1], vmin=0, vmax=stat2.max(), cmap='mako', cbar_ax=axs[2])
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    axs[0].set_xlabel(f'Resting State', fontsize=18)
    axs[1].set_xlabel(f'TMS', fontsize=18)
    fig.savefig(path + f'/{i}_heatmap.png')

# Calculate statistical significance (adjust windows as needed)
LZbefore = np.zeros(len(tep_microstates))
LZafter = np.zeros(len(tep_microstates))
for i in range(len(tep_microstates)):
    before = tep_microstates[i]['fit']['labels'][:,1900:2000]
    after = tep_microstates[i]['fit']['labels'][:,2000:2100]
    LZbefore_epoch = np.zeros(before.shape[0])
    LZafter_epoch = np.zeros(after.shape[0])
    for ii in range(before.shape[0]):
        LZbefore_epoch[ii] = LZ(before[ii,:])
        LZafter_epoch[ii] = LZ(after[ii,:])
    LZbefore[i] = LZbefore_epoch.mean()
    LZafter[i] = LZafter_epoch.mean()
stat, pval = scipy.stats.wilcoxon(LZbefore, LZafter, alternative='greater')