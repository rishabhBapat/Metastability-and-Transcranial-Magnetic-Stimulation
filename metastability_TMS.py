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
def coherence_timeseries(phase_timeseries): #voltage_epochxchannelxtime to coherence_epochxtime
    theta_nxt = np.angle(phase_timeseries)
    average_x_t = np.mean(np.cos(theta_nxt), axis = 1)
    average_y_t = np.mean(np.sin(theta_nxt), axis = 1)
    coherence_t = (average_x_t ** 2 + average_y_t ** 2) ** 0.5
    return coherence_t

def metastability_timeseries(coherence_timeseries, window_size, type): #coherence_epochxtime to metastability_epochxtime
    if type == 'sliding_future':
        indexes = np.arange(coherence_timeseries.shape[1] - window_size)
        meta_t = np.zeros((coherence_timeseries.shape[0], indexes.shape[0]))
        for i in indexes:
            meta_t[:,i] = np.std(coherence_timeseries[:, i:i+window_size], axis=1)
    elif type == 'sliding_past':
        indexes = np.arange(window_size, coherence_timeseries.shape[1])
        meta_t = np.zeros((coherence_timeseries.shape[0], indexes.shape[0] + window_size))
        for i in indexes:
            meta_t[:,i] = np.std(coherence_timeseries[:, i-window_size:i], axis=1)
    elif type == 'sliding_sym':
        interval = int(window_size / 2)
        indexes = np.arange(interval, coherence_timeseries.shape[1] - interval)
        meta_t = np.zeros((coherence_timeseries.shape[0], indexes.shape[0] + interval))
        for i in indexes:
            i = int(i)
            start = int(i - interval)
            end = int(i + interval)
            meta_t[:,i] = np.std(coherence_timeseries[:, start:end], axis=1)
    elif type == 'sliding_pulse_start_sym':
        interval = int(window_size / 2)
        pulse = (coherence_timeseries.shape[1] - 1) / 2
        first_indexes = np.arange(interval, pulse - interval)
        second_indexes = np.arange(pulse + interval, coherence_timeseries.shape[1] - interval)
        indexes = np.concatenate((first_indexes, second_indexes))
        meta_t = np.zeros((coherence_timeseries.shape[0], indexes))
        for i in indexes:
            i = int(i)
            start = int(i - interval)
            end = int(i + interval)
            meta_t[:,i] = np.std(coherence_timeseries[:, start:end], axis=1)
    elif type == 'sliding_pulse_start':
        pulse = (coherence_timeseries.shape[1] - 1) / 2
        indexes = np.arange(window_size, coherence_timeseries.shape[1] - window_size)
        meta_t = np.zeros((coherence_timeseries.shape[0], indexes.shape[0]))
        for i, index in enumerate(indexes):
            if index < pulse:
                meta_t[:,i] = np.std(coherence_timeseries[:, int(index-window_size):int(index)], axis=1)
            else:
                meta_t[:,i] = np.std(coherence_timeseries[:, int(index):int(index+window_size)], axis=1)
    elif type == 'binned':
        pulse = (coherence_timeseries.shape[1] - 1) / 2
        bins = int(coherence_timeseries.shape[1] / window_size)
        meta_t = np.zeros((coherence_timeseries.shape[0], bins))
        for i in range(bins):
            start = int(i * window_size)
            end = int(start + window_size)
            meta_t[:,i] = np.std(coherence_timeseries[:, start:end], axis=1)        
    return meta_t

# Import preprocessed data
directory120 = 'Datasets/TMS_EEG/derivatives/2sepoch_nobl_tagged' #120% RMT
pathlist120 = list(Path(directory120).glob('*.fif'))

directory110 = 'Datasets/TMS_EEG_2/derivatives/2sepoch_nobl_tagged/14ms' #110% RMT
pathlist110 = list(Path(directory110).glob(f'*110rmt_eeg.fif'))

directory100 = 'Datasets/TMS_EEG_2/derivatives/2sepoch_nobl_tagged/14ms' #100% RMT
pathlist100 = list(Path(directory100).glob(f'*100rmt_eeg.fif'))

montage = mne.channels.make_standard_montage('standard_1005')
samples = {'120%RMT': pathlist120, '110%RMT': pathlist110, '100%RMT': pathlist100}

# Derive channel groups
all_sample_groups = {}
for sample, pathlist in samples.items():
    evoked = []
    for path in pathlist:
        raw = mne.io.read_raw_fif(path, preload=True)
        new_names = dict((ch_name, ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp')) for ch_name in raw.ch_names)
        raw.rename_channels(new_names)
        raw.set_montage(montage)
        raw.interpolate_bads()
        events, event_ids = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, [3], baseline=(None,-1), tmin=-2, tmax=2) #'S  3'
        evoked.append(epochs.average())
    evoked_gr = mne.grand_average(evoked)
    evoked_cxt = evoked_gr.get_data()
    evoked_std_t = np.std(evoked_cxt, axis = 0)
    peak_std = np.where(evoked_std_t == np.max(evoked_std_t))
    evoked_peak_c = evoked_cxt[:,peak_std]
    mdiff_c = evoked_peak_c.mean() - evoked_peak_c
    channel_fraction = 10/mdiff_c.shape[0]
    group1 = np.where(mdiff_c < np.quantile(mdiff_c, channel_fraction))[0]
    group2 = np.where(mdiff_c > np.quantile(mdiff_c, 1 - channel_fraction))[0]
    all_sample_groups[f'{sample}_group1'] = group1
    all_sample_groups[f'{sample}_group2'] = group2
    all_sample_groups[f'{sample}_all'] = None
    if sum(group1) > sum(group2):
        all_sample_groups[f'{sample}_group1'] = group2
        all_sample_groups[f'{sample}_group2'] = group1
    evoked_gr.plot(gfp = True, spatial_colors = True)
    evoked_gr.plot(gfp = True, spatial_colors = True, picks=group1)
    evoked_gr.plot(gfp = True, spatial_colors = True, picks=group2)

# Calculate metastabilty and coherence
frequencies = {'all':None, 'delta':[1,3], 'theta':[4,7], 'alpha':[8,12], 'beta':[15,30], 'gamma':[35,42]}
window = 100
epoch_avg_coherence = {}
epoch_avg_meta = {}
gr_coherences = {}
gr_meta = {}
meta_std = {}
for sample, pathlist in samples.items():
    sample_groups = {name:group for name, group in all_sample_groups.items() if sample in name}
    for index, path in enumerate(pathlist):
        raw = mne.io.read_raw_fif(path, preload=True)
        new_names = dict((ch_name, ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp')) for ch_name in raw.ch_names)
        raw.rename_channels(new_names)
        raw.set_montage(montage)
        raw.interpolate_bads()
        events, event_ids = mne.events_from_annotations(raw)
        print(f'retreived participant {index}')
        for freq, value in frequencies.items():
            if freq != 'all':    
                raw_filt = raw.copy().filter(value[0], value[1])#, l_trans_bandwidth=value[1]*0.25, h_trans_bandwidth=value[1]*0.25
            else:
                raw_filt = raw.copy()
            raw_filt.apply_hilbert()
            epochs = mne.Epochs(raw_filt, events, [3], baseline=(None,-1), tmin=-2, tmax=2, preload=True) #10001
            print(f'{freq} data transformed and epoched')
            for group in ['group1', 'group2', 'all']:
                voltage_excxt = epochs.get_data(picks=sample_groups[f'{sample}_{group}'])
                coherence_ext = coherence_timeseries(voltage_excxt)
                coherence_avg_t = coherence_ext.mean(axis=0)
                meta_ext = metastability_timeseries(coherence_ext, window, 'sliding_past')
                meta_avg_t = meta_ext.mean(axis=0)
                epoch_avg_coherence[f'{sample}_{index}_{freq}_{group}'] = coherence_avg_t
                epoch_avg_meta[f'{sample}_{index}_{freq}_{group}'] = meta_avg_t
                print(f'grouped and epoched data of shape: {voltage_excxt.shape}') 
                print(f'epoch-wise coherence of shape: {coherence_ext.shape} epoch averaged coherence of shape: {coherence_avg_t.shape}')
                print(f'epoch-wise metastability of shape: {meta_ext.shape} epoch averaged metastability of shape: {meta_avg_t.shape}')

    # Subject averaging
    meta_arr = np.zeros([len(pathlist), epoch_avg_meta[f'{sample}_0_all_all'][epoch_avg_meta[f'{sample}_0_all_all'] != 0].shape[0]])
    for frequency in frequencies:
        for group in ['group1', 'group2', 'all']:
            keys = [key for key in epoch_avg_meta.keys() if key.split('_')[0] == sample and key.split('_')[2] == frequency and key.split('_')[3] == group]
            gr_avg_meta = np.zeros(epoch_avg_meta[f'{sample}_0_alpha_all'].shape[0])
            gr_avg_coherence = np.zeros(epoch_avg_coherence[f'{sample}_0_alpha_all'].shape[0])
            print(len(keys), f'{frequency}_{group}')
            for i, key in enumerate(keys):
                gr_avg_meta += epoch_avg_meta[key]
                gr_avg_coherence += epoch_avg_coherence[key]
                meta_arr[i,:] = epoch_avg_meta[key][epoch_avg_meta[key] != 0]
            gr_avg_meta /= len(keys)
            gr_avg_coherence /= len(keys)
            gr_coherences[f'{sample}_{frequency}_{group}'] = gr_avg_coherence
            gr_meta[f'{sample}_{frequency}_{group}'] = gr_avg_meta
            meta_std[f'{sample}_{frequency}_{group}'] = np.std(meta_arr, axis=0)

# Calculate statistical significance (adjust windows as needed)
samples = ['120%RMT', '110%RMT', '100%RMT']
normality = {}
significance = {}
for sample in samples:
    for frequency in frequencies:
        for group in ['all', 'group1', 'group2']:
            keys = [key for key in epoch_avg_meta if key.split('_')[0] == sample and key.split('_')[2] == frequency and key.split('_')[3] == group]
            pre_250 = np.zeros((len(keys), 250))
            post_250 = np.zeros((len(keys), 250))
            for i, key in enumerate(keys):
                pre_250[i,:] = epoch_avg_meta[key][1500:1750]
                post_250[i,:] = epoch_avg_meta[key][1750:2000]
            pre_means = pre_250.mean(axis=1)
            post_means = post_250.mean(axis=1)
            print(f'{pre_250.shape}, {post_250.shape}, {pre_means.shape}, {post_means.shape}')
            normal = scipy.stats.shapiro(pre_means)[1] > 0.05 and scipy.stats.shapiro(post_means)[1] > 0.05
            result = scipy.stats.wilcoxon(pre_means, post_means, axis=0, alternative='less')
            significance[f'{sample}_{frequency}_{group}'] = {'tstat': result.statistic, 'pvalue': result.pvalue, 'normality': normal}

# Plot metastability and coherence traces
pulse = 1000
baseline = [525,1525]
window= 50
newcycler = plt.cycler("color", plt.cm.plasma(np.linspace(0,1,6)))
for group in ['all', 'group1', 'group2']:
    for stim in ['120%RMT', '110%RMT', '100%RMT']:
        path = f'/media/cbdl/3c41c483-38c8-41c2-bc4f-d258e2f45c22/home/cbdl/Downloads/Mtsb_TMS/figures/test/hms/{group}_{stim}.png'
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=(16,10), tight_layout=True)
        axs[1].set_xlabel('Time(ms)', size=18)
        axs[1].set_ylabel(f'Metastability (Baselined)', size=18)
        axs[0].set_ylabel(f'KOP (Baselined)', size=18)
        axs[1].axvline(pulse, label='_pulse',color=pal['English Violet'], alpha=0.8)
        axs[0].axvline(pulse, label='_pulse',color=pal['English Violet'], alpha=0.8)
        axs[1].locator_params('y',nbins=5)
        axs[0].locator_params('y',nbins=5)
        axs[0].set_prop_cycle(newcycler)
        axs[1].set_prop_cycle(newcycler)
        keys = [key for key in gr_meta.keys() if key.split('_')[0] == stim and key.split('_')[2] == group]
        for i, key in enumerate(keys):
            frequency = key.split('_')[1]
            baseline_meta = gr_meta[key][baseline[0]:baseline[1]].mean()
            baselined_meta = (gr_meta[key][gr_meta[key] != 0] / baseline_meta) * 100
            baseline_coh = gr_coherences[key][baseline[0]:baseline[1]].mean()
            baselined_coh = (gr_coherences[key][gr_coherences[key] != 0] / baseline_coh) * 100
            meta_x = np.arange(int(window/2), gr_coherences[key].shape[0] - int(window/2))
            axs[0].plot(baselined_coh)
            axs[1].plot(meta_x, baselined_meta, label=frequency)
        axs[1].legend(loc='lower right',fontsize=15)
        fig.savefig(path, bbox_inches='tight')
