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

# Import preprocessed data
directory = 'Datasets/TMS_EEG/derivatives/2sepoch_nobl_tagged'
pathlist = list(Path(directory).glob('*.fif'))
montage = mne.channels.make_standard_montage('standard_1020')

# Filter data and identify channels showing negativity
evoked_gr = {}
channels = {}
for sample, pathlist in samples.items():
    evoked = []
    for path in pathlist:
        raw = mne.io.read_raw_fif(path, preload=True)
        new_names = dict((ch_name, ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp')) for ch_name in raw.ch_names)
        raw.rename_channels(new_names)
        raw.set_montage(montage)
        raw.interpolate_bads()
        raw.filter(None,2)
        events, event_ids = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, [3], baseline=(None,-1), tmin=-2, tmax=2, reject_by_annotation=True, picks=None) #'S  3'
        evoked.append(epochs.average())
    evoked_gr[sample] = mne.grand_average(evoked)
    evoked_cxt = evoked_gr[sample].get_data()
    evoked_std_t = np.std(evoked_cxt, axis = 0)
    peak_std = np.where(evoked_std_t == np.max(evoked_std_t[:1800]))
    evoked_peak_c = evoked_cxt[:,peak_std]
    mdiff_c = evoked_peak_c.mean() - evoked_peak_c
    group1 = np.where(mdiff_c < np.quantile(mdiff_c, 0.2))[0]
    group2 = np.where(mdiff_c > np.quantile(mdiff_c, 0.8))[0]
    channels[sample] = group2

# plot ERP
fig, axs = plt.subplots(nrows=3, ncols=3, sharex='col', figsize=(16,10), gridspec_kw={'width_ratios': [3, 1, 1]})
fig.suptitle('Stimulus Preceding Negativity', y=1.02)
for i, sample in enumerate(samples):
    evoked = evoked_gr[sample]
    chans = channels[sample]
    evoked.plot(picks=chans, spatial_colors=True, axes=axs[i,0], show=False)
    evoked.plot_topomap(times=-0.5, average=1, axes=[axs[i,1], axs[i,2]], show=False)
    axs[i,0].axvline(0, alpha=0.5, color=pal['English Violet'])
    axs[i,0].set_title(sample, loc='right')
fig.savefig(f'Derivatives/Mtsb_TMS/figures/SPN_3x2.png', bbox_inches='tight', dpi=300)
   