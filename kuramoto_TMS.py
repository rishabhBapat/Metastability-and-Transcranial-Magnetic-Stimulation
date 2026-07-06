import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from kuramoto_model import Kuramoto
import seaborn as sns

# Define functions
def moving_avg(x, window_size):
    indexes = np.arange(window_size/2, len(x) - window_size/2).astype('int')
    avgx = np.zeros(len(x))
    for index in indexes:
        avgx[index] = np.mean(x[int(index-window_size):int(index)])
    return avgx

# load matrices
labels_n = np.array([label.strip() for label in scipy.io.loadmat(r'C:\Users\risha\Projects\Code\Mtsb_TMS_frontiers\AAL_Matrices\AAL_labels.mat')['label90']])
conn_matrix = scipy.io.loadmat(r'C:\Users\risha\Projects\Code\Mtsb_TMS_frontiers\AAL_Matrices\AAL_matrices.mat')

# setup parameters
conn_nxn = conn_matrix['C'] / np.max(conn_matrix['C']) #normalised connectivity
dist_nxn = conn_matrix['D'] / 1000 #cortical distances scaled to meters
conduction_velocity = 25 #conduction velocity in m/s
dt = 0.001 #time step in seconds
delays_nxn = np.round(dist_nxn / conduction_velocity, 3) / dt #delays in unit dt

# simulate effect of TMS for various frequency ranges
window= 500 # window for metastabilty calculation
epoch= 10000 # length of data to extract around pulse  
n_seeds= 20 # number of rng seeds to average over

#simple reset
locus = 'R Precentral'; locus_i = np.argwhere(labels_n == 'R Precentral')
pulse_range = int(1 * len(conn_nxn))
stim_indices = np.argsort(dist_nxn[locus_i,:]).squeeze()[:pulse_range]

#Efield reset
with open(r'C:\Users\risha\Projects\Derivatives\Kuramoto_model\variables\Efield_simulations\indices_95.pkl', "rb") as input_file:
   stim_indices = pickle.load(input_file)

freqs = np.arange(5,40,10)
S_n = np.sum(conn_nxn, axis=1) 
meta_t = {freq:None for freq in freqs}
coh_t = {freq:None for freq in freqs}
for freq in freqs:
    high = freq; low = high-4
    omega_n = 2 * np.pi * (high - (high - low) * (((S_n - np.min(S_n))/(np.min(S_n) - np.max(S_n)))**2)) #assigns intrinic frequencies using method from lit
    meta_seedxt = np.zeros((n_seeds,epoch*2))
    coh_seedxt = np.zeros((n_seeds,epoch*2))
    for i in range(n_seeds): # repeats anlaysis for multiple seeds
        # find troughs in signal
        model = Kuramoto(n=len(dist_nxn),cv=conduction_velocity,k=270,timespan=75,dt=dt,noise=3,adjacency_nxn=conn_nxn,dist_nxn=dist_nxn,omega_n=omega_n,seed=i)
        theta = model.phase_timeseries()
        coh_all = model.coherence_timeseries(theta)
        troughs = scipy.signal.find_peaks(-1*coh_all, height=np.quantile(-1*coh_all,0.9), distance=10000)[0]

        # stimulate at trough
        pulse = troughs[1]
        model = Kuramoto(n=len(dist_nxn),cv=conduction_velocity,k=270,timespan=75,dt=dt,noise=3,adjacency_nxn=conn_nxn,dist_nxn=dist_nxn,omega_n=omega_n,seed=i)
        stim_nxt = model.make_stim(stim_time=pulse, stim_indices=stim_indices, stim_locus=locus_i, pulse_dur=1, stim_type='core_prop_reset')
        theta = model.phase_timeseries()
        coh_all = model.coherence_timeseries(theta)

        # find metastabilty and coherence
        group = np.where(abs(omega_n - omega_n[1]) <= 2*abs(omega_n[1] - omega_n[0]))[0]
        coh_group = model.coherence_timeseries(theta[group,:])
        meta_group = model.metastability_timeseries(window, pulse, coherence_t=coh_group)
        meta_seedxt[i,:] = meta_group[pulse-epoch:pulse+epoch]
        coh_seedxt[i,:] = coh_group[pulse-epoch:pulse+epoch]
    
    # average over seeds
    meta_t[freq] = np.mean(meta_seedxt,axis=0) 
    coh_t[freq] = np.mean(coh_seedxt,axis=0)

# coherence traces
fig, axs = plt.subplots(figsize=[8,6], tight_layout=True)
newcycler = plt.cycler("color", plt.cm.plasma(np.linspace(0,1,4)))
axs.set_prop_cycle(newcycler)
for freq in meta_t:
    axs.plot(coh_t[freq][8000:18000], label=f'{freq} Hz')
axs.set_ylabel('Coherence')
axs.set_xlabel('Time (ms)')
axs.axvline(2000, color='r', linestyle='--',alpha=0.3)
axs.legend(loc='upper left')

# metastability traces
fig, axs = plt.subplots(figsize=[8,6], tight_layout=True)
newcycler = plt.cycler("color", plt.cm.plasma(np.linspace(0,1,4)))
axs.set_prop_cycle(newcycler)
for freq in meta_t:
    meta_ts = moving_avg(meta_t[freq][8000:18000],50)
    axs.plot(meta_ts[:-50], label=f'{freq} Hz')
axs.set_ylabel('Metastability')
axs.set_xlabel('Time (ms)')
axs.axvline(2000, color='r', linestyle='--', alpha=0.3)
axs.legend(loc='upper left')