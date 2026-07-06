import numpy as np
from numba import jit
from time import perf_counter as clock

#timer
def timed(func):
    def wrapper(*args, **kwargs):
        tic = clock()
        result = func(*args, **kwargs)
        toc = clock()
        print(f'{func.__name__} took {toc-tic} s')
        return result
    return wrapper

#optimised core functions with JIT compilation
@jit('float64[:,:](float64[:,:], float64[:])', nopython=True, cache=True, parallel=False)
def coupling(theta_then, theta_now):
    return theta_then - theta_now[:,np.newaxis]

@jit('float64[:](float64[:,:], float64[:], int64, float64[:], float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:])', nopython=True, cache=True, parallel=True)
def _derivative(theta_then, theta_now, t, omega_n, normalised_coupling, adjacency_nxn, noise_nxt, stim_coupling_nxt, stim_nxt): #finds derivative of each oscillator at time 't'
    return omega_n + normalised_coupling * np.sum(adjacency_nxn * np.sin(coupling(theta_then, theta_now)), axis = 1) + noise_nxt[:,t] + stim_coupling_nxt[:,t] * np.sin(stim_nxt[:,t] - theta_now)

@jit('float64[:](float64[:,:], int64, int64, float64[:], float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:])', nopython=True, cache=True, parallel=True)    
def _fast_derivative(theta_nxt, t, delay, omega_n, normalised_coupling, adjacency_nxn, noise_nxt, stim_coupling_nxt, stim_nxt): #finds derivative of each oscillator at time 't'
    theta_then = theta_nxt[:,t-delay].repeat(len(omega_n)).reshape((-1,len(omega_n))).T
    return omega_n + normalised_coupling * np.sum(adjacency_nxn * np.sin(coupling(theta_then,theta_nxt[:,t])), axis = 1) + noise_nxt[:,t] + stim_coupling_nxt[:,t] * np.sin(stim_nxt[:,t] - theta_nxt[:,t])

#class that implements kuramoto model for use in whole brain modelling
class Kuramoto:

    def __init__(self, n, k, cv, timespan, dt, dist_nxn, noise = None, omega_n = None, adjacency_nxn = None, seed=None):
        self.n = n #number of neurons
        self.k = k #coupling constant
        self.cv = cv #conduction velocity
        self.normalised_coupling = k / n 
        self.timeseries = np.arange(0, timespan, dt) #timepoints at which to plot results
        self.dt = dt #time step
        self.delays_nxn = (np.round(dist_nxn / self.cv, 3) / dt).astype('int') #delays in unit dt
        self.dist_nxn = dist_nxn
        self.stim_type = None
        self.stim_nxt = np.zeros((self.n, len(self.timeseries)))
        self.stim_coupling_nxt = np.zeros((self.n, len(self.timeseries)))
        self.flags = []
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng(seed=42)
        if noise is not None:
            self.noise = noise #noise amplitude
        else:
            self.noise = 0
        if omega_n is not None: 
            self.omega_n = omega_n #intrinsic frequencies
        else:
            self.omega_n = abs(np.random.normal(loc = 1, scale = 0.1, size = n))
        if adjacency_nxn is not None:
            self.adjacency_nxn = adjacency_nxn #adjacency matrix
        else:
            adjacency_nxn = np.ones([n,n])
            np.fill_diagonal(adjacency_nxn, 0)
            self.adjacency_nxn = adjacency_nxn
    
    def make_stim(self, stim_time, stim_indices, stim_locus=None, pulse_dur=1, pulse_amp=1, pulse_freq=3, stim_type='reset'):
        self.stim_type = stim_type
        if stim_type == 'prop_reset':
            locus_delays_n = self.delays_nxn[stim_locus,stim_indices].squeeze()
            del_stimtimes_n = locus_delays_n + stim_time
            stim_nxt = np.zeros((self.n,len(self.timeseries))) #initialise stimulus array
            for i in stim_indices:
                stim_nxt[i,int(del_stimtimes_n[i]): int(del_stimtimes_n[i] + pulse_dur)] = 1
            self.stim_nxt = stim_nxt
            return stim_nxt
        
        if stim_type == 'core_prop_reset':
            locus_delays_n = self.delays_nxn[stim_locus,:].squeeze()
            del_stimtimes_n = locus_delays_n + stim_time
            stim_nxt = np.zeros((self.n,len(self.timeseries))) #initialise stimulus array
            for i,delay in enumerate(del_stimtimes_n):
                stim_nxt[i,int(delay): int(delay + pulse_dur)] = 1
            stim_nxt[stim_indices, stim_time:stim_time+pulse_dur] = 1
            stim_nxt[stim_indices, stim_time+pulse_dur+1:] = 0
            self.stim_nxt = stim_nxt
            return stim_nxt

        elif stim_type == 'reset':
            stim_nxt = np.zeros((self.n,len(self.timeseries))) #initialise stimulus array
            stim_nxt[stim_indices, stim_time:stim_time+pulse_dur] = 1
            self.stim_nxt = stim_nxt
            return stim_nxt

        elif stim_type == 'sine':
            stim_nxt = np.zeros((self.n,len(self.timeseries))) #initialise stimulus array
            self.stim_coupling_nxt = np.zeros_like(stim_nxt)
            if type(stim_time) == int:
                self.flags.append('single stimulus onset')
                self.stim_coupling_nxt[stim_indices,stim_time:stim_time+pulse_dur] = pulse_amp
                stim_nxt[stim_indices, stim_time:stim_time+pulse_dur] = np.sin(2 * np.pi * pulse_freq * np.linspace(0,pulse_dur*self.dt,pulse_dur))
            else:
                for t in stim_time:
                    self.flags.append('multiple stimulus onsets')
                    self.stim_coupling_nxt[stim_indices,t:t+pulse_dur] = pulse_amp
                    stim_nxt[stim_indices, t:t+pulse_dur] = np.sin(2 * np.pi * pulse_freq * np.linspace(0,pulse_dur*self.dt,pulse_dur))
            self.stim_nxt = stim_nxt
            return stim_nxt

    def derivative(self, theta_nxt, t): #finds derivative of each oscillator at time 't'
        theta_then = theta_nxt[np.arange(0,self.n,dtype=int), t - self.delays_nxn] #coordinate matrix with delayed values of theta
        return _derivative(theta_then, theta_nxt[:,t], t, self.omega_n, self.normalised_coupling, self.adjacency_nxn, self.noise_nxt, self.stim_coupling_nxt, self.stim_nxt)
    
    def fast_derivative(self, theta_nxt, t): #finds derivative of each oscillator at time 't'
        return _fast_derivative(theta_nxt, t, self.delays_nxn[0,0], self.omega_n, self.normalised_coupling, self.adjacency_nxn, self.noise_nxt, self.stim_coupling_nxt, self.stim_nxt)
    
    @timed
    def phase_timeseries(self):
        theta_nxt = 2 * np.pi * self.rng.random((self.n, len(self.timeseries))) #initialise with random values so system can run with delay at t=0
        self.noise_nxt = self.noise * self.rng.standard_normal((self.n,len(self.timeseries)))
        if (self.delays_nxn == self.delays_nxn[0,0]).all():
            delay = self.delays_nxn[0,0]
            derivative = self.fast_derivative
            self.flags.append(f'used fast derivative with homogenous delay of {delay} indices')
        else:
            derivative = self.derivative
            self.flags.append('used slow derviative with heterogenous delay')
        if self.stim_type in ['reset','prop_reset','core_prop_reset']:
            self.flags.append('checked stim_nxt')
            for t in range(len(self.timeseries)):
                d_theta_n = derivative(theta_nxt, t-1)
                theta_nxt[:,t] = theta_nxt[:,t-1] + d_theta_n * self.dt #euler integration
                if self.stim_nxt[:,t].any(): 
                    theta_nxt[np.where(self.stim_nxt[:,t]),t] = np.pi/2
        else:
            self.flags.append('didnt check stim_nxt')
            for t in range(len(self.timeseries)):
                d_theta_n = derivative(theta_nxt, t-1)
                theta_nxt[:,t] = theta_nxt[:,t-1] + d_theta_n * self.dt
        return theta_nxt
    
    @timed
    def coherence_timeseries(self, theta_nxt=None):
        if theta_nxt is None:
            theta_nxt = self.phase_timeseries()
        average_x_t = np.mean(np.cos(theta_nxt), axis = 0)
        average_y_t = np.mean(np.sin(theta_nxt), axis = 0)
        coherence_t = (average_x_t ** 2 + average_y_t ** 2) ** 0.5 #calcuates coherence as the magnitude of the resultant vector at a given time
        return coherence_t

    @timed
    def metastability_timeseries(self, window_size=100, pulse=None, coherence_t=None):
        indexes = np.arange(window_size, len(coherence_t) - window_size)
        meta_t = np.zeros(len(coherence_t))
        for index in indexes:
            if index < pulse:
                meta_t[index] = np.std(coherence_t[int(index-window_size):int(index)])
            else:
                meta_t[index] = np.std(coherence_t[int(index):int(index+window_size)])
        return meta_t