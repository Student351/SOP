# -*- coding: utf-8 -*-
"""
This code calculates the spectrogram of LFP in hippocampus, and plot it with 
the HDerror curve. This helps us to find the connection between Spike and LPF
"""

import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *

#load the function
#import sys
from multitaper_spectrogram_python import multitaper_spectrogram#to use this, you must install 'librosa'

#set the path to the data
path = 'C:/Users/Karl/McGill University/peyrache_Group - Yuxiao-SleepHDCells/Data/Mouse12-120806/Mouse12-120806/Mouse12-120806.eeg'
#load LFP data
#lfp = nap.load_lfp(path, filename='Mouse12-120806', channel=64, extension='.eeg', frequency=1250.0, precision='int16', bytes_size=2)
lfp = nap.loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision='int16')


ttlfp = lfp.index.values
ddlfp = lfp.values
lfp = nap.Tsd(t = ttlfp, d = ddlfp, time_units = 'us')


#length of the window
windowLg = 1.0

#set the calculating epoch
startT = [560]
endT = [600]
ep = nap.IntervalSet(startT, endT, time_units = 's')
ep

#restrict the lfp data
lfpEp = lfp.restrict(ep)
t = lfpEp.index.values
dLfp = lfpEp.values


#plot lfpEp here
plot(lfpEp.as_units('us'), '-')
show()

'''
If necessary, we detect ripples here

import pynacollada as pyna
rip_ep, rip_tsd = pyna.eeg_processing.detect_oscillatory_events(
                                            lfp = lfpEp,
                                            epoch = ep,
                                            freq_band = (100,300),
                                            thres_band = (1, 10),
                                            duration_band = (0.02,0.2),
                                            min_inter_duration = 0.02
                                            )

print(rip_tsd)
'''

'''
Then the calculation of spectrogram here
'''
#set the parameters of multitaper
Fs = 1250.0  #Sampling Frequency
frequency_range = [25, 200]  #Limit frequencies from 0 to 25 Hz or 25 to 200Hz
#Time bandwidth and number of tapers
time_bandwidth=5
num_tapers=3
window_params = [windowLg, 0.25]  #Window size is 1s with step size of 0.01s
min_nfft = 0  #No minimum nfft
detrend_opt = 'linear'  #detrend each window by subtracting the average


#get the HDerror (remember to check the type: DTW/error)
aT = np.array(startT)
bT = np.array(endT)
epoch = nap.IntervalSet(aT + windowLg/2, bT - windowLg/2, time_units = 's')
#bineddtw = error1.restrict(epoch)


#Z-score the dLfp
data = (dLfp-dLfp.mean(axis=0))/dLfp.std(axis=0)
data

spect, stimes, sfreqs = multitaper_spectrogram(data, Fs, frequency_range, time_bandwidth, num_tapers, window_params,
                           min_nfft, detrend_opt, multiprocess=False, cpus=False, plot_on = True, verbose = True)

from multitaper_spectrogram_python import nanpow2db
spect = nanpow2db(spect).T


#Plot the spectrogram
import librosa
import matplotlib.pyplot as plt
tt = stimes.A.ravel()
ff = sfreqs.A.ravel()
plt.figure(1, figsize=(10, 5))
librosa.display.specshow(spect, x_axis='time', y_axis='linear',
                          x_coords=tt, y_coords=ff, shading='auto',
                          cmap="jet")
plt.colorbar(label='Power (dB)')
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Frequency (Hz)")
#plt.vlines(rip_tsd.index.values-[startT], ymin=25, ymax=200, colors='k', linestyles='solid')
plt.show()
