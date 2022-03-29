# -*- coding: utf-8 -*-
"""
This code calculates the LFP power changement of different frequency at a certain time epoch
around ripples.
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
windowLg = 0.1

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
'''
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
Then the calculation of spectrogram here
'''
#set the parameters of multitaper
Fs = 1250.0  #Sampling Frequency
frequency_range = [10, 205]  #Limit frequencies from 0 to 25 Hz or 25 to 200Hz
#Time bandwidth and number of tapers
time_bandwidth=5
num_tapers=3
window_params = [windowLg, 0.001]  #Window size is 1s with step size of 0.01s
min_nfft = 0  #No minimum nfft
detrend_opt = 'linear'  #detrend each window by subtracting the average


#Z-score the dLfp
data = (dLfp-dLfp.mean(axis=0))/dLfp.std(axis=0)
data

spect, stimes, sfreqs = multitaper_spectrogram(data, Fs, frequency_range, time_bandwidth, num_tapers, window_params,
                           min_nfft, detrend_opt, multiprocess=False, cpus=False, plot_on = True, verbose = True)

from multitaper_spectrogram_python import nanpow2db
spectdb = nanpow2db(spect).T


#Plot the spectrogram
import librosa
import matplotlib.pyplot as plt
tt = stimes.A.ravel() 
ff = sfreqs.A.ravel()
plt.figure(1, figsize=(10, 5))
librosa.display.specshow(spectdb, x_axis='time', y_axis='linear',
                          x_coords=tt, y_coords=ff, shading='auto',
                          cmap="jet")
plt.colorbar(label='Power (dB)')
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Frequency (Hz)")
plt.vlines(rip_tsd.index.values-[startT], ymin=0, ymax=200, colors='k', linestyles='solid')
plt.show()


'''
Now we will calculates the LFP power changement of different frequency at a certain time epoch
around ripples
'''

#Get the center time of ripples
rip_ct = rip_tsd.index.values

#Decide the length of the time epoch we should use (the max rip_ep)
ripep = rip_ep.values
for i in range(len(rip_ep)):
    ripep[i,0] = ripep[i,1] - ripep[i,0]
    
max_rip_ep = max(ripep[:,0])

#Initialization and set the time index here
ct = rip_ct[0]
cst = ct - max_rip_ep/2 - startT
cet = ct + max_rip_ep/2 - startT
#Select the range of the power martix
idx = np.where((tt>=cst)&(tt<=cet))
ripower = np.zeros((len(idx),np.size(spect,1)))
ript = tt[idx] - (ct - startT)

#Now we calculate the averge power
for i in range(len(rip_ep)):
    ct = rip_ct[i]
    cst = ct - max_rip_ep/2 - startT
    cet = ct + max_rip_ep/2 - startT
    #Select the range of the power martix
    idx = np.where((tt>=cst)&(tt<=cet))
    if np.all(spect[idx,:]  == 0):
       print(i)
    else:
       ripower = spect[idx,:] + ripower
ripower = ripower.squeeze()
ripower = (ripower-ripower.mean(axis=0))/ripower.std(axis=0)
ripower = ripower.T



#Plot the spectrogram of power around ripples
import librosa
import matplotlib.pyplot as plt
plt.figure(1, figsize=(10, 5))
librosa.display.specshow(ripower, x_axis='time', y_axis='linear',
                          x_coords=ript, y_coords=ff, shading='auto',
                          cmap="jet")
plt.colorbar(label='Power (Z-scored)')
plt.xlabel("Time (HH:MM:SS)")
plt.ylabel("Frequency (Hz)")
plt.vlines([0], ymin=0, ymax=200, colors='k', linestyles='solid')
plt.show()

#Sum the power in the specified frequency interval
Low_freq = 100
High_freq = 200
row = np.where((ff>=Low_freq)&(ff<=High_freq))
ripower_freq = ripower[row,:]
ripower_freq = ripower_freq.squeeze()
ripower_freq = ripower_freq.sum[0]