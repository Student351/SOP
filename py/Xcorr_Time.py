# -*- coding: utf-8 -*-
"""
This code shows how the amplitude at high frequency(80-200Hz) of the result of xcorr
changes with time, and the output is "xcorrwt".
"""
'''
Set the time range
start time: T1
end time: T2
'''
T1 = 2800;
T2 = 3800;


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
#set the parameters of multitaper
Fs = 1250.0  #Sampling Frequency
frequency_range = [25, 200]  #Limit frequencies from 0 to 25 Hz or 25 to 200Hz
#Time bandwidth and number of tapers
time_bandwidth=5
num_tapers=3
window_params = [windowLg, 0.25]  #Window size is 1s with step size of 0.01s
min_nfft = 0  #No minimum nfft
detrend_opt = 'linear'  #detrend each window by subtracting the average



#Calculate xcorr in different time epoch
for T in range(T1, T2+1, 5):
    #set the calculating epoch
    startT = [560]
    endT = [600]
    ep = nap.IntervalSet(startT, endT, time_units = 's')
    ep
    
    #restrict the lfp data
    lfpEp = lfp.restrict(ep)
    t = lfpEp.index.values
    dLfp = lfpEp.values
    
    
    #get the HDerror (remember to check the type: DTW/error)
    aT = np.array(startT)
    bT = np.array(endT)
    epoch = nap.IntervalSet(aT + windowLg/2, bT - windowLg/2, time_units = 's')
    error2 = error1.restrict(epoch)
    
    
    #Z-score the dLfp
    data = (dLfp-dLfp.mean(axis=0))/dLfp.std(axis=0)
    data
    
    spect, stimes, sfreqs = multitaper_spectrogram(data, Fs, frequency_range, time_bandwidth, num_tapers, window_params,
                               min_nfft, detrend_opt, multiprocess=False, cpus=False, plot_on = True, verbose = True)
    
    
    #It is very important to zscore the power
    spect = spect.T
    power = (spect-spect.mean(axis=0))/spect.std(axis=0)
    power = power.T
    
    
    #Interpolate the error to make sure it has the same sampling rate as power
    i = np.linspace(1, len(error2), np.size(power,1), endpoint=True)
    x = range(1, len(error2))
    y = error2.values
    ierror = np.interp(i, x, y)
    
    
    '''
    Calculate cross-covariance
    C defines the area we focus on
    '''
    
    '''
    If necessary:
    fp=len(stimes)/(endT-startT)
    epxcov = np.zeros((len(power[:,1]),fp*(endT-startT)+1))
    '''
    
    
    c = 40
    for ip in range(len(power[:,1])):
        epxcov[ip,:] = np.correlate(ierror, power[ip,:], mode='valid')
    
    cen = (np.size(epxcov,1)-1)/2
    cen1 = cen-c
    cen2 = cen+c
    epxcov = epxcov[:,cen1:cen2]
    
    #xcorrwt 
    xcorrwt[(T-T1)/10] = sum(epxcov[123:287,:])/np.sum(epxcov)
    
    
xcorrwt