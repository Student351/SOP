# -*- coding: utf-8 -*-
"""
This function is designed to culculate the error between
decodeHD and trueHD with the given step size and window length.
Notice that the input decodeHD and trueHD must be in tsd format.
"""



def HDerror(decodeHD,trueHD,window,step=1):

    import numpy as np
    import pandas as pd
    import math
    pi = math.pi
    
    
    tt = decodeHD.index.values
    dataHD1 = decodeHD.values
    dataHD2 = trueHD.values
    
    dataHD1 = dataHD1[None]
    dataHD2 = dataHD2[None]
    dataHD1 = np.transpose(dataHD1)
    dataHD2 = np.transpose(dataHD2)
    error = np.zeros(len(tt))
    
    
    if np.mod(window,2) == 0:
        os1 = np.zeros((1,int(window/2-1)))
        os2 = np.zeros((1,int(window/2)))
        dataHD1 = np.hstack((os1,dataHD1.T,os2)).T
        dataHD2 = np.hstack((os1,dataHD2.T,os2)).T
        for i in range(len(tt)):
            d = 0
            ot = np.arange(i*step, i*step+window)
            for tt in ot:
                daa = np.absolute(np.mod(dataHD1[tt]-dataHD2[tt],2*pi))
                daa[daa>pi] = 2*pi - daa[daa>pi]
                d = d + daa
            d = d/window
            error[i] = d
    
    
    elif np.mod(window,2) != 0:
         os = np.zeros((1,int((window-1)/2)))
         dataHD1 = np.hstack((os,dataHD1.T,os)).T
         dataHD2 = np.hstack((os,dataHD2.T,os)).T
         for i in range(len(tt)):
             d = 0;
             ot = np.arange(i*step, i*step+window)
             for tt in ot:
                 daa = np.absolute(np.mod(dataHD1[tt]-dataHD2[tt],2*pi))
                 daa[daa>pi] = 2*pi - daa[daa>pi]
                 d = d + daa
             d = d/window
             error[i] = d
    
    
    error = np.tsd(t=tt, d = error, time_units = 's')

    return error

