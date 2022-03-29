# -*- coding: utf-8 -*-
"""
This code is written to calculate trueHD and decodedHD, and the smoothed error 
between them.
"""

import numpy as np
import pandas as pd
import pynapple as nap



#to plot HDerror and smooth it
decodeHD = decoded
trueHD = position['ry']
window = 11
step1 = 1



#calculate the error with set window and step
from HDerror import HDerror
error1 = HDerror(decodeHD,trueHD,window,step1)



import matplotlib.pyplot as plt
#plot the error
plt.figure()
plt.plot(error1.as_units('s'), color = 'red')
# always label the axes
plt.xlabel("Time")
plt.ylabel("Error")
# a title
plt.title('Error between decodeHD and trueHD ')
# and display the figure
plt.show()
