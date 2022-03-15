# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-26 21:18:16
# @Last Modified by:   gviejo
# @Last Modified time: 2022-01-26 21:27:38
#!/usr/bin/env python

# # Quick start
# 
# The example data to replicate the figure in the jupyter notebook can be found here :
# https://www.dropbox.com/s/1kc0ulz7yudd9ru/A2929-200711.tar.gz?dl=1
# 
# The data contain a short sample of a simultaneous recording during sleep and wake 
# from the anterodorsal nucleus of the thalamus and the hippocampus. 
# It contains both head-direction cells (i.e. cells that fire for a particular direction in the horizontal plane) and place cells (i.e. cells that fire for a particular position in the environment).
# 
# Preprocessing of the data was made with Kilosort 2.0 and spike sorting was made with Klusters.
# 
# Instructions for installing pynapple can be found here : 
# https://peyrachelab.github.io/pynapple/#installation
# 
#
# 
# This tutorial is meant to provide an overview of pynapple by going through:
# 1. **Input output (IO)**. In this case, pynapple will load a session containing data.
# 2. **Core functions** that handle time series, interval sets and group of time series.
# 3. **Process functions**. A small collection of high-level functions widely used in system neuroscience.


import numpy as np
import pandas as pd
import pynapple as nap


# The first step is to give the path to the data folder.

data_directory = 'C:/Users/Karl/McGill University/peyrache_Group - Yuxiao-SleepHDCells/Data/Mouse12-120806/Mouse12-120806'
#data_directory ='C:/Users/Karl/McGill University/peyrache_Group - Yuxiao-SleepHDCells/ADN/Mouse12/Mouse12-120806'

# The first step is to load the session with the function *load_session*. 
# When loading a session for the first time, pynapple will show a GUI 
# in order for the user to provide the information about the session, the subject, the tracking, the epochs and the neuronal data. 
# When informations has been entered, a [NWB file](https://pynwb.readthedocs.io/en/stable/) is created.
# In this example dataset, the NWB file already exists.

data = nap.load_session(data_directory, 'neurosuite')


# The object *data* contains the information about the session such as the spike times of all the neurons, 
# the tracking data and the start and ends of the epochs. We can check each object.

spikes = data.spikes
spikes


# *spikes* is a TsGroup object. 
# It allows to group together time series with different timestamps and associate metainformation about each neuron. 
# Under the hood, it wraps a dictionnary. 
# In this case, the location of where the neuron was recorded has been added when loading the session for the first time.
#
# In this case it holds 15 neurons and it is possible to access, similar to a dictionnary, the spike times of a single neuron: 



epochs = data.epochs
epochs
#Marge the epochs
Mep = pd.concat([epochs['sleep'],epochs['wake']],axis=0)
Mep

# Finally this dataset contains tracking of the animal in the environment. 
# It can be accessed through *data.position*. *rx, ry, rz* represent respectively 
# the roll, the yaw and the pitch of the head of the animal. *x* and *z* represent the position of the animal in the horizontal plane while *y* represent the elevation.

#position = data._make_position
position = data.position
print(position)



# # Tuning curves
# Let's do more advanced analysis. 
# Neurons from ADn (group 0 in the *spikes* group object) are know for firing for a particular direction. 
# Therefore, we can compute their tuning curves, i.e. their firing rates as a function of the head-direction 
# of the animal in the horizontal plane (*ry*). 
# We can use the function *compute_1d_tuning_curves*. 
# In this case, the tuning curves are computed over 120 bins and between 0 and 2$\pi$.

'''
tuning_curves = nap.compute_1d_tuning_curves(group=spikes, 
                                             feature=position['ry'], 
                                             ep=position['ry'].time_support, 
                                             nb_bins=121, 
                                             minmax=(0, 2*np.pi))

tuning_curves



# We can plot tuning curves in polar plots.
import matplotlib.pyplot as plt

neuron_location = spikes.get_info('location') # to know where the neuron was recorded
plt.figure(figsize=(12,9))

for i,n in enumerate(tuning_curves.columns):
    plt.subplot(4,6,i+1, projection = 'polar')
    plt.plot(tuning_curves[n])
    plt.title(neuron_location[n] + '-' + str(n), fontsize = 18)
    
plt.tight_layout()
plt.show()
'''

# While ADN neurons show obvious modulation for head-direction, it is not obvious for all CA1 cells. 
# Therefore we want to restrict the remaining of the analysis to only ADN neurons. 
# We can split the *spikes* group with the function *getby_category*.


spikes_by_location = spikes.getby_category('location')

print(spikes_by_location['adn'])

spikes_adn = spikes_by_location['adn']



# # Decoding
# 
# This last analysis shows how to use the function decoding of pynapple, in this case with head-direction cells.
# 
# The previous result indicates a persistent coordination of head-direction cells during sleep. 
# Therefore it is possible to decode a virtual head-direction signal even if the animal is not moving its head. 
# This example uses the function *decode_1d* which implements bayesian decoding (see : Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J. (1998). Interpreting neuronal population activity by reconstruction: unified framework with application to hippocampal place cells. Journal of neurophysiology, 79(2), 1017-1044.)
# 
# First we can validate the decoding function with the real position of the head of the animal during wake.


tuning_curves_adn = nap.compute_1d_tuning_curves(group=spikes_adn,feature=position['ry'],ep=position['ry'].time_support,nb_bins=121,minmax=(0, 2*np.pi))

decoded, proba_angle = nap.decode_1d(tuning_curves=tuning_curves_adn, 
                                     group=spikes_adn, 
                                     feature=position['ry'], 
                                     ep=Mep,
                                     bin_size=0.025 # second
                                    )
print(decoded)


# We can plot the decoded head-direction along with the true head-direction.
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
plt.plot(position['ry'].as_units('s'), linewidth=0.1, label = 'True')
plt.plot(decoded.as_units('s'), linewidth=0.1, label = 'Decoded')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Head-direction (rad)")
plt.savefig('1', dpi=600, format='svg')
plt.show()




