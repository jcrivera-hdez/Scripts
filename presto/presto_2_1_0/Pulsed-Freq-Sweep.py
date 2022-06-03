# -*- coding: utf-8 -*-

"""
Created on Mon Jun 14 16:15:59 2021
@author: JC

Last version: 2021-06-14

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import h5py
import scipy.signal as sig

from presto import pulsed
from presto import utils
from presto import version
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode

# Plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.titlesize'] = 23
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15



#%%

# from lab
# ADDRESS = 'bravo'
# PORT = 3490

# from office
ADDRESS = '130.237.35.90'
PORT = 42871

input_port = 1
output_port = 1

# NCO frequency in Hz
fNCO = 4.1e+9

# Instantiate lockin device
with pulsed.Pulsed( address = ADDRESS,
                    port = PORT,
                    adc_mode = AdcMode.Mixed,
                    adc_fsample = AdcFSample.G3_2,
                    dac_mode = DacMode.Mixed02,
                    dac_fsample = DacFSample.G6_4,
                    ) as pls:
    
    # Firmware version
    print( "Lockin: api = {}".format(version.version_api) )
    
    # Configure the built-in digital mixers for up and down conversion
    pls.hardware.configure_mixer( freq = fNCO,                  # NCO frequency
                                  in_ports = [input_port],      # input ports
                                  out_ports = [output_port],    # output ports
                                  )
    
    start = time.time()
    
    
    # Select inputs to store and the duration of each store
    pls.set_store_ports(input_port)
    pls.set_store_duration(2e-6)

    ######################################################################
    # Sinewave generator, template as envelope

    # The template defines the envelope, specify that it is an envelope
    # for carrier generator 1. This template will be scaled
    N = pulsed.MAX_TEMPLATE_LEN
    t = np.arange(N) / pls.get_fs('dac')
    s = np.hanning(N)
    template_1 = pls.setup_template(output_port, 1, s, envelope=1)

    # setup a list of frequencies for carrier generator 1
    NFREQ = 8
    f = np.logspace(6, 8, NFREQ)
    p = np.zeros(NFREQ)
    pls.setup_freq_lut(output_ports=output_port,
                       group=1,
                       frequencies=f,
                       phases=p)

    # setup a list of scales
    NSCALES = 8
    scales = np.linspace(1.0, .01, 8)
    pls.setup_scale_lut(output_ports=output_port,
                        group=1,
                        scales=scales,
                        )

    ######################################################################
    # define the sequence of pulses and data stores in time
    # At the end of the time sequence, increment frequency and scale index.
    # Since repeat_count for the scale lut is 8, scale index will actually
    # not increment every time, but only every 8 runs.
    # The frequency will increment every time, and wrap around every 8 runs.
    T = 0.0
    pls.output_pulse(T, template_1)
    pls.store(T)
    T = 5e-6
    pls.next_frequency(T, output_port)
    pls.next_scale(T, output_port)
    # repeat the time sequence 64 times. Run the total sequence 100 times and average the results.
    pls.run(period=10e-6,
            repeat_count=NFREQ * NSCALES,
            num_averages=100)
    
    t_arr, data = pls.get_store_data()
    