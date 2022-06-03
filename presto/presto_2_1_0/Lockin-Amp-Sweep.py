# -*- coding: utf-8 -*-

"""
Created on Thu Jan 21 16:13:56 2021
@author: JC

Last version: 2021-07-21

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import h5py
import scipy.signal as sig

from presto import lockin
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

# Current time
t = time.localtime()
print( time.strftime("%H:%M:%S", t) )


# from office
ADDRESS = '130.237.35.90'
PORT = 42871

# VIVACE physical ports
input_port = 1
output_port = 2     # IDT: port 1,  SQUID: port 2

# Frequency sweep in Hz
# res_freq = 8.00224603e+07
# freq_sweep = np.linspace(res_freq+0.1e6, res_freq-0.1e6, 401)
# power_sweep = np.logspace(-3, 0, 101, base=10)
n = 80001
freq_sweep = np.linspace( 50e6, 130e6, n )
power_sweep = np.logspace( -1, 0, 8, base=10 )
# power_sweep = np.logspace( -2, -0.3, 3, base=10 )


# Bandwidth in Hz
_df = 1000

# NCO frequency in Hz
fNCO = 4.1e+9

# Pixels for measurement
N = 24

# Note: approximate time for the measurement t = N * n / df


usb_array = np.zeros( ( len(freq_sweep), len(power_sweep) ), dtype = complex )
lsb_array = np.zeros_like( usb_array, dtype = complex )
f_tuned = np.zeros_like( freq_sweep )

# Instantiate lockin device
with lockin.Lockin( address = ADDRESS,
                    port = PORT,
                    adc_mode = AdcMode.Mixed,
                    adc_fsample = AdcFSample.G3_2,
                    dac_mode = DacMode.Mixed02,
                    dac_fsample = DacFSample.G6_4,
                    ) as lck:
    
    start = time.time()
    
    # Firmware version
    info_api = version.version_api
    print( "Lockin: api = {}".format(info_api) )
    
    
    # Configure the built-in digital mixers for up and down conversion
    lck.hardware.configure_mixer( freq = fNCO,                  # NCO frequency
                                  in_ports = [input_port],      # input ports
                                  out_ports = [output_port],    # output ports
                                  )
            
    # Create an output group
    og = lck.add_output_group( ports = output_port, nr_freq = 1 )
    
    # Create input group
    ig = lck.add_input_group( port = input_port, nr_freq = 1 )
    
    # Start the amplitude sweep
    for pow_ind, sig_pow in enumerate( power_sweep ):
        
        # Sanity check
        if pow_ind % 1 == 0:
            print( 'measuring... amplitude index: ' + str(pow_ind) )
        
        # Start frequency sweep
        for freq_index, sig_freq in enumerate( freq_sweep ):
            
            # Tuning
            f, df = lck.tune( sig_freq, _df )
            f_tuned[freq_index] = f
            lck.set_df( df )
            
            # Set the lock-in input and output frequencies
            og.set_frequencies( f )
            ig.set_frequencies( f )
        

            # Add pseudorandom noise to the generated output to improve harmonic and intermodulation distortion at low amplitudes
            if sig_pow < 0.015 :
                lck.set_dither(state=True, output_ports=output_port)
            
            # Set the lock-in output amplitudes
            og.set_amplitudes( sig_pow )
            
            # Set the lock-in output phases
            og.set_phases( phases = 0,
                           phases_q = -np.pi / 2 )
        
            # Apply the previous settings
            lck.apply_settings()
            
            # Get lock-in packets (pixels) from the local buffer
            data = lck.get_pixels( N )
            freqs, pixels_i, pixels_q = data[input_port]
    
            # Convert a measured IQ pair into a low/high sideband pair
            LSB, HSB = utils.untwist_downconversion( pixels_i, pixels_q )
    
            # Store data in array
            usb_array[freq_index, pow_ind] = np.mean( HSB )
            lsb_array[freq_index, pow_ind] = np.mean( LSB )


stop = time.time()

print( utils.format_sec(stop-start) )



#%%

# Saving data in the hdf5 file

folder = os.path.join( r'D:\VivaceData\SAW-KTH.hdf5' )
datorfile = h5py.File( folder, 'r+' )

measurement = datorfile.require_group( r'amplitude sweep' )
date = time.strftime('%Y-%m-%d-') + time.strftime('%H_%M_%S')
hf = measurement.create_group( date )

# Save the code
dt = h5py.special_dtype( vlen=str )
code_file = open( r'C:\Users\JC\PhD\Codes\Measurements-main\Lockin-Amp-Sweep.py', 'r' )
code = code_file.readlines()
code_set = hf.create_dataset( "source_code", (len(code),), dtype=dt )
for index in np.arange( 0, len(code) ):
    code_set[index] = code[index]
code_file.close()

# Save bandwidth and NCO frequency
hf.create_dataset( "df", (np.shape(df)), dtype=float, data=df )
hf.create_dataset( "fNCO", (np.shape(fNCO)), dtype=float, data=fNCO )

# Save frequency and amplitude sweep
hf.create_dataset( "freq sweep", (np.shape(f_tuned)), dtype=float, data=f_tuned )
hf.create_dataset( "amp sweep", (np.shape(power_sweep)), dtype=float, data=power_sweep )

# Save data
hf.create_dataset( "USB data", (np.shape(usb_array)), dtype=complex, data=usb_array )
hf.create_dataset( "LSB data", (np.shape(lsb_array)), dtype=complex, data=lsb_array )

# Close the hdf5 file
datorfile.close()


# Write it in the logbook

# Reescaling the frequencies
f_tuned_ghz = ( f_tuned + fNCO ) / 1e9

log_browser = open(r"D:\VivaceData\Logbook.txt","a")
log_browser.write( 'Date:                   ' + date + '\r' )
log_browser.write( 'Type of measurement:    Amplitude Sweep \r' )
log_browser.write( 'Type of measurement:    Reflection \r' )
log_browser.write( 'Description:            Wider amplitude sweep through all the resonances through the SQUID \r' )
log_browser.write( 'Pump frequencies:       %.5f - %.5f GHz \r' % (f_tuned_ghz[0], f_tuned_ghz[-1]) )
log_browser.write( 'Pump amplitudes:        %.3f - %.3f FS \r' % (power_sweep[0], power_sweep[-1]) )
log_browser.write( 'Signal frequencies:     - \r' )
log_browser.write( 'Signal amplitudes:      - \r' )
log_browser.write( 'df:                     %d Hz \r' % df )
log_browser.write( 'fNCO:                   %.1f GHz \r' % (fNCO/1e9) )
log_browser.write( 'N pixels:               %d \r' % N )
log_browser.write( 'Time:                   %s \r\n' % utils.format_sec(stop-start) )
log_browser.write( '--------------------------------------------------------------------------\n\n' )
     
log_browser.close()



#%%

# Plot of the data as a function of frequency and amplitude

# Phase
phase_data = sig.detrend( np.unwrap( np.angle(usb_array) ) )

# Amplitude normalised
amp_data = 20 * np.log10( np.abs(usb_array)/power_sweep )

# 2D plot of the normalised amplitude as a function of frequency and power    
fig, ax = plt.subplots(1)
a = ax.pcolormesh( f_tuned, 20 * np.log10(power_sweep), np.transpose(amp_data), shading='nearest', cmap='RdBu_r' )

fig.suptitle( 'Power sweep' )
ax.set_xlabel( 'frequency [Hz]' )
ax.set_ylabel( 'power [dBFS]' )
fig.colorbar( a )
