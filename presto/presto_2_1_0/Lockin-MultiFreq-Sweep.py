# -*- coding: utf-8 -*-

"""
Created on Wed Jun  9 15:13:30 2021
@author: JC

Last version: 2021-06-29

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

# Try to tune the whole comb at the same time

# from lab
# ADDRESS = 'bravo'
# PORT = 3490

# from office
ADDRESS = '130.237.35.90'
PORT = 42871


# Comb parameters

# Number of frequencies of the sweep
nfreqs = 32

# Frequency comb not tuned
_f0, _fn = 50e6, 130e6
_fcomb = np.linspace(_f0, _fn, nfreqs, endpoint=False)


# Frequency sweep in Hz
n = 1001
freq_sweep = np.linspace( 0, _fcomb[1] - _fcomb[0], n )
pump_amp = 1 / nfreqs

# Bandwidth in Hz
_df = 1e3

# NCO frequency in Hz
fNCO = 4.1e+9

# Pixels for measurement
N = 48

# Note: approximate time for the measurement t = N * n / df


usb_array = np.zeros( ( len(freq_sweep), nfreqs ), dtype = complex )
lsb_array = np.zeros_like( usb_array )

f_tuned = np.zeros(( len(freq_sweep), nfreqs ))

# Instantiate lockin device
with lockin.Lockin( address = ADDRESS,
                    port = PORT,
                    adc_mode = AdcMode.Mixed,
                    adc_fsample = AdcFSample.G3_2,
                    dac_mode = DacMode.Mixed02,
                    dac_fsample = DacFSample.G6_4,
                    ) as lck:
    
    # Firmware version
    print( "Lockin: api = {}".format(version.version_api) )
    
    # Configure the built-in digital mixers for up and down conversion
    lck.hardware.configure_mixer( freq = fNCO,      # NCO frequency
                                  in_ports = [1],   # input ports
                                  out_ports = [1],  # output ports
                                  )
    
    start = time.time()
    
    # Create an output group
    og = lck.add_output_group( ports = 1, nr_freq = nfreqs )
        
    # Set the lock-in output amplitudes
    og.set_amplitudes( np.ones(nfreqs) * 1/nfreqs )  
    # Set the lock-in output phases
    og.set_phases( phases = np.zeros( nfreqs),
                   phases_q = np.ones(nfreqs) * -np.pi / 2 )
    
    # Create input group
    ig = lck.add_input_group( port = 1, nr_freq = nfreqs )
    
    # Start frequency sweep
    for freq_index, sig_freq in enumerate( freq_sweep ):
        
        # Sanity check
        if freq_index % 250 == 0:
            print( 'measuring... frequency index: ' + str(freq_index) )
        
        # Tuning the first frequency and _df
        fcomb, df = lck.tune( _fcomb + sig_freq, _df )
        lck.set_df( df )

            
        # Save the tuned frequencies
        f_tuned[freq_index] = fcomb
        
        
        # Set the lock-in input and output frequencies
        og.set_frequencies( fcomb )
        ig.set_frequencies( fcomb )
        
        
        # Apply the previous settings
        lck.apply_settings()
        
        
        # Get lock-in packets (pixels) from the local buffer
        data = lck.get_pixels( N )
        freqs, pixels_i, pixels_q = data[1]

        # Convert a measured IQ pair into a low/high sideband pair
        LSB, HSB = utils.untwist_downconversion( pixels_i, pixels_q )

        # Store data in array
        usb_array[freq_index] = np.mean( HSB, axis=0 )
        lsb_array[freq_index] = np.mean( LSB, axis=0 )


# Sort and flatten the frequency comb and the data
f_tuned_flat = np.sort( f_tuned.flatten() )
usb_array_flat = np.transpose( usb_array ).flatten()
lsb_array_flat = np.transpose( lsb_array ).flatten()

stop = time.time()

print( utils.format_sec(stop-start) )



#%%

# Saving data in the hdf5 file

folder = os.path.join( r'D:\VivaceData\SAW-KTH.hdf5' )
datorfile = h5py.File( folder, 'r+' )

measurement = datorfile.require_group( r'frequency sweep' )
hf = measurement.create_group( time.strftime('%Y-%m-%d-') + time.strftime('%H_%M_%S') )

# Save the code
dt = h5py.special_dtype( vlen=str )
code_file = open( r'C:\Users\JC\PhD\Codes\Measurements-main\Lockin-MultiFreq-Sweep.py', 'r' )
code = code_file.readlines()
code_set = hf.create_dataset( "source_code", (len(code),), dtype=dt )
for index in np.arange( 0, len(code) ):
    code_set[index] = code[index]
code_file.close()

# Save bandwidth, NCO frequency and frequencies
hf.create_dataset( "df", (np.shape(df)), dtype=float, data=df )
hf.create_dataset( "fNCO", (np.shape(fNCO)), dtype=float, data=fNCO )
hf.create_dataset( "freq sweep", (np.shape(f_tuned_flat)), dtype=float, data=f_tuned_flat )

# Save data
hf.create_dataset( "USB data", (np.shape(usb_array_flat)), dtype=complex, data=usb_array_flat )
hf.create_dataset( "LSB data", (np.shape(lsb_array_flat)), dtype=complex, data=lsb_array_flat )

# Close the hdf5 file
datorfile.close()


# Write it in the logbook

# Reescaling the frequencies
f_tuned_ghz = ( f_tuned_flat + fNCO ) / 1e9

log_browser = open( r"D:\VivaceData\Logbook","a" )
log_browser.write( 'Date:                   ' + time.strftime('%Y-%m-%d-') + time.strftime('%H_%M_%S') + '\r' )
log_browser.write( 'Type of measurement:    Multifrequency Sweep \r' )
log_browser.write( 'Type of measurement:    Reflection \r' )
log_browser.write( 'Description:            Multifrequency sweep through all the SAW resonances. \r' )
log_browser.write( 'Pump frequencies:       %.3f - %.3f GHz \r' % (f_tuned_ghz[0], f_tuned_ghz[-1]) )
log_browser.write( 'Pump amplitudes:        %.1f FS \r' % pump_amp )
log_browser.write( 'Signal frequencies:     - \r' )
log_browser.write( 'Signal amplitudes:      - \r' )
log_browser.write( 'df:                     %d Hz \r' % df )
log_browser.write( 'fNCO:                   %.1f GHz \r' % (fNCO / 1e9) )
log_browser.write( 'N pixels:               %d \r\n' % N )
log_browser.write( '--------------------------------------------------------------------------\n\n' )

log_browser.close()



#%%

# Plot of the data as a function of frequency

# Phase
phase_data = sig.detrend( np.unwrap( np.angle(usb_array_flat) ) )

# Amplitude
amp_data = ( np.log10( np.abs(usb_array_flat) ) )
    
# Plot
fig, ax1 = plt.subplots(2, sharex = True)
ax1[0].plot( f_tuned_flat, amp_data )
ax1[1].plot( f_tuned_flat, phase_data )
fig.suptitle('Multifrequency sweep')
ax1[1].set_xlabel('frequency [Hz]')
ax1[1].set_ylabel('phase [rad]')
ax1[0].set_ylabel('amplitude [dB]')
