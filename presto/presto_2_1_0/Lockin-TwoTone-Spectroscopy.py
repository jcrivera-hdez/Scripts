# -*- coding: utf-8 -*-

"""
Created on Mon Jan 25 09:59:22 2021
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

# Current time
t = time.localtime()
print( time.strftime("%H:%M:%S", t) )


# from office
ADDRESS = '130.237.35.90'
PORT = 42871

# Pump and signal amplitudes
pump_amp = 0.01
signal_amp = 0.01

# Pump and signal frequency sweep
n = 2000
pump_freq_sweep = np.linspace( 78e6, 91e6, n )
signal_freq_sweep = np.linspace( 78e6, 91e6, n )

# Bandwidth in Hz
_df = 6e3

# NCO frequency in Hz
fNCO = 4.1e+9

# Pixels for measurement
N = 48

# Pump port
port_p = 2
# Signal port
port_s = 1

# Number of input frequencies
infreqs = 3

usb_array = np.zeros( ( len(pump_freq_sweep), len(signal_freq_sweep), N, infreqs ), dtype = complex )
lsb_array = np.zeros_like( usb_array, dtype = complex )

f_signal = np.zeros_like( signal_freq_sweep )
f_pump = np.zeros_like( signal_freq_sweep )
f_idler = np.zeros_like( signal_freq_sweep )

with lockin.Lockin( address = ADDRESS,
                    port = PORT,
                    adc_mode = AdcMode.Mixed,
                    adc_fsample = AdcFSample.G3_2,
                    dac_mode = DacMode.Mixed02,
                    dac_fsample = DacFSample.G6_4,
                    ) as lck:
    
    # Firmware version
    info_api = version.version_api
    print( "Lockin: api = {}".format(info_api) )
    
    
    # Configure the built-in digital mixers for up and down conversion
    lck.hardware.configure_mixer( freq = fNCO,      # NCO frequency
                                  in_ports = [1],   # input ports
                                  out_ports = [port_s, port_p],  # output ports
                                  )
    
    start = time.time()
    
    # Create an output group for the signal
    og_s = lck.add_output_group( ports = port_s, nr_freq = 1 )

    # Create an output group for the pump
    og_p = lck.add_output_group( ports = port_p, nr_freq = 1 )
    
    # Create input group
    ig = lck.add_input_group( port = 1, nr_freq = infreqs )
    
    
    # Set the lock-in output signal amplitudes
    og_s.set_amplitudes( amps = signal_amp )         
    # Set the lock-in output signal phases
    og_s.set_phases( phases = 0,
                     phases_q = -np.pi / 2 )
    
    # Set the lock-in output pump amplitudes
    og_p.set_amplitudes( amps = pump_amp )         
    # Set the lock-in output pump phases
    og_p.set_phases( phases = 0,
                     phases_q = -np.pi / 2 )
    
    # Pump frequency sweep
    for pump_freq_ind, _pump_freq in enumerate( pump_freq_sweep ):
        
        # Sanity check
        if pump_freq_ind % 500 == 0:
            print( 'measuring... frequency index: ' + str(pump_freq_ind) )
       
        # Signal frequency sweep
        for sig_freq_ind, _sig_freq in enumerate( signal_freq_sweep ):
            
            # Tuning
            sig_freq, df = lck.tune( _sig_freq, _df )
            lck.set_df( df )
            
            f_signal[sig_freq_ind] = sig_freq
            pump_freq = _pump_freq // df * df
            f_pump[pump_freq_ind] = pump_freq
            idler_freq = 2 * pump_freq - sig_freq
            
            freq_comb = np.array( [sig_freq, pump_freq, idler_freq] )
            
            # Set the lock-in output signal frequency
            og_s.set_frequencies( sig_freq )
            # Set the lock-in output signal frequency
            og_p.set_frequencies( pump_freq )
            
            # Set the lock-in input frequencies
            ig.set_frequencies( freq_comb )
            
            # Apply the previous settings
            lck.apply_settings()
            
            # Get lock-in packets (pixels) from the local buffer
            data = lck.get_pixels( N )
            freqs, pixels_i, pixels_q = data[1]
            
            # Convert a measured IQ pair into a low/high sideband pair
            LSB, HSB = utils.untwist_downconversion( pixels_i, pixels_q )
    
            # Store data in array
            usb_array[pump_freq_ind, sig_freq_ind, :, :] = HSB
            lsb_array[pump_freq_ind, sig_freq_ind, :, :] = LSB


stop = time.time()

print( utils.format_sec(stop-start) )



#%%

# Saving data in the hdf5 file

folder = os.path.join( r'D:\VivaceData\SAW-KTH.hdf5' )
datorfile = h5py.File( folder, 'r+' )

measurement = datorfile.require_group( r'two tone spectroscopy' )
date = time.strftime('%Y-%m-%d-') + time.strftime('%H_%M_%S')
hf = measurement.create_group( date )

# Save the code
dt = h5py.special_dtype( vlen=str )
code_file = open( r'C:\Users\JC\PhD\Codes\Measurements-main\Lockin-TwoTone-Spectroscopy.py', 'r' )
code = code_file.readlines()
code_set = hf.create_dataset( "source_code", (len(code),), dtype=dt )
for index in np.arange( 0, len(code) ):
    code_set[index] = code[index]
code_file.close()

# Save bandwidth and the NCO frequency
hf.create_dataset( "df", (np.shape(df)), dtype=float, data=[df] )
hf.create_dataset( "fNCO", (np.shape(fNCO)), dtype=float, data=[fNCO] )

# Save pump, signal and idler frequencies
hf.create_dataset( "signal freqs", (np.shape(f_signal)), dtype=float, data=f_signal )
hf.create_dataset( "pump freqs", (np.shape(f_pump)), dtype=float, data=f_pump)

# Save signal and pump amplitudes
hf.create_dataset( "pump amps", (np.shape(pump_amp)), dtype=float, data=pump_amp )
hf.create_dataset( "signal amps", (np.shape(signal_amp)), dtype=float, data=signal_amp )

# Save data
hf.create_dataset( "USB data", (np.shape(usb_array)), dtype=complex, data=usb_array )
hf.create_dataset( "LSB data", (np.shape(lsb_array)), dtype=complex, data=lsb_array )

# Close hdf5 file
datorfile.close()


# Write it in the logbook

# Reescaling the frequencies
f_pump_ghz = ( f_pump + fNCO ) / 1e9
f_signal_ghz = ( f_signal + fNCO ) / 1e9

log_browser = open( r"D:\VivaceData\Logbook.txt","a" )
log_browser.write( 'Date:                   ' + date + '\r' )
log_browser.write( 'Type of measurement:    Two Tone Spectroscopy \r' )
log_browser.write( 'Type of measurement:    Transmission (Pump) / Reflection (Signal) \r' )
log_browser.write( 'Description:            Sweep a pump and a signal frequency through all the SAW resonances \r' )
log_browser.write( 'Pump frequencies:       %.3f - %.3f GHz \r' % (f_pump_ghz[0], f_pump_ghz[-1]) )
log_browser.write( 'Pump amplitudes:        %.2f FS \r' % pump_amp )
log_browser.write( 'Signal frequencies:     %.3f - %.3f GHz \r' % (f_signal_ghz[0], f_signal_ghz[-1]) )
log_browser.write( 'Signal amplitudes:      %.1f FS \r' % signal_amp )
log_browser.write( 'df:                     %d Hz \r' % df )
log_browser.write( 'fNCO:                   %.1f GHz \r' % (fNCO / 1e9) )
log_browser.write( 'N pixels:               %d \r' % N )
log_browser.write( 'Time:                   %s \r\n' % utils.format_sec(stop-start) )
log_browser.write( '--------------------------------------------------------------------------\n\n' )

log_browser.close()



#%%

# Plotting data as a function of the signal and pump frequency

# Signal and idler amplitude
signal_amp = 20 * np.log10( np.abs( np.mean(usb_array[:,:,:,0], axis=2) ) )
idler_amp = 20 * np.log10( np.abs( np.mean(usb_array[:,:,:,2], axis=2) ) )


zmin_s = np.min(signal_amp)
zmax_s = np.max(signal_amp)
zmin_i = -110 #np.min(idler_amp)
zmax_i = -100 #np.max(idler_amp)

# Plotting
fig, ax = plt.subplots( 2, 1, sharex=True )
a = ax[0].pcolormesh( f_pump, f_signal, signal_amp, cmap='RdBu_r', vmin=zmin_s, vmax=zmax_s )
b = ax[1].pcolormesh( f_pump, f_idler, idler_amp, cmap='RdBu_r', vmin=zmin_i, vmax=zmax_i )
fig.suptitle( 'Two tone spectroscopy' )
ax[0].set_ylabel( 'signal freq [Hz]' )
ax[1].set_xlabel( 'pump freq [Hz]' )
ax[1].set_ylabel( 'idler freq [Hz]' )
fig.colorbar( a, ax=ax[0], label='dBFS' )
fig.colorbar( b, ax=ax[1], label='dBFS' )
