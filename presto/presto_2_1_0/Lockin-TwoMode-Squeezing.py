# -*- coding: utf-8 -*-

"""
Created on Fri Jul 23 13:58:14 2021
@author: JC & Sara

Last version: 2021-08-03

""" 

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import h5py

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


# Import pump amplitudes
folder_path = os.path.join( r'D:\VivaceData\SAW-KTH.hdf5' )
file = h5py.File( folder_path, 'r+' )

# Measurement type
meas_type = file[ 'amplitude sweep' ]
keys = list(meas_type.keys())

# Dataset of interest
hf = meas_type[ keys[2] ]

# Pump amplitude values
amp_sweep = np.array( hf['amp sweep'] )


# NCO frequency in Hz
fNCO = 4.1e+9

# Pixels for measurement
N = 50_000_000

# Number of pixels with pump on (off)
pix_on_off = 10_000

# Pump port
port_p = 2


# Tuning on a SHALLOW or DEEP resonance
# tuning = 'shallow'
tuning = 'deep'

# Import tuning results
folder_path_tune = os.path.join( r'D:\VivaceData\SAW-KTH-Parameters-v2.hdf5' )
hf_tune = h5py.File( folder_path_tune, 'r' )

if tuning == 'shallow':
    dim = 29
    # Pump index
    p = 17
else:
    dim = 15
    p = 9
    
# Listening frequencies
_IMP_freqs = np.zeros( ( len(amp_sweep), dim ) )

# Bandwidth
_df = np.zeros( len(amp_sweep) )

# Pump frequency
_pump_freq = np.zeros_like( _df )


# Import data
for i in range( len(amp_sweep) ):
    
    # Shallow resonance tuning
    if tuning == 'shallow':
        tuning_type = hf_tune[ r'tuning shallow' ]
    # Deep resonance tuning
    else:
        tuning_type = hf_tune[ r'tuning deep' ]
    
    # Amplitude group
    pump_group = tuning_type[ r'amp_ind_%d' % i ]
    
    # Listening frequencies
    _IMP_freqs[i] = np.array( pump_group['f IMPs'] )
    
    # Bandwidth
    _df[i] = np.array( pump_group['df'] )
    
    # Pump frequency
    if tuning == 'shallow':
        _pump_freq[i] = _IMP_freqs[i][p]           # Pump on a SHALLOW resonance
    else:
        _pump_freq[i] = _IMP_freqs[i][p]           # Pump on a DEEP resonance
                
# Number of input frequencies
infreqs = len( _IMP_freqs[0] )


# LOCKIN MEASUREMENT

# Create a folder to save data
folder_path_garage = os.path.join( r'D:\VivaceData\SAW-KTH-QuantumGarage.hdf5' )
file_garage = h5py.File( folder_path_garage, 'r+' )
measurement = file_garage.require_group( r'two mode squeezing' )
date = time.strftime('%Y-%m-%d-') + time.strftime('%H_%M_%S')
hf_meas = measurement.create_group( date )

# Upper and lower side band for pump on and off
usb_array_on = np.zeros( ( int(N/2), infreqs ), dtype = complex )
lsb_array_on = np.zeros_like( usb_array_on, dtype = complex )
usb_array_off = np.zeros_like( usb_array_on, dtype = complex )
lsb_array_off = np.zeros_like( usb_array_on, dtype = complex )


# Instantiate Lockin
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
    lck.hardware.configure_mixer( freq = fNCO,              # NCO frequency
                                  in_ports = 1,             # input ports
                                  out_ports = [port_p],     # output ports
                                  )

    start = time.time()


    # Create an output group for the pump
    og_p = lck.add_output_group( ports = port_p, nr_freq = 1 )
   

    # Set the lock-in output pump phases
    og_p.set_phases( phases = 0,
                     phases_q = -np.pi / 2 )

    # Create input group
    ig = lck.add_input_group( port = 1, nr_freq = infreqs )


    # Start the amplitude sweep
    for amp_ind, sig_amp in enumerate( amp_sweep ):
        
        # Create amplitude group
        amp_group = hf_meas.create_group( r'pump_amp_ind_' + str( amp_ind ) )

        # We do not need to tune the frequencies since they have already been tuned
        pump_freq = _pump_freq[amp_ind]
        df = _df[amp_ind]
        
        lck.set_df( df )
        
        # Generate the new frequency comb
        freq_comb = _IMP_freqs[amp_ind]
        

        # Save bandwidth
        amp_group.create_dataset( "df", (np.shape(df)), dtype=float, data=[df] )

        # Save pump and signal frequencies
        amp_group.create_dataset( "signal freqs", (np.shape(freq_comb)), dtype=float, data=freq_comb )
        amp_group.create_dataset( "pump freqs", (np.shape(pump_freq)), dtype=float, data=pump_freq )


        # Check if the idler and the signal are symmetrical
        if freq_comb[p] - freq_comb[p-1] != freq_comb[p+1] - freq_comb[p] :
            print( 'Signal and idler are NOT symmetrical' )

        else:
            print( 'Signal and idler are symmetrical' )

            # Set output frequency
            og_p.set_frequencies( pump_freq )

            # Set input frequencies
            ig.set_frequencies( freq_comb )


            # ALternating pump ON and OFF data
            pix_index = 0
            ii_on = 0
            ii_off = 0
   
            while pix_index < N/2:

                # Pump ON
                if pix_index % pix_on_off == 0:

                    # Pump amplitude
                    og_p.set_amplitudes( amps = sig_amp )

                    # Apply the previous settings
                    lck.apply_settings()

                    # Get lock-in packets (pixels) from the local buffer
                    data = lck.get_pixels( pix_on_off )
                    frequencies, pixels_i, pixels_q = data[1]

                    # Convert a measured IQ pair into a low/high sideband pair
                    LSB, HSB = utils.untwist_downconversion( pixels_i, pixels_q )

                    # Store data in array
                    usb_array_on[ ii_on:ii_on+pix_on_off, : ] = HSB
                    lsb_array_on[ ii_on:ii_on+pix_on_off, : ] = LSB

                    pix_index += 1
                    ii_on += pix_on_off
       

                # Pump OFF
                else:
                    
                    # Pump amplitude set to 0
                    og_p.set_amplitudes( amps = 0 )
   

                    # Apply the previous settings
                    lck.apply_settings()
   
                    # Get lock-in packets (pixels) from the local buffer
                    data = lck.get_pixels( pix_on_off )
                    frequencies, pixels_i, pixels_q = data[1]
           
                    # Convert a measured IQ pair into a low/high sideband pair
                    LSB, HSB = utils.untwist_downconversion( pixels_i, pixels_q )

                    # Store data in array
                    usb_array_off[ ii_off:ii_off+pix_on_off, : ] = HSB
                    lsb_array_off[ ii_off:ii_off+pix_on_off, : ] = LSB
           
                    pix_index += pix_on_off - 1
                    ii_off += pix_on_off

        # Save data
        amp_group.create_dataset( "USB data", (np.shape(usb_array_on)), dtype=complex, data=usb_array_on )
        amp_group.create_dataset( "LSB data", (np.shape(lsb_array_on)), dtype=complex, data=lsb_array_on )
        amp_group.create_dataset( "USB data OFF", (np.shape(usb_array_off)), dtype=complex, data=usb_array_off )
        amp_group.create_dataset( "LSB data OFF", (np.shape(lsb_array_off)), dtype=complex, data=lsb_array_off )


# Save pump amplitudes
hf_meas.create_dataset( "pump amps", (np.shape(amp_sweep)), dtype=float, data=amp_sweep )

# Save NCO frequency
hf_meas.create_dataset( "fNCO", (np.shape(fNCO)), dtype=float, data=[fNCO] )      

# Save the code
dt = h5py.special_dtype( vlen=str )
code_file = open( r'C:\Users\JC\PhD\Codes\Measurements-main\Lockin-TwoMode-Squeezing.py', 'r' )
code = code_file.readlines()
code_set = hf_meas.create_dataset( "source_code", (len(code),), dtype=dt )
for index in np.arange( 0, len(code) ):
    code_set[index] = code[index]
code_file.close()


# Close hdf5 file
file.close()


stop = time.time()

print( utils.format_sec(stop-start) )



#%%

# Write it in the logbook

# Reescale the pump frequency
pump_freq_ghz = ( pump_freq + fNCO ) / 1e9

log_browser = open( r"D:\VivaceData\Logbook.txt", "a" )
log_browser.write( 'Date:                   ' + date + '\r' )
log_browser.write( 'Type of measurement:    Two Mode Squeezing \r' )
log_browser.write( 'Type of measurement:    Transmission \r' )
log_browser.write( 'Description:            Pump the mirror SQUID at the %d th %s resonance,\n' % (p, tuning) )
log_browser.write( '                        and listen noise on all the deep resonance frequencies through the IDT.\n' )
log_browser.write( '                        Alternating pump ON and OFF \r' )
log_browser.write( 'Pump frequencies:       %.4f GHz \r' % (pump_freq_ghz) )
log_browser.write( 'Pump amplitudes:        %.3f - %.3f FS \r' % (amp_sweep[0], amp_sweep[-1]) )
log_browser.write( 'Signal frequencies:     - \r' )
log_browser.write( 'Signal amplitudes:      - \r' )
log_browser.write( 'df:                     %d Hz \r' % df )
log_browser.write( 'fNCO:                   %.1f GHz \r' % (fNCO / 1e9) )
log_browser.write( 'N pixels:               %d \r\n' % N )
log_browser.write( 'Time:                   %s \r\n' % utils.format_sec(stop-start) )
log_browser.write( '--------------------------------------------------------------------------\n\n' )

log_browser.close()