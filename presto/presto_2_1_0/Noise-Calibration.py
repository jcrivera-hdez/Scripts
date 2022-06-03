# -*- coding: utf-8 -*-

"""
Created on Sat Oct 17 11:27:29 2020
@author: Shan

Last version: 2021-06-03

"""

import numpy as np
import time
import os
import h5py

from presto import lockin
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import utils



#%%

# Saving the data in a hdf5 file
folder = os.path.join( r'D:\VivaceData\Calibration', time.strftime('%Y-%m-%d'),time.strftime('%H_%M_%S') )

if not os.path.isdir(folder):
    os.makedirs(folder)

# Filename
name = r'\Noise-calibration-test.hdf5'
datorfile = h5py.File( folder + name, 'w' )

# Save the code
dt = h5py.special_dtype( vlen=str )
code_file = open( r'C:\Users\JC\PhD\VivaceData\Codes\Noise-Calibration.py', 'r' )
code = code_file.readlines()
code_set = datorfile.create_dataset( "source_code", (len(code),), dtype=dt )
for index in np.arange( 0, len(code) ):
    code_set[index] = code[index]
code_file.close()


# VIVACE (bravo) address and port
ADDRESS = '130.237.35.90'
PORT = 42871

# NCO frequency
fNCO = 4.3e+9  # [Hz]
# Save the NCO frequency
datorfile.create_dataset("fNCO", data=fNCO )


# MEASUREMENT SETTINGS

# Set desired df
_df = 1e3  # [Hz]

# Central frequencies of the comb (set this frequencies away from the SAW resonances)
f1 = 4.417e9 - fNCO  # [Hz]
f2 = 4.433e9 - fNCO  # [Hz]
# Number of frequencies of the comb (the maximum is 96)
n_freqs = 32

# Pixels for measurement
N = 10*1000


# Temperature loop
temperature = 0
temp_ind = 0
temp_array = np.array([])
temp_array_after = np.array([])

while int(temperature) != -100:

    # Asking temperature
    print( 'To stop measurement input -100' )
    temperature = input( 'Enter mixing chamber temperature in mK: ' )
    print( 'The temperature in mK is: ', temperature )
    
    # Exit the loop in case we specify it
    if int(temperature) == -100:
        print('Measurement done!')
        break


    # Temperature group 
    temp_group = datorfile.create_group( 'temperature_ind' + str(temp_ind) )

    # Instantiate lockin device
    with lockin.Lockin( address=ADDRESS,
                        port=PORT,
                        adc_mode=AdcMode.Mixed,
                        adc_fsample=AdcFSample.G3_2,
                        dac_mode=DacMode.Mixed02,
                        dac_fsample=DacFSample.G6_4,
                        ) as lck:

        # Configure the built-in digital mixers for up and down conversion       
        lck.hardware.configure_mixer( freq = fNCO,
                                      in_ports = 1,
                                      out_ports = 1,
                                      )


        # Tuning of f1
        f1_tuned, df = lck.tune( f1, _df )
        lck.set_df( df )
        
        # Tuning of f2
        n1 = f1_tuned / df
        n2 = np.round( f2 / df )

        # This is to ensure the difference between both pumps is a multiple of 2*df
        if (n1%2==0 and n2%2==0) or (n1%2==1 and n2%2==1):
            f2_tuned = n2*df
        else:
            f2_tuned = (n2+1)*df
        
        
        # We generate the frequency comb (probably there's a better way to do it)
        freq_comb = np.zeros(n_freqs)
        
        freq_comb[int(n_freqs/2)] = (f1_tuned + f2_tuned)/2
        freq_comb[int(n_freqs/2)-1] = f1_tuned
        freq_comb[int(n_freqs/2)+1] = f2_tuned
        
        # Spacing of the comb
        delta = np.abs(f1_tuned - f2_tuned) / 2      
        # Sanity check
        if (freq_comb[int(n_freqs/2)+1] - delta) != freq_comb[int(n_freqs/2)]:
            print('center frequency is not tuned!')     
        # Build the remaining comb
        freq_comb[0:int(n_freqs/2)-1] = freq_comb[int(n_freqs/2)-1] + np.arange(-int(n_freqs/2)+1, 0, 1)*delta
        freq_comb[int(n_freqs/2)+2:] = freq_comb[int(n_freqs/2)+1] + np.arange(1, int(n_freqs/2)-1, 1)*delta
    
        
        # Create input group
        ig = lck.add_input_group( port = 1, nr_freq = n_freqs )
        
        # Set the lock-in input frequencies
        ig.set_frequencies( freq_comb )
        
        
        # Apply the settings
        lck.apply_settings()
 
        
        # Set up Quantum Garage
        data = lck.get_pixels( N )
        freqs, pixels_i, pixels_q = data[1]
        
        # Convert a measured IQ pair into a low/high sideband pair
        LSB, HSB = utils.untwist_downconversion( pixels_i, pixels_q )

        # Store data in array (Should I save raw data or the mean?)
        usb_array = np.mean( HSB )
        lsb_array = np.mean( LSB )
        
        # Save data
        temp_group.create_dataset("noise voltage high sideband", (np.shape([usb_array])), dtype=complex, data=[usb_array])
        temp_group.create_dataset("noise voltage low sideband", (np.shape([lsb_array])), dtype=complex, data=[lsb_array])


    # Asking temperature after measurement
    temperature_after = input('Enter mixing chamber temperature in mK: ')
    print('The temperature in mK is: ', temperature_after)
    print('Measurement done!')

    # Saving temperature before and after measurement
    temp_array = np.append( temp_array, int(temperature) )
    temp_array_after = np.append( temp_array_after, int(temperature_after) )


    temp_ind = temp_ind + 1
    


# Save the temperatures
datorfile.create_dataset('temperature array', np.shape(temp_array), dtype=float, data=temp_array)
datorfile.create_dataset('temperature array after', np.shape(temp_array_after), dtype=float, data=temp_array_after)

# Save the frequency comb and df
datorfile.create_dataset("frequency comb", (np.shape(freq_comb)), dtype=float, data=freq_comb)
datorfile.create_dataset("df", (np.shape([df])), dtype=float, data=[df])

# Close the hdf5 file
datorfile.close()

