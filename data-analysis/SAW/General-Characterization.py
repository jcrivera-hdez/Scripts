# -*- coding: utf-8 -*-

"""
Created on Mon Jun 14 10:57:32 2021
@author: JC

Last version: 2021-08-03

"""

import os
import sys
sys.path.append(r"C:\Users\JC\PhD\CircleFit-Notebook")
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import interpolate
from scipy.signal import peak_widths, find_peaks
import DataModule as dm
import pandas as pd

# Plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.titlesize'] = 23
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15



#%% Frequency sweep

##############################################################################
############################ FREQUENCY SWEEP #################################
##############################################################################

# TO DO: Add fit to the plot 

# Load the data
folder = os.path.join( r'D:\SAW\SAW-KTH.hdf5' )
file = h5py.File( folder, 'r' )
print( file.keys() )

# Type of measurement
meas_type = file[ 'frequency sweep' ]
keys = list(meas_type.keys())

# Dataset of interest
hf = meas_type[ keys[3] ]


# Frequency range
f = np.array( hf['freq sweep'][:] )

# Bandwidth
df = np.array( hf['df'] )

# NCO frequency
fNCO = np.array( hf['fNCO'] )

# Raw data
usb_data = np.array( hf['USB data'][:] )

# Phase
phase_data = sig.detrend( np.unwrap( np.angle(usb_data) ) )

# Amplitude
amp_data = 20 * np.log10( np.abs(usb_data) )
amp_data_linear = -10 ** (amp_data / 20)  # linear scale


# Plotting
fig, ax = plt.subplots(2, sharex = True)
ax[0].plot( f+fNCO, amp_data )
ax[1].plot( f+fNCO, phase_data )
ax[1].set_xlabel( 'frequency [Hz]' )
ax[1].set_ylabel( 'phase [rad]' )
ax[0].set_ylabel( 'amplitude [dBFS]' )


# CHARACTERIZING THE SAW SAMPLE 
# Author: Sara Persia

# Parameters to find the peaks
n = 3000    # number of data points we discard at the beggining and at the end
h = 59.4    # minimum high of the peaks
d = 2100    # mimimum distance between peaks

# Indices of the amplitude minima
min_ind = find_peaks( x = np.abs( amp_data[n:-n] ),
                      height = h,
                      distance = d,
                      )

# Width of each peak
w = peak_widths( amp_data_linear, min_ind[0] + n, rel_height=0.5 )

# Resonant frequencies and Q factors
res_freq = np.zeros( len(min_ind[0]) )
Q = np.zeros_like( res_freq )

# Resonant frequencies and Q factors using circlefit
fr = np.zeros_like(res_freq)
QL = np.zeros_like(res_freq)
Qc = np.zeros_like(res_freq)
Qint = np.zeros_like(res_freq)


# Plotting
fig, ax = plt.subplots(1)
ax.set_xlabel( 'frequency [Hz]' )
ax.set_ylabel( 'amplitude [dBFS]' )
ax.plot( f+fNCO, amp_data )


# Interpolate the resonance frequencies and calculate the Q factor for each resonance
for ii in range( len(min_ind[0]) ):
    
    # Three frequencies and amplitudes around the minima
    f0 = f[ min_ind[0][ii] + n - 3 ]
    f1 = f[ min_ind[0][ii] + n ]       
    f2 = f[ min_ind[0][ii] + n + 3 ]
    a0 = amp_data[ min_ind[0][ii] + n - 3 ]
    a1 = amp_data[ min_ind[0][ii] + n ]
    a2 = amp_data[ min_ind[0][ii] + n + 3 ]
    
    # Interpolation of the resonant frequency
    x = np.linspace( f0, f2, 201 )
    # Interpolation with a 2nd order polynomial
    y = interpolate.interp1d( np.array([f0,f1,f2]), np.array([a0,a1,a2]), kind = 2 )
    # Store the index of the minimum amplitude
    int_ind = np.argmin( y(x) )
    # Resonance frequency
    res_freq[ii] = x[int_ind]
    
    # Q factor
    Q[ii] = (fNCO + res_freq[ii]) / (w[0][ii]*1000)
    
    # Plotting
    ax.plot( x+fNCO, y(x), color = 'red' )
    ax.axvline( res_freq[ii]+fNCO, ls = '--', color = 'green' )
    
    # We chunck the data around the resonant frequencies
    fwindow = f[ min_ind[0][ii]+n-500:min_ind[0][ii]+n+500 ] + fNCO
    usb_window = usb_data[ min_ind[0][ii]+n-500:min_ind[0][ii]+n+500 ]
    data_window = dm.data_complex(fwindow, usb_window)
    
    ax.axvline( fwindow[0], ls = 'dotted', color = 'orange' )
    ax.axvline( fwindow[-1], ls = 'dotted', color = 'orange' )
    
    # Circle fit
    fit = data_window.circle_fit_reflection()
    result = data_window.fitresults
    
    # Convert the results into a pandas dataframe and later to a dictionary
    df = pd.DataFrame(result)
    dfdict = df.to_dict()
    
    # Resonant frequencies and Q factors for each resonance
    fr[ii] = dfdict['Value']['fr (GHz)']
    QL[ii] = dfdict['Value']['QL']
    Qc[ii] = dfdict['Value']['Qc']
    Qint[ii] = dfdict['Value']['Qint']

    
# Free Spectral Range
FSR = np.mean( res_freq[1:] - res_freq[:-1] )
FSR_std = np.std( res_freq[1:] - res_freq[:-1] )

# Plotting the external and internal Q factors
fig, ax = plt.subplots(1)
ax.plot( Qc, '.', label='Q$_c$' )
ax.plot( Qint, '.', label='Q$_{int}$' )
ax.set_xlabel( 'mode number' )
ax.set_ylabel( 'Q factor' )
ax.legend()


# Close the hdf5 file
file.close()



#%% Amplitude sweep

##############################################################################
############################ AMPLITUDE SWEEP #################################
##############################################################################

# Load the data
folder = os.path.join( r'D:\SAW\SAW-KTH.hdf5' )
file = h5py.File( folder, 'r' )
print( file.keys() )

# Type of measurement
meas_type = file[ 'amplitude sweep' ]
keys = list(meas_type.keys())

# Dataset of interest
hf = meas_type[ keys[8] ]

# Frequencies
f = np.array( hf['freq sweep'] )

# Bandwidth
df = np.array( hf['df'] )

# NCO frequency
fNCO = np.array( hf['fNCO'] )

# Pump amplitudes
amp_sweep = np.array( hf['amp sweep'] )
pamp = 20 * np.log10( np.array( hf['amp sweep'] ) )

# Raw data
usb_data = np.array( hf['USB data'] )

# Phase
phase_data = sig.detrend( np.unwrap( np.angle(usb_data) ) )

# Amplitude normalised
amp_data = 20 * np.log10( np.abs(usb_data) ) - pamp
amp_data_linear = -10 ** (amp_data / 20)  # linear scale


# 2D plot of the normalized amplitude as a function of frequency and power    
fig, ax = plt.subplots(1)
a = ax.pcolormesh( f+fNCO, pamp, np.transpose(amp_data), shading='nearest', cmap='RdBu_r' )
fig.suptitle( 'Power sweep' )
ax.set_xlabel( 'frequency [Hz]' )
ax.set_ylabel( 'power [dBFS]' )
fig.colorbar( a, label='dBFS' )


# CHARACTERIZING THE SAW SAMPLE 
# Author: Sara Persia

# Parameters to find the peaks
n = 3000    # number of data points we discard at the beggining and at the end
h = 57.4    # minimum high of the peaks
h = 53
d = 2100    # mimimum distance between peaks

# Resonant frequencies and Q factors
res_freq = np.zeros( (len(pamp), 29) )
Q = np.zeros_like( res_freq )
    
# Resonant frequencies and Q factors using circlefit
fr = np.zeros_like( res_freq )
QL = np.zeros_like( res_freq )
Qc = np.zeros_like( res_freq )
Qint = np.zeros_like( res_freq )

# Free spectral range
FSR = np.zeros_like( pamp )
FSR_std = np.zeros_like( FSR )

# Plotting
fig, ax = plt.subplots(1)
ax.set_xlabel( 'frequency [Hz]' )
ax.set_ylabel( 'amplitude [dBFS]' )

# Pump amplitude sweep
for pow_ind in range( len(pamp) ):

    # Indices of the amplitude minima
    min_ind = find_peaks( x = np.abs( amp_data[n:-n, pow_ind] ),
                          height = h,
                          distance = d,
                          )
    
    # Width of each peak
    w = peak_widths( amp_data_linear[:,pow_ind], min_ind[0] + n, rel_height=0.5 )
       
    # Plotting
    ax.plot( f+fNCO, amp_data[:,pow_ind], label=str(pamp[pow_ind])+'dB' )
    
    
    # Interpolate the resonance frequencies and calculate the Q factor for each resonance
    for ii in range( len(min_ind[0]) ):
        
        # Three frequencies and amplitudes around the minima
        f0 = f[ min_ind[0][ii] + n - 3 ]
        f1 = f[ min_ind[0][ii] + n ]       
        f2 = f[ min_ind[0][ii] + n + 3 ]
        a0 = amp_data[ min_ind[0][ii] + n - 3, pow_ind ]
        a1 = amp_data[ min_ind[0][ii] + n, pow_ind ]
        a2 = amp_data[ min_ind[0][ii] + n + 3, pow_ind ]
        
        # Interpolation of the resonant frequency
        x = np.linspace( f0, f2, 201 )
        # Interpolation with a 2nd order polynomial
        y = interpolate.interp1d( np.array([f0,f1,f2]), np.array([a0,a1,a2]), kind = 2 )
        # Store the index of the minimum amplitude
        int_ind = np.argmin( y(x) )
        # Resonance frequency
        res_freq[pow_ind, ii] = x[int_ind]
        
        # Q factor
        Q[pow_ind, ii] = (fNCO + res_freq[pow_ind, ii]) / (w[0][ii]*1000)
        
        # Plotting
        ax.plot( x+fNCO, y(x), color = 'red' )
        ax.axvline( res_freq[pow_ind,ii]+fNCO, ls = '--', color = 'green' )
        
        # We chunck the data around the resonant frequencies
        fwindow = f[ min_ind[0][ii]+n-500:min_ind[0][ii]+n+500 ] + fNCO
        usb_window = usb_data[ min_ind[0][ii]+n-500:min_ind[0][ii]+n+500, pow_ind ]
        data_window = dm.data_complex(fwindow, usb_window)
        
        # Circle fit
        fit = data_window.circle_fit_reflection()
        result = data_window.fitresults
        
        # Convert the results into a pandas dataframe and later to a dictionary
        df = pd.DataFrame(result)
        dfdict = df.to_dict()
        
        # Resonant frequencies and Q factors for each resonance
        fr[pow_ind,ii] = dfdict['Value']['fr (GHz)']
        QL[pow_ind,ii] = dfdict['Value']['QL']
        Qc[pow_ind,ii] = dfdict['Value']['Qc']
        Qint[pow_ind,ii] = dfdict['Value']['Qint']
    
        
    # Free Spectral Range
    FSR[pow_ind] = np.mean( res_freq[pow_ind,1:] - res_freq[pow_ind,:-1] )
    FSR_std[pow_ind] = np.std( res_freq[pow_ind,1:] - res_freq[pow_ind,:-1] )
    
ax.legend()

# Close the hdf5 file
file.close()


# # SAVING THE RESULTS IN A hdf5 FILE
# folder = os.path.join( r'D:\VivaceData\SAW-KTH-Parameters-v2.hdf5' )
# datorfile = h5py.File( folder, 'r+' )

# # Save data (use this in case the dataset is not created)
# datorfile.require_dataset( "res freqs", (np.shape(res_freq)), dtype=float, data=res_freq+fNCO )
# datorfile.require_dataset( "Q loaded", (np.shape(QL)), dtype=float, data=QL )
# datorfile.require_dataset( "Q external", (np.shape(Qc)), dtype=float, data=Qc )
# datorfile.require_dataset( "Q internal", (np.shape(Qint)), dtype=float, data=Qint )
# datorfile.require_dataset( "FSR", (np.shape(FSR)), dtype=float, data=FSR )
# datorfile.require_dataset( 'pump amps', (np.shape(amp_sweep)), dtype=float, data=amp_sweep)

# # # Save data (use this in case the dataset is already created)
# # freqs = datorfile["res freqs"]
# # freqs[...] = res_freq+fNCO
# # Qloaded = datorfile["Q loaded"]
# # Qloaded[...] = QL
# # Qext = datorfile["Q external"]
# # Qext[...] = Qc
# # Qintern = datorfile["Q internal"]
# # Qintern[...] = Qint
# # FSRange = datorfile["FSR"]
# # FSRange[...] = FSR


# # Close the hdf5 file
# datorfile.close()



#%% Two tone spectroscopy

##############################################################################
############################ TWO TONE SPECTROSCOPY ###########################
##############################################################################

# Load the data
folder = os.path.join( r'D:\SAW\SAW-KTH.hdf5' )
file = h5py.File( folder, 'r' )
print( file.keys() )

# Type of measurement
meas_type = file[ 'two tone spectroscopy' ]
keys = list(meas_type.keys())

# Dataset of interest
hf = meas_type[ keys[1] ]

# Pump, signal and idler frequencies
f_pump = hf['pump freqs'][:]
f_signal = hf['signal freqs'][:]
f_idler = 2*f_pump - f_signal

# Bandwidth
df = np.array( hf['df'] )

# NCO frequency
fNCO = np.array( hf['fNCO'] )

# Pump and signal amplitudes
pow_pump = np.array( hf['pump amps'] )
pow_sig = np.array( hf['signal amps'] )

# Raw data
usb_data = hf['USB data'][:]


# Signal and idler amplitude
signal_amp = 20 * np.log10( np.abs(usb_data[:,:,0]) )
idler_amp = 20 * np.log10( np.abs(usb_data[:,:,2]) )

zmin_s = np.min(signal_amp)
zmax_s = np.max(signal_amp)
zmin_i = np.min(idler_amp)
zmax_i = np.max(idler_amp)

# Plotting
fig, ax = plt.subplots( 2, 1, sharex=True )
a = ax[0].pcolormesh( f_pump, f_signal, signal_amp, cmap='RdBu_r', vmin=zmin_s, vmax=zmax_s )
b = ax[1].pcolormesh( f_pump, f_idler, idler_amp, cmap='RdBu_r', vmin=zmin_i, vmax=zmax_i )
fig.suptitle( 'Two tone spectroscopy' )
ax[0].set_title( 'Signal Amplitude' )
ax[1].set_title( 'Idler Amplitude' )
ax[0].set_ylabel( 'signal freq [Hz]' )
ax[1].set_xlabel( 'pump freq [Hz]' )
ax[1].set_ylabel( 'signal freq [Hz]' )
fig.colorbar( a, ax=ax[0], label='dBFS' )
fig.colorbar( b, ax=ax[1], label='dBFS' )

# Close the hdf5 file
file.close()



#%% Two mode squeezing

##############################################################################
############################ TWO MODE SQUEEZING ##############################
##############################################################################

# Load the data
folder = os.path.join( r'D:\SAW\SAW-KTH.hdf5' )
file = h5py.File( folder, 'r' )
print( file.keys() )

# Type of measurement
meas_type = file[ 'two mode squeezing' ]
keys = list(meas_type.keys())

# Dataset of interest
hf = meas_type[ keys[0] ] 

# Amplitude group
keys2 = list(hf.keys())
hf2 = hf[ 'pump_amp_ind_1' ]

# Pump, signal and idler frequencies
f_pump = np.array( hf2['pump freqs'] )
f_signal = np.array( hf2['signal freqs'][:] )

# Bandwidth
df = np.array( hf2['df'] )

# NCO frequency
# fNCO = np.array( hf['fNCO'] )

# Raw data
usb_data = hf2['USB data'][:]
usb_data_off = hf2['USB data OFF'][:]

# Generating the covariance matrix for the pump on data
nmodes = np.arange( 0, len(usb_data[0]), 1 )

real_part = np.real( usb_data )
im_part = np.imag( usb_data )
coord_array = [ ( real_part[:, v], im_part[:, v] ) for v in nmodes  ]
coord_array = np.concatenate( coord_array, axis = 0 )
cov_matrix = np.cov( coord_array ) * 6.33645961**2


# Set the labels
listI = ['I$_{'+str(nmodes[i])+'}$' for i in range( len(nmodes) ) ]
xlabels = np.arange( -0.5+nmodes[0], 2*nmodes[-1]+1, 2 )
ylabels = np.arange( 0.5+nmodes[0], 2*nmodes[-1]+1, 2 )
x = np.arange( -1 + nmodes[0], 2*nmodes[-1]+2, 1)
grid_arr = np.arange( -1 + nmodes[0], 2*nmodes[-1]+2, 1 )

zmax = np.max( cov_matrix ) / 1000
zmin = -zmax 

fig1, ax1 = plt.subplots(1)
a = ax1.pcolormesh( x, x, np.flipud( cov_matrix ), shading='nearest', cmap='RdBu_r', vmin=zmin, vmax=zmax )
ax1.set_title( 'Experimental Covariance Matrix', fontsize='large' )
fig1.colorbar( a )
plt.xticks( xlabels, listI, fontsize='large' )
plt.yticks( ylabels, np.flipud(listI), fontsize='large' )
plt.grid( True, which='minor', axis='both', linestyle='-', color='b', linewidth=1.5 )



# Potting histogram for center mode
mode = 9
x1 = np.real( usb_data[:, mode] )
x2 = np.imag( usb_data[:, mode] )
x1_off = np.real( usb_data_off[ : len(x1), mode] )
x2_off = np.imag( usb_data_off[ : len(x1), mode] )
binr = np.max( np.append(x1, x2) )*1.1
n_bins = 201
II_corr, xedgesh, yedgesh = np.histogram2d( x1, x2, 
                                            bins = ( np.linspace(-binr,binr,n_bins), np.linspace(-binr,binr,n_bins) ) )
II_corr_off, xedgesh, yedgesh = np.histogram2d(x1_off, x2_off, 
                                                bins = ( np.linspace(-binr,binr,n_bins), np.linspace(-binr,binr,n_bins) ) )
center_hist = (II_corr - II_corr_off)/np.max( II_corr - II_corr_off )
fig3, ax3 = plt.subplots(1, 1)
cent_hist_fig = ax3.contourf( xedgesh[0:n_bins-1], yedgesh[0:n_bins-1], II_corr, cmap = 'RdBu_r' )    
ax3.set_aspect('equal')
fig3.colorbar(cent_hist_fig)



# Plotting histogram for a symmetric pair around some pump
mode1 = 8
mode2 = 10
x1_o = np.real( usb_data[:, mode1] )
x2_o = np.real( usb_data[:, mode2] )
x1_o_off = np.real( usb_data_off[:len(x1_o), mode1] )
x2_o_off = np.real( usb_data_off[:len(x1_o), mode2] )
n_bins = 201
II_corr, xedgesh, yedgesh = np.histogram2d( x1_o, x2_o, 
                                            bins = ( np.linspace(-binr,binr,n_bins), np.linspace(-binr,binr,n_bins) ) )
II_corr_off, xedgesh, yedgesh = np.histogram2d( x1_o_off, x2_o_off,
                                                bins = ( np.linspace(-binr,binr,n_bins), np.linspace(-binr,binr,n_bins) ) )
twomode_hist = (II_corr - II_corr_off)/np.max( II_corr - II_corr_off )
fig3, ax3 = plt.subplots(1, 1)
twomode_hist_fig = ax3.contourf( xedgesh[0:n_bins-1], yedgesh[0:n_bins-1], II_corr, cmap = 'RdBu_r' )    
ax3.set_aspect('equal')
fig3.colorbar(twomode_hist_fig)




# Close the hdf5 file
file.close()
