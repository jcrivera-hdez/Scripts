# -*- coding: utf-8 -*-

"""
Created on Thu Nov  4 17:14:44 2021
@author: JC

Last version: 2022-01-25

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import h5py


# Plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.constrained_layout.use'] = True
# plt.rcParams['figure.titlesize'] = 23
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["legend.frameon"] = False



#%%

# Objective: Reconstruction of a non-linear resonator from their intermodulation spectrum by harmonic balance
# Check TLS Project pdf for more theory details

# Load data file
file = h5py.File(r"D:\TLS\TLS-Data\TLS_IMP_amped_DCAW23_200amps_11points.hdf5", "r")


# Resonance frequency of the system
f_res = 4.1105253e9

# Order of the higher non-linear damping term
reconstruction_order = 2

# Model
# model = 'Z'                   # m(d2y/dt2) + c(dy/dt) + c2(y)^2 (dy/dt)     + ky = F0*Vdrive
model = 'Zdot'                # m(d2y/dt2) + c(dy/dt) + c2(dy/dt)^2 (dy/dt) + ky = F0*Vdrive

# Frequency and amplitude sweep arrays
mixer_frequencies = file[ 'mixer frequencies' ][:]
output_amp_array = file[ 'signal amplitudes' ][:]

# Amplitude in dBm
output_amp_array_dBm = 16.612 * np.log10(output_amp_array) - 16.743

# Fit parameters
m_arr = np.zeros(( len(mixer_frequencies), len(output_amp_array) ))
gamma_arr = np.zeros_like( m_arr )
alpha_arr = np.zeros_like( m_arr )
k0_arr = np.zeros_like( m_arr )
f0_arr = np.zeros_like( m_arr )
gamma_nl_arr = np.zeros(( len(mixer_frequencies), len(output_amp_array), reconstruction_order//2 ))

# Experimental data
spectrum_arr = np.zeros(( len(mixer_frequencies), len(output_amp_array), 61 ), dtype=complex )

# Frequency loop
for mix_freq_ind, mix_freq_val in enumerate( mixer_frequencies ):
    
    print( "mix_freq_ind", mix_freq_ind )
    
    # Figure
    fig = plt.figure( figsize=(12,12) )
    
    gs = fig.add_gridspec(3,2)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    
    
    # Amplitude loop
    for amp_ind, amp_val in enumerate( output_amp_array ):
        
        folder_amp = file[ 'signal_amp_' + str(amp_val) ]
        
        # Bandwidth of the measurement
        df = folder_amp[ 'tuned df, Npix, mixerfreq' ][0]
        # Number of pixels
        N_pix = folder_amp[ 'tuned df, Npix, mixerfreq' ][1]
        # Mixer frequency (center frequency of our comb)
        mixer_freq = mix_freq_val
        
        # USB and LSB frequency combs 
        comb_freqs_usb = folder_amp[ 'frequency combs' ][0][:-1]    # We remove the last column due to some mistake (?) in the measurement
        comb_freqs_lsb = -np.flip( comb_freqs_usb[1:] )             # We remove the first column because it is already in the USB
        
        comb_freqs = np.concatenate( (comb_freqs_lsb, comb_freqs_usb) )
        
        # Raw data, we remove the same indices as for the frequency comb
        spectrum_usb = folder_amp[ 'USB data' ][mix_freq_ind][:-1]
        spectrum_lsb = np.flip( folder_amp[ 'LSB data' ][mix_freq_ind][1:-1] )
        
        spectrum = np.concatenate( (spectrum_lsb, spectrum_usb) )
        
        spectrum_arr[mix_freq_ind, amp_ind] = spectrum
        
        # We take out only the IMPs
        comb_freqs_imp = comb_freqs[::2]
        spectrum_imp = spectrum[::2]
        
        # Trimmering the noise
        # comb_freqs_imp = comb_freqs_imp[10:21]
        # spectrum_imp = spectrum_imp[10:21]
        
        # Inspired by the use of mixing frequency I realize we can mix down to an even lower frequency to speed up calculation
		# We should just make sure that lowest karray is fairly large with respect to ... what? Either separation or total width of karray
        karray = np.int_( np.around(comb_freqs_imp / df) )
        karray_width = np.max(karray) - np.min(karray)
        karray_IF = np.min(karray) - karray_width * 10
        karray_IF_freq = karray_IF * df
        karray = karray - karray_IF
        
        # Create zero-paded spectrum
        Y = np.zeros( np.max(karray)*10, dtype=np.complex ) # this has to be significantly larger than the largest value in the karray
        Y[karray] = spectrum_imp 
        # Time-domain signal
        y = np.fft.irfft(Y)
        y *= len(y) # scaling because we divide by integration length in the lockin        
        
        # Frequency range (just to plot in the frequency domain)
        freqs = df * np.arange( len(Y) )
        # Angular frequency
        w = 2*np.pi*freqs
        
        # Derivative of the signal in time domain
        ydot = np.fft.irfft( 1.0j * w * Y )
        # ydot = np.gradient( y, 1/df )
        
        # karray of only non-driven tones
        karray_non_driven = np.hstack( (karray[:int(len(karray)/2)], karray[int(len(karray)/2)+2:]) )
        
        
        # Create H-matrix
        # First column has the mass term, i.e. -w**2*Y
        # Second column has the linear damping term, i.e iw*Y
        # Third column has the linear stiffness, i.e Y
        # Fourth column has the non-linear Duffing, i.e. FFT(y**3)
        # Fifth column has the non-linear damping
        
        col1 = (-w**2 * Y)[karray_non_driven]
        col2 = (1.0j * w * Y)[karray_non_driven]
        col3 = (Y)[karray_non_driven]
        col4 = ( np.fft.rfft(y**3) / len(y) )[karray_non_driven]
        
        # Merge all columns
        H = np.vstack( (col1, col2, col3, col4) )
        
        # Higher order in non-linear damping
        if model == 'Z':
            for i in range( reconstruction_order//2 ):
                col = (np.fft.rfft( ydot * y**(2*i+2) ) / len(y))[karray_non_driven]
                H = np.vstack((H, col))
        elif model == 'Zdot':
            for i in range( reconstruction_order//2 ):
                col = (np.fft.rfft( ydot * ydot**(2*i+2) ) / len(y))[karray_non_driven]
                H = np.vstack((H, col))
        
            
        # Making the matrix real instead of complex
        Hcos = np.real( H )
        Hsin = np.imag( H )
        H = np.hstack( (Hcos, Hsin) )
        
        # Normalize H for a more stable inversion
        N = np.diag( 1. / np.max(np.abs(H), axis=1) )
        H_norm = np.dot( N, H ) # normalized H-matrix
        
        # Solve system H*p=0 by finding the null space
        u, s, vh = scipy.linalg.svd( H_norm.T )  
        
        # Take the smallest singular value to be null vector
        p_norm = vh[-1]
        
        # Re-normalize p-values
		# Note: we have actually solved Q = H * N * Ninv * p
		# Thus we obtained Ninv*p and multiply by N to obtain p
        p = np.dot( N, p_norm ) # re-normalize parameter values
        
        # Forward calculation to check result, should be zero vector
        Q_fit = np.dot( p, H )
        
        # Scale result to known m
        m = 1.
        p = m / p[0] * p
        
        # Scale parameters by drive force assuming known mass
        m, c, k, a = p[:4]
        g = p[4:]               # nonlinear polynomial parameters, 2, 4 etc
        
        # Calculate resonance frequency and Q-factor, note in both of these the unknown F0 scaling falls out
        w0 = np.sqrt( k/m )
        # Mix up to get true resonance frequency
        w0 += 2 * np.pi * mixer_freq + 2 * np.pi * karray_IF_freq 
        f0 = w0 / 2 / np.pi
        
        # Q factor
        Q_factor = m * w0 / c
        
        
        # Save parameter values
        m_arr[mix_freq_ind,amp_ind] = m
        gamma_arr[mix_freq_ind,amp_ind] = c
        k0_arr[mix_freq_ind,amp_ind] = k
        alpha_arr[mix_freq_ind,amp_ind] = a
        f0_arr[mix_freq_ind,amp_ind] = f0
        for i in range(reconstruction_order//2):
            gamma_nl_arr[mix_freq_ind, amp_ind, i] = g[i]        
        
        # Plot experimental data
        ax0.semilogy( freqs[karray] + mixer_freq + karray_IF_freq, np.abs( Y[karray] )*1e6, '-' )
        ax0.axvline( freqs[karray[int(len(karray)/2)]] + mixer_freq + karray_IF_freq )
        ax0.axvline( freqs[karray[int(len(karray)/2)+1]] + mixer_freq + karray_IF_freq )
        
        # Plot reconstruction of the damping nonlinear force vs velocity in the measured range
        ys = np.linspace( np.min(y), np.max(y) )
        ys_dot = np.linspace( np.min(ydot), np.max(ydot) )
        
        # Conservative force terms
        # Non-linear Duffing
        nonlin_duffing = np.zeros_like( ys )
        for i in range( reconstruction_order//2 ):
            nonlin_duffing += a * ys**3
        # Total conservative force
        tot_duffing = k * ys + nonlin_duffing
        
        # Damping terms
        # Non-linear damping
        nonlin_damping = np.zeros_like( ys )
        for i in range( reconstruction_order//2 ):
            if model == 'Z':
                nonlin_damping += g[i] * ys_dot * ys**(2*i+2)
            elif model == 'Zdot':
                nonlin_damping += g[i] * ys_dot * ys_dot**(2*i+2)
        # Total damping (linear + non-linear)        
        tot_damping = c * ys_dot + nonlin_damping
        
        # Plotting
        ax1.plot( ys*1e3, nonlin_duffing, '-''')  
        ax2.plot( ys*1e3, tot_duffing, '-''')        
        ax3.plot( ys_dot*1e3, nonlin_damping, '-' )
        ax4.plot( ys_dot*1e3, tot_damping, '-' )
    
    # Sanity check
    # print( 'Pump 1 frequency: ' + str(freqs[karray[int(len(karray)/2)]] + mixer_freq + karray_IF_freq) )
    # print( 'Pump 2 frequency: ' + str(freqs[karray[int(len(karray)/2)+1]] + mixer_freq + karray_IF_freq) )
    # print( 'Pump 1 frequency - v2: ' + str(comb_freqs[30] + mixer_freq) )
    # print( 'Pump 2 frequency - v2: ' + str(comb_freqs[32] + mixer_freq) )
    
    
    # Axis labels
    ax0.set_ylabel( "Amplitude (a.u)" )
    ax0.set_xlabel( "Frequency (GHz)" )
    
    ax1.set_title( "Conservative Force", fontsize=23 )
    ax1.set_ylabel( "F$_{nl}$" )
    
    ax2.set_ylabel( "F$_{tot}$" )
    ax2.set_xlabel( "Deflection (mm)" )
    
    ax3.set_title( "Damping Force", fontsize=23 )
    ax3.set_ylabel( "F$_{nl}$" )
    
    ax4.set_ylabel( "F$_{tot}$" )
    ax4.set_xlabel( "Velocity (mm/s)" )
    

# Phase along mixer frequency at the two drives frequencies
fig, ax = plt.subplots(2)
ax[0].plot( (comb_freqs[2*int(len(karray)/2)]+mixer_frequencies)/1e9, np.angle( spectrum_arr[:, 8, 2*int(len(karray)/2)] ) )
ax[1].plot( (comb_freqs[2*int(len(karray)/2)]+mixer_frequencies)/1e9, np.abs( spectrum_arr[:, 8, 2*int(len(karray)/2)] ), label='Pump 1' )
ax[0].plot( (comb_freqs[2+2*int(len(karray)/2)]+mixer_frequencies)/1e9, np.angle( spectrum_arr[:, 8, 2+2*int(len(karray)/2)] ) )
ax[1].plot( (comb_freqs[2+2*int(len(karray)/2)]+mixer_frequencies)/1e9, np.abs( spectrum_arr[:, 8, 2+2*int(len(karray)/2)] ), label='Pump 2' )
ax[1].set_xlabel( 'center frequency (GHz)' )
ax[0].set_ylabel( 'Phase (rad)' )
ax[1].set_ylabel( 'Amplitude (a.u)' )
ax[1].legend() 

# Complex plane plot at a fixed amplitude
fig, ax = plt.subplots(1)
ax.plot( np.real( spectrum_arr[:,3,2*int(len(karray)/2)] ), np.imag( spectrum_arr[:,3,2*int(len(karray)/2)] ), '.' )


# Experimental raw data plot at a fixed mixer frequency
fig, ax = plt.subplots(1)
for i in range( len(output_amp_array) ):
    
    # Full spectrum
    ax.semilogy( (comb_freqs+mixer_frequencies[0])/1e9, np.abs( spectrum_arr[0,i,:] )*1e6 )
    
    # Only IMPs spectrum
    # ax.semilogy( (comb_freqs_imp+mixer_frequencies[0])/1e9, np.abs( spectrum_arr[0,i,::2] )*1e6 )

ax.set_ylabel( "Amplitude (a.u.)" )
ax.set_xlabel( "Frequency (GHz)" )



#%%

# 2D plot of the whatever value as a function of the mixer frequency and the comb frequency

# Figure parameters
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


fig2, ax2 = plt.subplots( 4, 3, constrained_layout=True )
ax2 = ax2.flatten()

# Color range
zmax = np.max( 20 * np.log10( np.abs(spectrum_arr[:,:,:])) )
zmin = -180 # np.min( 20 *  np.log( np.abs(spectrum_arr[:,:,:])) )

for axi in range( 4*3 ):
    if file == h5py.File(r"D:\TLS\TLS-Data\TLS_IMP_amped_DCAW23_200amps_11points.hdf5", "r"):
        amp_ind = 18*axi
    a = ax2[axi].pcolormesh(
        comb_freqs, mixer_frequencies/1e9, 20*np.log10( np.abs(spectrum_arr[:,amp_ind,:]) ), cmap='RdBu_r', vmax=zmax, vmin=zmin,
        )
    ax2[axi].set_title(f'amp_ind = {output_amp_array_dBm[18*axi]:.3f} FS')
fig.colorbar( a, ax=ax2[:], location='right', label=r'mag (a.u.)', shrink=0.6 )
[ax2[axi].set_xlabel('Comb frequencies (Hz)') for axi in [9, 10, 11]]
[ax2[axi].set_ylabel('Mixer frequency (GHz)') for axi in [0, 3, 6, 9]]