# -*- coding: utf-8 -*-

"""
Created on Tue Nov  2 14:29:18 2021
@author: JC

Last version: 2021-12-20

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import h5py
from vivace import lockin


# Plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.constrained_layout.use'] = True
# plt.rcParams['figure.titlesize'] = 23
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["legend.frameon"] = False



#%%

# Objective: Reconstruction of a non-linear resonator from their intermodulation spectrum by harmonic balance
# Check TLS Project pdf for more theory details

# Load data file
file = h5py.File(r"D:\TLS\TLS-Data\TLS_IMPspec.hdf5", "r")


# Resonance frequency of the system
f_res = 4.1105253e9

# Order of the higher non-linear Duffing term
reconstruction_order = 3

# Parameters for Vivace
wanted_df = 10
N_pix = 50
mixer_freq = 4.05e9
comb_spacing = 500

# Amplitude and frequency sweep arrays 
output_amp_array = np.logspace( -3, -0.5, 21 )
f_center_array = np.linspace( f_res-40e3, f_res+40e3, 161 )


# Figure
plt.figure( figsize=(6,12) )
ax1 = plt.subplot( 511 )
ax2 = plt.subplot( 512 )
ax3 = plt.subplot( 513 )
ax4 = plt.subplot( 514 )
ax5 = plt.subplot( 515 )

# Since in the measurement the bandwidth (df) was not saved, 
# we need to call Vivace to tune everthing and obtain an estimate of df
# Note that this is not the latest version of Vivace, but it works
with lockin.MixLockin( dac_freq=6.0, dry_run=True ) as lck:
	# Frequency loop (one frequency selected)
	for f_center_idx in [72]:
		# Check
		print( "f_center_idx", f_center_idx )
		
		# Center frequency of the comb
		f_center = f_center_array[ f_center_idx ]
		f_center_comb = f_center - mixer_freq
		# Pump frequencies
		p1_freq = f_center_comb - comb_spacing
		p2_freq = f_center_comb + comb_spacing
		# Tuning and obtaining df
		p1_tuned, _df = lck.tune( p1_freq, wanted_df )
		df = lck.set_df( _df )
		
		# Building the frequency comb
		n1 = p1_tuned / df
		n2 = np.round( p2_freq / df )
		
		# We ensure the difference between both pumps is a multiple of 2*df
		if ( n1%2==0 and n2%2==0 ) or ( n1%2==1 and n2%2==1 ):
			p2_tuned = n2 * df
		else:
			p2_tuned = (n2+1) * df
		
		# Fundamental building block of the comb
		Delta = np.abs( p1_tuned - p2_tuned ) / 2
		center_freq = ( p1_tuned + p2_tuned ) / 2
		comb_freqs = np.zeros(32)
		# Center of the comb
		comb_freqs[16] = center_freq
		# Pump frequencies
		comb_freqs[15] = p1_tuned
		comb_freqs[17] = p2_tuned
		
		# Sanity check
		if ( comb_freqs[17] - Delta ) != comb_freqs[16]:
			print( 'center frequency is not tuned!' )
		
		# Build the remaining comb
		comb_freqs[0:15] = np.arange( -15, 0, 1 ) * Delta + comb_freqs[15]
		comb_freqs[18:] = np.arange( 1, 15, 1 ) * Delta + comb_freqs[17]
		
		# I and Q quadratures
		phases_I = np.zeros_like( comb_freqs )
		phases_Q = -np.ones_like( comb_freqs ) * np.pi / 2
		drive_phase =  0	# -np.pi/2?
		
		# amplitude loop
		for amp_idx in np.arange( len(output_amp_array) )[:1:-2]: # plot small amp curves on top of large amp
			# Check
			print( "amp_idx", amp_idx )
			
			# Pump amplitudes
			signal_amp = output_amp_array[amp_idx]
			signal_amp_array = np.zeros( 32, dtype=np.complex )
			signal_amp_array[15] = signal_amp * np.exp( 1j*drive_phase )
			signal_amp_array[17] = signal_amp * np.exp( 1j*drive_phase )
			
			# Intermodulation spectra
			spectrum = file['USB data'][amp_idx, f_center_idx]
			
			# Take out only the IMPs, and use names matching previous script
			comb_freqs_imp = comb_freqs[1::2]
			spectrum_imp = spectrum[1::2]
			signal_amp_array_imp = signal_amp_array[1::2]
			
			# Bandwidth (the original measurement is smaller but for the reconstruction it is enough to just use one beat
			df = Delta
			
			# inspired by the use of mixing frequency I realize we can mix down to an even lower frequency to speed up calculation
			# We should just make sure that lowest karray is fairly large with respect to ... what? Either separation or total width of karray
			karray = np.int_( np.around(comb_freqs_imp / df) )
			karray_width = np.max(karray) - np.min(karray)
			karray_IF = np.min(karray) - karray_width * 10
			karray_IF_freq = karray_IF * df
			karray = karray - karray_IF
			
			# Create zero-paded spectrum and time-domain signal
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
			
			# karray of only non-driven tones
			# Note: original drive tones are element 15 and 17, I then take only every odd inded [1::2]
			# such that those elements end up on 7 and 8.
			karray_non_driven = np.hstack( (karray[:7], karray[9:]) )
			
			# Create H-matrix
			# Model: m(d2y/dt2) + c(dy/dt) + ky + d(dy/dt)|dy/dt| + gy**3 = F0*Vdrive
			# => m/F0*(d2y/dt2) + c/F0*(dy/dt) + k/F0*y + d/F0 (dy/dt)|dy/dt| + g/F0*y**3 = V1*cos(w1t) + V2*cos(w2t)
			# First column has mass term, i.e. -w**2*Y
			# Second column has damping term, i.e iw*Y
			# Third column has linear stiffness, i.e Y
			# Fourth column has the non-linear damping
			# Fifth column has 3th order duffing, i.e. FFT(y**3)
			
			col1 = (-w**2 * Y)[karray_non_driven]
			col2 = (1.0j * w * Y)[karray_non_driven]
			col3 = (Y)[karray_non_driven]
			col4 = (np.fft.rfft( ydot * np.abs(ydot) ) / len(y))[karray_non_driven]
			
			# Merge all columns
			H = np.vstack( (col1, col2, col3, col4) )
			
			# Higher order Duffing terms
			for i in range(reconstruction_order//2):
				col = (np.fft.rfft(y**(2*i+3))/len(y))[karray_non_driven]
				H = np.vstack((H, col))
				
			# Make matrix real instead of complex
			Hcos = np.real( H )
			Hsin = np.imag( H )
			H = np.hstack( (Hcos, Hsin) )
			
			# The drive vector, Q (from the yasuda paper)
			# Qcos = np.real(signal_amp_array_imp)
			# Qsin = np.imag(signal_amp_array_imp)
			# Q = np.hstack((Qcos, Qsin))
			
			# Normalize H for more stable inversion
			N = np.diag( 1./np.max(np.abs(H), axis=1) )
			H_norm = np.dot( N, H ) # normalized H-matrix
			
			# Solve system 0 = H*p by finding the null space
			u, s, vh = scipy.linalg.svd( H_norm.T )
			
			# Take smallest singluar value to be null vector
			p_norm = vh[-1]
			
			# Re-normalize p-values
			# note, we have actually solved Q = H * N * Ninv * p
			# thus we obtained Ninv*p and multiply by N to obtain p
			p = np.dot( N, p_norm ) # re-normalize parameter values
			
			# Forward calculation to check result, should be zero vector
			Q_fit = np.dot( p, H )
			
			# Scale result to known m
			m = 1.
			p = m / p[0] * p
			
			# Scale parameters by drive force assuming known mass
			m, c, k, d = p[:4]
			g = p[4:]               # nonlinear polynomial parameters, 3, 5 etc
			
			# Calculate resonance frequency and Q-factor, note in both of these the unknown F0 scaling falls out
			w0 = np.sqrt( k/m )
			# Mix up to get true resonance frequency
			w0 += 2 * np.pi * mixer_freq + 2 * np.pi * karray_IF_freq 
			f0 = w0 / 2 / np.pi
			# gamma = c/2m, Q~= w0/(2gamma) = w0 /(c/m)
			Q_factor = m * w0 / c
			
			# Check of the reconstructed values
			label = "amp {:.3f}: f0={:.1f} Hz, Q={:.3g}, m={:.3g}, k={:.3g}, c={:.3g}, d={:.3g}, g={:.3g}".format(signal_amp, f0, Q_factor, m, k, c, d, g[0])
			print( label )
			
			# Plot experimental data
			ax1.semilogy( freqs[karray] + mixer_freq + karray_IF_freq, np.abs( Y[karray] )*1e6, '-' )
			
			# Plot reconstruction of nonlinear force vs deflection in the measured range
			ys = np.linspace(min(y), max(y))
			nl_reconstructed = np.zeros_like(ys)
			for i in range(reconstruction_order//2):
				nl_reconstructed += g[i]*ys**(2*i+3)
			static_reconstructed = k*ys**1 + nl_reconstructed
			ax2.plot(ys*1e3, static_reconstructed, '-', label=label)
			ax3.plot(ys*1e3, nl_reconstructed, '-', label="reconstructed")
			
			# Plot reconstruction of the damping nonlinear force vs velocity in the measured range
			ys = np.linspace( np.min(ydot), np.max(ydot) )
			nl_reconstructed = d * ys * np.abs(ys)
			static_reconstructed = c * ys + nl_reconstructed
			ax4.plot( ys*1e3, static_reconstructed, '-', label=label )
			ax5.plot( ys*1e3, nl_reconstructed, '-', label="reconstructed")


# Axis labels
ax1.set_ylabel( "Defl. amplitude ($\mu$m)" )
ax1.set_xlabel( "Frequency (Hz)" )

ax2.set_ylabel("Total static force (N)")
ax2.set_xlabel("Deflection (mm)")

ax3.set_ylabel("Nonlinear static force (N)")
ax3.set_xlabel("Deflection (mm)")

ax4.set_ylabel( "Total damping force (N)" )
ax4.set_xlabel( "Velocity (mm/s)" )

ax5.set_ylabel( "Nonlinear damping force (N)" )
ax5.set_xlabel( "Velocity (mm/s)" )

plt.tight_layout()

plt.show()
