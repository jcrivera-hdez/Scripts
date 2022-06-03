# -*- coding: utf-8 -*-

"""
Created on Thu Nov 25 16:24:47 2021
@author: JC

Last version: 2022-01-20

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

# %%

# Objective: Reconstruction of a non-linear resonator from their intermodulation spectrum by harmonic balance
# Check TLS Project pdf for more theory details

# Load data file
file = h5py.File(r"D:\TLS\TLS-Data\TLS_IMP_amped_DCAW23_200amps_11points.hdf5", "r")
# file = h5py.File(r"D:\TLS\TLS-Data\TLS_IMP_amped_DCAW23_10amps_201points.hdf5", "r")


# Resonance frequency of the system
f_res = 4.1105253e9  # Hz
w_res = 2 * np.pi * f_res

# Waveguide-resonator coupling rate
Lambda = 0.5

# Gain of the output line
gain_output_line = 80  # dB
gain_output_line_linear = 10 ** (gain_output_line / 10)

# Attenuation of the inut line
att_input_line = 55 + 9 + 0.2 + 4.7 - 20  # dB
att_input_line_linear = 10 ** (att_input_line / 10)

# Order of the higher non-linear TLS term
reconstruction_order = 1

# Frequency and amplitude sweep arrays
mixer_frequencies = file['mixer frequencies'][:]
output_amp_array = file['signal amplitudes'][:]  # Vivace units

# Amplitude in dBm
output_amp_array_dBm = 16.612 * np.log10(output_amp_array) - 16.743

# Amplitude in W
output_amp_array_W = 10 ** ((output_amp_array_dBm - 30) / 10)

# Fit parameters
m_arr = np.zeros((len(mixer_frequencies), len(output_amp_array)))
gamma_arr = np.zeros_like(m_arr)
kappa0_arr = np.zeros_like(m_arr)
kappa_arr = np.zeros((len(mixer_frequencies), len(output_amp_array), reconstruction_order))

# Experimental data
spectrum_arr = np.zeros((len(mixer_frequencies), len(output_amp_array), 61), dtype=complex)
spectrum_arr_dBm = np.zeros_like

# Figure
fig = plt.figure(figsize=(12, 12))

gs = fig.add_gridspec(3, 2)
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])

# Nice colors for the figure
colors = plt.cm.rainbow(np.linspace(0, 1, 10 * len(mixer_frequencies)))

# Frequency loop
for mix_freq_ind, mix_freq_val in enumerate(mixer_frequencies):

    # Check
    print("mix_freq_ind", mix_freq_ind)

    # Amplitude loop
    for amp_ind, amp_val in enumerate(output_amp_array):

        # Amplitude folder
        folder_amp = file['signal_amp_' + str(amp_val)]

        # Bandwidth of the measurement
        df = folder_amp['tuned df, Npix, mixerfreq'][0]
        # Number of pixels
        N_pix = folder_amp['tuned df, Npix, mixerfreq'][1]
        # Mixer frequency (center frequency of our comb)
        mixer_freq = mix_freq_val

        # USB and LSB frequency combs
        comb_freqs_usb = folder_amp['frequency combs'][0][:-1]  # We remove the last column due to some mistake in the measurement
        comb_freqs_lsb = -np.flip(comb_freqs_usb[1:])  # We remove the first column because it is already in the USB
        # All frequencies in one array
        comb_freqs = np.concatenate((comb_freqs_lsb, comb_freqs_usb))

        # Raw data, we remove the same indices as for the frequency comb
        spectrum_usb = folder_amp['USB data'][mix_freq_ind][:-1]
        spectrum_lsb = np.flip(folder_amp['LSB data'][mix_freq_ind][1:-1])

        # All spectrum in one array and output line gain substracted
        spectrum = np.concatenate((spectrum_lsb, spectrum_usb))

        # Spectrum conversion from Vivace units to dBm
        spectrum_dBm = 16.612 * np.log10(spectrum) - 16.743

        # Spectrum conversion from dBm to W
        spectrum_W = 10 ** ((spectrum_dBm - 30) / 10)

        # Saving the spectra to plot it later
        spectrum_arr[mix_freq_ind, amp_ind] = spectrum_W

        # We take out only the IMPs
        comb_freqs_imp = comb_freqs[::2]
        spectrum_imp = spectrum_W[::2]

        # Trimmering the noise
        # comb_freqs_imp = comb_freqs_imp[8:23]
        # spectrum_imp = spectrum_imp[8:23]

        # Inspired by the use of mixing frequency I realize we can mix down to an even lower frequency to speed up calculation
        # We should just make sure that lowest karray is fairly large with respect to ... what? Either separation or total width of karray
        karray = np.int_(np.around(comb_freqs_imp / df))
        karray_width = np.max(karray) - np.min(karray)
        karray_IF = 0  # np.min(karray) - karray_width * 10
        karray_IF_freq = karray_IF * df
        karray = karray - karray_IF

        # Intra-cavity spectrum (we correct attenuation and gain on the input and output line respectively)
        spectrum_intracav = 1.0j * np.sqrt(2 / Lambda) * (-spectrum_imp / np.sqrt(gain_output_line_linear))
        spectrum_intracav[15] = 1.0j * np.sqrt(2 / Lambda) * (
                    output_amp_array_W[amp_ind] / np.sqrt(att_input_line_linear) - spectrum_imp[15] / np.sqrt(
                gain_output_line_linear))
        spectrum_intracav[16] = 1.0j * np.sqrt(2 / Lambda) * (
                    output_amp_array_W[amp_ind] / np.sqrt(att_input_line_linear) - spectrum_imp[16] / np.sqrt(
                gain_output_line_linear))

        # Create zero-paded spectrum
        Y = np.zeros(np.max(karray) * 10,
                     dtype=np.complex)  # This has to be significantly larger than the largest value in the karray
        Y[karray] = spectrum_intracav
        # Time-domain signal
        y = np.fft.ifft(Y)
        y *= len(y)  # Scaling because we divide by integration length in the lockin

        # Frequency range (just to plot in the frequency domain)
        # freqs = df * np.arange( len(Y) )
        freqs = np.fft.fftfreq(len(Y), d=1 / (df * len(Y)))
        # Angular frequency
        w = 2 * np.pi * freqs

        # Derivative of the signal in time domain
        ydot = np.fft.fft(1.0j * w * Y)
        # ydot = np.gradient( y, 1/df )

        # karray of only non-driven tones
        karray_non_driven = np.hstack((karray[:int(len(karray) / 2)], karray[int(len(karray) / 2) + 2:]))

        # Create H-matrix
        # First column has the damping linear term
        # Second column has the linear term
        # Third column has the polynomial TLS non-linear term

        col1 = (-1.0j * (w - w_res) * Y)[karray_non_driven]
        col2 = (Y)[karray_non_driven]

        # Merge all columns
        H = np.vstack((col1, col2))

        # Non-linear TLS columns
        for i in range(reconstruction_order):
            cola = (np.fft.fft( np.abs(y**(1 + i)) * y ) / len(y))[karray_non_driven]
            # colb = np.real( np.fft.fft(y**(2+i)) / len(y) )[karray_non_driven]
            H = np.vstack((H, cola))

        # Making the matrix real instead of complex
        Hcos = np.real(H)
        Hsin = np.imag(H)
        H = np.hstack((Hcos, Hsin))

        # Normalize H for a more stable inversion
        N = np.diag(1. / np.max(np.abs(H), axis=1))
        H_norm = np.dot(N, H)  # normalized H-matrix

        # Solve system H*p=0 by finding the null space
        u, s, vh = scipy.linalg.svd(H_norm.T)

        # Take the smallest singular value to be null vector
        p_norm = vh[-1]

        # Re-normalize p-values
        # Note: we have actually solved Q = H * N * Ninv * p
        # Thus we obtained Ninv*p and multiply by N to obtain p
        p = np.dot(N, p_norm)  # re-normalize parameter values

        # Forward calculation to check result, should be zero vector
        Q_fit = np.dot(p, H)

        # Scale result to known m
        # m = 1.
        # p = m / p[0] * p

        # Scale parameters by drive force assuming known mass
        m, kappa0 = p[:2]
        kappa = p[2:]

        # Save parameter values
        m_arr[mix_freq_ind, amp_ind] = m
        kappa0_arr[mix_freq_ind, amp_ind] = kappa0
        kappa_arr[mix_freq_ind, amp_ind] = kappa

        # Plot experimental data
        ax0.semilogy((freqs[karray] + mixer_freq + karray_IF_freq) / 1e9, np.abs(Y[karray]), '-',
                     color=colors[10 * mix_freq_ind])
    # ax0.axvline( (freqs[karray[int(len(karray)/2)]] + mixer_freq + karray_IF_freq)/1e9 )
    # ax0.axvline( (freqs[karray[int(len(karray)/2)+1]] + mixer_freq + karray_IF_freq)/1e9 )

# Sanity check
# print( 'Pump 1 frequency: ' + str(freqs[karray[int(len(karray)/2)]] + mixer_freq + karray_IF_freq) )
# print( 'Pump 2 frequency: ' + str(freqs[karray[int(len(karray)/2)+1]] + mixer_freq + karray_IF_freq) )
# print( 'Pump 1 frequency - v2: ' + str(comb_freqs[30] + mixer_freq) )
# print( 'Pump 2 frequency - v2: ' + str(comb_freqs[32] + mixer_freq) )

# Axis labels
ax0.set_ylabel("Intra-cavity Spectra (W)")
ax0.set_xlabel("Frequency (GHz)")

ax1.set_title("Conservative Force", fontsize=23)
ax1.set_ylabel("F$_{nl}$")

ax2.set_ylabel("F$_{tot}$")
ax2.set_xlabel("Deflection (mm)")

ax3.set_title("Damping Force", fontsize=23)
ax3.set_ylabel("F$_{nl}$")

ax4.set_ylabel("F$_{tot}$")
ax4.set_xlabel("Velocity (mm/s)")

# Phase along mixer frequency at the two drives frequencies
fig, ax = plt.subplots(2)
ax[1].plot((comb_freqs[2 * int(len(karray) / 2)] + mixer_frequencies) / 1e9,
           np.angle(spectrum_arr[:, 8, 2 * int(len(karray) / 2)]))
ax[0].plot((comb_freqs[2 * int(len(karray) / 2)] + mixer_frequencies) / 1e9,
           np.abs(spectrum_arr[:, 8, 2 * int(len(karray) / 2)]), label='Pump 1')
ax[1].plot((comb_freqs[2 + 2 * int(len(karray) / 2)] + mixer_frequencies) / 1e9,
           np.angle(spectrum_arr[:, 8, 2 + 2 * int(len(karray) / 2)]))
ax[0].plot((comb_freqs[2 + 2 * int(len(karray) / 2)] + mixer_frequencies) / 1e9,
           np.abs(spectrum_arr[:, 8, 2 + 2 * int(len(karray) / 2)]), label='Pump 2')
ax[1].set_xlabel('Frequency (GHz)')
ax[1].set_ylabel('Phase (rad)')
ax[0].set_ylabel(r'|$\alpha_{out}$| (W)')
ax[0].legend()

# Complex plane plot
# fig, ax = plt.subplots(1)
# ax.plot( np.real( spectrum_arr[:,3,2*int(len(karray)/2)] ), np.imag( spectrum_arr[:,3,2*int(len(karray)/2)] ), '.' )


# Experimental data plot at a fixed pump amplitudes
fig, ax = plt.subplots(1)
for mix_freq_ind, mix_freq_val in enumerate(mixer_frequencies):
    # Full spectrum
    # ax.semilogy( (comb_freqs + mix_freq_val)/1e9, np.abs( spectrum_arr[mix_freq_ind,6,:]), color=colors[10*mix_freq_ind] )

    # Only IMPs spectrum
    ax.semilogy((comb_freqs_imp + mix_freq_val) / 1e9, np.abs(spectrum_arr[mix_freq_ind, 6, ::2]),
                color=colors[10 * mix_freq_ind])

ax.set_ylabel(r'|$\alpha_{out}$| (W)')
ax.set_xlabel("Frequency (GHz)")

# IMP vs output power
fig, ax = plt.subplots(1)
ax.semilogy(output_amp_array_dBm, np.abs(spectrum_arr[5, :, 30]), '-', label='Drive 1')
ax.semilogy(output_amp_array_dBm, np.abs(spectrum_arr[5, :, 32]), '-', label='Drive 2')
ax.semilogy(output_amp_array_dBm, np.abs(spectrum_arr[5, :, 34]), label='3rd order')
ax.semilogy(output_amp_array_dBm, np.abs(spectrum_arr[5, :, 36]), label='5rd order')
ax.semilogy(output_amp_array_dBm, np.abs(spectrum_arr[5, :, 38]), label='7rd order')
ax.semilogy(output_amp_array_dBm, np.abs(spectrum_arr[5, :, 59]), label='Background')
ax.set_xlabel(r'|$\alpha_{in}$| (dBm)')
ax.set_ylabel('IMP peak (W)')
ax.legend()

# Parameters vs output power
fig, ax = plt.subplots(2 + reconstruction_order, sharex=(True))
ax[0].plot(output_amp_array_dBm, m_arr[5], label='m')
ax[1].plot(output_amp_array_dBm, kappa0_arr[5], label='$\kappa_0$')
ax[0].set_ylabel("1/$\lambda'$")
ax[1].set_ylabel('$\kappa_0$')
for i in range(reconstruction_order):
    ax[i + 2].plot(output_amp_array_dBm, kappa_arr[5, :, i], label='$\kappa_' + str(i + 1) + '$')
    ax[i + 2].set_ylabel('$\kappa_' + str(i + 1) + '$')
ax[-1].set_xlabel(r'|$\alpha_{in}$| (dBm)')

# %%

# 2D plot of the signal amplitude as a function of the mixer frequency and the comb frequency

# Figure parameters
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

fig2, ax2 = plt.subplots(4, 3)
ax2 = ax2.flatten()

# Color range
zmax = np.max(10 * np.log10(np.abs(spectrum_arr[:, :, :])) + 30)
zmin = -170  # np.min( 10*np.log10( np.abs(spectrum_arr[:,:,:])) + 30 )

for axi in range(4 * 3):
    amp_ind = axi
    if file == h5py.File(r"D:\TLS\TLS-Data\TLS_IMP_amped_DCAW23_200amps_11points.hdf5", "r"):
        amp_ind = int(len(output_amp_array) / 11) * axi
    a = ax2[axi].pcolormesh(
        comb_freqs, mixer_frequencies / 1e9, 10 * np.log10(np.abs(spectrum_arr[:, amp_ind, :])) + 30,
        cmap='RdBu_r',
        vmax=zmax, vmin=zmin,
    )
    ax2[axi].set_title(r'|$\alpha_{in}$|' + f' = {output_amp_array_dBm[amp_ind]:.3f} dBm')
fig.colorbar(a, ax=ax2[:], location='right', label=r'|$\alpha_{out}$| (dBm)', shrink=0.6)
[ax2[axi].set_xlabel('Comb frequencies (Hz)') for axi in [9, 10, 11]]
[ax2[axi].set_ylabel('Mixer frequency (GHz)') for axi in [0, 3, 6, 9]]

# %%

fig, ax = plt.subplots(1)
ax.plot(np.real(y), '-')
