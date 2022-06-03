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

# Greek alphabet: α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, ς, σ, τ, υ, φ, χ, ψ, ω

π = np.pi


# Objective: Reconstruction of a non-linear resonator from its intermodulation spectrum by harmonic balance
# Check TLS Project pdf for more theory details

# Load data    
# file = r'D:\TLS\TLS-Data\TLS_IMP_amped_DCAW23_200amps_11points.hdf5'
file = r'D:\TLS\TLS-Data\TLS_IMP_amped_DCAW23_10amps_201points.hdf5'


# Open hdf5 file
with h5py.File(file, "r") as dataset:

    # Mixer frequency array
    fNCO_arr = np.asarray( dataset["mixer frequencies"] )     # Hz
    # Pump amplitudes array 
    pamp_arr = np.asarray( dataset["signal amplitudes"] )     # fsu
    
    # SYSTEM PARAMETERS
    # Resonance frequency of the system
    f_res = 4.1105253e9  # Hz
    ω_res = 2*π * f_res
    
    # Waveguide-resonator coupling rate
    λ = 0.5
    
    # Gain of the output line
    gain_output_line = 80  # dB
    gain_output_line_linear = 10 ** (gain_output_line / 10)
    
    # Attenuation of the inut line
    att_input_line = 55 + 9 + 0.2 + 4.7 - 20  # dB
    att_input_line_linear = 10 ** (att_input_line / 10)

    # Amplitude in dBm
    pamp_arr_dBm = 16.612 * np.log10(pamp_arr) - 16.743
    # Amplitude in W
    pamp_arr_W = 10 ** ((pamp_arr_dBm - 30) / 10)
    
    # Data array
    spectrum_arr = np.zeros( (len(fNCO_arr), len(pamp_arr), 61), dtype=complex )
    
    # Paraeters array
    F0_recon_arr = np.zeros(( len(fNCO_arr), len(pamp_arr) ))
    f0_recon_arr = np.zeros_like( F0_recon_arr )
    κ0_recon_arr = np.zeros_like( F0_recon_arr )
    κ1_recon_arr = np.zeros_like( F0_recon_arr )
    κ2_recon_arr = np.zeros_like( F0_recon_arr )
    κ3_recon_arr = np.zeros_like( F0_recon_arr )
    κ4_recon_arr = np.zeros_like( F0_recon_arr )


    
    # Mixer frequency loop
    for fNCO_ind, fNCO_val in enumerate(fNCO_arr):
        
        # Check
        print("fNCO index", fNCO_ind)
        
        # Amplitude loop
        for amp_ind, amp_val in enumerate(pamp_arr):
            
            # Amplitude folder
            folder_amp = dataset['signal_amp_' + str(amp_val)]
            
            # Bandwidth of the measurement in Hz
            df = folder_amp['tuned df, Npix, mixerfreq'][0]
            # Number of pixels
            N_pix = folder_amp['tuned df, Npix, mixerfreq'][1]
            # Mixer frequency (center frequency of our comb) in Hz
            fNCO = fNCO_val
            

            # USB and LSB frequency combs
            comb_freqs_usb = folder_amp['frequency combs'][0][:-1]  # We remove the last column due to some mistake in the measurement
            comb_freqs_lsb = -np.flip(comb_freqs_usb[1:])           # We remove the first column because it is already in the USB
            # All frequencies in one array
            comb_freqs = np.concatenate(( comb_freqs_lsb, comb_freqs_usb ))
    

            # Raw USB and LSB output data from Vivace, we remove the same indices as for the frequency comb
            spectrum_usb = np.asarray( folder_amp['USB data'][fNCO_ind][:-1] )
            spectrum_lsb = np.flip( np.asarray( folder_amp['LSB data'][fNCO_ind][1:-1] ) )
            # All spectrum in one array 
            spectrum = np.concatenate(( spectrum_lsb, spectrum_usb ))
            # Spectrum conversion from Vivace units to dBm
            spectrum_dBm = 16.612 * np.log10(spectrum) - 16.743
            # Spectrum conversion from dBm to W
            spectrum_W = 10 ** ((spectrum_dBm - 30) / 10)
            
            
            # Save output spectrum
            # spectrum_arr[fNCO_ind, amp_ind] = spectrum_W
            
            
            # Intra-cavity spectrum (we correct attenuation and gain on the input and output line respectively using input-output theory)
            spectrum_intracav = 1.0j * np.sqrt(2 / λ) * (-spectrum_W / np.sqrt(gain_output_line_linear))
            spectrum_intracav[30] = ( 1.0j * np.sqrt(2 / λ) * 
                                     (pamp_arr_W[amp_ind] / np.sqrt(att_input_line_linear) - 
                                     spectrum_W[30] / np.sqrt(gain_output_line_linear)) )
            spectrum_intracav[32] = ( 1.0j * np.sqrt(2 / λ) * 
                                     (pamp_arr_W[amp_ind] / np.sqrt(att_input_line_linear) - 
                                     spectrum_W[32] / np.sqrt(gain_output_line_linear)) )
            
            # IMP frequencies and IMP spectrum
            comb_freqs_imp = comb_freqs[::2]
            spectrum_imp = spectrum_intracav[::2]
            
            
            # # Plot output and intra-cavity spectrum as a function of frequency
            # fig, ax = plt.subplots( 3 )
            # ax[0].plot( comb_freqs, np.real(spectrum), '.-' )
            # ax[0].set_ylabel( 'A$_{output}$ [fsu]' )
            # ax[1].plot( comb_freqs, np.real(spectrum_intracav), '.-' )
            # ax[1].set_ylabel( 'A$_{intra-cavity}$ [W]' )
            # ax[2].plot( comb_freqs_imp, np.real(spectrum_imp), '.-' )
            # ax[2].set_ylabel( 'A$_{intra-cavity-IMP}$ [W]' )
            

            # Zero paded spectrum
            # Inspired by the use of mixing frequency I realize we can mix down to an even lower frequency to speed up calculation
            # We should just make sure that lowest karray is fairly large with respect to ... what? Either separation or total width of karray
            k_arr = np.int_( np.around(comb_freqs_imp / df) )
            k_arr_width = np.max(k_arr) - np.min(k_arr)
            k_arr_IF = 5 #-1500
            k_arr_IF_freq = k_arr_IF * df
            k_arr = k_arr - k_arr_IF
            
            # Indices for the drives
            k_arr_drives = np.array(( k_arr[int(len(k_arr)/2)], k_arr[int(len(k_arr)/2)+1] ))
            
            
            # Create zero-paded spectrum
            A = np.zeros( np.max(k_arr) * 10, dtype=np.complex )  # This has to be significantly larger than the largest value in the karray
            A[k_arr] = spectrum_imp
            # Time-domain signal
            a = np.fft.ifft(A)
            a *= len(a)  # Scaling because we divide by integration length in the lockin
            
            
            # Drives array
            signal_amp_array_imp = np.zeros_like( A, dtype=complex )
            signal_amp_array_imp[k_arr_drives] = pamp_arr_W[amp_ind] / np.sqrt(att_input_line_linear)
    
            # Frequency range (just to plot in the frequency domain)
            # freqs = df * np.arange( len(Y) )
            f = np.fft.fftfreq( len(a), d=1 / (df * len(a)) )
            
            # # Plot zero-paded intra-cavity spectrum in frequency and time domain
            # fig, ax = plt.subplots( 2 )
            # ax[0].semilogy( f, np.abs(A), '.-' )
            # ax[0].set_ylabel( 'A$_{intra-cavity-IMP}$ [W]' )
            # ax[0].axvline(f[k_arr_drives[0]], 0, 1, ls='--')
            # ax[0].axvline(f[k_arr_drives[1]], 0, 1, ls='--')
            # ax[0].set_xlabel( 'frequency [Hz]' )
            # ax[1].plot( np.real(a) )
            # ax[1].plot( np.imag(a) )
            # ax[1].plot( np.abs(a) )
            # ax[1].set_ylabel( 'a$_{intra-cavity-IMP}$ [W]' )
                        
            
            # HARMONIC BALANCE SECTION

            # Angular frequency
            ω = 2*π * f

            # Create H-matrix
            col1 = (1.0j * ω * A)
            col2 = (1.0j * A)
            col3 = A
            col4 = ( np.fft.fft( a**2 ) ) / len(a)
            col5 = ( np.fft.fft( a**3 ) ) / len(a)
            
            # Merge all columns
            H = np.vstack(( col1, col2, col3, col4, col5 ))
            
            # Making the matrix real instead of complex
            Hcos = np.real( H )
            Hsin = np.imag( H )
            H = np.hstack(( Hcos, Hsin ))
            
            # Normalize H for a more stable inversion
            Nm = np.diag( 1. / np.max(np.abs(H), axis=1) )
            H_norm = np.dot( Nm, H )  # normalized H-matrix
            
            # The drive vector, Q (from the Yasuda paper)
            Qcos = np.real( signal_amp_array_imp )
            Qsin = np.imag( signal_amp_array_imp ) 
            Q = np.hstack(( Qcos, Qsin ))
            
            # Solve system Q = H*p
            H_norm_inv = scipy.linalg.pinv( H_norm )
            p_norm = np.dot( Q, H_norm_inv )
            
            # Re-normalize p-values
            # Note: we have actually solved Q = H * Nm * Ninv * p
            # Thus we obtained Ninv*p and multiply by Nm to obtain p
            p = np.dot( Nm, p_norm )  # re-normalize parameter values
            
            # Forward calculation to check result, should be almost everything zero vector
            Q_fit = np.dot( p, H )
            
            # Scale parameters by drive force assuming known mass
            AA, B, C, D, E = p
            
            # Assuming a known drive
            m_recon = 1.
            
            # Parameters reconstructed
            F0_recon = 1/AA
            f0_recon = F0_recon * B / (2*π)
            κ0_recon = F0_recon * C
            κ1_recon = F0_recon * D
            κ2_recon = F0_recon * E
            
            # print( "f0_recon = {} Hz".format(f0_recon) )
            # print( "f0 = = {} Hz".format(np.abs(f0_recon)-f[k_arr_IF]+fNCO) )
            # print( "kappa0_recon = {}".format(κ0_recon) )
            # print( "kappa1_recon = {}".format(κ1_recon) )
            
            F0_recon_arr[fNCO_ind,amp_ind] = F0_recon
            f0_recon_arr[fNCO_ind,amp_ind] = f0_recon-f[k_arr_IF]+fNCO
            κ0_recon_arr[fNCO_ind,amp_ind] = κ0_recon
            κ1_recon_arr[fNCO_ind,amp_ind] = κ1_recon
            κ2_recon_arr[fNCO_ind,amp_ind] = κ2_recon
            

fNCO_ind = 5
# for fNCO_ind in range(len(fNCO_arr)):
fig, ax = plt.subplots( 5, sharex=True )
ax[0].plot( pamp_arr_dBm, F0_recon_arr[fNCO_ind] )
ax[1].plot( pamp_arr_dBm, f0_recon_arr[fNCO_ind] )
ax[1].axhline( f_res, color='black', ls='--' )
ax[2].plot( pamp_arr_dBm, κ0_recon_arr[fNCO_ind] )
ax[3].plot( pamp_arr_dBm, κ1_recon_arr[fNCO_ind] )
ax[4].plot( pamp_arr_dBm, κ2_recon_arr[fNCO_ind] )
ax[0].set_ylabel( '$F_0$' )
ax[1].set_ylabel( '$f_0$' )
ax[2].set_ylabel( '$\kappa_0$' )
ax[3].set_ylabel( '$\kappa_1$' )
ax[4].set_ylabel( '$\kappa_2$' )
ax[4].set_xlabel( '$A_p$ [dBm]' )
fig.suptitle( r'$f_{NCO}$' + f' = {fNCO_arr[fNCO_ind]:.0f} GHz', fontsize=16 )

