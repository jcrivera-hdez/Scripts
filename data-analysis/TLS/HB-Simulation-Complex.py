import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.integrate import ode
from scipy.signal import find_peaks
import scipy
import scipy.linalg

# Greek alphabet: α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, ς, σ, τ, υ, φ, χ, ψ, ω

π = np.pi

# Functions
def eom_driven( t, state, fr, κ0, drive ):
    ωr = 2*π * fr
    return -1.0j*ωr*state - κ0*state + drive(t)


def eom_driven_real( t, state_real, fr, κ0, drive ):
    state_complex = state_real[::2] + 1j*state_real[1::2]
    ret_complex = eom_driven( t, state_complex, fr, κ0, drive )
    tmp = np.zeros( len(ret_complex)*2 )
    tmp[::2] = np.real( ret_complex )
    tmp[1::2] = np.imag( ret_complex )
    return tmp


def eom_nonlinear( t, state, fr, κ0, kappa1, drive ):
    ωr = 2 * np.pi * fr
    return -1.0j*ωr*state - κ0*state - kappa1*state**3 + drive(t)


def eom_nonlinear_real( t, state_real, fr, κ0, κ1, drive ):
    state_complex = state_real[::2] + 1j*state_real[1::2]
    ret_complex = eom_nonlinear( t, state_complex, fr, κ0, κ1, drive )
    tmp = np.zeros( len(ret_complex)*2 )
    tmp[::2] = np.real( ret_complex )
    tmp[1::2] = np.imag( ret_complex )
    return tmp


def imp_drive( t, f1, f2, F0 ):
    ω1 = 2*π * f1
    ω2 = 2*π * f2
    return F0 * ( np.cos(ω1*t) + np.cos(ω2*t) )


def recon_driven( A, f ):
    # Angular frequency
    ω = 2*π * f
    
    # Create H-matrix
    col1 = (1.0j * ω * A)[max_ind[0]]
    col2 = (1.0j * A)[max_ind[0]]
    col3 = A[max_ind[0]]
    
    # Merge all columns
    H = np.vstack(( col1, col2, col3 ))
    
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
    param_recon = p
    
    # Parameters reconstructed
    F0_recon = 1 / param_recon[0]
    f0_recon = F0_recon * param_recon[1] / (2*π)
    κ0_recon = F0_recon * param_recon[2]
    
    return F0_recon, f0_recon, κ0_recon, Q_fit


def recon_nonlinear( A, f ):
    # Angular frequency
    ω = 2*π * f
    
    a_filtered = np.fft.ifft( A )
    a_filtered *= len(a_filtered)
    
    # Create H-matrix
    col1 = (1.0j * ω * A)[max_ind]
    col2 = (1.0j * A)[max_ind]
    col3 = A[max_ind]
    col4 = ( np.fft.fft( a_filtered**3 ) )[max_ind] / len(a_filtered)
    
    # Merge all columns
    H = np.vstack(( col1, col2, col3, col4 ))
    
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
    param_recon = p
    
    # Parameters reconstructed
    F0_recon = 1 / param_recon[0]
    f0_recon = F0_recon * param_recon[1] / (2*π)
    κ0_recon = F0_recon * param_recon[2]
    κ1_recon = F0_recon * param_recon[3]
    
    return F0_recon, f0_recon, κ0_recon, κ1_recon, Q_fit


# Drive parameters
F0 = 1
f1 = 0.954
f2 = 0.956
df = 0.001
fs = 20
N = int(round(fs/df))

# Integration parameters
T = 1. / df
T_relax = 10 * T
dt = 1. / fs
t_all = dt * (np.arange( (T+T_relax)/dt ) + 1) 

# Initial conditions
y0 = [0,0]

# Drive tones
drive = partial( imp_drive, f1=f1, f2=f2, F0=F0 )


#%% DRIVEN HARMONIC OSCILLATOR

# System parameters
m = 1.
c = 0.03
f0 = 0.955
k = (2*π*f0)**2 * m 
κ0 = c / (2 * np.sqrt(m*k))

print( "DRIVEN HARMONIC OSCILLATOR")
print( "f0 = {} Hz".format(f0) )
print( "κ0 = {}".format(κ0) )
print( "F0 = {}".format(F0))


# Integrator
o = ode( eom_driven_real ).set_integrator( 'lsoda', atol=1e-12 , rtol=1e-12 )
o.set_f_params( f0, κ0, drive )
o.set_initial_value( y0, 0 )

# Time-domain solution
y_all = np.zeros(( len(t_all), len(y0) ))
for i,t in enumerate( t_all ):
    o.integrate(t)
    y_all[i] = o.y
    

# Merge the results onto the complex plane
a_all = y_all[:,0] + 1.0j*y_all[:,1]

# We save one oscillation once we reached the steady state
a = a_all[-N-1:-1]
t = t_all[-N-1:-1]

# Fourier domain solution
A = np.fft.fft( a ) / len(a)           # rfft or fft? Now a is complex
f = np.fft.fftfreq( len(t), d=dt )


# Plotting stuff
fig, ax = plt.subplots( 3, 1 )
ax[0].plot( t_all, np.abs(a_all) )
ax[1].plot( t, np.abs(a) )
ax[2].semilogy( f, np.abs(A) )
ax[0].set_ylabel( '|a|' )
ax[1].set_xlabel( 'time [s]' )
ax[1].set_ylabel( '|a|' )
ax[2].set_xlabel( 'frequency [Hz]' )
ax[2].set_ylabel( '$|\mathcal{F}(a)|$' )
ax[2].set_ylim( 1e-8, 1e2 )


# Indices of the amplitude maxima
max_ind = find_peaks( x = np.abs(A),
                      height = 2e-2,
                      )

print( 'Number of peaks: ', len(max_ind[0]) )
print( max_ind[0] )

# for i in range( len(A) ):
#     if i not in max_ind[0]:
#         A[i] = 0
        
# Drives indices
ind_drives = np.array([ 954, 956, 19044, 19046 ])

# Drives array
signal_amp_array_imp = np.zeros_like( max_ind[0], dtype=complex )
for i, index in enumerate( max_ind[0] ):
    if index in ind_drives:
        signal_amp_array_imp[i] = 0.5
    
    
# # Sanity check
# fig2, ax2 = plt.subplots( 1 )
# ax2.semilogy( f, np.abs(A) )
# ax2.set_xlabel( 'frequency [Hz]' )
# ax2.set_ylabel( '$\mathcal{F}(x)$' )


# Reconstruction section
F0_recon, f0_recon, κ0_recon, Q_fit = recon_driven( A, f )
    
print( "f0_recon = {} Hz".format(f0_recon) )
print( "κ0_recon = {}".format(κ0_recon) )
print( "F0 = {}".format(F0_recon) )
print( "----------------------------------")


#%% NON-LINEAR DAMPING

# System parameters
m = 1.
c = 0.3
c1 = c/10
f0 = 0.955
k = (2*π*f0)**2 * m 
κ0 = c / (2 * np.sqrt(m*k))
κ1 = c1 / (2 * np.sqrt(m*k))
Q_factor = m * 2*π*f0 / c

print( "NON-LINEAR DAMPING OSCILLATOR")
print( "f0 = {} Hz".format(f0) )
print( "κ0 = {}".format(κ0) )
print( "κ1 = {}".format(κ1) )
print( "F0 = {}".format(F0) )


# Integrator
o = ode( eom_nonlinear_real ).set_integrator( 'lsoda', atol=1e-12 , rtol=1e-12 )
o.set_f_params( f0, κ0, κ1, drive )
o.set_initial_value( y0, 0 )

# Time-domain solution
y_all = np.zeros(( len(t_all), len(y0) ))
for i,t in enumerate( t_all ):
    o.integrate(t)
    y_all[i] = o.y
    

# Merge the results onto the complex plane
a_all = y_all[:,0] + 1.0j*y_all[:,1]


# We save one oscillation once we reached the steady state
a = a_all[-N-1:-1]
t = t_all[-N-1:-1]

# Fourier domain solution
A = np.fft.fft( a ) / len(a)
f = np.fft.fftfreq( len(t), d=dt )


# Plotting stuff
fig, ax = plt.subplots( 3, 1 )
ax[0].plot( t_all, np.abs(a_all) )
ax[1].plot( t, np.abs(a) )
ax[2].semilogy( f, np.abs(A) )
ax[0].set_ylabel( '|a|' )
ax[1].set_xlabel( 'time [s]' )
ax[1].set_ylabel( '|a|' )
ax[2].set_xlabel( 'frequency [Hz]' )
ax[2].set_ylabel( '$|\mathcal{F}(a)|$' )
ax[2].set_ylim( 1e-8, 1e2 )


# Indices of the amplitude minima
max_ind_pos = find_peaks( x = np.abs(A[:2000]),
                      height = 1e-10,
                      )
max_ind_neg = find_peaks( x = np.abs(A[-2000:]),
                      height = 1e-8,
                      )

max_ind = np.append( max_ind_pos[0], len(A)-2000+max_ind_neg[0] )

print( 'Number of peaks: ', len(max_ind) )
# print( max_ind )

for i in range( len(A) ):
    if i not in max_ind:
        A[i] = 0
        
# Drives indices
ind_drives = np.array([ 954, 956, 19044, 19046 ])

# Drives array
signal_amp_array_imp = np.zeros_like( max_ind, dtype=complex )
for i, index in enumerate( max_ind ):
    if index in ind_drives:
        signal_amp_array_imp[i] = 0.5
    
    
# Sanity check
fig, ax = plt.subplots( 1 )
ax.semilogy( f, np.abs(A) )
ax.set_xlabel( 'frequency [Hz]' )
ax.set_ylabel( '$\mathcal{F}(a)$' )


# Reconstruction section
F0_recon, f0_recon, κ0_recon, κ1_recon, Q_fit = recon_nonlinear( A, f )

print( "f0_recon = {} Hz".format(f0_recon) )
print( "κ0_recon = {}".format(κ0_recon) )
print( "κ1_recon = {}".format(κ1_recon) )
print( "F0_recon = {}".format(F0_recon) )
print( "----------------------------------")
