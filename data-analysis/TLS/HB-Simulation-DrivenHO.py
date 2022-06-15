import numpy as np
from functools import partial
from scipy.integrate import ode
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from scipy.signal import find_peaks

# Greek alphabet: α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, ς, σ, τ, υ, φ, χ, ψ, ω

π = np.pi

# Plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.constrained_layout.use'] = True
# plt.rcParams['figure.titlesize'] = 23
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["legend.frameon"] = False



# Equation of motion
def eom_driven_HO( t, y, m, c, k, drive ):
    return [ y[1],
             1./m * ( -c*y[1] - k*y[0] + drive(t))
             ]


# Equation of motion
def eom_nonlinear_damp( t, y, m, c, cnl, k, drive ):
    return [ y[1],
             1./m * ( -c*y[1] - cnl*y[1]**3 - k*y[0] + drive(t))
             ]


# Drives
def imp_drive( t, f1, f2, F0 ):
    return F0 * ( np.cos(2*π*f1*t) + np.cos(2*π*f2*t) )


def recon_driven_HO( Y, f ):
    
    # Angular frequency
    ω = 2*π * f
    
    # Create H-matrix
    col1 = (-1 * ω**2 * Y)[max_ind[0]]
    col2 = (1.0j * ω * Y)[max_ind[0]]
    col3 = Y[max_ind[0]]
    
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
    
    # Forward calculation to check result, should be q
    Q_fit = np.dot( p, H )
    
    # Scale parameters by drive force assuming known mass
    A, B, C = p
    
    # Assuming a known drive
    m_recon = 1.
    
    # Parameters reconstructed
    F0_recon = 1/A
    c_recon = F0_recon * B
    k_recon = F0_recon * C
    
    f0_recon = np.sqrt(k_recon/m_recon) / (2*π)
    κ0_recon = c_recon / ( 2*np.sqrt(m_recon*k_recon) )
    Qfactor_recon = m_recon*np.sqrt(k_recon/m_recon)/c_recon
    
    return F0_recon, f0_recon, κ0_recon, Qfactor_recon


def recon_nonlinear_damping( Y, f ):
    # Angular frequency
    ω = 2*π * f
    
    # Derivative of the signal in time domain
    ydot = np.fft.irfft(1.0j * ω * Y)
    ydot *= len(ydot)
    
    # Create H-matrix
    col1 = (-1 * ω**2 * Y)[max_ind[0]]
    col2 = (1.0j * ω * Y)[max_ind[0]]
    col3 = ( np.fft.rfft( ydot**3 ) )[max_ind[0]]/len(ydot)
    col4 = Y[max_ind[0]]
    
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
    A, B, C, D = p
    
    # Assuming a known drive
    m_recon = 1.
    
    # Parameters reconstructed
    F0_recon = 1/A
    c_recon = F0_recon * B
    cnl_recon = F0_recon * C
    k_recon = F0_recon * D
    
    f0_recon = np.sqrt(k_recon/m_recon) / (2*π)
    κ0_recon = c_recon / ( 2*np.sqrt(m_recon*k_recon) )
    κ1_recon = cnl_recon / ( 2*np.sqrt(m_recon*k_recon) )
    
    return F0_recon, f0_recon, κ0_recon, κ1_recon


# Drive parameters
F0 = 1
f1 = 0.954
f2 = 0.956
df = 0.001
fs = 20
N = int(fs/df)

# Integration parameters
T = 1. / df
T_relax = 10 * T
dt = 1. / fs
t_all = dt * ( np.arange( (T+T_relax)/dt ) + 1 )

# Initial conditions
y0 = [0,0]

# Drive tones
drive = partial( imp_drive, f1=f1, f2=f2, F0=F0 )


#%% DRIVEN HARMONIC OSCILLATOR

# System parameters
m = 1.
c = 0.3
f0 = 0.955
k = (2*π*f0)**2 * m 
κ0 = c / (2 * np.sqrt(m*k))
Qfactor = m * np.sqrt(k/m) / c

print( "DRIVEN HARMONIC OSCILLATOR")
print( "f0 = {} Hz".format(f0) )
print( "F0 = {}".format(F0) )
print( "κ0 = {}".format(κ0) )
print( "Q = {}".format(Qfactor) )


# Integrator
o = ode( eom_driven_HO )
o.set_f_params( m, c, k, drive )
o.set_initial_value( y0, 0 )

# Time-domain solution
y_all = np.zeros(( len(t_all), len(y0) ))
for i,t in enumerate( t_all ):
    o.integrate(t)
    y_all[i] = o.y
    

# We save one oscillation once we reached the steady state
y = y_all[-N:,0]
t = t_all[-N:]
q = drive(t)

# Fourier domain solution
Y = np.fft.fft( y ) / len(y)
f = np.fft.fftfreq( len(t), d=dt )

# Plotting stuff
fig, ax = plt.subplots( 3, 1 )
ax[0].plot( t_all, y_all[:,0] )
ax[1].plot( t, y )
ax[2].semilogy( f, np.abs(Y) )
ax[0].set_ylabel( 'x' )
ax[1].set_xlabel( 'time [s]' )
ax[1].set_ylabel( 'x' )
ax[2].set_xlabel( 'frequency [Hz]' )
ax[2].set_ylabel( '$\mathcal{F}(x)$' )


# Indices of the amplitude minima
max_ind = find_peaks( x = np.abs(Y[:2000]),
                      height = 1e-3,
                      )

print( 'Number of peaks: ', len(max_ind[0]) )
print( max_ind[0] )

for i in range( len(Y) ):
    if i not in max_ind[0]:
        Y[i] = 0
        
# Drives indices
ind_drives = np.array([ 954, 956 ])
# ind_drives = np.array([ 1908, 1912 ]) # Indices when having 2T

signal_amp_array_imp = np.zeros_like( max_ind[0], dtype=complex )
for i, index in enumerate( max_ind[0] ):
    if index in ind_drives:
        signal_amp_array_imp[i] = 0.5
        

# Sanity check
fig, ax = plt.subplots( 1 )
ax.semilogy( f[0:2000], np.abs(Y[0:2000]) )
ax.set_xlabel( 'frequency [Hz]' )
ax.set_ylabel( '$\mathcal{F}(x)$' )


# Reconstruction section
F0_recon, f0_recon, κ0_recon, Qfactor_recon = recon_driven_HO( Y, f )

print( "f0_recon = {} Hz".format(f0_recon) )
print( "κ0_recon = {}".format(κ0_recon) )
print( "F0_recon = {}".format(F0_recon) )
print( "Q_recon = {}".format(Qfactor_recon) )
print( "----------------------------------")


#%% NON-LINEAR DAMPING

# System parameters
m = 1.
c = 0.003
cnl = c/100
f0 = 0.955
k = (2*π*f0)**2 * m 
κ0 = c / (2 * np.sqrt(m*k))
κ1 = cnl / (2 * np.sqrt(m*k))

print( "NON-LINEAR DAMPING OSCILLATOR")
print( "f0 = {} Hz".format(f0) )
print( "kappa0 = {}".format(κ0) )
print( "kappa1 = {}".format(κ1) )
print( "F0 = {}".format(F0) )


# Integrator
o = ode( eom_nonlinear_damp )
o.set_f_params( m, c, cnl, k, drive )
o.set_initial_value( y0, 0 )

# Time-domain solution
y_all = np.zeros(( len(t_all), len(y0) ))
for i,t in enumerate( t_all ):
    o.integrate(t)
    y_all[i] = o.y
    

# We save one oscillation once we reached the steady state
y = y_all[-N:,0]
t = t_all[-N:]
q = drive(t)

# Fourier domain solution
Y = np.fft.rfft( y ) / len(y)
f = np.fft.rfftfreq( len(t), d=dt )


# Plotting stuff
fig, ax = plt.subplots( 3, 1 )
ax[0].plot( t_all, y_all[:,0] )
ax[1].plot( t, y )
ax[2].semilogy( f, np.abs(Y) )
ax[0].set_ylabel( 'x' )
ax[1].set_xlabel( 'time [s]' )
ax[1].set_ylabel( 'x' )
ax[2].set_xlabel( 'frequency [Hz]' )
ax[2].set_ylabel( '$\mathcal{F}(x)$' )


# Indices of the amplitude minima
max_ind = find_peaks( x = np.abs(Y[:2000]),
                      height = 1e-6,
                      )

print( 'Number of peaks: ', len(max_ind[0]) )
print( max_ind[0] )

for i in range( len(Y) ):
    if i not in max_ind[0]:
        Y[i] = 0
        
# Drives indices
ind_drives = np.array([ 954, 956 ])

signal_amp_array_imp = np.zeros_like( max_ind[0], dtype=complex )
for i, index in enumerate( max_ind[0] ):
    if index in ind_drives:
        signal_amp_array_imp[i] = 0.5
    
    
# Sanity check
fig, ax = plt.subplots( 1 )
ax.semilogy( f[:2000], np.abs(Y[:2000]) )
ax.set_xlabel( 'frequency [Hz]' )
ax.set_ylabel( '$\mathcal{F}(x)$' )


# Reconstruction section
F0_recon, f0_recon, κ0_recon, κ1_recon = recon_nonlinear_damping( Y, f )

print( "f0_recon = {} Hz".format(f0_recon) )
print( "κ0_recon = {}".format(κ0_recon) )
print( "κ1_recon = {}".format(κ1_recon) )
print( "F0_recon = {}".format(F0_recon) )

