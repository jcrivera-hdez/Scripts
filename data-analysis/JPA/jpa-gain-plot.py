import h5py
import numpy as np
import matplotlib.pyplot as plt

# Greek alphabet: α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, ς, σ, τ, υ, φ, χ, ψ, ω

def dB( data ):
    return 20 * np.log10( np.abs(data**2) )

def plot_traces( trace, freq_arr ):
    ax.plot( freq_arr/1e9, dB(trace) )
    ax.set_xlabel( 'frequency [GHz]' )
    ax.set_ylabel( 'gain [dB]' )

def plot_gain( gain, freq_arr, bias_arr ):
    cutoff = 0.1  # %
    zmin = np.percentile( gain, cutoff )
    zmax = np.percentile( gain, 100. - cutoff )
    
    fig, ax = plt.subplots( 1 )
    a = ax.pcolormesh(bias_arr,
                      freq_arr/1e9, 
                      gain.T, 
                      vmin=zmin, 
                      vmax=zmax, 
                      cmap="RdBu_r",
                            )
    fig.colorbar( a, label=r'gain [dB]' )
    ax.set_xlabel( 'DC bias [V]' )
    ax.set_ylabel( 'frequency [GHz]' )

def plot_gainsweep( gain_arr, freq_arr, bias_arr ):
    cutoff = 0.1  # %
    zmin = np.percentile( gain_arr, cutoff )
    zmax = np.percentile( gain_arr, 100. - cutoff )
    
    nr_rows = 4
    nr_columns = 3
    nr_plots = nr_rows * nr_columns
    
    n = nr_pump_pwr // nr_plots
    
    fig, ax = plt.subplots( nr_rows, nr_columns )
    ax = ax.flatten()
    
    for axi in range(nr_plots):
        amp_ind = n * axi
        a = ax[axi].pcolormesh( bias_arr,
                                freq_arr/1e9,
                                gain_arr[amp_ind].T, 
                                vmin=zmin, 
                                vmax=zmax, 
                                cmap="RdBu_r",
                                )
        ax[axi].set_title( r'$A_p$' + f' = {pump_pwr_arr[amp_ind+1]:.3f} fsu' )
        ax[axi].axvline( x=bias_arr[bias_ind], linestyle='--', color='black' )
    fig.colorbar( a, ax=ax[:], location='right', label=r'gain [dB]', shrink=0.6 )
    [ax[axi].set_xlabel( 'DC bias [V]' ) for axi in [9, 10, 11]]
    [ax[axi].set_ylabel( 'Frequency [GHz]' ) for axi in [0, 3, 6, 9]]
    

# Load data    
file = r'D:\JPA\JPA-Data\QuantumGarage.hdf5'
run = '2022-06-02_12_54_22'
idx_str = "JPA/{}".format(run)

# Open hdf5 file
with h5py.File(file, "r") as dataset:
    
    # Attributes
    df = dataset[idx_str].attrs["df"]

    # Data
    freq_arr = np.asarray(dataset[idx_str]["freq sweep"])
    pump_pwr_arr = np.asarray(dataset[idx_str]["pump pwr sweep"])
    bias_arr = np.asarray(dataset[idx_str]["bias sweep"])
    usb_arr = np.asarray(dataset[idx_str]["USB"])

nr_pump_pwr = len(pump_pwr_arr)  
nr_bias = len(bias_arr) 
bias_ind = 19

# Amplitude normalised
gain_ref = dB(usb_arr[0])
gain_arr = dB(usb_arr[1:])  - dB(usb_arr[0])


# PLOTTING

# Background (no-pump) response
fig, ax = plt.subplots(1)
plot_traces( usb_arr[0][bias_ind], freq_arr )

# Plot gain at a given pump power
plot_gain( gain_arr[-1], freq_arr, bias_arr )

# Plot gain sweep for all pump powers
plot_gainsweep( gain_arr, freq_arr, bias_arr )

fig, ax = plt.subplots(1)
for pump_idx in range(nr_pump_pwr-1):
    plot_traces( usb_arr[pump_idx, bias_ind], freq_arr )

# JPA linewidth for a given dc bias and JPA pump power.

fig, ax = plt.subplots( 1 )
for i in range(nr_pump_pwr-1):
    ax.plot( freq_arr/1e9, gain_arr[i,bias_ind], label=f'{pump_pwr_arr[i+1]:.3f} fsu' )
ax.set_title( r'DC bias ' + f'= {bias_arr[bias_ind]:.3f} V' )
ax.set_xlabel( 'frequency [GHz]' )
ax.set_ylabel( 'gain [dB]' )
ax.legend( title=r'$A_p$' )