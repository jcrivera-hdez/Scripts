import h5py
import numpy as np
import matplotlib.pyplot as plt

# Greek alphabet: α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, ς, σ, τ, υ, φ, χ, ψ, ω

def dB( data ):
    return 20 * np.log10( np.abs(data**2) )


def plot_traces( trace, freq_arr, ax ):
    
    ax.plot( freq_arr/1e9, dB(trace), label=f'{pump_pwr_arr[pump_idx+1]:.3f} fsu' )
    ax.set_title( r'DC bias ' + f'= {bias_arr[bias_ind]:.3f} V' )
    ax.set_xlabel( 'frequency [GHz]' )
    ax.set_ylabel( 'gain [dB]' )


def plot_gain( gain, freq_arr, bias_arr, fig, ax ):

    cutoff = 0.1
    zmin = np.percentile( gain, cutoff )
    zmax = np.percentile( gain, 100. - cutoff )
    
    a = ax.pcolormesh( bias_arr,
                       freq_arr/1e9, 
                       gain.T, 
                       vmin=zmin, 
                       vmax=zmax, 
                       cmap="RdBu_r",
                       )
    fig.colorbar( a, label=r'gain [dB]' )
    ax.set_xlabel( 'DC bias [V]' )
    ax.set_ylabel( 'frequency [GHz]' )


def plot_gain_imshow( gain, x_arr, y_arr, fig, ax):
    
    xmin, xmax = np.min(x_arr), np.max(x_arr)
    ymin, ymax = np.min(y_arr)/1e9, np.max(y_arr)/1e9
    
    cutoff = 0.1
    zmin = np.percentile( gain, cutoff )
    zmax = np.percentile( gain, 100. - cutoff )
    
    a = ax.imshow( gain.T,
                   origin = "lower",
                   aspect = "auto",
                   extent = [xmin, xmax, ymin, ymax],
                   vmin = zmin,
                   vmax = zmax,
                   cmap = "RdBu_r",
                   interpolation = None,
                   )
    fig.colorbar( a, label=r'gain [dB]' )
    ax.set_xlabel( 'DC bias [V]' )
    ax.set_ylabel( 'frequency [GHz]' )


def plot_gainsweep( gain_arr, freq_arr, bias_arr ):
    
    nr_rows = 4
    nr_columns = 3
    nr_plots = nr_rows * nr_columns
        
    n = nr_pump_pwr // nr_plots
    
    fig, ax = plt.subplots( nr_rows, nr_columns, figsize=[19, 9.5], constrained_layout=True )
    ax = ax.flatten()
       
    cutoff = 0.1  # %
    zmin = np.percentile( gain_arr, cutoff )
    zmax = np.percentile( gain_arr, 100. - cutoff )
    
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

# Gain normalised
gain_ref = dB(usb_arr[0])
gain_arr = dB(usb_arr[1:])  - gain_ref


# Plot background (no-pump) response
pump_idx = 0
fig0, ax0 = plt.subplots(1, constrained_layout=True)
plot_traces( usb_arr[pump_idx][bias_ind], freq_arr, ax0 )

# Plot gain at a given pump power
fig1, ax1 = plt.subplots(1, constrained_layout=True)
plot_gain_imshow( gain_arr[-1], bias_arr, freq_arr, fig1, ax1 )

# Plot gain sweep for all pump powers
plot_gainsweep( gain_arr, freq_arr, bias_arr )

fig2, ax2 = plt.subplots(1, constrained_layout=True)
for pump_idx in range(nr_pump_pwr-1):
    plot_traces( usb_arr[pump_idx, bias_ind], freq_arr, ax2 )
ax2.legend( title=r'$A_p$' )
