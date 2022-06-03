import h5py
import numpy as np
import matplotlib.pyplot as plt

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



#%% DC Bias sweep


# Load data    
file = r'D:\JPA\JPA-Data\QuantumGarage-JPA.hdf5'
meas_type = 'DC bias sweep'
run = '2022-04-06_13_01_56'
run = '2022-05-25_16_05_57'


run_str = "{}/{}".format(meas_type, run)
idx_str = "{}/{}".format(meas_type, run)

# Open hdf5 file
with h5py.File(file, "r") as dataset:
    
    # Attributes
    df = dataset[run_str].attrs["df"]

    # Data
    freq_arr = np.asarray(dataset[idx_str]["freq sweep"])
    bias_arr = np.asarray(dataset[idx_str]["bias sweep"])
    usb_arr = np.asarray(dataset[idx_str]["USB"])


# Phase
φ_arr = np.unwrap( np.unwrap( np.angle(usb_arr), axis = 1 ), axis = 0 )

# Amplitude normalised
amp_arr = 20 * np.log10( np.abs(usb_arr) )


# dω
dω = 2 * np.pi * df

# DC bias step
dbias = bias_arr[1]-bias_arr[0]

# 'Derivative' in the frequency direction
dφx_arr = np.diff( φ_arr, axis=1 )

# 'Derivative' with respect DC bias
dφy_arr = np.diff( φ_arr, axis=0 )

# Group delay and d(phase)/d(DC)
gdelay_arr = -dφx_arr / dω
dφdbias_arr = dφy_arr/ dbias


# PLOTTING

# S11 as a function of the frequency and the DC bias 
fig, ax = plt.subplots(1)
a = ax.pcolormesh( bias_arr, freq_arr/1e9, np.swapaxes(amp_arr,0,1), shading='nearest', cmap='RdBu_r' )
ax.set_ylabel( 'DC bias [V]' )
ax.set_xlabel( 'frequency [GHz]' )
fig.colorbar( a, label='amplitude [a.u.]' )

# # Phase as a function of the frequency and the DC bias
# fig, ax = plt.subplots(1)
# a = ax.pcolormesh( freq_arr/1e9, bias_arr, φ_arr, shading='nearest', cmap='RdBu_r' )
# ax.set_ylabel( 'DC bias [V]' )
# ax.set_xlabel( 'frequency [GHz]' )
# fig.colorbar( a, label='phase unwraped' )

# Group delay as a function of the frequency and the DC bias
fig, ax = plt.subplots( 1 )
a = ax.pcolormesh( bias_arr, freq_arr/1e9, np.swapaxes(gdelay_arr,0,1), shading='nearest', cmap='RdBu_r' )
ax.set_ylabel( 'DC bias [V]' )
ax.set_xlabel( 'frequency [GHz]' )
fig.colorbar( a, ax=ax, label='group delay [s]' )

# # d(phase)/d(DC) as a function of the frequency and the DC bias
# # choose limits for colorbar
# cutoff = .1  # %
# zmax = np.percentile( dφdbias_arr, cutoff )
# zmin = np.percentile( dφdbias_arr, 100. - cutoff )
        
# fig, ax = plt.subplots( 1 )
# b = ax.pcolormesh( freq_arr/1e9, bias_arr, dφdbias_arr, shading='nearest', cmap='RdBu_r', vmin=zmin, vmax=zmax )
# ax.set_ylabel( 'DC bias [V]' )
# ax.set_xlabel( 'frequency [GHz]' )
# fig.colorbar( b, ax=ax, label=r'$\dfrac{\partial\phi}{\partial bias}$' )


#%% Pump power sweep

# Load data    
file = r'D:\JPA\JPA-Data\QuantumGarage.hdf5'
meas_type = 'JPA'
run = '2022-04-08_13_25_50'   # Stepping fNCO
# run = '2022-04-20_13_21_37'     # Fixing fNCO
run = '2022-06-02_12_54_22'
run = '2022-06-02_15_19_51'

run_str = "{}/{}".format(meas_type, run)
idx_str = "{}/{}".format(meas_type, run)

# Open hdf5 file
with h5py.File(file, "r") as dataset:
    
    # Attributes
    df = dataset[run_str].attrs["df"]

    # Data
    freq_arr = np.asarray(dataset[idx_str]["freq sweep"])
    pump_pwr_arr = np.asarray(dataset[idx_str]["pump pwr sweep"])
    bias_arr = np.asarray(dataset[idx_str]["bias sweep"])
    usb_arr = np.asarray(dataset[idx_str]["USB"])

nr_pump_pwr = len(pump_pwr_arr)  
nr_bias = len(bias_arr) 
 
# Amplitude normalised
# gain_arr = 20 * ( np.log10( np.abs(usb_arr[1:]**2) ) - np.log10( np.abs(usb_arr[0]**2) ) )
gain_arr = 20 * ( np.log10( np.abs(usb_arr[1:]**2) ) )

bias_ind = 27
bias_ind = 19

# PLOTTING

# Background (no-pump) response
fig, ax = plt.subplots( 1 )
ax.plot( freq_arr/1e9, 20*np.log10( np.abs(usb_arr[0]**2) )[bias_ind] )
ax.set_title( 'pump off' )
ax.set_xlabel( 'frequency [GHz]' )
ax.set_ylabel( 'gain [dB]' )

# Gain as a function of the frequency, dc bias and JPA pump power.
# Figure parameters
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

cutoff = 0.1  # %
zmax = -150 #np.percentile( gain_arr, cutoff )
zmin = -180#np.percentile( gain_arr, 100. - cutoff )
# absz = max( abs(zmax), abs(zmin) )

nr_rows = 4
nr_columns = 3
nr_plots = nr_rows * nr_columns

n = nr_pump_pwr // nr_plots

fig, ax = plt.subplots( nr_rows, nr_columns )
ax = ax.flatten()

for axi in range(nr_plots):
    amp_ind = n * axi
    a = ax[axi].pcolormesh(freq_arr/1e9, 
                            bias_arr, 
                            gain_arr[amp_ind], 
                            vmin=zmin, 
                            vmax=zmax, 
                            cmap="RdBu_r",
                            )
    ax[axi].set_title( r'$A_p$' + f' = {pump_pwr_arr[amp_ind+1]:.3f} fsu' )
    ax[axi].axhline( y=bias_arr[bias_ind], linestyle='--', color='black' )
fig.colorbar( a, ax=ax[:], location='right', label=r'gain [dB]', shrink=0.6 )
[ax[axi].set_xlabel( 'Frequency [Hz]' ) for axi in [9, 10, 11]]
[ax[axi].set_ylabel(' DC bias [V]' ) for axi in [0, 3, 6, 9]]


# JPA linewidth for a given dc bias and JPA pump power.
# Figure parameters
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

fig, ax = plt.subplots( 1 )
for i in range(nr_pump_pwr-1):
    ax.plot( freq_arr[:-1]/1e9, gain_arr[i,bias_ind,:-1], label=f'{pump_pwr_arr[i+1]:.3f} fsu' )
ax.set_title( r'DC bias ' + f'= {bias_arr[bias_ind]:.3f} V' )
ax.set_xlabel( 'frequency [GHz]' )
ax.set_ylabel( 'gain [dB]' )
ax.legend( title=r'$A_p$' )


# Gain ratio between the phase sensitive gain and the "normal" gain
gain_ratio = gain_arr[:,bias_ind,400] - gain_arr[:,bias_ind,401]

fig, ax = plt.subplots( 1 )
ax.plot( pump_pwr_arr[1:], gain_ratio, '.' )
ax.set_xlabel( '$A_p$' )
ax.set_ylabel( 'gain difference' )


#%% Two-mode squeezing

# Load data    
file = r'D:\JPA\JPA-Data\QuantumGarage-JPA.hdf5'
meas_type = 'TMS'
run = '2022-04-08_14_41_51'


run_str = "{}/{}".format(meas_type, run)
idx_str = "{}/{}".format(meas_type, run)

# Open hdf5 file
with h5py.File(file, "r") as dataset:
    
    # Attributes
    df_arr = dataset[run_str].attrs["df"]

    # Data
    freq_arr = np.asarray(dataset[idx_str]["freq sweep"])
    pump_pwr_arr = np.asarray(dataset[idx_str]["pump pwr sweep"])
    usb_arr = np.asarray(dataset[idx_str]["USB"])
    lsb_arr = np.asarray(dataset[idx_str]["LSB"])

nr_pump_pwr = len(pump_pwr_arr)  
nr_df = len(df_arr)


for df_ind, df_val in enumerate(df_arr):
    
    # Pump off data at a given df
    Ioff_arr = np.real( usb_arr[df_ind,0,:,0] )
    Qoff_arr = np.imag( usb_arr[df_ind,0,:,0] )
    
    
    # PLOTTING
    
    # Noise histograms for the center mode
    # Figure parameters
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    cutoff = 0.1  # %
    zmax = np.percentile(usb_arr[df_ind], cutoff)
    zmin = np.percentile(usb_arr[df_ind], 100. - cutoff)
    absz = max( abs(zmax), abs(zmin) )
    
    # Number of rows and columns of our figure
    nr_rows = 4
    nr_columns = 3
    nr_plots = nr_rows * nr_columns
    
    n = nr_pump_pwr // nr_plots
    
    fig, ax = plt.subplots( nr_rows, nr_columns )
    ax = ax.flatten()
    
    for axi in range(nr_plots):
        
        amp_ind = n * axi
        
        # I and Q quadratures
        I_arr = np.real( usb_arr[df_ind,amp_ind+1,:,0] )
        Q_arr = np.imag( usb_arr[df_ind,amp_ind+1,:,0] )
        
        # Number of bins of the histogram
        binr = np.max( np.append(I_arr, Q_arr) )*1.1
        nr_bins = 201
        bins = np.linspace( -binr, binr, nr_bins )
        
        # Arange data to plot on an histogram
        IQ_arr, xedge, yedge = np.histogram2d( I_arr, Q_arr, bins=(bins,bins) )
        IQ_arr_off, xedgesh, yedgesh = np.histogram2d( Ioff_arr, Qoff_arr, bins=(bins,bins) )
        center_hist = (IQ_arr - IQ_arr_off) / np.max( IQ_arr - IQ_arr_off )
        
        # Plot data
        a = ax[axi].pcolormesh(xedge[0:nr_bins-1]*1e3,
                               yedge[0:nr_bins-1]*1e3,
                               center_hist, 
                               vmin=-absz, 
                               vmax=absz, 
                               cmap="RdBu_r",
                               )
        
        ax[axi].set_title( r'pump amp' + f' = {pump_pwr_arr[amp_ind+1]:.2f} fsu', fontsize=9 )
        ax[axi].set_aspect( 'equal' )
        
    fig.suptitle( r'df' + f' = {df_arr[df_ind]:.0f} Hz')
    fig.colorbar( a, ax=ax[:], location='right', label=r'noise [a.u.]', shrink=0.6 )
    [ax[axi].set_xlabel('I [a.u]') for axi in [9, 10, 11]]
    [ax[axi].set_ylabel('Q[a.u]') for axi in [0, 3, 6, 9]]


#%% Two mode squeezing (to be revised)

# Load data    
file = r'D:\JPA\JPA-Data\QuantumGarage-JPA.hdf5'
meas_type = 'TMS'
run = '2022-04-19_14_39_43'
# run = '2022-04-08_14_41_51'


run_str = "{}/{}".format(meas_type, run)
idx_str = "{}/{}".format(meas_type, run)

# Open hdf5 file
with h5py.File(file, "r") as dataset:
    
    # Attributes
    df_arr = dataset[run_str].attrs["df"]
    fNCO = dataset[run_str].attrs["fNCO"]

    # Data
    freq_usb_arr = np.asarray(dataset[idx_str]["freq sweep"])
    pump_pwr_arr = np.asarray(dataset[idx_str]["pump pwr sweep"])
    usb_arr = np.asarray(dataset[idx_str]["USB"][:,1])
    lsb_arr = np.asarray(dataset[idx_str]["LSB"][:,1])
    
    usb_arr_off = np.asarray(dataset[idx_str]["USB"][:,0])
    lsb_arr_off = np.asarray(dataset[idx_str]["LSB"][:,0])
    

a_off = np.abs( np.mean( lsb_arr_off, axis=1) )
a = np.abs( np.mean( lsb_arr, axis=1) )

b = np.zeros(30)
b_off = np.zeros(30)
for i in range( 30 ):
    b[i] = a[i,i]
    b_off[i] = a_off[i,i]
    
fig, ax = plt.subplots( 1 )
ax.plot(b)
ax.plot(b_off)
    
#%%

freq_arr = np.append( np.flip(freq_usb_arr), freq_usb_arr )
data_arr = np.append( np.flip(lsb_arr), usb_arr, axis=1 )
data_arr_off = np.append( np.flip(lsb_arr_off), usb_arr_off, axis=1 )

real_data_arr = np.real( data_arr )
imag_data_arr = np.imag( data_arr )
real_data_arr_off = np.real( data_arr_off )
imag_data_arr_off = np.imag( data_arr_off )

coord_arr = [ real_data_arr, imag_data_arr ]
coord_arr = np.concatenate( coord_arr, axis=0 )
coord_arr = np.swapaxes( coord_arr, 0, 1 )
coord_arr_off = [ real_data_arr_off, imag_data_arr_off ]
coord_arr_off = np.concatenate( coord_arr_off, axis=0 )
coord_arr_off = np.swapaxes( coord_arr_off, 0, 1 )

cov_matrix = np.cov( coord_arr )
cov_matrix_off = np.cov( coord_arr_off )

fig, ax = plt.subplots( 1 )
a = ax.pcolormesh( np.flipud(cov_matrix), shading='nearest', cmap='RdBu_r', vmax=0.2e-10 )
fig.colorbar( a )

diag = np.diag(cov_matrix)
diag_off = np.diag(cov_matrix_off)
fig, ax = plt.subplots( 1 )
ax.plot( diag-diag_off )


mean_data = np.abs( np.mean(data_arr, axis=0) )
mean_data_off = np.abs( np.mean(data_arr_off, axis=0) )
fig, ax = plt.subplots( 1 )
ax.plot( mean_data )
ax.plot( mean_data_off )


























