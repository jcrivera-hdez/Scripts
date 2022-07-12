# -*- coding: utf-8 -*-

"""
Created on Wed Mar 10 16:19:50 2021
@author: JC

Last version: 2021-03-11

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from scipy.constants import Planck, Boltzmann
from scipy.optimize import curve_fit
from vivace import utils
import time

# Plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.titlesize'] = 23
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15


# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------

# Defining thermal noise function
def thermal_noise_func(temperature, gain_signal, gain_idler, ns, f_signal, df):
    """
    Calculates the thermal noise as a function of temperature
    Parameters
    ----------
    temperature : NP.ARRAY
        Temperature in mK
    gain_signal : FLOAT
        Gain of the signal in dB
    gain_idler: FLOAT
        Gain of the idler in dB
    ns : FLOAT
        Added number of photons from the signal
    f_signal : FLOAT
        Signal frequency
    df: FLOAT
        Bandwidth
    Returns
    -------
    V2 : NP.ARRAY
        Thermal Noise
    """
    # Transmission line impedance
    Zc = 50

    # Signal and idler gain
    Gs = 10 ** np.abs(gain_signal / 10)
    Gi = 10 ** np.abs(gain_idler / 10)

    # Pump frequency
    f_pump = 6.05e9
    f_idler = 2 * f_pump - f_signal

    # Conv factor
    factor_signal = 2 * Zc * Planck * f_signal * df
    factor_idler = 2 * Zc * Planck * f_idler * df

    # Thermal Noise
    V2_signal = factor_signal * (
                Gs / np.tanh(Planck * f_signal / (2 * Boltzmann * temperature * 1e-3)) + (Gs - 1) * (2 * ns + 1))
    V2_idler = factor_idler * Gi / np.tanh(Planck * f_idler / (2 * Boltzmann * temperature * 1e-3))
    V2 = (V2_signal + V2_idler) * 1e12

    return V2


# Defining thermal noise function with a constrain
def thermal_noise_func_constrained(temperature, gain_signal, ns, f_signal, df):
    """
    Calculates the thermal noise as a function of temperature
    Parameters
    ----------
    temperature : NP.ARRAY
        Temperature in mK
    gain_signal : FLOAT
        Gain of the signal in dB
    gain_idler: FLOAT
        Gain of the idler in dB
    ns : FLOAT
        Added number of photons from the signal
    f_signal : FLOAT
        Signal frequency
    df: FLOAT
        Bandwidth
    Returns
    -------
    V2 : NP.ARRAY
        Thermal Noise
    """
    # Transmission line impedance
    Zc = 50

    # Signal and idler gain
    Gs = 10 ** np.abs(gain_signal / 10)
    Gi = Gs - 1

    # Pump frequency
    f_pump = 6.05e9
    f_idler = 2 * f_pump - f_signal

    # Conv factor
    factor_signal = 2 * Zc * Planck * f_signal * df
    factor_idler = 2 * Zc * Planck * f_idler * df

    # Thermal Noise
    V2_signal = factor_signal * (
                Gs / np.tanh(Planck * f_signal / (2 * Boltzmann * temperature * 1e-3)) + (Gs - 1) * (2 * ns + 1))
    V2_idler = factor_idler * Gi / np.tanh(Planck * f_idler / (2 * Boltzmann * temperature * 1e-3))
    V2 = (V2_signal + V2_idler) * 1e12

    return V2


# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# OCTOBER 2020 CALIBRATION DATA LOADING
# ---------------------------------------------------------------------------------------------------------------------------------

# Note: The October data takes around 30 s to load
start = time.time()

calibration = 'oct'

# Loading of the noise power data

# File path
common_path = r'D:\VivaceData\2020-10-23'
time_path = os.listdir(common_path)

folders = []
for path_ind, path_name in enumerate(time_path):
    data_path = os.path.join(common_path, time_path[path_ind])
    filename = os.listdir(data_path)
    path = os.path.join(common_path, time_path[path_ind], filename[0])
    folders.append(path)

hf_noise = h5py.File(folders[0], 'r')

# NCO frequency
fNCO = 3.8e9
# Frequency comb
freq_comb = np.array(hf_noise['comb_freqs']) + fNCO
# Bandwidth
df = hf_noise['df'][0]

# Noise data
noise_power = np.zeros((len(freq_comb), len(folders)))
noise_power_std = np.zeros_like(noise_power)

for freq_ind in range(len(freq_comb)):
    # Loading the data
    for ii in range(len(folders)):
        # Loading of the noise data
        hf_high = h5py.File(folders[ii], 'r')
        complex_usb_data = hf_high['complex high sideband'][:, freq_ind] * np.sqrt(2) * 6.33645961
        noise_power[freq_ind, ii] = np.mean(np.abs(complex_usb_data) ** 2) * 1e12
        noise_power_std[freq_ind, ii] = np.std(np.abs(complex_usb_data) ** 2) * 1e12 / np.sqrt(len(complex_usb_data))

# Loading the temperature data

# Temperature calibration data folder
temp_folder = os.path.join(
    r'D:\VivaceData\Vivace-Calibration\vivace_temp_noise_calibration\2020-10\vivace_temp_noise_calibration-20201023.hdf5')
hf_temp = h5py.File(temp_folder, 'r')

# Temperature data load
raw_temp = hf_temp['Data'][:, 10]

# Total measurement time
t_tot = np.linspace(0, len(raw_temp), len(raw_temp))

# Reference time in seconds
t0 = 11 * 3600 + 49 * 60 + 52

# Duration of each measurement
t_window = 21 * 60

# Temperatures values
T = np.zeros(len(time_path))
T_std = np.zeros_like(T)

for time_ind, time_value in enumerate(time_path):
    t = int(time_value[0:2]) * 3600 + int(time_value[3:5]) * 60 + int(time_value[6:])
    # Time interval in which we estimate the average temperature and its std deviation
    tmin = t - t0
    tmax = tmin + t_window
    temp_meas = raw_temp[(t_tot > tmin) & (t_tot < tmax)]
    # Average temperature in mK
    T[time_ind] = 1000 * np.mean(temp_meas)
    T_std[time_ind] = 1000 * np.std(temp_meas)

stop = time.time()
print(f"Time spent loading OCT data: {utils.format_sec(stop - start)}")

# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# JANUARY 2021 CALIBRATION DATA LOADING
# ---------------------------------------------------------------------------------------------------------------------------------

# Note: The January data takes around 30 min to load
start = time.time()

calibration = 'jan'

# Temperature calibration data folder
folder = os.path.join(
    r'D:\VivaceData\Vivace-Calibration\vivace_temp_noise_calibration_2021-01\SAW_calibration_data_Jan_2021.hdf5')
hf = h5py.File(folder, 'r')

# List with the names of all the folders inside the hdf5 file
temp_list = list(hf.keys())

temp_grp = hf.require_group(temp_list[0])

df = temp_grp['df'][0]  # bandwidth
fNCO = temp_grp['VLO frequency'][...]  # local oscillator frequency
freq_comb = temp_grp['frequency comb'][:]  # frequency comb

usb_freq = freq_comb + fNCO  # upper side band
lsb_freq = -freq_comb + fNCO  # lower side band

freq_comb = np.concatenate((np.flipud(lsb_freq), usb_freq))  # complete frequency comb

# Loading the temperature data

# Temperature data
T = np.zeros(len(temp_list))
T_std = np.zeros_like(T)

for folder_ind, folder_name in enumerate(temp_list):
    temp_grp = hf.require_group(folder_name)
    T[folder_ind] = temp_grp['temperature'][:]

# Sort from low to high temperatures
T = np.sort(T)

# Loading of the noise power data

# Noise data
noise_power = np.zeros((len(freq_comb), len(T)))
noise_power_std = np.zeros_like(noise_power)

for temp_ind, temp_value in enumerate(T):
    temp_grp = hf.require_group('temperature ' + str(temp_value))

    usb_raw = temp_grp['complex high sideband OFF'][:] * np.sqrt(2) * 6.33645961
    usb_noise = np.mean(np.abs(usb_raw) ** 2, axis=0) * 1e12
    usb_noise_err = np.std(np.abs(usb_raw) ** 2, axis=0, ddof=1) / np.sqrt(len(usb_raw)) * 1e12

    lsb_raw = temp_grp['complex low sideband OFF'][:] * np.sqrt(2) * 6.33645961
    lsb_noise = np.mean(np.abs(lsb_raw) ** 2, axis=0) * 1e12
    lsb_noise_err = np.std(np.abs(lsb_raw) ** 2, axis=0, ddof=1) / np.sqrt(len(lsb_raw)) * 1e12

    noise_power[:, temp_ind] = np.concatenate((np.flipud(lsb_noise), usb_noise))
    noise_power_std[:, temp_ind] = np.concatenate((np.flipud(lsb_noise_err), usb_noise_err))

stop = time.time()
print(f"Time spent loading JAN data: {utils.format_sec(stop - start)}")

# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# CALIBRATION - Calculation for the gain and added noise for all frequencies in our frequency comb
# ---------------------------------------------------------------------------------------------------------------------------------

# Parameters initial guess
g0_signal = 83
n0 = 3
g0_idler = 88

# Fit parameters
gs = np.zeros(len(freq_comb))
gs_std = np.zeros_like(gs)

gi = np.zeros_like(gs)
gi_std = np.zeros_like(gs)

ns = np.zeros_like(gs)
ns_std = np.zeros_like(gs)

sGsGi = np.zeros_like(gs)
sGsn = np.zeros_like(gs)
sGin = np.zeros_like(gs)

ss_res = np.zeros_like(gs)
R2 = np.zeros_like(gs)

for freq_ind in range(len(freq_comb)):

    freq_used = freq_comb[freq_ind]

    # Fit to temperature data
    popt, pcov = curve_fit(lambda T, Gsignal, Gidler, n_a: thermal_noise_func(T, Gsignal, Gidler, n_a, freq_used, df),
                           T,
                           noise_power[freq_ind],
                           p0=(g0_signal, g0_idler, n0),
                           sigma=noise_power_std[freq_ind],
                           # bounds = ( [50, 50, 2.5], [100, 100, 10] ),
                           maxfev=10000,
                           )

    # Sanity check to have nan in the standard deviations
    if pcov[0, 0] < 0:
        pcov[0, 0] = np.inf
        pcov[2, 2] = np.inf

    # Extraction of the fitting parameters, i.e. gains and added number of photons
    gs[freq_ind] = popt[0]
    gs_std[freq_ind] = np.sqrt(pcov[0, 0])

    gi[freq_ind] = popt[1]
    gi_std[freq_ind] = np.sqrt(pcov[1, 1])

    ns[freq_ind] = popt[2]
    ns_std[freq_ind] = np.sqrt(pcov[2, 2])

    # Cross correlations in the errors
    sGsGi[freq_ind] = pcov[0, 1]
    sGsn[freq_ind] = pcov[0, 2]
    sGin[freq_ind] = pcov[1, 2]

    # Fit quality parameters
    residuals = noise_power[freq_ind] - thermal_noise_func(T, gs[freq_ind], gi[freq_ind], ns[freq_ind], freq_used, df)
    ss_res[freq_ind] = np.sum(residuals ** 2)
    ss_tot = np.sum((noise_power[freq_ind] - np.mean(noise_power[freq_ind])) ** 2)
    R2[freq_ind] = 1 - ss_res[freq_ind] / ss_tot

# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------------------------------------------------------------

# Idler contribution
Tb = 30e-3
f_pump = 6.05e9
f_idler = 2 * f_pump - freq_comb
ni = (1 / np.tanh(Planck * f_idler / (2 * Boltzmann * Tb)) - 1) / 2
ns_min = (-gi / gs * (2 * ni + 1)) / 2

# Plotting voltage noise vs temperature for a single frequency
freq_ind = 28
freq_used = freq_comb[freq_ind]

fig, ax = plt.subplots(1)
ax.errorbar(T, noise_power[freq_ind], xerr=T_std, yerr=noise_power_std[freq_ind], fmt='.', label='measurement')
ax.plot(np.linspace(1, 700, 1000),
        thermal_noise_func(np.linspace(1, 700, 1000), gs[freq_ind], gi[freq_ind], ns[freq_ind], freq_used, df), '--',
        label='noise fit')
ax.set_xlabel('Temperature (mK)')
ax.set_ylabel('Voltage noise ($\mu V^2$)')
ax.set_title('Frequency = ' + str("{:.4f}".format(freq_used / 1e9)) + ' GHz (' + str(freq_ind) + ')')
ax.legend()

# Plotting residuals
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(ss_res, '.')
ax[0].set_ylabel('Squared sum of residuals')
ax[0].set_title('Calibration')
ax[1].plot(R2, '.')
ax[1].set_xlabel('Frequency index')
ax[1].set_ylabel('R$^2$')

# Plotting signal gain, n and idler gain as a function of frequency
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.errorbar(freq_comb, gs, yerr=gs_std, fmt='.')
ax1.set_ylabel('Gain Signal [dB]')
ax1.set_title('Calibration')
ax2.errorbar(freq_comb, ns, yerr=ns_std, fmt='.')
ax2.plot(freq_comb, ns_min, '.', label='min')
ax2.set_ylabel('n$_s$')
ax2.legend()
ax3.errorbar(freq_comb, gi, yerr=gi_std, fmt='.')
ax3.set_ylabel('Gain Idler [dB]')
ax3.set_xlabel('Frequency [Hz]')

# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# SAVE THE CALIBRATION VALUES
# ---------------------------------------------------------------------------------------------------------------------------------

# Only keep the "reasonable" values for the fitting parameters
gs_filter = gs[(gi_std <= 10) & (gs_std <= 10)]
gs_std_filter = gs_std[(gi_std <= 10) & (gs_std <= 10)]
gi_filter = gi[(gi_std <= 10) & (gs_std <= 10)]
gi_std_filter = gi_std[(gi_std <= 10) & (gs_std <= 10)]
ns_filter = ns[(gi_std <= 10) & (gs_std <= 10)]
ns_std_filter = ns_std[(gi_std <= 10) & (gs_std <= 10)]

sGsGi_filter = sGsGi[(gi_std <= 10) & (gs_std <= 10)]
sGsn_filter = sGsn[(gi_std <= 10) & (gs_std <= 10)]
sGin_filter = sGin[(gi_std <= 10) & (gs_std <= 10)]

freq_comb_filter = freq_comb[(gi_std <= 10) & (gs_std <= 10)]

ni_filter = ni[(gi_std <= 10) & (gs_std <= 10)]
ns_min_filter = ns_min[(gi_std <= 10) & (gs_std <= 10)]

# Plotting the "reasonable" values for the signal gain, n and idler gain as a function of frequency
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].errorbar(freq_comb_filter, gs_filter, yerr=gs_std_filter, fmt='.')
ax[0].set_ylabel('Gain Signal (dB)')
ax[0].set_title('Calibration')
ax[1].errorbar(freq_comb_filter, ns_filter, yerr=ns_std_filter, fmt='.')
ax[1].plot(freq_comb_filter, ns_min_filter, '.', label='min')
ax[1].set_ylabel('n$_s$')
ax[1].legend()
ax[2].errorbar(freq_comb_filter, gi_filter, yerr=gi_std_filter, fmt='.')
ax[2].set_ylabel('Gain Idler (dB)')
ax[2].set_xlabel('Frequency [Hz]')

# Plot of the correlation errors between the fitted parameters
fig2, ax2 = plt.subplots(3, 1, sharex=True)
ax2[0].plot(freq_comb_filter, sGsGi_filter, '.')
ax2[0].set_ylabel('cov(Gs,Gi)')
ax2[0].set_title('Calibration')
ax2[1].plot(freq_comb_filter, sGsn_filter, '.')
ax2[1].set_ylabel('cov(Gs,n)')
ax2[2].plot(freq_comb_filter, sGin_filter, '.')
ax2[2].set_ylabel('cov(Gi,n)')
ax2[2].set_xlabel('Frequency [Hz]')

# Saving the values to use them in th covariance matrix reconstruction
# np.savez( r'D:\VivaceData\Vivace-Calibration\Calib-Parameters-2020-10',
#           gain_signal = gs_filter,
#           gain_signal_std = gs_std_filter,
#           gain_idler = gi_filter,
#           gain_idler_std = gi_std_filter,
#           ns = ns_filter,
#           ns_std = ns_std_filter,
#           sGsGi = sGsGi_filter,
#           sGsn = sGsn_filter,
#           sGin = sGin_filter,
#           freq_comb = freq_comb_filter )

# Example of how to load data
# npzfile = np.load(r'D:\VivaceData\Vivace-Calibration\Calib-Parameters.npz')
# print( npzfile['gain_signal'])


# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# SPECTRAL NOISE DENSITY AS A FUNCTION OF FREQUENCY
# ---------------------------------------------------------------------------------------------------------------------------------

# Load 4 pump data to compare it with the calibration
folder_high = os.path.join(r'D:\VivaceData\2020-10-22\23_39_30\fourpumps_ON_OFF_cycle_2khz_df_modejumping.hdf5')
hf_base = h5py.File(folder_high, 'r')

f1_ind = 1
p1_ind = 5

nmodes = np.arange(0, 32, 2)

freq1_grp = hf_base.require_group('pump_freq_ind_' + str(f1_ind))
power1_grp = freq1_grp.require_group('pump_amp_ind' + str(p1_ind))

# Bandwidth of the 4 pump data
df2 = freq1_grp['df'][0]

# Frequency comb of the 4 pump data
freq_comb_pumpoff = freq1_grp['frequency comb'][:] + 3.8e+9

# We upload the raw data for the SAW pump off
off_raw = power1_grp['complex high sideband OFF'][:] * np.sqrt(2) * 6.33645961
partial_set_off = off_raw[:, nmodes]
real_part = np.real(partial_set_off)
im_part = np.imag(partial_set_off)
coord_array = [(real_part[:, v], im_part[:, v]) for v in range(len(nmodes))]
coord_array = np.concatenate(coord_array, axis=0)

# Voltage noise at base temperature
base_noise = np.mean(np.abs(off_raw) ** 2, axis=0) * 1e12
base_noise_std = np.std(np.abs(off_raw) ** 2, axis=0) * 1e12 / np.sqrt(len(off_raw))

# Base temperature
base_temp = 15

# Plotting the spectral noise density as a function of frequency for the calibration + SAW pump off data
fig, ax = plt.subplots(1, 1)
for i in range(len(T)):
    ax.semilogy(freq_comb, noise_power[:, i] / df, '.-', label="{:.1f}".format(T[i]))
ax.semilogy(freq_comb_pumpoff, base_noise / df2, '.-', label="15")
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Spectral Noise Density')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plotting the frequency combs
fig, ax = plt.subplots()
ax.plot(freq_comb, '.', label='Calibration')
ax.plot(freq_comb_pumpoff, '.', label='SAW pump off')
ax.hlines(freq_comb_pumpoff[12], 0, 64)
ax.set_title('Frequency combs')
ax.set_xlabel('Frequency index')
ax.set_ylabel('Frequency [Hz]')
ax.legend()

# Plotting the spectral noise density as a function of temperature including the base point for a single frequency
freq_ind = 0

if calibration == 'jan':
    freq_disp = 11
elif calibration == 'oct':
    freq_disp = 2

fig, ax = plt.subplots(1)
ax.errorbar(T, noise_power[freq_ind] / df, xerr=T_std, yerr=noise_power_std[freq_ind] / df, fmt='.',
            label='Calibration')
ax.plot(np.linspace(1, 700, 1000),
        thermal_noise_func(np.linspace(1, 700, 1000), gs[freq_ind], gi[freq_ind], ns[freq_ind], freq_comb[freq_ind],
                           df) / df, '--')
ax.errorbar(base_temp, base_noise[freq_ind + freq_disp] / df2, yerr=base_noise_std[freq_ind + freq_disp] / df2, fmt='.',
            label='SAW pump off')
ax.set_xlabel('Temperature (mK)')
ax.set_ylabel('Spectral Noise Density')
ax.legend()

# Plotting the pump off noise and gain as a function of frequency
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].errorbar(freq_comb_pumpoff, base_noise / df2, yerr=base_noise_std / df2, fmt='.-')
ax[0].set_ylabel('Spectral noise density')
ax[1].errorbar(freq_comb_filter, gs_filter, yerr=gs_std_filter, fmt='.-')
ax[1].set_ylabel('Signal gain [dB]')
ax[1].set_xlabel('Frequency [Hz]')

# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# CALIBRATION ADDING THE CONSTRAINT: Gi = Gs - 1
# ---------------------------------------------------------------------------------------------------------------------------------

# Parameters initial guess (The fit is not very sensitive to the initial conditions)
g0_signal = 80
n0 = 11

# discard the upper x points
x = 1

# lower boundary for ns (-2 does not affect the fitting)
ns_boundary = np.linspace(-2, 10, 13, endpoint=True)

# Fit parameters
gs_cons = np.zeros((len(ns_boundary), len(freq_comb)))
gs_std_cons = np.zeros_like(gs_cons)

ns_cons = np.zeros_like(gs_cons)
ns_std_cons = np.zeros_like(gs_cons)

gi_cons = np.zeros_like(gs_cons)
gi_std_cons = np.zeros_like(gs_cons)

sGsn_cons = np.zeros_like(gs_cons)

ss_res_cons = np.zeros_like(gs_cons)
R2_cons = np.zeros_like(gs_cons)

# Loop for the lowe boundary for ns
for ns_ind, nslim in enumerate(ns_boundary):

    for freq_ind in range(len(freq_comb)):
        freq_used = freq_comb[freq_ind]

        # Fit to temperature data
        popt, pcov = curve_fit(lambda T, Gsignal, n: thermal_noise_func_constrained(T[:-x], Gsignal, n, freq_used, df),
                               T,
                               noise_power[freq_ind, :-x],
                               p0=(g0_signal, n0),
                               sigma=noise_power_std[freq_ind, :-x],
                               bounds=([-np.inf, nslim], [np.inf, np.inf]),
                               maxfev=10000)

        # Extraction of the fitting parameters, i.e. gains and added number of photons
        gs_cons[ns_ind, freq_ind] = popt[0]
        gs_std_cons[ns_ind, freq_ind] = np.sqrt(pcov[0, 0])

        ns_cons[ns_ind, freq_ind] = popt[1]
        ns_std_cons[ns_ind, freq_ind] = np.sqrt(pcov[1, 1])

        gi_cons[ns_ind, freq_ind] = 10 * np.log10(10 ** np.abs(gs_cons[ns_ind, freq_ind] / 10) - 1)
        gi_std_cons[ns_ind, freq_ind] = gs_std_cons[ns_ind, freq_ind]

        # Cross correlations in the errors
        sGsn_cons[ns_ind, freq_ind] = pcov[0, 1]

        # Fit quality parameters
        residuals = noise_power[freq_ind, :-x] - thermal_noise_func_constrained(T[:-x], gs_cons[ns_ind, freq_ind],
                                                                                ns_cons[ns_ind, freq_ind], freq_used,
                                                                                df)
        ss_res_cons[ns_ind, freq_ind] = np.sum(residuals ** 2)
        ss_tot = np.sum((noise_power[freq_ind] - np.mean(noise_power[freq_ind])) ** 2)
        R2_cons[ns_ind, freq_ind] = 1 - ss_res_cons[ns_ind, freq_ind] / ss_tot

# Plotting residuals
zmax = np.max(ss_res_cons)
zmin = 0

fig, ax = plt.subplots(1)
a = ax.pcolormesh(freq_comb, ns_boundary, ss_res_cons, shading='nearest', cmap='RdBu_r', vmin=zmin, vmax=zmax)
ax.set_title('Squared sum of residuals', fontsize='large')
ax.set_ylabel('n$_s$ lower boundary')
ax.set_xlabel('Frequency [Hz]')
fig.colorbar(a)

# Plotting signal gain
zmax = np.max(gs_cons)
zmin = np.min(gs_cons)

fig, ax = plt.subplots(1)
a = ax.pcolormesh(freq_comb, ns_boundary, gs_cons, shading='nearest', cmap='RdBu_r', vmin=zmin, vmax=zmax)
ax.set_title('G$_s$', fontsize='large')
ax.set_ylabel('n$_s$ lower boundary')
ax.set_xlabel('Frequency [Hz]')
fig.colorbar(a)

# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------------------------------------------------------------

# Voltage noise vs temperature for a single frequency
freq_ind = 21
freq_used = freq_comb[freq_ind]

fig, ax = plt.subplots(1)
ax.errorbar(T, noise_power[freq_ind], xerr=T_std, yerr=noise_power_std[freq_ind], fmt='.', label='measurement')
ax.plot(np.linspace(1, 700, 1000),
        thermal_noise_func_constrained(np.linspace(1, 700, 1000), gs_cons[0, freq_ind], ns_cons[0, freq_ind], freq_used,
                                       df), '--', label='noise fit')
ax.set_xlabel('Temperature (mK)')
ax.set_ylabel('Voltage noise ($\mu V^2$)')
ax.set_title('Frequency = ' + str("{:.4f}".format(freq_used / 1e9)) + ' GHz (' + str(freq_ind) + ')')
ax.legend()

# Residuals
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(ss_res_cons[0], '.')
ax[0].set_ylabel('Squared sum of residuals')
ax[0].set_title('Calibration - Constrained')
ax[1].plot(R2_cons[0], '.')
ax[1].set_xlabel('Frequency index')
ax[1].set_ylabel('R$^2$')

# Signal gain, n and idler gain as a function of frequency
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].errorbar(freq_comb, gs_cons[0], yerr=gs_std_cons[0], fmt='.')
ax[0].set_ylabel('Gain Signal (dB)')
ax[0].set_title('Calibration - Constrained')
ax[1].errorbar(freq_comb, ns_cons[0], yerr=ns_std_cons[0], fmt='.')
ax[1].set_ylabel('n$_s$')
ax[2].errorbar(freq_comb, gi_cons[0], yerr=gi_std_cons[0], fmt='.')
ax[2].set_ylabel('Gain Idler (dB)')
ax[2].set_xlabel('Frequency [Hz]')

# Plots of the pump off noise and gain as a function of frequency
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].errorbar(freq_comb_pumpoff, base_noise / df2, yerr=base_noise_std / df2, fmt='.-')
ax[0].set_ylabel('Spectral noise density')
ax[1].errorbar(freq_comb, gs_cons[0], yerr=gs_std_cons[0], fmt='.-')
ax[1].set_ylabel('Signal gain [dB]')
ax[1].set_xlabel('Frequency [Hz]')

# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# SAVE THE "CONSTRAINED" CALIBRATION VALUES
# ---------------------------------------------------------------------------------------------------------------------------------

# Only keep the "reasonable" values for the fitting parameters
gs_cons_filter = gs_cons[0][(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]
gs_std_cons_filter = gs_std_cons[0][(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]
gi_cons_filter = gi_cons[0][(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]
gi_std_cons_filter = gi_std_cons[0][(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]
ns_cons_filter = ns_cons[0][(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]
ns_std_cons_filter = ns_std_cons[0][(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]

sGsn_cons_filter = sGsn_cons[0][(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]

freq_comb_cons_filter = freq_comb[(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]

ni_filter_cons = ni[(gi_std_cons[0] <= 1) & (gs_std_cons[0] <= 1)]

# Plot of the reasonable values for the signal gain, n and idler gain as a function of frequency
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.errorbar(freq_comb_cons_filter, gs_cons_filter, yerr=gs_std_cons_filter, fmt='.')
ax1.set_ylabel('Gain Signal [dB]')
ax1.set_title('Calibration - Constrained')
ax2.errorbar(freq_comb_cons_filter, ns_cons_filter, yerr=ns_std_cons_filter, fmt='.')
ax2.set_ylabel('n$_s$')
ax3.errorbar(freq_comb_cons_filter, gi_cons_filter, yerr=gi_std_cons_filter, fmt='.')
ax3.set_ylabel('Gain Idler [dB]')
ax3.set_xlabel('Frequency [Hz]')

# Saving the values to use them in th covariance matrix reconstruction
# np.savez( r'D:\VivaceData\Vivace-Calibration\Calib-Parameters-2020-10-Constrained-gi',
#           gain_signal = gs_cons_filter,
#           gain_signal_std = gs_std_cons_filter,
#           gain_idler = gi_cons_filter,
#           gain_idler_std = gi_std_cons_filter,
#           ns = ns_cons_filter,
#           ns_std = ns_std_cons_filter,
#           sGsn = sGsn_cons_filter,
#           freq_comb = freq_comb_cons_filter )


# %%

# ---------------------------------------------------------------------------------------------------------------------------------
# CHECK
# ---------------------------------------------------------------------------------------------------------------------------------

npzfile = np.load(r'D:\VivaceData\Vivace-Calibration\Calib-Parameters-2020-10-Constrained-gi.npz')

gs = npzfile['gain_signal']
gs_std = npzfile['gain_signal_std']
gi = npzfile['gain_idler']
gi_std = npzfile['gain_idler_std']
ns = npzfile['ns']
ns_std = npzfile['ns_std']
sGsGi = npzfile['sGsGi']
freqs_calibration = npzfile['freq_comb']

# Plotting the gain and n
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].errorbar(freqs_calibration, gs, yerr=gs_std, fmt='.', label='Calibration')
ax[0].set_ylabel('Gain Signal [dB]')
ax[1].errorbar(freqs_calibration, ns, yerr=ns_std, fmt='.', label='Calibration')
ax[1].set_ylabel('n signal')
ax[2].errorbar(freqs_calibration, gi, yerr=gi_std, fmt='.', label='Calibration')
ax[2].set_ylabel('Gain Idler [dB]')
ax[2].set_xlabel('frequencies [Hz]')
