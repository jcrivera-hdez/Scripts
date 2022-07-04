import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['figure.constrained_layout.use'] = True


def dB(Sjk):
    return 20 * np.log10(np.abs(Sjk))


def plot_gain_cut(gain, farr, pump_val, bias_val, axis):
    axis.plot(farr / 1e9, gain, label=f'{pump_val:.3f} fsu')
    axis.set_title(r'DC bias ' + f'= {bias_val:.3f} V')
    axis.set_xlabel('frequency [GHz]')
    axis.set_ylabel('gain [dB]')


def plot_gain(gain, x_arr, y_arr, figure, axis):
    xmin, xmax = np.min(x_arr), np.max(x_arr)
    ymin, ymax = np.min(y_arr) / 1e9, np.max(y_arr) / 1e9

    cutoff = 0.01
    zmin = np.percentile(gain, cutoff)
    zmax = np.percentile(gain, 100. - cutoff)

    a = axis.imshow(gain.T,
                    origin="lower",
                    aspect="auto",
                    extent=[xmin, xmax, ymin, ymax],
                    vmin=zmin,
                    vmax=zmax,
                    cmap="RdBu_r",
                    interpolation=None,
                    )
    figure.colorbar(a, label=r'gain [dB]')
    axis.set_xlabel('DC bias [V]')
    axis.set_ylabel('frequency [GHz]')
    axis.set_title('pump power = ' + f'{pump_pwr_arr[pump_idx]:.3f} fsu')

    plt.show()


def plot_gainsweep(gain_arr_, freq_arr_, bias_arr_):
    nr_rows = 4
    nr_columns = 3
    nr_plots = nr_rows * nr_columns

    n = nr_pump_pwr // nr_plots

    fig, ax = plt.subplots(nr_rows, nr_columns, figsize=[19, 9.5], constrained_layout=True)
    ax = ax.flatten()

    xmin, xmax = np.min(bias_arr_), np.max(bias_arr_)
    ymin, ymax = np.min(freq_arr_) / 1e9, np.max(freq_arr_) / 1e9

    cutoff = 0.01  # %
    zmin = np.percentile(gain_arr_, cutoff)
    zmax = np.percentile(gain_arr_, 100. - cutoff)

    for axi in range(nr_plots):
        amp_ind = n * axi
        a = ax[axi].imshow(gain_arr_[amp_ind].T,
                           origin="lower",
                           aspect="auto",
                           extent=[xmin, xmax, ymin, ymax],
                           vmin=zmin,
                           vmax=zmax,
                           cmap="RdBu_r",
                           interpolation=None,
                           )
        ax[axi].set_title(r'$A_p$' + f' = {pump_pwr_arr[amp_ind + 1]:.3f} fsu')
        ax[axi].axvline(x=bias_arr_[bias_idx], linestyle='--', color='black')
    fig.colorbar(a, ax=ax[:], location='right', label=r'gain [dB]', shrink=0.6)
    [ax[axi].set_xlabel('DC bias [V]') for axi in [9, 10, 11]]
    [ax[axi].set_ylabel('Frequency [GHz]') for axi in [0, 3, 6, 9]]

    plt.show()


# Load data
file = r'D:\JPA\JPA-Data\QuantumGarage-JPA.hdf5'
cooldown = r'2022-06-30'
run = '2022-07-01_12_00_50'
idx_str = "{}/{}".format(cooldown, run)

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
bias_idx = 55

# Load data    
run = '2022-06-09_10_46_19'
idx_str = "{}/{}".format(cooldown, run)

# Open hdf5 file
with h5py.File(file, "r") as dataset:
    # Attributes
    df_ref = dataset[idx_str].attrs["df"]
    bias_ref = dataset[idx_str].attrs["DC bias"][0]

    # Data
    freq_ref_arr = np.asarray(dataset[idx_str]["freq sweep"])
    usb_ref_arr = np.asarray(dataset[idx_str]["USB"])

# Gain normalised
gain_ref = dB(usb_ref_arr) - 20  # Check this (the amplitude in fsu of the reference meas was 10 bigger)
gain_arr = dB(usb_arr[1:]) - gain_ref

# Plot background (no-pump) response
fig0, ax0 = plt.subplots(1)
plot_gain_cut(gain_ref, freq_arr, 0, bias_ref, ax0)
plt.show()

# Plot gain sweep for all pump powers
plot_gainsweep(gain_arr, freq_arr, bias_arr)

# Plot gain at a given pump power
pump_idx = -1
fig1, ax1 = plt.subplots(1)
plot_gain(gain_arr[pump_idx], bias_arr, freq_arr, fig1, ax1)


fig2, ax2 = plt.subplots(1)
for pump_idx in range(nr_pump_pwr - 1):
    plot_gain_cut(gain_arr[pump_idx, bias_idx], freq_arr, pump_pwr_arr[pump_idx + 1], bias_arr[bias_idx], ax2)
ax2.legend(title=r'$A_p$')
plt.show()
