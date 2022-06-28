import matplotlib.pyplot as plt
import numpy as np
import h5py

plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['figure.constrained_layout.use'] = True


def dB(Sjk):
    return 20 * np.log10(np.abs(Sjk))


def plotSjk(farr, Sjk):
    plt.plot(farr / 1e9, dB(Sjk), 'darkred')
    plt.grid(True)
    plt.xlabel('f [GHz]')
    plt.ylabel('$S_{11}$[dB]')


def groupdelayFromS(freqarr, Sjk, windowSize=3):

    df = np.mean(np.diff(freqarr))
    phase = np.unwrap(np.angle(Sjk))
    smoothphase = np.zeros((len(bias_arr), len(freq_arr) - 2))
    for idx in range(len(phase[:, 0])):
        smoothphase[idx] = np.convolve(phase[idx, :], np.ones((windowSize,)) / windowSize, mode='valid')
    gd_ = np.abs(np.diff(smoothphase, axis=1) / 2 / np.pi / df) * 1e9

    lendiff = len(freqarr) - len(gd_)

    freq_gd_ = freqarr[:-lendiff] + df * lendiff / 2

    return {'freq_gd': freq_gd_, 'gd': gd_, 'phasegd': smoothphase}


def plotgd(groupdelay, x_arr, y_arr, figure, axis, label):
    xmin, xmax = np.min(x_arr), np.max(x_arr) + 0.01
    ymin, ymax = np.min(y_arr) / 1e9, np.max(y_arr) / 1e9
    zmax = 65
    zmin = 50

    a = axis.imshow(groupdelay.T,
                    origin="lower",
                    aspect="auto",
                    extent=[xmin, xmax, ymin, ymax],
                    vmin=zmin,
                    vmax=zmax,
                    cmap="Spectral",
                    interpolation=None,
                    )

    figure.colorbar(a, label="Group delay [ns]")

    if label == 'flux':
        axis.set_xlabel("$\Phi/\Phi_0$")
    else:
        axis.set_xlabel("DC bias [V]")
    axis.set_ylabel("Frequency [GHz]")

    plt.show()


# Load data
file = r'D:\JPA\JPA-Data\QuantumGarage-JPA.hdf5'
cooldown = '2022-06-07'
run = '2022-06-07_17_01_41'
idx_str = "{}/{}".format(cooldown, run)

# Open hdf5 file
with h5py.File(file, "r") as dataset:
    # Data
    freq_arr = np.asarray(dataset[idx_str]["freq_arr"])
    bias_arr = np.asarray(dataset[idx_str]["bias_arr"])
    S11_arr = np.asarray(dataset[idx_str]["s11_arr"])

# Group delay
group_delay = groupdelayFromS(freq_arr, S11_arr)
gd = group_delay['gd']
freq_gd = group_delay['freq_gd']

# Parameters values
M = 7.65e-13  # H
Rb = 1000  # Ohm
Flux_quanta = 2.07e-15  # Wb

# Flux in terms of the flux quanta
flux_arr = M / Rb * bias_arr / Flux_quanta
# Plot group delay
fig0, ax0 = plt.subplots(1)
plotgd(gd, bias_arr, freq_gd, fig0, ax0, 'bias')
fig1, ax1 = plt.subplots(1)
plotgd(gd, flux_arr, freq_gd, fig1, ax1, 'flux')

# Group delay at a given dc bias value
bias_idx = 555
fig2, ax2 = plt.subplots(1)
ax2.plot(freq_gd / 1e9, gd[:, bias_idx])
ax2.set_xlabel('frequency [GHz]')
ax2.set_ylabel('group delay [ns]')
ax2.set_title('DC bias ' + f'= {bias_arr[bias_idx]:.3f} V')
plt.show()
