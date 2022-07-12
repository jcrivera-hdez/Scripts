import numpy as np
import matplotlib.pyplot as plt


def plot_xbrokenaxis(x, y, xlim1_, xlim2_, ylim_, xlabel_, ylabel_, figure, axis):
    # Set axis labels
    axis[0].set_ylabel(str(ylabel_))
    axis[1].set_xlabel(str(xlabel_))

    # Plot same data on both axis
    axis[0].semilogy(x, y, '.-')
    axis[1].semilogy(x, y, '.-')

    # limit each x-axis to the chosen range
    axis[0].set_xlim(*xlim1_)
    axis[1].set_xlim(*xlim2_)
    axis[0].set_ylim(*ylim_)

    # hide spines between both axes
    axis[0].spines['right'].set_visible(False)
    axis[1].spines['left'].set_visible(False)
    axis[1].yaxis.tick_right()

    # how big to make the diagonal lines in axes coordinates
    d = .015

    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axis[0].transAxes, color='k', clip_on=False)
    axis[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axis[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=axis[1].transAxes)  # switch to the bottom axes
    axis[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axis[1].plot((-d, +d), (-d, +d), **kwargs)

    # vary the distance between plots
    figure.subplots_adjust(wspace=0.2)

    return figure


# Example of use for plot_xbrokenaxis
# Parameters
fs = 10                 # Sampling frequency
df = 0.001              # Bandwidth
T = 1. / df             # Period
T_relax = 5 * T
dt = 1. / fs
time = dt*(np.arange((T+T_relax)/dt) + 1)
omega1 = 2*np.pi*1      # Drive 1 angular frequency
omega2 = 2*np.pi*1.1    # Drive 1 angular frequency

# signal and frequencies
signal = np.abs(np.fft.fft(np.cos(omega1 * time) + np.cos(omega2 * time)))
frequency = np.fft.fftfreq(len(time), d=dt)

# Plot
fig, ax = plt.subplots(1, 2, sharey=True, constrained_layout=False)
xlim1 = np.array([-1.2, -0.9])
xlim2 = np.array([0.9, 1.2])
ylim = np.array([1e-12, 1e6])
xlabel = str('frequency [Hz]')
ylabel = str('signal [a.u.]')
plot_xbrokenaxis(frequency, signal, xlim1, xlim2, ylim, xlabel, ylabel, fig, ax)
plt.show()
