from typing import List, Union, Any

import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader
from scipy.constants import Planck

# Read all txt files in this folder:
path = r'D:\TLS\TLS-Data'
all_files = glob.glob(path + "/*.txt")

# concatenate all files into a single np.array called frame:
li: List[Union[Union[TextFileReader, Series, DataFrame, None], Any]] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None, sep=" ")
    li.append(df)
frame = pd.concat(li, ignore_index=True, axis=0)
frame = np.array(frame)

# rows in order: Qi, Qi_err. Qc, Qc_err, Ql, Ql_err, fr, fr_err, pow_span, n
Qi = np.array([frame[k][:] for k in range(0, len(frame), 10)])
Qi_err = np.array([frame[k + 1][:] for k in range(0, len(frame), 10)])
fr = np.array([frame[k + 6][:] for k in range(0, len(frame), 10)])
n = np.array([frame[k + 9][:] for k in range(0, len(frame), 10)])
power = np.array([frame[k + 8][:] for k in range(0, len(frame), 10)])

plt.style.use('bmh')  # to explore your options: print(plt.style.available)
markers = ['s', 'v', 'X', 'D', 'o', '^', 'p', 'P']

# %% Qi vs n

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xscale('log')
ax.set_yscale('log')
for h in range(0, len(Qi)):
    freq = fr[h, 0]
    ax.errorbar(n[h], Qi[h], yerr=Qi_err[h], fmt='o', label=str(np.round(freq * 1e-9, 3)) + ' GHz', markersize=10,
                color='#6CB0E4')

# ax.legend()
ax.grid(which='both', axis='y', color='darkgrey')
ax.grid(which='major', axis='x', color='darkgrey')
plt.xlim([1e-2, 1e8])

ax.set_xlabel(r'$\langle n\rangle$', fontsize=24)
ax.set_ylabel('$Q_i$', fontsize=24)

# %% Qi vs P

fig, ax = plt.subplots(figsize=(15, 10))
for h in range(0, len(Qi)):
    freq = fr[h, 0]
    ax.plot(power[h] + 55, fr[h], 'o', label=str(np.round(freq * 1e-9, 3)) + ' GHz', markersize=10, color='#6CB0E4')
ax.set_xlabel('VNA output ower [dBm]', fontsize=24)
ax.set_ylabel('$f_r$', fontsize=24)

# %% n vs P

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_yscale('log')

for h in range(0, len(Qi)):
    freq = fr[h, 0]
    ax.plot(power[h] + 55, n[h], 'o', label=str(np.round(freq * 1e-9, 3)) + ' GHz', markersize=10, color='#6CB0E4')

ax.set_xlabel('VNA output power [dBm]', fontsize=24)
ax.set_ylabel(r'$\langle n\rangle$', fontsize=24)

# %% 

# Manually calculating photons number.

Ql = np.array([frame[k + 4][:] for k in range(0, len(frame), 10)])
Ql_err = np.array([frame[k + 5][:] for k in range(0, len(frame), 10)])

Qc = np.array([frame[k + 2][:] for k in range(0, len(frame), 10)])
Qc_err = np.array([frame[k + 3][:] for k in range(0, len(frame), 10)])

nphot = 2 * Ql ** 2 * 10 ** ((power - 30) / 10) / (2 * np.pi * Planck * fr ** 2 * Qc)

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_yscale('log')

for h in range(0, len(Qi)):
    freq = fr[h, 0]
    ax.plot(power[h] + 55, nphot[h], 'o', label=str(np.round(freq * 1e-9, 3)) + ' GHz', markersize=10, color='#6CB0E4')
    # ax.plot(power[h]+55,n[h],'o',label=str(np.round(freq*1e-9,3))+' GHz', markersize=10)

ax.set_xlabel('VNA output power [dBm]', fontsize=24)
ax.set_ylabel(r'$\langle n\rangle$', fontsize=24)
plt.show()