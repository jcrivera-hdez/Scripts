# -*- coding: utf-8 -*-

"""
Created on Tue Jan 26 09:26:48 2021
@author: JC

Last update: 2021-12-20
    
"""

import time
import pyvisa as visa
import h5py
import os
import inspect
import numpy as np
import pandas as pd



###########################################################################
# Save folder location, save file name and run name

save_folder = r"C:/Users/Admin/Desktop/Local_JC/"
save_file = r"QuantumGarage.hdf5"
myrun = time.strftime("%Y-%m-%d_%H_%M_%S")
t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

###########################################################################
# Sample name, temperature and total attentuation along measurement chain

sample = "JPA"
temperature = 0.010     # K
atten = 60              # dB
comment_str = "forward"

###########################################################################
# Signal sweep values

f_start = 3.06                              # GHz
f_stop = 3.66                               # GHz
intbw = 100e+3                              # Hz
f_delta = 100e+3                            # Hz
p_in_arr = np.linspace( -40, 0, 10 )        # dBm
Navg = 10

# Check
if f_stop <= f_start:
    raise ValueError("f_stop needs to be larger than f_start. Current values are f_start = {} GHz, f_stop = {} GHz".format(f_start, f_stop))

# Points in frequency array, adjust f_stop
Npts = int((f_stop * 1e+9 - f_start * 1e+9) / f_delta)

# VNA limitation: Npts cannot exceed 100_001
if Npts > 100_001:
    raise ValueError("Npts cannot exceed 100001. It is {}".format(Npts))

# Update f_stop
f_stop = (f_start * 1e+9 + Npts * f_delta) / 1e+9

# Empty arrays for storaging data
freq_arr = np.zeros(( len(p_in_arr), Npts ))
amp_arr = np.zeros_like(freq_arr)
pha_arr = np.zeros_like(freq_arr)

############################################################################
# Setup VNA

# Initaite power sweep
for pow_ind, p_in in enumerate(p_in_arr):

    # Resource manager for VNA
    rm = visa.ResourceManager('C:\\Program Files (x86)\\IVI Foundation\\VISA\\WinNT\\agvisa\\agbin\\visa32.dll')
    myVNA = rm.open_resource('TCPIP0::DESKTOP-HCDF01N::hislip0::INSTR')
    myVNA.timeout = 10_000
    myVNA.write("*CLS")
    myVNA.write("*IDN?")
    
    # Verify
    print("VNA identification: {}".format(myVNA.read()))

    # Reset VNA settings
    myVNA.write("SYSTem:PRESet")
    myVNA.write("FORM:DATA ASCii,0")

    # Setup a S21
    myVNA.write("CALCulate1:MEASure1:PARameter 'S21'")
    myVNA.write("SENSe1:SWE:MODE HOLD")
        
    # Apply sweep settings
    myVNA.write("SENSe1:BANDwidth {}".format(intbw))
    myVNA.write("SENSe1:FREQuency:START {}ghz".format(f_start))
    myVNA.write("SENSe1:FREQuency:STOP {}ghz".format(f_stop))
    myVNA.write("SENSe1:SWEep:POINts {}".format(Npts))
    myVNA.write("SOURce1:POWer:AMPlitude {}".format(p_in))
    myVNA.write("SENSe1:AVERage:MODE POINt")
    myVNA.write("SENSe1:AVERage:STATe ON")
    myVNA.write("SENSe1:AVERage:COUNt {}".format(int(Navg)))
    myVNA.write("SENSe1:SWEep:GENeration ANAL")
    
    # Reference oscillator
    print("VNA ref. oscillator: {}".format(myVNA.query("SENS:ROSC:SOUR?")))
    
    # Debug
    print("VNA setup for S21 sweep")

    # Headers for VNA data processing
    headers = ['Frequency [Hz]', 'S11mag [dB]', 'S11phase [deg]', 'S21mag [dB]',
    'S21phase [deg]', 'S12mag [dB]', 'S12phase [deg]', 'S22mag [dB]', 'S22phase [deg]']

    #############################################################################
    # Sweep

    # Sweep time
    myVNA.write("SENSe1:SWEep:TIME:AUTO ON")
    myVNA.write("SENSe1:SWEep:TYPE LIN")
    myVNA.write("SENS:SWE:TIME?")
    sweeptime = round(float(myVNA.read()), 3)
    print("Sweep time of {} s".format(sweeptime))

    # Start sweep
    myVNA.write("SENS:SWE:MODE SING")
    print("Sweep started")

    # Polling
    sweepdone = False
    myVNA.write("*ESE 1")
    myVNA.write("*CLS")
    myVNA.write("*OPC")
    while (not sweepdone):
        x = myVNA.query("*ESR?")
        sweepdone = int(x.strip("+-\n"))
        time.sleep(0.1)
    print("Sweep completed")

    # Read data in Touchstone format
    myVNA.write("CALC1:MEAS1:DATA:SNP? 2")
    time.sleep(0.1)
    snp = myVNA.read()
    print("Read data")

    # Process data
    snplst = snp.split(',')
    lst = list(map(list, zip(*([iter(snplst)] * Npts))))
    datalst = [list(map(float, sublst)) for sublst in lst]
    dframe = pd.DataFrame(datalst, index = headers).T
    freq_arr[pow_ind] = dframe["Frequency [Hz]"].to_numpy()
    amp_arr[pow_ind] = dframe["S21mag [dB]"].to_numpy()
    pha_arr[pow_ind] = dframe["S21phase [deg]"].to_numpy()

#######################################################################
# RF off

# VNA RF off
myVNA.write("OUTPut:STATe OFF")
print("VNA RF off")
rm.close()
print("VNA disconnected")

######################################################################
# Create a dictionary with run attributes

# Get current time
t_end = time.strftime("%Y-%m-%d_%H_%M_%S")

# Attributes for this run
myrun_attrs = {"Meas": "S21_sweep",
               "Instr": "VNA",
               "Sample": sample,
               "T": temperature,
               "att": atten,
               "RT-amp_out": 20,
               "RT-amp_in": 0,
               "Comment": comment_str,
               "p_in_start": p_in_arr[0],
               "p_in_stop": p_in_arr[-1],
               "intbw": intbw,
               "delta_f" : f_delta,
               "Npoints": Npts,
               "Navg": Navg,
               "f_start": f_start,
               "f_stop": f_stop,
               "t_start": t_start,
               "t_end": t_end,
               "Script name": os.path.basename(__file__)}

########################################################################
# Save data and attributes

# Create folders if they do not exist
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

# String handles for data
freq_data_str = "{}/{}/Frequency".format(sample, myrun)
amp_data_str = "{}/{}/Amplitude".format(sample, myrun)
pha_data_str = "{}/{}/Phase".format(sample, myrun)
pow_data_str = "{}/{}/Power".format(sample, myrun)

# String handles for script
run_str = "{}/{}".format(sample, myrun)
source_str = "{}/{}/Source code".format(sample, myrun)

# Read lines of the script
filename = inspect.getframeinfo(inspect.currentframe()).filename
with open(filename, "r") as codefile:
    code_lines = codefile.readlines()

# Open the save file (.hdf5) in append mode
with h5py.File(os.path.join(save_folder, save_file), "a") as savefile:
    
    # Save data
    savefile.create_dataset(freq_data_str, (np.shape(freq_arr[0])), dtype=float, data=(freq_arr[0]))
    savefile.create_dataset(amp_data_str, (np.shape(amp_arr)), dtype=float, data=(amp_arr))
    savefile.create_dataset(pha_data_str, (np.shape(pha_arr)), dtype=float, data=(pha_arr))
    savefile.create_dataset(pow_data_str, (np.shape(p_in_arr)), dtype=float, data=(p_in_arr))

    # Write dataset attributes
    savefile[freq_data_str].attrs["Unit"] = "Hz"
    savefile[amp_data_str].attrs["Unit"] = "dB"
    savefile[pha_data_str].attrs["Unit"] = "deg"
    savefile[pow_data_str].attrs["Unit"] = "dBm"
    
    # Check
    print("Saved data")
    
    # Save script
    dt = h5py.special_dtype(vlen=str)
    code_set = savefile.create_dataset(source_str.format(sample, myrun), (len(code_lines),), dtype=dt)
    for i in range(len(code_lines)):
        code_set[i] = code_lines[i]

    # Save the attributes of the run
    for key in myrun_attrs:
        savefile[run_str].attrs[key] = myrun_attrs[key]

    # Ckeck
    print("Saved script and run attributes.")

# Check
print("Finished on:", time.strftime("%Y-%m-%d_%H%M%S"))
print("Done")