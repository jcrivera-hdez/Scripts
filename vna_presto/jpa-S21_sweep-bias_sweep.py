import os
import time
import h5py
import inspect

import pyvisa as visa
import numpy as np
import pandas as pd
from tqdm import tqdm

from presto import lockin

###########################################################################
# Save folder location, save file name and run name

save_folder = r"C:/Users/Admin/Desktop/Local_JC/"
save_file = r"test.hdf5"
myrun = time.strftime("%Y-%m-%d_%H_%M_%S")

t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

###########################################################################
# Sample name, temperature and total attenuation along measurement chain

sample = "JPA"
temperature = 0.012
atten = 80
comment_str = "forward"

############################################################################
# Basic setup

# Presto configuration
ADDRESS = "delta"
bias_port = 1

# VNA configuration
os.add_dll_directory('C:\\Program Files (x86)\\IVI Foundation\\VISA\\WinNT\\agvisa\\agbin\\visa32.dll')
rm = visa.ResourceManager()
myVNA = rm.open_resource('TCPIP0::DESKTOP-HCDF01N::hislip0::INSTR')

# Headers for VNA data processing
headers = ['Frequency [Hz]', 'S11mag [dB]', 'S11phase [deg]', 'S21mag [dB]',
           'S21phase [deg]', 'S12mag [dB]', 'S12phase [deg]', 'S22mag [dB]', 'S22phase [deg]']


############################################################################
# VNA methods

def save_script(folder, file, sample, myrun, myrun_attrs, verbose=False):
    # Create folders if they do not exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # String handles
    run_str = "{}/{}".format(sample, myrun)
    source_str = "{}/{}/Source code".format(sample, myrun)

    # Read lines of the script
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    with open(filename, "r") as codefile:
        code_lines = codefile.readlines()

    # Write lines of script and run attributes
    with h5py.File(os.path.join(folder, file), "a") as savefile:

        dt = h5py.special_dtype(vlen=str)
        code_set = savefile.create_dataset(source_str.format(sample, myrun), (len(code_lines),), dtype=dt)
        for i in range(len(code_lines)):
            code_set[i] = code_lines[i]

        # Save the attributes of the run
        for key in myrun_attrs:
            savefile[run_str].attrs[key] = myrun_attrs[key]

    if verbose:
        print("Saved script and run attributes.")


def save_data(folder, file, sample, myrun, freq_arr, s21_arr, bias_arr, verbose=False):
    # Create folders if they do not exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # String handles
    freq_data_str = "{}/{}/freq_arr".format(sample, myrun)
    s21_data_str = "{}/{}/s11_arr".format(sample, myrun)
    bias_data_str = "{}/{}/bias_arr".format(sample, myrun)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(folder, file), "a") as savefile:
        savefile.create_dataset(freq_data_str, (np.shape(freq_arr)), dtype=np.float64, data=freq_arr)
        savefile.create_dataset(s21_data_str, (np.shape(s21_arr)), dtype=np.complex64, data=s21_arr)
        savefile.create_dataset(bias_data_str, (np.shape(bias_arr)), dtype=np.float64, data=bias_arr)

        # Write dataset attributes
        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[bias_data_str].attrs["Unit"] = "V"

    if verbose:
        print("Saved data")


def setvolt(bias_val):
    with lockin.Lockin(address=ADDRESS) as lck:
        # Set DC bias on JPA
        lck.hardware.set_dc_bias(bias=bias_val,
                                 port=bias_port,
                                 range_i=3,
                                 )

        time.sleep(0.1)


def setup_vna(verbose=False):
    # standard settings
    myVNA.timeout = 10000
    myVNA.write("*CLS")
    myVNA.write("*IDN?")

    if verbose:
        print("VNA identification: {}".format(myVNA.read()))

    # Reset VNA settings
    myVNA.write("SYSTem:PRESet")
    myVNA.write("FORM:DATA ASCii,0")

    # Setup a S21
    myVNA.write("CALCulate1:MEASure1:PARameter 'S21'")
    myVNA.write("SENSe1:SWE:MODE HOLD")

    # Set sweep generation mode to analog
    myVNA.write("SENSe1:SWEep:GENeration ANAL")

    if verbose:
        print("VNA ref. oscillator: {}".format(myVNA.query("SENS:ROSC:SOUR?")))
        print("VNA setup for S21 sweep")


def sweep_vna(f_start, f_stop, intbw, f_delta, p_in, Navg, verbose=False):
    # Check
    if f_stop <= f_start:
        raise ValueError(
            "f_stop needs to be larger than f_start. Current values are f_start = {} GHz, f_stop = {} GHz".format(
                f_start, f_stop))

    # Points in frequency array, adjust f_stops
    Npts = int((f_stop * 1e+9 - f_start * 1e+9) / f_delta)

    # VNA limitation: Npts cannot exceed 100_001
    if Npts > 100_001:
        raise ValueError("Npts cannot exceed 100001. It is {}".format(Npts))

    # Update f_stop
    f_stop = (f_start * 1e+9 + Npts * f_delta) / 1e+9

    # power
    if verbose:
        print(f"Set power: {p_in} dBm")

    ############################################################################
    # Setup VNA

    # Reset VNA settings
    myVNA.write("SYSTem:FPRESet")
    myVNA.write("FORM:DATA ASCii,0")

    # Setup Sij for two-port devices
    myVNA.write("CALC1:PAR:DEF:EXT 'ch1_S11', 'S11'")
    myVNA.write("CALC1:PAR:DEF:EXT 'ch1_S21', 'S21'")

    # Setup windows
    myVNA.write("DISP:WIND1:STAT ON")

    # Setup traces in windows
    myVNA.write("DISPlay:WIND1:TRACe1:FEED 'ch1_S11'")
    myVNA.write("DISPlay:WIND1:TRACe2:FEED 'ch1_S21'")

    # Hold trigger
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

    # Debug
    if verbose:
        print("VNA setup for S21 sweep")

    # Headers for VNA data processing
    headers = ['Frequency [Hz]', 'S11mag [dB]', 'S11phase [deg]', 'S21mag [dB]',
               'S21phase [deg]', 'S12mag [dB]', 'S12phase [deg]', 'S22mag [dB]', 'S22phase [deg]']

    # Sweep time
    myVNA.write("SENSe1:SWEep:TIME:AUTO ON")
    myVNA.write("SENSe1:SWEep:TYPE LIN")
    myVNA.write("SENS:SWE:TIME?")
    sweeptime = round(float(myVNA.read()), 3)
    if verbose:
        print("Sweep time of {} s".format(sweeptime))

    # Start sweep
    myVNA.write("SENS:SWE:MODE SING")
    if verbose:
        print("Sweep started")

    # Polling
    sweepdone = False
    myVNA.write("*ESE 1")
    myVNA.write("*CLS")
    myVNA.write("*OPC")
    while not sweepdone:
        x = myVNA.query("*ESR?")
        sweepdone = int(x.strip("+-\n"))
        time.sleep(0.1)
    if verbose:
        print("Sweep completed")

    # Read data in Touchstone format
    myVNA.write("CALC1:MEAS1:DATA:SNP? 2")
    time.sleep(0.1)
    snp = myVNA.read()
    if verbose:
        print("Read data")

    # Process data
    snplst = snp.split(',')
    lst = list(map(list, zip(*([iter(snplst)] * Npts))))
    datalst = [list(map(float, sublst)) for sublst in lst]
    dframe = pd.DataFrame(datalst, index=headers).T

    # Convert data to numpy arrays
    freq_arr = dframe["Frequency [Hz]"].to_numpy()
    # amp_arr = dframe["S21mag [dB]"].to_numpy()
    # pha_arr = dframe["S21phase [deg]"].to_numpy()

    # Convert to complex
    # s11 = 10**((1/20)*(dframe["S11mag [dB]"].to_numpy())) * np.exp(1j*(dframe["S11phase [deg]"].to_numpy() * np.pi / 180 ))
    s21 = 10 ** ((1 / 20) * (dframe["S21mag [dB]"].to_numpy())) * np.exp(
        1j * (dframe["S21phase [deg]"].to_numpy() * np.pi / 180))
    # s12 = 10**((1/20)*(dframe["S12mag [dB]"].to_numpy())) * np.exp(1j*(dframe["S12phase [deg]"].to_numpy() * np.pi / 180 ))
    # s22 = 10**((1/20)*(dframe["S22mag [dB]"].to_numpy())) * np.exp(1j*(dframe["S22phase [deg]"].to_numpy() * np.pi / 180 ))

    return {'freq_arr': freq_arr, 's11_arr': s21}


def disconnect_vna(verbose=False):
    # VNA RF off
    myVNA.write("OUTPut:STATe OFF")
    if verbose:
        print("VNA RF off")
    rm.close()
    if verbose:
        print("VNA disconnected")


############################################################################

f_start = 3.5
f_stop = 8
intbw = 1e3
f_delta = 1e+6
Navg = 10
p_in = -43
Npts = int((f_stop * 1e+9 - f_start * 1e+9) / f_delta)

############################################################################

verbose = False
print(f"Run name: {myrun}")

# setup VNA
setup_vna(verbose)

# DC bias values
bias_arr = np.arange(-4, 4, 0.01)
s11_arr = np.zeros((len(bias_arr), Npts), dtype=complex)

# loop over dc bias
with tqdm(total=len(bias_arr), ncols=80) as pbar:
    for bias_idx, bias_val in enumerate(bias_arr):
        # Set Presto DC bias
        setvolt(bias_val)

        # VNA frequency sweep
        data = sweep_vna(f_start, f_stop, intbw, f_delta, p_in, Navg, verbose)
        freq_arr = data['freq_arr']
        s11_arr[bias_idx] = data['s11_arr']

        pbar.update(1)

# Save data        
save_data(save_folder, save_file, sample, myrun, freq_arr, s11_arr, bias_arr, verbose)

# disconnect VNA
disconnect_vna(verbose)

# Get current time
t_end = time.strftime("%Y-%m-%d_%H_%M_%S")

# Attributes for this run
myrun_attrs = {"Meas": "S21-sweep_p-sweep",
               "Instr": "VNA",
               "Sample": sample,
               "T": temperature,
               "att": atten,
               "4K-amp_out": 42,
               "RT-amp_out": 41,
               "RT-amp_in": 0,
               "Comment": comment_str,
               "intbw": intbw,
               "delta_f": f_delta,
               "Npoints": Npts,
               "Navg": Navg,
               "f_start": f_start,
               "f_stop": f_stop,
               "t_start": t_start,
               "t_end": t_end,
               "Stepping": 'p_in',
               "Script name": os.path.basename(__file__)}

# Save script and run attributes
save_script(save_folder, save_file, sample, myrun, myrun_attrs)

# Debug
print("Done")
