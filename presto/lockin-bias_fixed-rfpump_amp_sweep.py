import numpy as np
import time
import os
import h5py
import inspect

import presto
from presto import lockin, utils
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode


############################################################################
# Saving methods

# Save script function
def save_script(folder, file, sample, myrun, myrun_attrs):
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
        code_set = savefile.create_dataset(source_str.format(myrun), (len(code_lines),), dtype=dt)
        for i in range(len(code_lines)):
            code_set[i] = code_lines[i]

        # Save the attributes of the run
        for key in myrun_attrs:
            savefile[run_str].attrs[key] = myrun_attrs[key]

    # Debug
    print("Saved script and run attributes.")


# Save data function
def save_data(folder, file, sample, myrun, freq, usb_arr, lsb_arr):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(folder, file), "a") as savefile:
        # String as handles
        freq_data_str = "{}/{}/freq sweep".format(sample, myrun)
        usb_data_str = "{}/{}/USB".format(sample, myrun)
        lsb_data_str = "{}/{}/LSB".format(sample, myrun)

        # Write data to datasets
        savefile.create_dataset(freq_data_str, (np.shape(freq)),
                                dtype=float, data=freq)
        savefile.create_dataset(usb_data_str, (np.shape(usb_arr)),
                                dtype=complex, data=usb_arr)
        savefile.create_dataset(lsb_data_str, (np.shape(lsb_arr)),
                                dtype=complex, data=lsb_arr)

        # Write dataset attributes
        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[usb_data_str].attrs["Unit"] = "fsu complex"
        savefile[lsb_data_str].attrs["Unit"] = "fsu complex"


# Saving folder location, saving file and run name
save_folder = r'D:/JPA/JPA-Data'
save_file = r'test.hdf5'
myrun = time.strftime("%Y-%m-%d_%H_%M_%S")
t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

# Sample name and total attenuation along measurement chain
sample = 'JPA'
atten = 80
temperature = 0.012

# Type of measurement
reverse = False
fwd_and_rev = False

# Lab Network
ADDRESS = '130.237.35.90'  # from Office
# PORT = 42870              # Vivace ALFA
# PORT = 42871              # Vivace BRAVO
PORT = 42873  # Presto DELTA

if PORT == 42870:
    Box = 'Vivace ALFA'
elif PORT == 42871:
    Box = 'Vivace BETA'
elif PORT == 42873:
    Box = 'Presto DELTA'

# Physical Ports
input_port = 1
output_port = 1
flux_port = 5
bias_port = 1

# Pseudorandom noise (only when working with small amplitudes)
dither = False

# DC BIAS PARAMETERS
# DC bias value in V
bias_val = 0.807

# MEASUREMENT PARAMETERS
# NCO frequency
fNCO = 4.2e9
# Bandwidth in Hz
df = 15e3
# Number of pixels
Npix = 1_000_000
# Number pf pixels we discard
Nskip = 10

# RF PUMP PARAMETERS
# Pump frequency in Hz
fp = 8.4e9
amp_pump_arr = np.array([0, 1])
nr_pump = len(amp_pump_arr)

# Instantiate lockin device
with lockin.Lockin(address=ADDRESS,
                   port=PORT,
                   adc_mode=AdcMode.Mixed,
                   adc_fsample=AdcFSample.G2,
                   dac_mode=[DacMode.Mixed02, DacMode.Mixed04, DacMode.Mixed02, DacMode.Mixed02],
                   dac_fsample=[DacFSample.G6, DacFSample.G10, DacFSample.G6, DacFSample.G6],
                   ) as lck:
    # Start timer
    t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

    # Print Presto version
    print("Presto version: " + presto.__version__)

    # Set measurement comb
    fs_start = fp / 2.
    # tune center frequency
    # fs_start, df = lck.tune( fs_start, df )
    # number of frequencies
    nr_freq = 30
    # frequency comb
    freq_arr = fs_start + df * np.linspace(0, nr_freq, nr_freq + 1) - fNCO

    # pump frequency tuned
    fp = 2 * fs_start

    # Data
    usb_arr = np.zeros((nr_pump, Npix, nr_freq + 1), dtype=np.complex128)
    lsb_arr = np.zeros_like(usb_arr)

    # Set DC bias value
    lck.hardware.set_dc_bias(bias_val, bias_port)
    lck.hardware.sleep(1.0, False)

    # Configure mixer for the JPA pump
    lck.hardware.configure_mixer(freq=fp,
                                 out_ports=flux_port,
                                 sync=False,
                                 )

    # Configure mixer just to be able to create output and input groups
    lck.hardware.configure_mixer(freq=fNCO,
                                 in_ports=input_port,
                                 out_ports=output_port,
                                 sync=True,
                                 )

    # Set df
    lck.set_df(df)

    # Create output group for the JPA pump frequency
    og_pump = lck.add_output_group(ports=flux_port, nr_freq=1)
    og_pump.set_frequencies(0.0)
    og_pump.set_phases(phases=0,
                       phases_q=-np.pi / 2)

    # Create input group
    ig = lck.add_input_group(port=input_port, nr_freq=nr_freq + 1)
    ig.set_frequencies(freq_arr)

    # Add pseudorandom noise if needed
    lck.set_dither(dither, output_port)

    lck.apply_settings()
    time.sleep(0.1)

    for amp_idx, amp_val in enumerate(amp_pump_arr):
        # Pump amplitude
        og_pump.set_amplitudes(amp_val)

        lck.apply_settings()
        time.sleep(0.01)

        # Get lock-in packets (pixels) from the local buffer
        data = lck.get_pixels(Nskip + Npix)
        freqs, pixels_i, pixels_q = data[input_port]

        # Convert a measured IQ pair into a low/high sideband pair
        LSB, HSB = utils.untwist_downconversion(pixels_i, pixels_q)

        # Store data in array
        usb_arr[amp_idx] = HSB[-Npix:]
        lsb_arr[amp_idx] = LSB[-Npix:]

    # Mute outputs at the end of the sweep
    og_pump.set_amplitudes(0.0)
    lck.apply_settings()
    lck.hardware.set_dc_bias(0.0, bias_port)

# Stop timer
t_end = time.strftime("%Y-%m-%d_%H_%M_%S")

# Create dictionary with attributes
myrun_attrs = {"Meas": 'gain',
               "Instr": Box,
               "T": temperature,
               "Sample": sample,
               "att": atten,
               "4K-amp_out": 42,
               "RT-amp_out": 41,
               "RT-amp_in": 0,
               "f_start": fs_start,
               "f_stop": freq_arr[-1] + fNCO,
               "df": df,
               "nr_freq": nr_freq,
               "f_pump": fp,
               "amp_pump": amp_pump_arr[-1],
               "DC bias": bias_val,
               "Npixels": Npix,
               "Nskip": Nskip,
               "Dither": dither,
               "t_start": t_start,
               "t_end": t_end,
               "Script name": os.path.basename(__file__),
               }

# Save script and attributes    
save_script(save_folder, save_file, sample, myrun, myrun_attrs)

# Save data
save_data(save_folder, save_file, sample, myrun, freq_arr + fNCO, usb_arr, lsb_arr)
