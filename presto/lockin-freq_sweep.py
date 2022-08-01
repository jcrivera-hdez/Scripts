import numpy as np
import time
import os
import h5py
import inspect
from tqdm import tqdm

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
def save_data(folder, file, sample, myrun, freq_arr, usb_arr):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(folder, file), "a") as savefile:
        # String as handles
        freq_data_str = "{}/{}/freq sweep".format(sample, myrun)
        usb_data_str = "{}/{}/USB".format(sample, myrun)

        # Write data to datasets
        savefile.create_dataset(freq_data_str, (np.shape(freq_arr)),
                                dtype=float, data=freq_arr)
        savefile.create_dataset(usb_data_str, (np.shape(usb_arr)),
                                dtype=complex, data=usb_arr)

        # Write dataset attributes
        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[usb_data_str].attrs["Unit"] = "fsu complex"


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

# MEASUREMENT PARAMETERS
# NCO frequency
fNCO = 4.2e9
# Bandwidth in Hz
df = 15e3
# Number of pixels
Npix = 5000
# Number pf pixels we discard
Nskip = 1

# DC bias in V
bias_val = 0

# SIGNAL PARAMETERS
# Signal output amplitude from Vivace/Presto
amp_sig = 0.05

# Instantiate lockin device
with lockin.Lockin(address=ADDRESS,
                   port=PORT,
                   adc_mode=AdcMode.Mixed,
                   adc_fsample=AdcFSample.G2,
                   dac_mode=DacMode.Mixed02,
                   dac_fsample=DacFSample.G6,
                   ) as lck:
    # Start timer
    t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

    # Print Presto version
    print("Presto version: " + presto.__version__)

    # Set measurement comb
    fs_center = 4.2e9
    # tune center frequency
    fs_center, df = lck.tune(fs_center, df)
    fs_span = (200e6 / 2) // df * 2 * df
    fs_start = fs_center - fs_span / 2
    fs_stop = fs_center + fs_span / 2
    n_start = int(round(fs_start / df))
    n_stop = int(round(fs_stop / df))
    n_arr = np.arange(n_start, n_stop + 1)
    nr_freq = len(n_arr)
    freq_arr = df * n_arr - fNCO

    # Data
    usb_arr = np.zeros(nr_freq, dtype=np.complex128)

    # Set DC bias
    lck.hardware.set_dc_bias(bias_val, bias_port)
    lck.hardware.sleep(1.0, False)

    # Configure mixer just to be able to create output and input groups
    lck.hardware.configure_mixer(freq=fNCO,
                                 in_ports=input_port,
                                 out_ports=output_port,
                                 sync=True,
                                 )

    # Set df
    lck.set_df(df)

    # Create output group for the signal frequency
    og = lck.add_output_group(ports=output_port, nr_freq=1)
    # Set the lock-in output amplitudes
    og.set_amplitudes(amp_sig)
    # Set the lock-in output phases
    og.set_phases(phases=0,
                  phases_q=-np.pi / 2)

    # Create input group
    ig = lck.add_input_group(port=input_port, nr_freq=1)

    # Add pseudorandom noise if needed
    lck.set_dither(dither, output_port)

    lck.apply_settings()

    # Display nice progress bar
    with tqdm(total=nr_freq, ncols=80) as pbar:
        # Signal frequency sweep
        for sig_ind, sig_val in enumerate(freq_arr):
            og.set_frequencies(sig_val)
            ig.set_frequencies(sig_val)

            lck.apply_settings()

            # Get lock-in packets (pixels) from the local buffer
            data = lck.get_pixels(Nskip + Npix)
            freqs, pixels_i, pixels_q = data[input_port]

            # Convert a measured IQ pair into a low/high sideband pair
            LSB, HSB = utils.untwist_downconversion(pixels_i[:, 0], pixels_q[:, 0])

            # Store data in array
            usb_arr[sig_ind] = np.mean(HSB[-Npix:])

            # Update progress bar
            pbar.update(1)

    # Mute outputs at the end of the sweep
    og.set_amplitudes(0.0)
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
               "f_stop": fs_stop,
               "df": df,
               "nr_freq": nr_freq,
               "amp": amp_sig,
               "Npixels": Npix,
               "Nskip": Nskip,
               "DC bias": bias_val,
               "Dither": dither,
               "t_start": t_start,
               "t_end": t_end,
               "Script name": os.path.basename(__file__),
               }

# Save script and attributes    
save_script(save_folder, save_file, sample, myrun, myrun_attrs)

# Save data
save_data(save_folder, save_file, sample, myrun, freq_arr + fNCO, usb_arr)
