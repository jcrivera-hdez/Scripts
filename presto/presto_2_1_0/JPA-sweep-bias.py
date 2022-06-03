import numpy as np
import time
import os
import h5py
import inspect
from tqdm import tqdm

import presto
from presto import lockin, utils
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode



#%% Functions

# Save script function
def save_script( folder, file, meas_type, myrun, myrun_attrs ):
    
    # Create folders if they do not exist
    if not os.path.isdir( folder ):
        os.makedirs( folder )

    # String handles
    run_str = "{}/{}".format( meas_type, myrun )
    source_str = "{}/{}/Source code".format( meas_type, myrun )

    # Read lines of the script
    filename = inspect.getframeinfo( inspect.currentframe() ).filename
    with open( filename, "r" ) as codefile:
        code_lines = codefile.readlines()

    # Write lines of script and run attributes
    with h5py.File(os.path.join(folder, file), "a") as savefile:

        dt = h5py.special_dtype( vlen=str )
        code_set = savefile.create_dataset( source_str.format(myrun), (len(code_lines),), dtype=dt )
        for i in range(len(code_lines)):
            code_set[i] = code_lines[i]

        # Save the attributes of the run
        for key in myrun_attrs:
            savefile[run_str].attrs[key] = myrun_attrs[key]

    # Debug
    print("Saved script and run attributes.")
    
    
# Save data function
def save_data( folder, file, meas_type, myrun, freq, dc_bias, usb_arr ):
    if not os.path.isdir( folder ):
        os.makedirs( folder )

    # Open the save file (.hdf5) in append mode
    with h5py.File( os.path.join(folder, file ), "a") as savefile:
                
        # String as handles
        freq_data_str = "{}/{}/freq sweep".format( meas_type, myrun )
        bias_data_str = "{}/{}/bias sweep".format( meas_type, myrun )
        usb_data_str = "{}/{}/USB".format( meas_type, myrun )

        # Write data to datasets
        savefile.create_dataset( freq_data_str, (np.shape(freq)),
                                 dtype=float, data=(freq))
        savefile.create_dataset( bias_data_str, (np.shape(dc_bias)),
                                 dtype=float, data=(dc_bias))
        savefile.create_dataset( usb_data_str, (np.shape(usb_arr)),
                                 dtype=complex, data=(usb_arr))
        
        # Write dataset attributes
        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[bias_data_str].attrs["Unit"] = "fsu"
        savefile[usb_data_str].attrs["Unit"] = "fsu complex"



#%%

# Saving folder location, saving file and run name
save_folder = r'D:/JPA/JPA-Data'
save_file = r'QuantumGarage-JPA.hdf5'
myrun = time.strftime("%Y-%m-%d_%H_%M_%S")
t_start = time.strftime("%Y-%m-%d_%H_%M_%S")


# Sample name and total attenuation along measurement chain
sample = 'JPA'
atten = 60

# Type of measurement
meas_type = 'DC bias sweep'
reverse = False
fwd_and_rev = False

# Lab Network
ADDRESS = '130.237.35.90'   # from Office 
# PORT = 42870              # Vivace ALFA
# PORT = 42871              # Vivace BRAVO
PORT = 42873                # Presto DELTA

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
dither = True
ext_ref_clk = False


# DC BIAS PARAMETERS
# DC bias values in Volts
bias_min = 0
bias_max = 4
nr_bias = 301
bias_arr = np.linspace( bias_min, bias_max, nr_bias )


# MEASUREMENT PARAMETERS
# Bandwidth in Hz
df = 10e6
# Number of pixels
Npix = 10
# Number pf pixels we discard
Nskip = 1


# SIGNAL PARAMETERS
# Signal frequency sweep values in Hz
fs_center = 4.2e9
fs_span = 400e6
# Signal output amplitude from Vivace/Presto
amp_sig = 0.01


# Instantiate lockin device
with lockin.Lockin( address = ADDRESS,
                    port = PORT,
                    ext_ref_clk = ext_ref_clk,
                    adc_mode = AdcMode.Mixed,
                    adc_fsample = AdcFSample.G2,
                    dac_mode = DacMode.Mixed02,
                    dac_fsample = DacFSample.G6,
                    ) as lck:
    
    # Start timer
    t_start = time.strftime("%Y-%m-%d_%H_%M_%S")
    
    # Print Presto version
    print("Presto version: " + presto.__version__)
    
    # Set bandwidth
    _, df = lck.tune(0.0, df)
    
    # Set measurement comb
    fs_start = fs_center - fs_span / 2
    fs_stop = fs_center + fs_span / 2
    n_start = int(round(fs_start / df))
    n_stop = int(round(fs_stop / df))
    n_arr = np.arange(n_start, n_stop + 1)
    nr_freq = len(n_arr)
    freq_arr = df * n_arr
    
    # Data
    usb_arr = np.zeros( (nr_bias, nr_freq), dtype=np.complex128 )
    
    # Ask Riccardo why to set a initial value of dc bias. 
    # Is is because Presto needs some time to output the first dc bias?
    lck.hardware.set_dc_bias( bias_arr[0], bias_port )
    lck.hardware.sleep( 1.0, False )
    
    # Configure mixer just to be able to create output and input groups
    lck.hardware.configure_mixer(
        freq = freq_arr[0],
        in_ports = input_port,
        out_ports = output_port,
        )
    
    # Set df
    lck.set_df( df )
    
    # Create output group for the signal frequency
    og = lck.add_output_group( ports = output_port, nr_freq = 1 )
    og.set_frequencies( 0.0 )
    # Set the lock-in output amplitudes
    og.set_amplitudes( amp_sig )
    # Set the lock-in output phases
    og.set_phases( phases = 0,
                   phases_q = 0 )
    
    # Add pseudorandom noise if needed
    lck.set_dither( dither, output_port )
    
    # Create input group
    ig = lck.add_input_group( port = input_port, nr_freq = 1 )
    ig.set_frequencies( 0.0 )
    
    lck.apply_settings()
    
    # Display a nice progress bar
    with tqdm(total=(nr_bias * nr_freq), ncols=80) as pbar:
    
        # Bias loop
        for bias_ind, bias_val in enumerate(bias_arr):
            
            # Set DC bias on JPA
            lck.hardware.set_dc_bias( bias_val, bias_port )
            lck.hardware.sleep(1.0, False)
            
            # Signal frequency sweep
            for sig_ind, sig_val in enumerate(freq_arr):
                
                # Configurate mixer
                lck.hardware.configure_mixer(
                    freq = sig_val,
                    in_ports = input_port,
                    out_ports = output_port,
                    )
                
                lck.apply_settings()
                lck.hardware.sleep(1e-3, False)
                
                # Get lock-in packets (pixels) from the local buffer
                data = lck.get_pixels( Nskip + Npix )
                freqs, pixels_i, pixels_q = data[input_port]
        
                # Convert a measured IQ pair into a low/high sideband pair
                LSB, HSB = utils.untwist_downconversion( pixels_i[:,0], pixels_q[:,0] )
        
                # Store data in array
                usb_arr[bias_ind, sig_ind] = np.mean( HSB[-Npix:] )
                
                # Update progress bar
                pbar.update(1)
    
   
    # Mute outputs at the end of the sweep
    og.set_amplitudes(0.0)
    lck.apply_settings()
    lck.hardware.set_dc_bias(0.0, bias_port)
            
    
# Stop timer
t_end = time.strftime("%Y-%m-%d_%H_%M_%S")
        
        
# Create dictionary with attributes
# Attributes for this run
myrun_attrs = {"Meas": meas_type,
               "Instr": Box,
               "Sample": sample,
               "att": atten,
               "f_start": fs_start,
               "f_stop": fs_stop,
               "df": df,
               "nr_freq": nr_freq,
               "amp": amp_sig,
               "Npixels": Npix,
               "Nskip": Nskip,
               "Dither": dither,
               "t_start": t_start,
               "t_end": t_end,
               "Script name": os.path.basename(__file__),
               }

# Save script and attributes    
save_script( save_folder, save_file, meas_type, myrun, myrun_attrs )

# Save data
save_data(save_folder, save_file, meas_type, myrun, freq_arr, bias_arr, usb_arr )