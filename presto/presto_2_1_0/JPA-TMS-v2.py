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
def save_data( folder, file, meas_type, myrun, freq, pump_pwr, usb_arr, lsb_arr ):
    if not os.path.isdir( folder ):
        os.makedirs( folder )

    # Open the save file (.hdf5) in append mode
    with h5py.File( os.path.join(folder, file ), "a") as savefile:
                
        # String as handles
        freq_data_str = "{}/{}/freq sweep".format( meas_type, myrun )
        pump_pwr_data_str = "{}/{}/pump pwr sweep".format( meas_type, myrun )
        usb_data_str = "{}/{}/USB".format( meas_type, myrun )
        lsb_data_str = "{}/{}/LSB".format( meas_type, myrun )

        # Write data to datasets
        savefile.create_dataset( freq_data_str, (np.shape(freq)),
                                 dtype=float, data=(freq))
        savefile.create_dataset( pump_pwr_data_str, (np.shape(pump_pwr)),
                                 dtype=float, data=(pump_pwr))
        savefile.create_dataset( usb_data_str, (np.shape(usb_arr)),
                                 dtype=complex, data=(usb_arr))
        savefile.create_dataset( lsb_data_str, (np.shape(lsb_arr)),
                                 dtype=complex, data=(lsb_arr))
        
        # Write dataset attributes
        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[pump_pwr_data_str].attrs["Unit"] = "fsu"
        savefile[usb_data_str].attrs["Unit"] = "fsu complex"
        savefile[lsb_data_str].attrs["Unit"] = "fsu complex"
        


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
meas_type = 'TMS'
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


# DC BIAS PARAMETERS
# DC bias values in Volts
bias_val = 0.367


# MEASUREMENT PARAMETERS
# Bandwidth in Hz
# df_arr = np.array([ 100e3, 50e3, 25e3, 20e3, 12.5e3, 10e3, 5e3, 1e3  ])
df_arr = np.array([ 100e3 ])
nr_df = len(df_arr)
# Number of pixels
Npix = 100000//2
# Number pf pixels we discard
Nskip = 10
# NCO frequency in Hz
fNCO = 4.1e9-30*df_arr[0]


# RF PUMP PARAMETERS
# Pump frequency in Hz (SHOULD IT BE A SWEEP?)
fp = 8.4e9
# Pump amplitude sweep (fsu units between 0 and 1)
amp_pump_min = 0
amp_pump_max = 1
nr_amp_pump = 2
amp_pump_arr = np.linspace( amp_pump_min, amp_pump_max, nr_amp_pump )


# Instantiate lockin device
with lockin.Lockin( address = ADDRESS,
                    port = PORT,
                    adc_mode = AdcMode.Mixed,
                    adc_fsample = AdcFSample.G2,
                    dac_mode = [DacMode.Mixed02, DacMode.Mixed04, DacMode.Mixed02, DacMode.Mixed02],
                    dac_fsample = [DacFSample.G6, DacFSample.G10, DacFSample.G6, DacFSample.G6],
                    ) as lck:
    
    # Start timer
    t_start = time.strftime("%Y-%m-%d_%H_%M_%S")
    
    # Print Presto version
    print("Presto version: " + presto.__version__)
    
    # Set DC bias
    lck.hardware.set_dc_bias( bias_val, bias_port )
    lck.hardware.sleep( 1.0, False )
    

    
    # Configure mixer just to be able to create output and input groups
    lck.hardware.configure_mixer(
        freq = fNCO,
        in_ports = input_port,
        out_ports = output_port,
        sync = True,
        )
    
    
    # Set the measurement comb (we tune it with respect the highest df in df_arr)
    fs_center = fp/2
    fs_center, df = lck.tune(fs_center, df_arr[0])
    fs_span = (200e6 / 2) // df * 2 * df
    comb_spacing = 60*df
    nr = 30
    freq_arr = np.linspace( fs_center-fs_span/2, fs_center+fs_span/2, nr ) - fNCO
    freq_arr = freq_arr // df * df
    nr_freq = len(freq_arr)
    
    fp = 2 * (freq_arr[len(freq_arr)//2] + fNCO)
    
    lck.set_df( df )
    
    # Configure mixer for the JPA pump
    lck.hardware.configure_mixer(
                        freq = fp,
                        out_ports = flux_port,
                        sync = False,
                        )
    
    # Data
    usb_arr = np.zeros( (nr_freq, nr_amp_pump, Npix, nr_freq), dtype=np.complex128)
    lsb_arr = np.zeros_like( usb_arr )
    
    # Create output group for the JPA pump frequency
    og_pump = lck.add_output_group( ports = flux_port, nr_freq = 1 )
    og_pump.set_frequencies( 0.0 )
    og_pump.set_phases( phases = 0,
                        phases_q = 0 )
    
    # Create output group for the signal frequency
    og = lck.add_output_group( ports = output_port, nr_freq = nr_freq )
    og.set_frequencies( freq_arr )
    # Set the lock-in output amplitudes
    
    # Set the lock-in output phases
    og.set_phases( phases = np.zeros_like(freq_arr),
                   phases_q = np.ones( len(freq_arr ))*(-np.pi/2) )
    
  
    # Create input group
    ig = lck.add_input_group( port = input_port, nr_freq = nr_freq )
    ig.set_frequencies( freq_arr )
    
    # Add pseudorandom noise if needed
    lck.set_dither( dither, output_port )
    
    lck.apply_settings()
    
    # Display a nice progress bar
    with tqdm( total=(nr_amp_pump * nr_df), ncols=80 ) as pbar:
        
        # df loop
        for amp_ind in range( len(freq_arr) ):
            
            amp_arr = np.zeros_like( freq_arr )
            amp_arr[amp_ind] = 0.01
            
            og.set_amplitudes( amp_arr )
            
            
            lck.apply_settings()
            lck.hardware.sleep(1e-3, False)
            
            # Pump power loop
            for pamp_ind, pamp_val in enumerate(amp_pump_arr):
                
                # Set JPA pump amplitudes
                og_pump.set_amplitudes( pamp_val )
                
                lck.apply_settings()
                
                # Get lock-in packets (pixels) from the local buffer
                data = lck.get_pixels( Nskip + Npix )
                freqs, pixels_i, pixels_q = data[input_port]
        
                # Convert a measured IQ pair into a low/high sideband pair
                LSB, HSB = utils.untwist_downconversion( pixels_i, pixels_q )
        
                # Store data in array
                usb_arr[amp_ind, pamp_ind] = HSB[-Npix:]
                lsb_arr[amp_ind, pamp_ind] = LSB[-Npix:]
                
                # Update progress bar
                pbar.update(1)
    
   
    # Mute outputs at the end of the sweep
    og_pump.set_amplitudes(0.0)
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
               "RT-amp_out": 20,
               "RT-amp_in": 41,
               "df": df_arr,
               "nr_freq": nr_freq,
               "pump_freq": fp,
               "fNCO": fNCO,
               "bias": bias_val,
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
save_data(save_folder, save_file, meas_type, myrun, freqs, amp_pump_arr, usb_arr, lsb_arr )