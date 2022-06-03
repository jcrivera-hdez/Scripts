# -*- coding: utf-8 -*-

import sys
import time
import h5py
import os
import inspect
import numpy as np
import pyvisa as visa
from tqdm import tqdm

import presto
from presto import lockin, utils

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
        code_set = savefile.create_dataset(source_str.format(sample, myrun), (len(code_lines),), dtype=dt)
        for i in range(len(code_lines)):
            code_set[i] = code_lines[i]

        # Save the attributes of the run
        for key in myrun_attrs:
            savefile[run_str].attrs[key] = myrun_attrs[key]

    # Debug
    print("Saved script and run attributes.")

def save_data(folder, file, sample, myrun, freq, data, pump_powers, calibration, index_dct):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Determine index number under run
    idx = index_dct["idx"]

    # String as handles
    freq_data_str = "{}/{}/{}/Frequency".format(sample, myrun, str(idx))
    data_data_str = "{}/{}/{}/Data".format(sample, myrun, str(idx))
    pump_data_str = "{}/{}/{}/Pump_power".format(sample, myrun, str(idx))
    cali_data_str= "{}/{}/{}/Calibration".format(sample, myrun, str(idx))
    # idx_attrs_str = "{}/{}/{}".format(sample, myrun, str(idx))

    # Open the save file (.hdf5) in append mode
    with h5py.File(os.path.join(folder, file), "a") as savefile:

        # Write data to datasets
        savefile.create_dataset(freq_data_str, (np.shape(freq)), dtype=float, data=(freq))
        savefile.create_dataset(data_data_str, (np.shape(data)), dtype=complex, data=(data))
        savefile.create_dataset(pump_data_str, (np.shape(pump_powers)), dtype=float, data=(pump_powers))
        savefile.create_dataset(cali_data_str, (np.shape(calibration)), dtype=float, data=(calibration))

        # Write dataset attributes
        savefile[freq_data_str].attrs["Unit"] = "Hz"
        savefile[data_data_str].attrs["Unit"] = "fsu complex"
        savefile[pump_data_str].attrs["Unit"] = "dBm"
        savefile[cali_data_str].attrs["Unit"] = "fsu, V"
        
        # Write index attributes
        # savefile[idx_attrs_str].attrs[""] = index_dct[""]
        
    # Debug
    print("Saved data")

def main():
    ###########################################################################
    # Save folder location, save file name and run name
    
    save_folder = "/home/august/Documents/qafm_local/"
    # save_file = "MPAF4_4_9.hdf5"
    save_file = "test.h5"
    myrun = time.strftime("%Y-%m-%d_%H_%M_%S")
    t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

    ###########################################################################
    # Sample name, temperature and total attentuation along measurement chain

    sample = "MPAF4_4_9"
    temperature = 0.010
    atten = 60
    comment_str = "HP+LP_filters"

    ###########################################################################
    # Vivace

    BOX = "bravo"
    BOX_ADDRESS = "130.237.35.90"
    BOX_PORT = 42871

    ###########################################################################
    # Frequency values

    f_nco = 3.0 * 1e+9

    f_center = 3.332285 * 1e+9
    f_span = 150e+6
    
    f_start = f_center - f_span / 2
    f_stop = f_center + f_span / 2

    df = 500e+3
    Npxs = 1

    v_amp = 0.01

   ###########################################################################
    # JPA bias values
    
    # ********************************************************************** #
    # Calibration data
    # Measured using multimeter on port 3 of Bravo 2022-01-10
    # ********************************************************************** #
    
    cal_data = np.asarray([[-1.000e+00, -1.242e+00],
               [-5.000e-01, -6.200e-01],
               [-2.000e-01, -2.478e-01],
               [-1.000e-01, -1.236e-01],
               [-5.000e-02, -6.150e-02],
               [-2.000e-02, -2.430e-02],
               [-1.000e-02, -1.180e-02],
               [ 0.000e+00,  5.000e-04],
               [ 1.000e-02,  1.290e-02],
               [ 2.000e-02,  2.530e-02],
               [ 5.000e-02,  6.260e-02],
               [ 1.000e-01,  1.247e-01],
               [ 2.000e-01,  2.489e-01],
               [ 5.000e-01,  6.220e-01],
               [ 1.000e+00,  1.243e+00]])
    
    # NOTE: full scale corresponds to [-1.25, +1.25] V on high-impendance load
    bias_set = -0.2914 #fsu
    
    ###########################################################################
    # JPA pump values

    pump_freq = 2 * f_center
    pump_address = "130.237.35.90"
    ref_osc_str = "external_10MHz_Vivace"

    nr_pump_pow = 12
    pump_pow_start = -11
    pump_pow_stop = -8
    pump_pow_arr = np.round(np.linspace(pump_pow_start, pump_pow_stop, nr_pump_pow), 2)
    pump_pow_arr = np.r_[[-100], pump_pow_arr]

    ###########################################################################
    # Ports
    
    v_input_port = 1
    v_output_port = 1
    v_bias_port = 3

    ###########################################################################
    # Setup signal generator (AgilentE8247C)

    # Resource manager for signal generator
    rm = visa.ResourceManager('@py')
    inst = rm.open_resource(f"TCPIP::{pump_address:s}::5025::SOCKET")
    inst.read_termination = "\n"
    inst.write_termination = "\n"
    time.sleep(0.1)
    print("Instrument: {}".format(inst.query("*IDN?")))
    
    # Turn off output
    inst.write("OUTP {:d}".format(int(bool(0))))
    time.sleep(0.1)
    print("Agilent output: {}".format(inst.query("OUTP?").strip()))
    
    # Set reference to external clock, if valid source present, otherwise use internal
    inst.write("ROSC:SOUR:AUTO {:d}".format(1))
    time.sleep(0.1)
    print("Agilent clock type: {}".format(inst.query("ROSC:SOUR?").strip()))
    

    ###########################################################################
    # Instantiate a Presto Lockin-object in mixed mode

    with lockin.Lockin(
        address=BOX_ADDRESS,
        port=BOX_PORT,
        adc_mode=lockin.AdcMode.Mixed,
        adc_fsample=lockin.AdcFSample.G4,
        dac_mode=lockin.DacMode.Mixed02,
        dac_fsample=lockin.DacFSample.G4
        ) as lck:

        # Start time
        t_start = time.strftime("%Y-%m-%d_%H_%M_%S")

        # Print "myrun"
        print("Run: {}".format(myrun))

        # Print Presto version
        print("Presto version: " + presto.__version__)

        #######################################################################
        # Setup the input and output group for EM resonator, as well as the
        # frequency of the NCO used to drive the EM resonator.
    
        # Configure the NCO for pump frequencies
        lck.hardware.configure_mixer(
            freq=f_nco,
            in_ports=v_input_port,
            out_ports=v_output_port)
        print("EM mixer configured.")

        # Create a lockin output group object, for one frequency
        outgroup = lck.add_output_group(v_output_port, 1)
        outgroup.set_amplitudes(v_amp)
        outgroup.set_phases(phases=0.0, phases_q=-np.pi / 2)

        # Create a lockin input group object, with for one frequencies
        ingroup = lck.add_input_group(v_input_port, 1)
        ingroup.set_phases(0.0)
        lck.apply_settings()
        time.sleep(.1)

        # Verify applied settings
        print("EM frequencies applied: {} Hz".format(outgroup.get_frequencies()))
        print("EM phases applied: {}".format(outgroup.get_phases()))
        print("EM amplitudes applied: {} fsu".format(outgroup.get_amplitudes()))

        #######################################################################
        # Prepare data arrays

        fs = lck.get_fs("dac")
        nr_samples = int(round(fs / df))
        df = fs / nr_samples
        n_start = int(round(f_start / df))
        n_stop = int(round(f_stop / df))
        n_arr = np.arange(n_start, n_stop + 1)
        nr_freq = len(n_arr)
        freq_arr = df * n_arr
        
        # Raise error if offset exceeds the Vivace's bandwidth
        f_offset = freq_arr[-1] - f_nco
        if f_offset > 500e+6:
            raise ValueError("f_offset is too large. It is {} Hz".format(f_offset))

        # Shift
        freqs = freq_arr - f_nco
        
        # Tune
        f_arr, df = lck.tune(freqs, df)
        
        # Center frequency
        f_center_tuned = f_arr[int(len(f_arr)/2)] + f_nco
        pump_freq = 2 * f_center_tuned
        print(pump_freq)
        
        # Set frequency
        inst.write("FREQ {:.3f} HZ".format(float(pump_freq)))
        time.sleep(0.1)
        print("Agilent set freq: {:e} Hz".format(float(inst.query("FREQ?").strip())))
        
        # Response array
        resp_arr = np.zeros((len(pump_pow_arr), nr_freq), np.complex128)

        #######################################################################
        # Start measurement

        # Index dictionary
        idx_dct = dict(idx = 0)
        
        with tqdm(total=(len(pump_pow_arr)*nr_freq), ncols=80) as pbar:

            lck.hardware.set_dc_bias(bias_set, v_bias_port)
            time.sleep(1.0)

            for pp, pump_pow in enumerate(pump_pow_arr):
            
                if pump_pow == -100:
                    # Turn off output
                    inst.write("OUTP {:d}".format(int(bool(0))))
                    time.sleep(0.1)
                    print("Agilent output: {}".format(inst.query("OUTP?").strip()))
                else:
                    # Set power
                    inst.write("POW {:.2f} dBm".format(float(pump_pow)))
                    time.sleep(0.1)
                    print("Agilent set power: {:.2f} dBm".format(float(inst.query("POW?").strip())))
                    
                    # Turn on output
                    inst.write("OUTP {:d}".format(int(bool(1))))
                    time.sleep(0.1)
                    print("Agilent output: {}".format(inst.query("OUTP?").strip()))
                time.sleep(1.0)
    
                for ii, freq in enumerate(f_arr):
                    outgroup.set_frequencies(freq)
                    ingroup.set_frequencies(freq)
                    lck.apply_settings()
                    time.sleep(.1)
                    
                    data = lck.get_pixels(Npxs)
                    freqs, pixels_I, pixels_Q = data[1]
                    _, HSB = utils.untwist_downconversion(pixels_I[:, 0], pixels_Q[:, 0])
                    
                    resp_arr[pp, ii] = np.mean(HSB)
                    
                    pbar.update(1)

        #######################################################################
        # RF off 
        outgroup.set_amplitudes(0.0)  # amplitude [0.0 - 1.0]
        lck.apply_settings()
        time.sleep(.1)

        # Verify
        print("Amplitude output group: {}".format(outgroup.get_amplitudes()))
        
        # Bias off
        lck.hardware.set_dc_bias(0.0, v_bias_port)
        time.sleep(0.1)
        print("Bias: OFF")
        
        # Agilent RF off
        inst.write("POW {:.2f} dBm".format(float(-20)))
        time.sleep(0.1)
        print("Agilent set power: {:.2f} dBm".format(float(inst.query("POW?").strip())))
        time.sleep(0.1)
        inst.write("OUTP {:d}".format(int(bool(0))))
        time.sleep(0.1)
        print("Agilent output: {}".format(inst.query("OUTP?").strip()))
        time.sleep(0.1)
        rm.close()
        print("Agilent disconnected")

    ######################################################################
    # Create a dictionary with run attributes

    # Get current time
    t_end = time.strftime("%Y-%m-%d_%H_%M_%S")

    # Attributes for this run
    myrun_attrs = {"Meas": "JPA_pwr_sweep",
                   "Instr": "Vivace+Agilent",
                   "Box": BOX,
                   "Sample": sample,
                   "T": temperature,
                   "att": atten,
                   "RT-amp_out": 20,
                   "RT-amp_in": 41,
                   "Comment": comment_str,
                   "amp": v_amp,
                   "df": df,
                   "Npoints": nr_freq,
                   "Npixels": Npxs,
                   "jpa_pump_freq": pump_freq,
                   "nr_pump_power": nr_pump_pow,
                   "bias": bias_set,
                   "ref_osc": ref_osc_str,
                   "f_start": f_start,
                   "f_stop": f_stop,
                   "t_start": t_start,
                   "t_end": t_end,
                   "Script name": os.path.basename(__file__)}

    ###########################################################################
    # Save data
    save_data(save_folder, save_file, sample, myrun, freq_arr, resp_arr, pump_pow_arr, cal_data, idx_dct)
    
    # Save script and run attributes
    save_script(save_folder, save_file, sample, myrun, myrun_attrs)

    # Debug
    print("Finished on:", time.strftime("%Y-%m-%d_%H%M%S"))
    print('Done')

if __name__ == "__main__":
    
    # Launch presto
    for i in range(10):
        try:
            main()
        except RuntimeError as err_msg:
            print("Error: {}".format(err_msg))
            print("Restarting...")
            time.sleep(1)
            continue
        else:
            break
