import pandas as pd
import pyvisa as visa
import numpy as np
import time
import os
from presto import lockin


# Resource manager for VNA
os.add_dll_directory('C:\\Program Files (x86)\\IVI Foundation\\VISA\\WinNT\\agvisa\\agbin\\visa32.dll')
rm = visa.ResourceManager()
myVNA = rm.open_resource('TCPIP0::DESKTOP-HCDF01N::hislip0::INSTR')
myVNA.timeout = 10000
myVNA.write("*CLS")
myVNA.write("*IDN?")

# Verify
print("VNA identification: {}".format(myVNA.read()))


# Presto configuration
ADDRESS = "delta"
bias_port = 1

def setvolt(bias_val):
    with lockin.Lockin( address = ADDRESS,) as lck:
        # Set DC bias on JPA
        lck.hardware.set_dc_bias( bias_val, bias_port, range_i = 3 )
        

# Standard sweep parameters
prm = dict( fst=3.5, fend=8, df=1e-3, pin=-43, IFBW=1e3, nave=1 )    
        

def sweep( prm, verbose = True ):
    
    intbw = prm['IFBW']
    f_delta = prm['df'] * 1e9
    Navg = prm['nave']

    f_start,f_stop = prm['fst'], prm['fend'] 
    p_in = prm['pin']
    
    # Check
    if f_stop <= f_start:
        raise ValueError("f_stop needs to be larger than f_start. Current values are f_start = {} GHz, f_stop = {} GHz".format(f_start, f_stop))
    
    # Points in frequency array, adjust f_stops
    Npts = int((f_stop * 1e+9 - f_start * 1e+9) / f_delta)
    
    # VNA limitation: Npts cannot exceed 100_001
    if Npts > 100_001:
        raise ValueError("Npts cannot exceed 100001. It is {}".format(Npts))
    
    # Update f_stop
    f_stop = (f_start * 1e+9 + Npts * f_delta) / 1e+9
    
    ############################################################################
    # Setup VNA
    
    # Reset VNA settings
    myVNA.write("SYSTem:FPRESet")
    myVNA.write("FORM:DATA ASCii,0")
    
    # Setup Sij for two-port devices
    myVNA.write("CALC1:PAR:DEF:EXT 'ch1_S11', 'S11'")
    myVNA.write("CALC1:PAR:DEF:EXT 'ch1_S21', 'S21'")
    myVNA.write("CALC1:PAR:DEF:EXT 'ch1_S12', 'S12'")
    myVNA.write("CALC1:PAR:DEF:EXT 'ch1_S22', 'S22'")
    
    # Setup windows
    myVNA.write("DISP:WIND1:STAT ON")
    myVNA.write("DISP:WIND2:STAT ON")
    
    # Setup traces in windows
    myVNA.write("DISPlay:WIND1:TRACe1:FEED 'ch1_S11'")
    myVNA.write("DISPlay:WIND1:TRACe2:FEED 'ch1_S21'")
    myVNA.write("DISP:WIND2:TRACe3:FEED 'ch1_S12'")
    myVNA.write("DISP:WIND2:TRACe4:FEED 'ch1_S22'")
    
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
    
    # Reference oscillator
    if verbose:
        print("VNA ref. oscillator: {}".format(myVNA.query("SENS:ROSC:SOUR?")))
    
    # Debug
    if verbose: 
        print("VNA setup for Sij sweep")
    
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
    while (not sweepdone):
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
    dframe = pd.DataFrame(datalst, index = headers).T
    
    # Frequency array
    fre_arr = dframe["Frequency [Hz]"].to_numpy()
    
    # Convert to complex
    # s11 = 10**((1/20)*(dframe["S11mag [dB]"].to_numpy())) * np.exp(1j*(dframe["S11phase [deg]"].to_numpy() * np.pi / 180 ))
    s21 = 10**((1/20)*(dframe["S21mag [dB]"].to_numpy())) * np.exp(1j*(dframe["S21phase [deg]"].to_numpy() * np.pi / 180 ))
    # s12 = 10**((1/20)*(dframe["S12mag [dB]"].to_numpy())) * np.exp(1j*(dframe["S12phase [deg]"].to_numpy() * np.pi / 180 ))
    # s22 = 10**((1/20)*(dframe["S22mag [dB]"].to_numpy())) * np.exp(1j*(dframe["S22phase [deg]"].to_numpy() * np.pi / 180 ))

    return fre_arr, s21     

def rf_off( verbose = True ):
    myVNA.write("OUTPut:STATe OFF")
    if verbose:
        print("VNA RF off")
    rm.close()
    if verbose:
        print("VNA disconnected")