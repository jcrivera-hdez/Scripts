import time
from presto import lockin

ADDRESS = "delta"
CONVERTER_RATES = dict(
    adc_mode=lockin.AdcMode.Mixed,
    adc_fsample=lockin.AdcFSample.G2,
    dac_mode=lockin.DacMode.Mixed04,
    dac_fsample=lockin.DacFSample.G10,
    )

output_port = 5

f_d = 8.4e+9
out_amp = 0.0
df = 10e+3


with lockin.Lockin(address=ADDRESS, **CONVERTER_RATES) as lck:
    
    # mixer
    lck.hardware.configure_mixer(
        freq=f_d,
        out_ports=output_port,
    )
    
    # output group
    outgroup = lck.add_output_group(output_port, 1)
    outgroup.set_frequencies(0.0)
    outgroup.set_phases(phases=0.0, phases_q=0.0)
    outgroup.set_amplitudes(out_amp)
    
    # df
    lck.set_df(df)
    lck.apply_settings()
    time.sleep(0.1)   
    
    # read
    print("Amplitude: {} FS".format(outgroup.get_amplitudes()))
    
    
