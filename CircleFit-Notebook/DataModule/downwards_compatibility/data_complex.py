# -*- coding: utf-8 -*-
"""
    !!!DEPRECIATED!!!
    ONLY HERE FOR COMPATIBLITY TO OLD DATAMODULES. DO NOT USE OR MODIFY IT

    PLEASE USE DATA_C.py

"""

from DataModule.base import data_module_base
import numpy as np
from CircleFit.plotting import plot_rawdata
import CircleFit.plotting as cp
from CircleFit.circuit import Notch, Reflection
from .data_line import data_line
from CircleFit.fit_toolbox import get_delay


class data_complex(data_module_base):
    """Complex DataModule Class

    Stores data as complex voltage phasors Frequency, Re[V] + i*Im[V]

    Parameters
    ----------
    x : list
        List/Numpy array of x values
    value : list
        List/Numpy array of complex y values [1+1j, 2+2j, ....]
    """

    def __init__(self, x=None, value=None):
        super().__init__()
        if x is None:
            self.x = []
            self.value = []
        else:
            if value is None:
                print('ERROR: no array inserted')
                raise Exception('EMPTYARRAY')
            self.load_cplx_var(x, value)
        print('!!"data_complex" depreciated. Please use "data_c"!!')

    def load_var(self, x, re, im):
        """Load values in form of three arrays into data:complex

        Parameters
        -----------
        x : list
            X Array. Typically frequencies
        re : list
            Real part of y values
        im : list
            Imaginary part of y values
        """
        self.x = np.array(x)
        self.value = np.array(re + 1j*im)
        #self.dB = data_line(self.x, 20*np.log10(np.abs(self.value)))
        #self.phase = data_line(self.x,
        #                       np.unwrap(np.angle(self.value))*180/np.pi)
        self.select()  # check this needs some work

    def load_cplx_var(self, x, y):
        """Load values in form of two arrays into data_complex

        Parameters
        -----------
        x : list
            X Array. Typically frequencies
        y : list(complex)
            Y values as a list of complex numbers [1 + 1j, 2 + 0j, ...]
        """
        self.x = np.array(x)
        self.value = np.array(y)
        #self.dB = data_line(self.x, 20*np.log10(np.abs(self.value)))
        #self.phase = data_line(self.x,
        #                      np.unwrap(np.angle(self.value))*180/np.pi)
        self.select()

    def select(self, xrng=None):
        """Select range of data.

        Plots, fits, etc will then only be applied on this range.
        If nothing is specified all the data will be select

        Parameters
        ----------
        xrng : list
            Start and Stop values of the range in a list [start, stop].
        """
        if xrng is None:
            self.idx_min = 0
            self.idx_max = len(self.x)
        else:
            idx = np.where((self.x >= xrng[0]) & (self.x <= xrng[1]))[0]
            self.idx_min = idx[0]
            self.idx_max = idx[-1]

        # Apply for submodules
        #self.dB.select(xrng)
        #self.phase.select(xrng)

    def get_dB(self):
        """This function returns a data_line module with the selected data of
        this data module converted in amplitude (dB). Data Module parameters
        are copied as well.
        """

        tmp=data_line(self.return_xsel(),20*np.log10(np.abs(self.return_vsel())))
        tmp.par=self.par
        tmp.time_start=self.time_start
        tmp.time_stop=self.time_stop
        tmp.temp_start=self.temp_start
        tmp.temp_stop=self.temp_stop
        tmp.temp_start_time=self.temp_start_time
        tmp.temp_stop=self.temp_stop_time
        tmp.comments=self.comments
        tmp.date_format = self.date_format
        tmp.save_date_format = self.save_date_format

        return tmp

    def get_phase(self,unit='deg', unwrap = True):
        """This function returns a data_line module with the selected data of
        this data module converted in amplitude (dB). Data Module parameters
        are copied as well.

        arguments:
         - unit:
             'd' or 'deg' for degrees (def)
             'r' or 'rad' for radians
        - unwrap:
            True (def): phase is continuous
            False: phase is contained in 1 period
        """

        u = unit.lower()
        tmp = np.angle(self.return_vsel())
        if unwrap is True:
            tmp = np.unwrap(tmp)
        elif unwrap is False:
            pass
        else:
            print('Wrong unwrap setting inserted')
            raise ValueError

        if u == 'd' or u == 'deg':
            tmp *= 180/np.pi

        elif u=='r' or u=='rad':
            pass
        else:
            print('Wrong unit inserted')
            raise ValueError

        tmp = data_line(self.return_xsel(),tmp)
        tmp.par=self.par
        tmp.time_start=self.time_start
        tmp.time_stop=self.time_stop
        tmp.temp_start=self.temp_start
        tmp.temp_stop=self.temp_stop
        tmp.temp_start_time=self.temp_start_time
        tmp.temp_stop=self.temp_stop_time
        tmp.comments=self.comments
        tmp.date_format = self.date_format
        tmp.save_date_format = self.save_date_format

        return tmp

    def return_vsel(self):
        """Returns currently selected data value array (without frequencies)"""
        return self.value[self.idx_min:self.idx_max]
    '''
    def update(self):
        """Update self.value with values from dB and phase."""
        V_re = 10**(self.dB.y/20)*np.cos(self.phase.y*np.pi/180)
        V_im = 10**(self.dB.y/20)*np.sin(self.phase.y*np.pi/180)
        self.value = np.array(V_re + 1j*V_im)
    '''
    def correct_delay(self, delay=None, comb_slopes=False, f_range=.25):
        """Correct the electric delay of the cables.

        Note
        -----
        Give delay in nanoseconds.

        Parameters
        -----------
        delay : float
            Electric delay of in ns.
            If set to None, a fit will estimate the electric delay.
        comb_slopes : bool
            Do one fit for the whole range (use this, only if the resonance
            features in the middle seem to be symmetric)
        f_range : float
            Percentage of the data (first and last) to use for the fit. This
            is done to get rid of resonance features in the middle
        """
        try:
            # Check if already set if yes, unset first delay
            old_delay = self.delay
            self.value *= np.exp(-2j*np.pi*old_delay*self.x)
            self.phase.y = np.unwrap(np.angle(self.value))*180/np.pi
            del self.delay
            del self.value_raw
            del self._delayOffset

        except AttributeError:
            pass
        # Try to estimate delay with linear fit of phase if no delay is given
        if delay is None:
            delay, offset = get_delay(self, comb_slopes=comb_slopes,
                                      f_range=f_range)
            self._delayOffset = offset
        # Correct new delay
        self.value_raw = self.value.copy()
        self.value *= np.exp(2j*np.pi*delay*self.x)
        # Save set delay
        self.delay = delay
        # Update phase subdatamodule
        self.phase.y = np.unwrap(np.angle(self.value))*180/np.pi

    def plot(self, engine='bokeh'):
        """Plot data

        Plots raw data if no circle fit is present. If a circlefit was done,
        it plots the data and fits.

        Parameters
        -----------
        engine : str
            Chose the plot engine between 'bokeh' (default) and 'pyplot'
        """
        try:
            self.plot_fitted_data(engine=engine)

        except:
            self.plot_rawdata(engine=engine)

    def plot_rawdata(self, **kwargs):
        """Plots plain data"""
        plot_rawdata(self.return_xsel(), self.return_vsel(), **kwargs)

    def circle_fit_notch(self, delay=None, a=None, alpha=None, phi0=None,
                         subtract_bg=True, fr_init=None, Ql_init=None,
                         weight_width=1, print_res=True, plot_res=True,
                         comb_slopes=False, final_mag=False, maxfev=10000):
        """Circle fit for notch configuration.

        This function fits the complex data to a notch-configuration circuit
        model. By default it will print and plot the fit results. To disable
        this automation, use the `print_res` and `plot_res` keywords.

        All the fit parameters and data is stored in a .circuit subobject of
        this datamodule.

        Note
        -----
            If something goes wrong with the fit, you can for example use the
            .circuit.plot_steps() function to show the individual steps of the
            fit routine.

        Measurement Setup::

        |            --[Z1]-- --[Z2]--
        |                    |
        |    OUT            [Z3]         IN
        |                    |
        |            -----------------

        Parameters
        -----------
        delay : float
            Set cable delay manually, otherwise determined by the fitting
            routine. Use large span to determine delay. Can be obtained by
            .get_delay()
        a : float
            Set a manually
        alpha : float
            Set alpha manually
        phi0 : float
            Impedance mismatch expressed in radians
        substract_bg : bool
            Subtract linear background
        fr_init : float
            Initial guess for resonance frequency. In GHz.
        Ql_init : float
            Initial guess for Lorentzian fit
        weight_width : float
            Multiply of FWHM frequency for which points are weighted equally.
            Outside the points are weighted less.
        print_res : bool
            Print fit results
        plot_res : bool
            Plot fit results
        final_mag : bool
            Use just magnitude data for final fit of Ql and fr. This is useful,
            since sometimes the phase data is very noisy in comparison to the
            magnitude data. To minimize the error, therefore the final fit can
            be just performed on the magnitude data.
        comb_slopes : bool
            Fit through whole phase instead of start and end separately
            (with a hole in the middle) for obtaining the delay.
        maxfev : int
            Maximum number of iterations for fits
        """
        kwargs = {'delay': delay, 'a': a, 'alpha': alpha, 'phi0': phi0,
                  'subtract_bg': subtract_bg, 'fr_init': fr_init,
                  'Ql_init': Ql_init, 'weight_width': weight_width,
                  'print_res': print_res, 'plt_res': plot_res,
                  'maxfev': maxfev, 'comb_slopes': comb_slopes,
                  'final_mag': final_mag}
        self.circuit = Notch(self, **kwargs)

        # Further save fitresults in data_complex class
        self.fitresults = self.circuit.fitresults

    def circle_fit_reflection(self,  delay=None, a=None, alpha=None, phi0=0,
                              subtract_bg=True, fr_init=None, Ql_init=None,
                              weight_width=1, print_res=True, plt_res=True,
                              comb_slopes=False, final_mag=False, maxfev=1e6):
        """Circle fit for reflection configuration.

        This function fits the complex data to a reflection-configuration
        circuit model. By default it will print and plot the fit results.
        To disable this automation, use the `print_res` and `plot_res`
        keywords.

        All the fit parameters and data is stored in a .circuit subobject of
        this datamodule.

        Note
        -----
            If something goes wrong with the fit, you can for example use the
            .circuit.plot_steps() function to show the individual steps of the
            fit routine.

        Measurement Setup::

        |            --[Z1]---
        |    IN              |
        |    OUT            [Z3]
        |                    |
        |            ---------

        Parameters
        -----------
        delay : float
            Set cable delay manually, otherwise determined by the fitting
            routine. Use large span to determine delay. Can be obtained by
            .get_delay()
        a : float
            Set a manually
        alpha : float
            Set alpha manually
        phi0 : float
            Impedance mismatch expressed in radians
        substract_bg : bool
            Subtract linear background
        fr_init : float
            Initial guess for resonance frequency. In GHz.
        Ql_init : float
            Initial guess for Lorentzian fit
        weight_width : float
            Multiply of FWHM frequency for which points are weighted equally.
            Outside the points are weighted less.
        print_res : bool
            Print fit results
        plot_res : bool
            Plot fit results
        comb_slopes : bool
            Fit through whole phase instead of start and end separately
            (with a hole in the middle) for obtaining the delay.
        final_mag : bool
            Use just magnitude data for final fit of Ql and fr. This is useful,
            since sometimes the phase data is very noisy in comparison to the
            magnitude data. To minimize the error, therefore the final fit can
            be just performed on the magnitude data.
        maxfev : int
            Maximum number of iterations for fits
        """
        kwargs = {'delay': delay, 'a': a, 'alpha': alpha, 'phi0': phi0,
                  'subtract_bg': subtract_bg, 'fr_init': fr_init,
                  'Ql_init': Ql_init, 'weight_width': weight_width,
                  'print_res': print_res, 'plt_res': plt_res, 'maxfev': maxfev,
                  'comb_slopes': comb_slopes, 'final_mag': final_mag}
        self.circuit = Reflection(self, **kwargs)

        # Further save fitresults in data_complex class
        self.fitresults = self.circuit.fitresults

    def plot_fitted_data(self, **kwargs):
        """Plot fitted data"""
        cp.plot_cfit(self.circuit, **kwargs)

    def plot_steps(self):
        """Plot each step of the circlefit routine"""
        cp.plot_steps(self.circuit)

    def get_delay(self):
        """Outputs the delay. Alias for .circuit.delay"""
        return self.circuit.delay

    def get_no_of_photons_full_info(self, power_dbm=None):
        return self.circuit.get_no_of_photons_full_info(power_dbm)

    def get_no_of_photons(self, power_dbm=None):
        return self.circuit.get_no_of_photons(power_dbm)

    def circle_fit_dc(self, **kwargs):
        """Alias for circle_fit_reflection(phi=None)"""
        return self.circle_fit_reflection(phi0=None, **kwargs)

    def do_circle_fit_dc(self, **kwargs):
        """Alias to circle_fit_reflection for downwards compatibility."""
        print('Abbreviated function name. Use circle_fit_dc in future!')
        return self.circle_fit_reflection(phi0=None, **kwargs)

    def do_circle_fit_notch(self, **kwargs):
        """Alias to circle_fit_reflection for downwards compatibility."""
        print('Abbreviated function name. Use circle_fit_notch() instead')
        return self.circle_fit_notch(**kwargs)

    def do_circle_fit_reflection(self, **kwargs):

        print('Abbreviated function name. Use circle_fit_reflection in' +
              'future!')
        return self.circle_fit_reflection(**kwargs)

    # Assign lazy variables
    dB = property(get_dB)
    phase = property(get_phase)
# Alias for compatibility #####################################################
data_cplx = data_complex
