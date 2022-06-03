# -*- coding: utf-8 -*-
"""Class for complex data (Frequency, Re + i*Im)"""

from .data_table import data_table
import numpy as np
from CircleFit.plotting import plot_rawdata
import CircleFit.plotting as cp
from CircleFit.circuit import Notch, Reflection
from CircleFit.fit_toolbox import get_delay


class data_complex(data_table):
    """Complex DataModule Class

    Stores data as complex voltage phasors Frequency, Re[V] + i*Im[V]

    Parameters
    ----------
    x : list
        List/Numpy array of x values
    values : list, array, ndarray
        List/Numpy array of complex y values [1+1j, 2+2j, ....]
    """
    def __init__(self, x=None, values=None):

        if (x is None) or (values is None):
            super(data_complex, self).__init__([[np.nan], [np.nan], [np.nan]],
                                               ['Freq (GHz)', 'Real (V)',
                                                'Imag (V)'])
        else:
            re = np.real(values)
            im = np.imag(values)
            super(data_complex, self).__init__([x, re, im],
                                               ['Freq (GHz)', 'Real (V)',
                                                'Imag (V)'])
        self.delay = None
        self._delayOffset = None
        self.circuit = None
        self.fitresults = None
        self.fitresults_full_model = None

    def _repr_html_(self):
        """Show pandas dataframe as default representation"""
        print_str = '<h3>data_complex</h3>\n'
        if self.delay is not None:
            print_str += 'delay = {:.2e} s\n'.format(self.delay)
        if self.fitresults is not None:
            print_str += '\n<h4>Fitresults</h4>'
            print_str += self.fitresults.to_html()
            print_str += '\n<h4>Head of data </h4>'

        return print_str + self.df.head().to_html()

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
        self.import_data([x, re, im], ['Freq (GHz)', 'Real (V)', 'Imag (V)'])

    def load_cplx_var(self, x, y):
        """Load values in form of two arrays into data_complex

        Parameters
        -----------
        x : list
            X Array. Typically frequencies
        y : list(complex)
            Y values as a list of complex numbers [1 + 1j, 2 + 0j, ...]
        """
        re = np.real(y)
        im = np.imag(y)
        self.import_data([x, re, im], ['Freq (GHz)', 'Real (V)', 'Imag (V)'])

    @property
    def x(self):
        """Return frequencies"""
        return np.array(self.df['Freq (GHz)'][self.idx_min:self.idx_max])

    @x.setter
    def x(self, value):
        if len(value) == len(self.x):
            # Replace select (idx_min to idx_max)
            self.df[self.name_x][self.idx_min:self.idx_max] = value
        elif len(value) == len(self.df[self.name_x]):
            # Replace whole dataframe
            self.df[self.name_x] = value
        else:
            # Reindex
            self.df = self.df.reindex(range(len(value)))
            self.df[self.name_x] = value
            # Reset min idx and max idx
            self.idx_min = 0
            self.idx_max = None

    @property
    def y(self):
        return self.value

    @y.setter
    def y(self, value):
        """Set real and imag values"""
        self.df['Real (V)'][self.idx_min:self.idx_max] = np.real(value)
        self.df['Imag (V)'][self.idx_min:self.idx_max] = np.imag(value)

    @property
    def value(self):
        """Return complex values"""
        re = self.df['Real (V)'][self.idx_min:self.idx_max]
        im = self.df['Imag (V)'][self.idx_min:self.idx_max]

        if self.delay:
            return np.array(re + 1j*im)*np.exp(2j*np.pi*self.delay*self.x)
        else:
            return np.array(re + 1j*im)

    @value.setter
    def value(self, value):
        """Set real and imag values"""
        self.df['Real (V)'][self.idx_min:self.idx_max] = np.real(value)
        self.df['Imag (V)'][self.idx_min:self.idx_max] = np.imag(value)

    @property
    def value_raw(self):
        re = self.df['Real (V)'][self.idx_min:self.idx_max]
        im = self.df['Imag (V)'][self.idx_min:self.idx_max]
        return np.array(re + 1j * im)

    @property
    def dB(self):
        """This function returns a data_table module with the selected data of
        this data module converted in amplitude (dB). Data Module parameters
        are copied as well.
        """
        tmp = data_table([self.x, 20*np.log10(np.abs(self.value))],
                         ['Frequency (GHz)', 'Mag (dB)'])
        # TODO Make a function for copying datamodules
        tmp.par = self.par
        tmp.time_start = self.time_start
        tmp.time_stop = self.time_stop
        tmp.temp_start = self.temp_start
        tmp.temp_stop = self.temp_stop
        tmp.temp_start_time = self.temp_start_time
        tmp.temp_stop = self.temp_stop_time
        tmp.comments = self.comments
        tmp.date_format = self.date_format
        tmp.save_date_format = self.save_date_format
        return tmp

    @property
    def phase(self):
        return self.get_phase()

    def get_phase(self, unit='deg', unwrap=True):
        """This function returns a data_table module with the selected data of
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
        tmp = np.angle(self.value)
        if unwrap is True:
            tmp = np.unwrap(tmp)
        elif unwrap is False:
            pass
        else:
            print('Wrong unwrap setting inserted')
            raise ValueError

        if u == 'd' or u == 'deg':
            tmp *= 180 / np.pi

        elif u == 'r' or u == 'rad':
            pass
        else:
            print('Wrong unit inserted')
            raise ValueError

        tmp = data_table([self.x, tmp],
                         ['Frequency (GHz)', 'Phase ({})'.format(unit)])
        tmp.par = self.par
        tmp.time_start = self.time_start
        tmp.time_stop = self.time_stop
        tmp.temp_start = self.temp_start
        tmp.temp_stop = self.temp_stop
        tmp.temp_start_time = self.temp_start_time
        tmp.temp_stop = self.temp_stop_time
        tmp.comments = self.comments
        tmp.date_format = self.date_format
        tmp.save_date_format = self.save_date_format
        return tmp

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
        # Check if already set. If yes, unset the previous delay
        if self.delay is not None:
            self.value *= np.exp(-2j*np.pi*self.delay*self.x)
            self.delay = None
            self._delayOffset = None

        # Try to estimate delay with linear fit of phase if no delay is given
        if delay is None:
            delay, offset = get_delay(self, comb_slopes=comb_slopes,
                                      f_range=f_range)
            self._delayOffset = offset

        else:
            _, offset = get_delay(self, comb_slopes=comb_slopes,
                                  f_range=f_range)
            self._delayOffset = offset

        # Save set delay
        self.delay = delay

    def plot(self, engine='bokeh'):
        """Plot data

        Plots raw data if no circle fit is present. If a circlefit was done,
        it plots the data and fits.

        Parameters
        -----------
        engine : str
            Chose the plot engine between 'bokeh' (default) and 'pyplot'
        """
        # TODO Rewrite without try and except
        try:
            self.plot_fitted_data(engine=engine)

        except:
            self.plot_rawdata(engine=engine)

    def plot_rawdata(self, **kwargs):
        """Plots plain data"""
        plot_rawdata(self.x, self.value, **kwargs)

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
            Impedance mismatch expressed in radians. Use phi0=None to
            automatically determine impedance mismatch.
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
