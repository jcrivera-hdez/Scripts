# -*- coding: utf-8 -*-
"""
Created on Feb2017
@author: Christian Schneider, David Zoepfl
"""

# Python utilities
import numpy as np
import pandas as pd
# Display pandas table nicely
from IPython.display import display
# Circle Fit utilities
import CircleFit.fit_toolbox as ft
import CircleFit.plotting as cp
import CircleFit.error_calculation as errcalc
# Constants
from scipy.constants import hbar


class circuit(object):
    """
       Base class for circlefit.
       The initiator will do all the operations to cancel the environment.

       Afterwards the configuration specific operations are done in the
       subclasses (Notch or Reflection)
    """

    def __init__(self, data, delay=None, a=None, alpha=None,
                 subtract_bg=True, fr_init=None, Ql_init=None, weight_width=1,
                 maxfev=1e6, comb_slopes=False, **kwargs):

        # 1. Correct delay
        data.correct_delay(delay, comb_slopes=comb_slopes)

        # Get data for circle fit
        self.freq = data.x
        self.value = data.value
        self.value_raw = data.value_raw
        self.delay = data.delay
        try:
            self.power = data.par['power']
        except KeyError:
            self.power = None
        try:
            self._delayOffset = data._delayOffset
        except AttributeError:
            self._delayOffset = None
        # Init needed variables
        self.fr = None
        self.Qc_real = None
        self.Ql = None
        self.Qi = None
        self.value_calc = None

        # 2. Subtract background
        #    Save rotation frequency and bg slope for recalculation later
        if subtract_bg:
            self.value, self._bg_pars = ft.subtract_linear_bg(self.freq,
                                                              self.value)
            f_rotate, bg_slope, _ = self._bg_pars

        # 3. Estimate Ql and resonance frequency for weights
        self._lor_pars = ft.fit_lorentzian(self.freq, self.value, Ql_init,
                                           fr_init, maxfev=maxfev)
        _, _, Ql_est, fr_est = self._lor_pars[0]
        
        #fr_est= fr_init
        # 4. Get weights
        self._weights = ft.get_weights(self.freq, Ql_est, fr_est, weight_width)

        # 5. Circle Fit I
        self._circle_pars1 = ft.fit_circle_weights(self.freq, self.value,
                                                   fr_est, Ql_est,
                                                   self._weights)
        xc, yc, r0 = self._circle_pars1
        zc = np.complex(xc, yc)

        # 6. Detect theto_0,to get the offresonant point
        #    The fit could also return Ql and fr. However these parameters are
        #    better fitted with a lorentzian.
        self.theta0, self._theta_pars = ft.fit_theta0(self.freq, self.value,
                                                      Ql_est, fr_est, zc)
        self.offrespoint = zc + r0 * np.exp(1j * self.theta0)
        if not a:
            self.a = np.abs(self.offrespoint)
        else:
            self.a = a
        if not alpha:
            self.alpha = np.angle(self.offrespoint)
        else:
            self.alpha = alpha

    def process_results(self, print_res=True, plt_res=True):
        # Save coefficient of determination
        # Cf. https://en.wikipedia.org/wiki/Coefficient_of_determination
        res_real = (self.value_raw.real - self.value_calc.real) ** 2
        res_imag = (self.value_raw.imag - self.value_calc.imag) ** 2
        res = np.append(res_real, res_imag)
        mean_re = np.mean(self.value_raw.real)
        mean_im = np.mean(self.value_raw.imag)
        S_tot_real = (self.value_raw.real - mean_re) ** 2
        S_tot_imag = (self.value_raw.imag - mean_im) ** 2
        S_tot = np.append(S_tot_real, S_tot_imag)
        self.det_coeff = 1 - res.sum() / S_tot.sum()

        # Format errors
        errors, errors_full_model, chisqr, resid = self.e

        # Fitresults are stored in pandas dataframe
        fres = {'Value': np.array([self.Ql, self.Qc_real, self.Qi, self.fr,
                                   self.phi0, self.det_coeff]),
                'Error': np.append(errors, [np.NaN])}
        fres_full = {'Value': np.array([self.a, self.alpha, self.delay,
                                        self.Ql, self.absQc, self.phi0,
                                        self.fr, self.theta0, self.xc_norm,
                                        self.yc_norm, self.r_norm,
                                        self.offrespoint.real,
                                        self.offrespoint.imag]),
                     'Error': np.append(errors_full_model, [0, 0, 0, 0, 0, 0])}
        self.fitresults = pd.DataFrame(data=fres,
                                       index=['QL', 'Qc', 'Qint', 'fr (GHz)',
                                              'Phi0 (rad)', 'R^2'],
                                       columns=['Value', 'Error'])

        self.fitresults_full_model = pd.DataFrame(data=fres_full,
                                                  index=['a', 'alpha', 'delay',
                                                         'QL', 'absQc', 'phi0',
                                                         'fr', 'Theta0', 'xc',
                                                         'yc', 'r',
                                                         'x_offres',
                                                         'y_offres'],
                                                  columns=['Value',
                                                           'Error'])

        # Print results
        if print_res:
            pd.set_option('precision', 3)
            display(self.fitresults)
            pd.set_option('precision', 8)

        # Plot of data, raw data and (successfully) fitted data if wanted
        if plt_res:
            cp.plot_cfit(self)

    def plot_steps(self):
        """
            Plots steps of circlefit
        """
        cp.plot_steps(self)

    def plot_linear_slope(self):
        cp.plot_linear_slope(self)

    def plot_lorentzian(self):
        cp.plot_lorentzian(self)

    def plot_circle_fit_I(self):
        cp.plot_circle_fit_I(self)

    def plot_phase_fit(self):
        cp.plot_phase_fit(self)

    def plot_final_circle(self):
        cp.plot_final_circle(self)

    def plot_residuals(self):
        """
            Plots residuals.
        """
        cp.plot_residuals(self)

    def plot_weights(self):
        """
            Plots residuals.
        """
        cp.plot_weights(self)

    def get_sigma(self):
        """Gets the normalised least square parameter of the fit"""
        pnt_err = np.abs(self.value - self.value_calc) ** 2
        chi_sqr = 1. / float(len(self.value) - 8) * pnt_err.sum()
        sigma = np.sqrt(chi_sqr)
        return sigma

    def get_no_of_photons_full_info(self, power_dbm=None):
        """
            Prints estimation for the number of photons in the resonator and
            the single photon limit, used formula according to paper below.

            Source: Bruno et al., "Reducing intrinsic loss in superconducting
            resonators by surface treatment and deep etching of silicon
            substrates", APL 106 (2015)
        """
        if power_dbm is None:
            pow_dbm = self.power
        else:
            pow_dbm = power_dbm

        if pow_dbm is not None:
            power_watts = 10 ** ((pow_dbm / 10) - 3)
            no_of_photons = (2 / (
                    hbar * (2 * np.pi * self.fr * 1e9) ** 2) * self.Ql ** 2
                             / self.Qc_real * power_watts)
            single_ph_limit_watts = (
                    (hbar * (2 * np.pi * self.fr * 1e9) ** 2) / 2 *
                    self.Qc_real / self.Ql ** 2)

            single_ph_limit_dbm = 10 * np.log10(single_ph_limit_watts / 1e-3)

            print("Power in dbm: {}".format(pow_dbm))
            print("Number of photons in resonator: {}".format(no_of_photons))
            print("Single photon limit (dbm): {}".format(single_ph_limit_dbm))

        else:
            print("To evaluate no of photons give used power as parameter")

    def get_no_of_photons(self, power_dbm=None):
        """Returns estimation for the number of photons in the resonator"""

        if power_dbm is None:
            power_dbm = self.power

        if power_dbm is not None:
            power_watts = 10 ** ((power_dbm / 10) - 3)
            no_of_photons = (2 / (hbar * (2 * np.pi * self.fr * 1e9) ** 2) * (
                self.Ql) ** 2 / self.Qc_real * power_watts)
            return no_of_photons

        else:
            print("To evaluate no of photons give used power as parameter")


class Notch(circuit):

    def __init__(self, data, **kwargs):
        self.type = 'Notch'
        # Standard corrections (see Circuit)
        super(Notch, self).__init__(data, **kwargs)

        # Notch specific
        # Get variables from base class
        xc, yc, r0 = self._circle_pars1
        zc = np.complex(xc, yc)
        _, _, Ql_est, fr_est = self._lor_pars[0]

        # Normalize circle by moving offres point to (1,0) and scale
        # using parameters a and alpha
        self.circle_norm = self.value * np.exp(-1j * self.alpha) / self.a
        zc_norm = zc * np.exp(-1j * self.alpha) / self.a
        self.r_norm = r0 / self.a
        self.yc_norm = zc_norm.imag
        self.xc_norm = zc_norm.real

        # Impedance mismatch rotation
        if kwargs['phi0'] is not None:
            self.phi0 = kwargs['phi0']
        else:
            self.phi0 = -np.arcsin(self.yc_norm / self.r_norm)

        absQc_est = Ql_est / (2 * self.r_norm)

        # Fit final circle
        if not kwargs['final_mag']:
            # Final fit to complex data
            self._cir_fit_pars = ft.fit_model_notch(self.freq,
                                                    self.circle_norm,
                                                    Ql_est, absQc_est, fr_est,
                                                    self.phi0,
                                                    self._weights)
            self.Ql, self.absQc, self.fr, self.phi1 = \
                np.abs(self._cir_fit_pars[0])
            self.finalfit = 'full'
        else:
            # Final fit to just magnitude (sometimes phase is much more noisy)
            self._cir_fit_pars = ft.fit_mag_notch(self.freq,
                                                  self.circle_norm,
                                                  Ql_est, absQc_est, fr_est,
                                                  self.phi0,
                                                  self._weights)
            self.Ql, self.absQc, self.fr = np.abs(self._cir_fit_pars[0])
            self.phi1 = self.phi0
            self.finalfit = 'mag'

        # Calculate Quality factors
        self.Qc_real = self.Ql / (2 * self.r_norm * np.exp(1j * self.phi0).real)
        self.Qi = 1. / (1. / self.Ql - (1. / self.Qc_real))

        # Recalculate values to illustrate fit and calculate errors
        self.value_calc = self.calc(self.freq)

        # Calculate errors
        self.e = errcalc.get_errors_notch(self)
        self.process_results(kwargs['print_res'], kwargs['plt_res'])

    def calc(self, freqs):
        # Notch model
        value = (self.a*np.exp(1j*self.alpha)*(1 - self.Ql/self.absQc*
                                               np.exp(1j*self.phi0)/
                                               (1. + 2j * self.Ql*(self.freq /
                                                self.fr - 1))))
        # Subtract Background
        try:
            f_rotate, bg_slope, _ = self._bg_pars
            value = (10 ** (0.05 * (20 * np.log10(np.abs(value)) +
                                    (freqs - f_rotate) * bg_slope)) *
                     np.exp(1j * np.angle(value)))
        except:
            pass

        # Electric delay
        value *= np.exp(-2j * np.pi * self.freq * self.delay)

        return value


class Reflection(circuit):

    def __init__(self, data, **kwargs):
        self.type = 'Reflection'

        # Do Standard corrections (see Circuit)
        super(Reflection, self).__init__(data, **kwargs)

        # Get variables from base class
        xc, yc, r0 = self._circle_pars1
        zc = np.complex(xc, yc)
        _, _, Ql_est, fr_est = self._lor_pars[0]
        # Reflection specific operations
        self.circle_norm = self.value * np.exp(
            1j * (np.pi - self.alpha)) / self.a
        zc_norm = zc * np.exp(1j * (np.pi - self.alpha)) / self.a
        self.r_norm = r0 / self.a
        self.yc_norm = zc_norm.imag
        self.xc_norm = zc_norm.real
        absQc_est = Ql_est / self.r_norm  # Factor 2 different to Notch!

        # Impedance mismatch rotation
        # Phi is zero for true reflection. However with using
        # circulators or directional coupler, there could be a impedance
        # mismatch
        if kwargs['phi0'] is not None:
            self.phi0 = kwargs['phi0']
        else:
            self.phi0 = np.arcsin(self.yc_norm / self.r_norm)

        # Fit final circle
        if not kwargs['final_mag']:
            self._cir_fit_pars = ft.fit_model_refl(self.freq,
                                                   self.circle_norm,
                                                   Ql_est, absQc_est, fr_est,
                                                   self.phi0,
                                                   self._weights)
            # Calculate quality factors
            self.Ql, self.absQc, self.fr, self.phi1 = \
                np.abs(self._cir_fit_pars[0])
            self.finalfit = 'full'
        # Use just magnitude (phase seems to be more sensitive to noise)
        else:
            self._cir_fit_pars = ft.fit_mag_refl(self.freq,
                                                 self.circle_norm,
                                                 Ql_est, absQc_est, fr_est,
                                                 self.phi0,
                                                 self._weights)
            # Calculate quality factors
            self.Ql, self.absQc, self.fr, self.phi1, self.c = \
                np.abs(self._cir_fit_pars[0])
            self.finalfit = 'mag'  # Indicator which final fit was used
        self.Qc_real = self.absQc
        self.Qi = 1. / (1. / self.Ql - 1. / self.Qc_real)

        # Recalculate values to illustrate fit and calculate errors
        self.value_calc = self.calc(self.freq)

        # Calculate errors
        self.e = errcalc.get_errors_refl(self)
        self.process_results(kwargs['print_res'], kwargs['plt_res'])

    def calc(self, freqs):
        value = -(self.a * np.exp(-1j * (np.pi - self.alpha)) *
                  (1 - (2 * self.Ql / self.absQc * np.exp(1j * self.phi0)) /
                   (1 + 2j * self.Ql * (freqs - self.fr) / self.fr)))

        # Background
        try:
            f_rotate, bg_slope, _ = self._bg_pars
            value = (10 ** (0.05 * (20 * np.log10(np.abs(value)) +
                                    (freqs - f_rotate) * bg_slope)) *
                     np.exp(1j * np.angle(value)))
        except:
            pass

        # Electric delay
        value *= np.exp(-2j * np.pi * self.freq * self.delay)

        return value
