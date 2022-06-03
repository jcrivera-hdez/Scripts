# -*- coding: utf-8 -*-
"""
Created on Apr 2016
@author: Christian Schneider, David Zoepfl
"""

import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt

def linear_fit_bg(freq, data, f_range=.33):
    """
        Fit a linear slope to the data.

        freq:       [array] Frequency
        data:       [array] Complex data
        f_range:    [Percent] Just f_range are used for the fit (get rid of the
                    resonance)
    """
    fit_range = int(np.round(len(data))*f_range)
    w = np.zeros(len(data))
    w[0:fit_range] = 1
    w[-fit_range:-1] = 1
    lin_fit = np.polyfit(freq, 20*np.log10(np.abs(data)), 1, w=w)
    return lin_fit


def fit_lorentzian(freq, data, Ql_init=None, fr_init=None, maxfev=10000):
    """
        Lorentzian Fit.

        Parameters:
        freq:       [Float] Frequency array
        data:       [Complex Array] Data

        Ql_init:    [Float] Initial guess for Ql
        fr_init:    [Float] Initial guess for resonance frequency
    """
    # Parameters

    mag = np.abs(data)

    if Ql_init is None:
        Ql_init = 1e4

    if fr_init is None:
        fr_init = freq[np.argmin(mag)]  # Guess minimum as resonance frequency
    A0 = mag[0]  # Guess for background

    # Fit
    p, pcovs = spopt.curve_fit(lorentzian_abs, freq, mag,
                               p0=(A0, fr_init/Ql_init, Ql_init, fr_init),
                               maxfev=int(maxfev))
    return p, pcovs


def lorentzian_abs(x, A0, A1, Ql, x0):
    """
        Lorentzian function
    """
    return np.abs((A0 - A1/(np.pi * x0/Ql + 1j*2*np.pi * (x-x0))))


def get_delay(data, comb_slopes=True, f_range=.3):
    """
        Function to determine the delay.
        Two different versions are available:
            1. Standard version (comb_slopes=True), where the middle part of
            the phase (resonance) is weighted less and a fit is done over the
            whole phase data
            2. comb_slopes=False, where the phase is fitted at the beginning
            and at the end separately (works better, if electric delay is
            small)

        Parameters:
        comb_slopes:    [True, False] Use fit for whole data or separate start
                        and beginning
        f_range:        [Float] Percentage of data for fit. E.g. 0.3 means use
                        first and last 30% of the data for fitting.
    """
    fit_range = int(np.round(len(np.angle(data.value))*f_range))
    w = np.zeros_like(data.x)
    w[0:fit_range] = 1
    w[-fit_range:-1] = 1
    if comb_slopes:
        lin_fit, res, _, _, _ = np.polyfit(data.x,
                                           data.phase.y, 1, w=w,
                                           full=True)
        offset = lin_fit[1]
        delay = -lin_fit[0]/360  # Phase data is in degrees
        mean = np.mean(data.phase.y)
        SS_tot = np.sum((data.phase.y - mean)**2)
        R_sq = 1 - res/SS_tot
        if R_sq < .9:
            print('Delay fit shows high unexplained variability.' +
                  ' Please check!\nR^2 = {.2f}'.format(R_sq))

    else:
        lin_fit_first = np.polyfit(data.x[0:fit_range],
                                   data.phase.y[0:fit_range], 1)
        lin_fit_last = np.polyfit(data.x[-fit_range:-1],
                                  data.phase.y[-fit_range:-1], 1)
        mean_slope = (lin_fit_first[0]+lin_fit_last[0])/2
        delay = -mean_slope/360  # Phase is in deg
        offset = (lin_fit_first[1] + lin_fit_last[1])/2

    return delay, offset


def tan_phase(x, theta0, Ql, fr, consRa):
        """
            Circuit Model phase behavior which is used for the fit.
        """
        return theta0 + consRa*np.unwrap(2.*np.arctan(2.*Ql*(1. - x/fr)))


def phase_fit(f_data, z_data, theta0, Ql, fr, maxfev=10000):
    """
        Arctan fit of phase.
        Information is used to get offresonant point
    """
    p0 = (theta0, Ql, fr, 1)
    thetas = np.unwrap(np.angle(z_data))
    # Fit
    popt, pcov = spopt.curve_fit(tan_phase, f_data, thetas, p0=p0,
                                 maxfev=maxfev)
    return popt, pcov


def periodic_boundary(x, bound):
    return np.fmod(x, bound) - np.trunc(x/bound)*bound


def fit_circle_weights(freq, data, fr, Ql, weights):

    def res(params, data, weights):
        r, xc, yc = params
        r_calc = (data.real-xc)**2 + (data.imag-yc)**2
        return (r_calc - r**2)**2*weights**2

    # Estimate r (half distance between minimal abs data and start
    f_index = np.argwhere(freq >= fr)[0]
    r_guess = np.abs(data[0] - data[f_index])/2
    # Estimate middle point as the point in the middle between start point and
    # minimum point
    mid_vec = (data[0] + data[f_index])/2
    xc_guess = mid_vec.real
    yc_guess = mid_vec.imag
    # Least squares fit
    f = spopt.leastsq(res, [r_guess, xc_guess, yc_guess], args=(data, weights),
                      full_output=True)
    r, xc, yc = f[0]
    return xc, yc, r


def notch_model(x, Ql, absQc, x0, phi0):
    return (1 - np.exp(1j*phi0)*Ql/absQc/(1 + 2j*Ql*(x-x0)/x0))


def notch_model_mag(x, Ql, absQc, x0):
    """Magnitude notch model.

    Analytical solution for magnitude
    """
    return np.abs(Ql)/np.abs(absQc)/np.sqrt(1 + 4*(x-x0)**2/x0**2*Ql**2)


def fit_model_notch(freq, data, Ql, absQc, fr, phi0, weights):
    """
        Final fit of the model to get information about Ql, Qc and fr.
    """
    def res(params, f, data):
        Ql, Qc, x0, phi0 = params
        diff = notch_model(f, Ql, Qc, x0, phi0) - data
        z1d = np.zeros(data.size*2, dtype=np.float64)
        z1d[0:z1d.size:2] = diff.real*weights
        z1d[1:z1d.size:2] = diff.imag*weights
        return z1d

    f = spopt.leastsq(res, [Ql, absQc, fr, phi0], args=(freq, data),
                      full_output=True)
    return f[0], f[1]


def fit_mag_notch(freq, data, Ql, absQc, fr, phi0, weights):
    """
        Final fit of the model to get information about Ql, Qc and fr.
    """
    def res(params, f, d, weights):
        Ql, Qc, x0 = params
        diff = (notch_model_mag(f, Ql, Qc, x0) - d)**2
        return diff*weights

    # First make the model "easier"
    data1 = np.abs((1 - data)*np.exp(-1j*phi0))
    f = spopt.leastsq(res, [Ql, absQc, fr], args=(freq, data1, weights),
                      full_output=True)
    return f[0], f[1]


def reflection_model(x, Ql, Qc, x0, phi0):
    return -1*(1 - np.exp(1j*phi0)*(2*Ql/Qc)/(1. + 2j*Ql*((x-x0)/x0)))


def reflection_model_mag(x, Ql, Qc, x0, phi0, c):
    return c - (2*Ql/Qc)**2/(1. + 4*Ql**2*((x-x0)/x0)**2)


def fit_model_refl(freq, data, Ql, Qc, fr, phi0, weights):
    """
        Final circle fit of the model to get information about Ql, Qc, fr, phi0
    """
    def res(params, f, data):
        Ql, Qc, x0, phi0 = params
        diff = reflection_model(f, Ql, Qc, x0, phi0) - data
        z1d = np.zeros(data.size*2, dtype=np.float64)
        z1d[0:z1d.size:2] = diff.real*weights
        z1d[1:z1d.size:2] = diff.imag*weights
        return z1d

    f = spopt.leastsq(res, [Ql, Qc, fr, phi0], args=(freq, data),
                      full_output=True)
    return f[0], f[1]


def fit_mag_refl(freq, data, Ql, Qc, fr, phi0, weights):
    """
        Final circle fit of the model to get information about Ql, Qc, fr, phi0
    """
    def res(params, f, data):
        Ql, Qc, x0, phi0, c = params
        diff = (reflection_model_mag(f, Ql, Qc, x0, phi0, c) -
                np.abs(data)**2)
        return diff*weights

    f = spopt.leastsq(res, [Ql, Qc, fr, phi0, 1], args=(freq, data),
                      full_output=True)
    return f[0], f[1]


def subtract_linear_bg(freq, data):
    """
      Subtracts background determined with a linear fit.
    """
    bg_slope, offset = linear_fit_bg(freq, data)
    mag = 20*np.log10(np.abs(data))
    f_rotate = freq[np.argmin(mag)]  # Rotate around resonance
    data_norm = (10**(0.05*(mag - (freq - f_rotate)*bg_slope)) *
                 np.exp(1j*np.angle(data)))
    return data_norm, (f_rotate, bg_slope, offset)


def get_weights(freq, Ql, fr, weight_width):
    """
        Weighting of the function. In range of FWHM, determined by
        (Ql/fr)*weight_width. Outer values are weighted prop. to 1/abs(f-fr)
    """

    width = fr/Ql
    weights = np.ones(len(freq))
    outer_idx = np.abs(freq - fr) > width*weight_width
    weights[outer_idx] = (width*weight_width)/(np.abs(freq[outer_idx] - fr))
    return weights


def fit_theta0(freq, data, Ql, fr, zc):
    """
        Move fitted circle to the origin to detect rotation
    """
    z_data_moved = (data - zc)
    theta0 = ((np.unwrap(np.angle(z_data_moved))[0] +
              np.unwrap(np.angle(z_data_moved))[-1])/2)
    # Arctan fit which gives theta0, being the phase of the offres point.
    # In other words: Find out the rotaation of the circle
    fitparams, pcop = phase_fit(freq, z_data_moved, theta0, np.absolute(Ql),
                                fr)
    theta0, Ql, fr, tmp = fitparams
    theta0 = periodic_boundary(theta0 + np.pi, 2*np.pi)
    return theta0, (fitparams, pcop)
