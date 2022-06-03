# -*- coding: utf-8 -*-
"""Standard fit functions used in the data module."""
import numpy as np
kb = 1.3806488e-23 #m2 Kg s-2 K-1
hbar = 1.054571726e-34 #Js


bose_par_labels = ['Amplitude', 'Temperature']
def bose(E, A0, T):
    """Bose Einstein distribution for energies in units of GHz"""       
    return A0*1/(np.exp(E*hbar*2*np.pi*1e9/(kb*T))-1)



boltzmann_par_labels = ['Amplitude', 'Temperature']
def exp_boltzmann(E, A, T):
    """Boltzmann factor, energies in units of GHz""" 
    return A*np.exp(-E*hbar*2*np.pi*1e9/(kb*T))



def gaussian_2d(xy, center_x, center_y, height, width):
    """2d Gaussian Function

    Parameters
    -----------
    xy : [array, array]
        X and Y array
    center_x : float
        Center of gaussian in x direction
    center_y : float
        Center of gaussian in y dircetion
    height : float
        Height of gaussian
    width : float
        Width of gaussian
    """
    x, y = xy
    return height*np.exp(-0.5*((x - center_x)/width)**2 -
                         0.5*((y - center_y)/width)**2)


def gaussian_2d_mult(xy, *pars):
    """Sum of multiple 2d gaussians"""
    tmp = 0
    for i in np.arange(0, len(pars), 4):
        tmp += gaussian_2d(xy, pars[i], pars[i+1],
                           pars[i+2], pars[i+3])
    return np.array(tmp)


mode_fit_pars_labels = ['offset', 'Qc', 'df', 'f0', 'Qi']
def mode_fit(x,*p):
    """Fit for obtaining Quality factors out of resonances.

    .. math::
        y = p[0] + 20\log(|1 - (1/p[1] - 2i p[2]/p[3])/(1/Ql + 2i(x-p[3]-p[2]\
            )/(p[3]+p[2]))

    Function parameters:
        | p[0]: = offset
        | p[1] = Qc
        | p[2] = dF
        | p[3] = resonance frequency
        | p[4] = Qint

    Todo
    -------
        Sources missing

    Parameters
    -----------
    x : float
        x value
    *p : args
        Initial parameter guesses

    Returns
    --------
    float, np.array, list
        Function values
    """
    s0 = p[0]
    Qext = p[1]
    dF = p[2]
    Fres = p[3]
    Qint = p[4]
    F = x

    Qtot = 1./(1./Qext + 1./Qint)
    Fres2 = Fres+dF

    return s0 + 20.*np.log10(np.absolute(1. - (1./Qext - 2*1j*dF/Fres) /
                                         (1./Qtot + 2*1j*(F-Fres2)/Fres2)))


exp_fit_pars_labels = ['offset', 'Amplitude', 'delay', 'tau']
def exp_fit(x, *p):
    """Exponential function

    .. math::
        y = p[0] + p[1]\exp(-(x-p[2])/p[3])

    Function parameters:
        | p[0]: = vertical offset
        | p[1] = Amplitude
        | p[2] = horizontal offset
        | p[3] = tau (default exp is negative)

    Parameters
    -----------
    x : float
        x value
    *p : args
        Initial parameter guesses

    Returns
    --------
    float, np.array, list
        Function values
    """
    return p[0] + p[1]*np.exp(-(x - p[2])/p[3])


def exp_fit_rev(x, *p):
    """Declining exponential function

    .. math::
        y= p[0] + p[1](1-\exp(-(x-p[2])/p[3]))

    Function parameters:
        | p[0] = vertical offset
        | p[1] = Amplitude
        | p[2] = horizontal offset
        | p[3] = tau (default exp is negative)

    Parameters
    -----------
    x : float
        x value
    *p : args
        Initial parameter guesses

    Returns
    --------
    float, np.array, list
        Function values
    """
    return p[0] + p[1]*(1.-np.exp(-(x-p[2])/p[3]))


def poly_fit(x, *p):
    """Polynomial function.

     .. math::
        y = p[0] + p[1]x + p[2]x^2 + ...

    Function parameters:
        | p[0] = offset
        | p[1] = coefficient for x
        | p[2] = coefficient for x^2
        | ...

    Note
    -----
        Function infers degree of polynom from number if guessed initial
        parameters

    Example
    --------
        >>> poly_fit(x,[1.,1.,1.,1.])
        p[0]+p[1]*x+p[2]*x**2+p[3]*x**3

    Parameters
    -----------
    x : float
        x value
    *p : args
        Initial parameter guesses

    Returns
    --------
    float, np.array, list
        Function values. List size depends of given number of parameters
    """
    pol = 0
    for i in range(len(p)):
        pol += p[i]*(x**i)

    return pol


lorentzian_fit_pars_labels = ['center', 'width', 'offset', 'amplitude']
def lorentzian_fit(x,*p):
    """Lorentzian function

    .. math::
        y(x) = p[3]/\pi \cdot p[1]/2/((x-p[0])^2 + (p[1]/2)^2 ) + p[2]

    Function parameters:
        | p[0] = center
        | p[1] = width
        | p[2] = baseline
        | p[3] = amplitude
        | p[4] = scale

    Parameters
    -----------
    x : float
        x value
    *p : args
        Initial parameter guesses

    Returns
    --------
    float, np.array, list
        Function values. List size depends of given number of parameters
    """
    y=p[1]**2/((x-p[0])**2 +p[1]**2 )   
    
    return p[3] * y/y.max() + p[2]


cos_fit_pars_labels = ['amplitude', 'period', 'phase', 'offset']
def cos_fit(x, *p):
    """Cosine function

    .. math::
        y(x) = p[0] \cos(2\pi/p[1] x+p[2])+p[3]

    Function parameters:
        | p[0] = amplitude
        | p[1] = period
        | p[2] = phase
        | p[3] = offset

    Parameters
    -----------
    x : float
        x value
    *p : args
        Initial parameter guesses

    Returns
    --------
    float, np.array, list
        Function values. List size depends of given number of parameters
    """
    return p[0]*np.cos(2*np.pi/p[1]*x+p[2])+p[3]


T2_fit_pars_labels = ['amplitude', 'period', 'phase', 'tau', 'offset']
def T2_fit(x, *p):
    """T2 function

    .. math::
        y(x) = p[0] \sin(2\pi/p[1] x+p[2]) \exp(-x/p[3])+p[4]

    Function parameters:
        | p[0] = amplitude
        | p[1] = period
        | p[2] = phase
        | p[3] = tau
        | p[4] = offset

    Parameters
    -----------
    x : float
        x value
    *p : args
        Initial parameter guesses

    Returns
    --------
    float, np.array, list
        Function values. List size depends of given number of parameters
    """
    return p[0]*np.sin(2*np.pi/p[1]*x+p[2])*np.exp(-x/p[3])+p[4]


T2_beating_fit_pars_labels = ['amplitude', 'period1', 'phase1', 'period2',
                              'phase2', 'tau', 'offset']
def T2_beating_fit(x,*p):
    """T2 beating function

    .. math::
        y(x) = p[0] \sin(2\pi/p[1] x + p[2]) \sin(2\pi/p[3] x + p[4]) \
        \exp(-x/p[5])+p[6]

    Function parameters:
        | p[0] = amplitude1
        | p[1] = period1
        | p[2] = phase1
        | p[3] = period2
        | p[4] = phase2
        | p[5] = tau
        | p[6] = offset

    Parameters
    -----------
    x : float
        x value
    *p : args
        Initial parameter guesses

    Returns
    --------
    float, np.array, list
        Function values. List size depends of given number of parameters
    """
    return p[0]*np.sin(2*np.pi/p[1]*x+p[2])*np.sin(2*np.pi/p[3]*x+p[4]) *\
        np.exp(-x/p[5])+p[6]
