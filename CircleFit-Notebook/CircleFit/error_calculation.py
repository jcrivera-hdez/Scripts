# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:09:15 2016
@author: David Zoepfl, Christian Schneider
"""
import numpy as np


def get_errors_notch(cc):
    """
        Error calculation for notch circuit.
        Requires circuit class (cc).

        Follows the ansatz
        C_pars = J^-1*sigma_exp^2*J^T^-1

        cf. Kai Oliver Arras:
        An Introduction To Error Propagation:
        Derivation, Meaning and Examples
        of Equation
    """
    errors = []
    errors_full_model = []

    f = cc.freq
    d = cc.delay
    environment = cc.a*np.exp(1j*cc.alpha)*np.exp(-2j*np.pi*f*d)

    d_a = cc.value_calc/cc.a
    d_alpha = cc.value_calc * 1j
    d_delay = cc.value_calc * (-2j*np.pi*f)
    d_Ql = -(environment*np.exp(1j*cc.phi0) /
             (cc.absQc * (1 + 2j*cc.Ql*(f-cc.fr)/cc.fr)**2))
    d_absQc = (environment*(np.exp(1j*cc.phi0)*cc.Ql) /
               (cc.absQc**2*(1+2j*cc.Ql*(f-cc.fr)/cc.fr)))
    d_phi0 = (-1*environment*(1j*cc.Ql*cc.fr*np.exp(1j*cc.phi0)) /
              (2j*(f-cc.fr)*cc.absQc*cc.Ql+cc.absQc*cc.fr))
    d_fr = (-1*environment*(2j*cc.Ql**2*f*np.exp(1j*cc.phi0)) /
            (cc.absQc*(cc.fr+2j*cc.Ql*f-2j*cc.Ql*cc.fr)**2))

    # Residuals
    res = cc.value_raw - cc.value_calc

    # Jacobi matrix is calculated, real, "scalar product" between direction of
    # residual and value itself is taken
    Jac = np.array([np.append(d_a.real*cc._weights, d_a.imag*cc._weights),
                    np.append(d_alpha.real*cc._weights,
                              d_alpha.imag*cc._weights),
                    np.append(d_delay.real*cc._weights,
                              d_delay.imag*cc._weights),
                    np.append(d_Ql.real*cc._weights, d_Ql.imag*cc._weights),
                    np.append(d_absQc.real*cc._weights,
                              d_absQc.imag*cc._weights),
                    np.append(d_phi0.real*cc._weights,
                              d_phi0.imag*cc._weights),
                    np.append(d_fr.real*cc._weights, d_fr.imag*cc._weights)])

    sigma_real = 1./float(len(f)-7)*(cc._weights*res.real**2).sum()
    sigma_imag = 1./float(len(f)-7)*(cc._weights*res.imag**2).sum()
    # 1/sigma because matrix gets inverted later
    sigma = np.diagflat(np.repeat([1./sigma_real, 1./sigma_imag], len(f)))

    rhs = np.dot(sigma, np.transpose(Jac))
    rhs = np.dot(Jac, rhs)

    try:
        cov = np.linalg.inv(rhs)
    except:
        cov = None

    if (cov is not None):
        # Errors are calculated according to Gaussian error propagation
        a_err, alpha_err, delay_err, Ql_err, absQc_err, phi0_err, fr_err =\
         np.sqrt(np.abs(np.diagonal(cov)))

        dQi_Ql = (1./((1./cc.Ql-np.cos(cc.phi0)/cc.absQc)*cc.Ql))**2
        dQi_absQc = -np.cos(cc.phi0)/((1./cc.Ql-np.cos(cc.phi0) /
                                       cc.absQc)**2*cc.absQc**2)
        dQi_phi0 = (-cc.Ql**2 * cc.absQc*np.sin(cc.phi0) /
                    (cc.absQc - cc.Ql*np.cos(cc.phi0))**2)
        err_diag = (dQi_Ql**2 * Ql_err**2 + dQi_absQc**2*absQc_err**2 +
                    dQi_phi0**2 * phi0_err**2)
        err_corr = (2*dQi_Ql*dQi_absQc*cov[3, 4]+2*dQi_Ql*dQi_phi0*cov[3, 5] +
                    2*dQi_absQc*dQi_phi0 * cov[4, 5])
        Qi_err = np.sqrt(err_diag + err_corr)
        dQc_real_absQc = 1./np.cos(cc.phi0)
        dQc_real_phi0 = (-1)*np.sin(cc.phi0)*cc.absQc/(np.cos(cc.phi0)**2)
        err_diag = dQc_real_absQc**2*absQc_err**2+dQc_real_phi0**2*phi0_err**2
        err_corr = 2*dQc_real_absQc*dQc_real_phi0*cov[4, 5]
        Qc_real_err = np.sqrt(err_diag + err_corr)
        
        errors = np.array([Ql_err, Qc_real_err, Qi_err, fr_err, phi0_err])
        errors_full_model = np.array([a_err, alpha_err, delay_err, Ql_err,
                                     absQc_err, phi0_err, fr_err])
        return errors, errors_full_model, [sigma_real, sigma_imag], res

    else:
        print('Error calculation failed!')
        return np.zeros(5), np.zeros(7), [0, 0], None


def get_errors_refl(cc):
    errors = []
    errors_full_model = []

    f = cc.freq
    d = cc.delay
    environment = -cc.a*np.exp(1j*(cc.alpha - np.pi))*np.exp(-2j*np.pi*f*d)

    # Derivatives
    d_a = cc.value_calc/cc.a
    d_alpha = cc.value_calc*1j
    d_delay = cc.value_calc*(-2j*np.pi*f)
    d_Ql = -(environment*2*np.exp(1j*cc.phi0) /
             (cc.absQc * (1 + 2j*cc.Ql*(f-cc.fr)/cc.fr)**2))
    d_absQc = (-environment*2*cc.Ql*np.exp(1j*cc.phi0)/
               (cc.absQc**2*(1+2j*cc.Ql*(f-cc.fr)/cc.fr)))                                        
    d_phi0 = (-1*environment*(1j*cc.Ql*cc.fr*np.exp(1j*cc.phi0)) /
              (2j*(f-cc.fr)*cc.absQc*cc.Ql+cc.absQc*cc.fr))
    d_fr = (environment*(2j*cc.Ql**2*f*np.exp(1j*cc.phi0)*2) /
            (cc.absQc * (cc.fr+2j*cc.Ql*f-2j*cc.Ql*cc.fr)**2))
    # Residuals
    res = cc.value_raw - cc.value_calc
    Jac = np.array([np.append(d_a.real*cc._weights, d_a.imag*cc._weights),
                    np.append(d_alpha.real*cc._weights,
                              d_alpha.imag*cc._weights),
                    np.append(d_delay.real*cc._weights,
                              d_delay.imag*cc._weights),
                    np.append(d_Ql.real*cc._weights, d_Ql.imag*cc._weights),
                    np.append(d_absQc.real*cc._weights,
                              d_absQc.imag*cc._weights),
                    np.append(d_phi0.real*cc._weights,
                              d_phi0.imag*cc._weights),
                    np.append(d_fr.real*cc._weights, d_fr.imag*cc._weights)])

    sigma_real = 1./float(len(f)-7)*(cc._weights*res.real**2).sum()
    sigma_imag = 1./float(len(f)-7)*(cc._weights*res.imag**2).sum()
    # 1/sigma because matrix gets inverted later
    sigma = np.diagflat(np.repeat([1./sigma_real, 1./sigma_imag], len(f)))

    rhs = np.dot(sigma, np.transpose(Jac))
    rhs = np.dot(Jac, rhs)
    
    try:
        cov = np.linalg.inv(rhs)
    
    except:
        cov = None

    if (cov is not None):
        # Uncertainties are calcualted according to Gaussian error propagation
        a_err, alpha_err, delay_err, Ql_err, absQc_err, phi0_err, fr_err =\
         np.sqrt(np.abs(np.diagonal(cov)))
        dQi_Ql = (1./((1./cc.Ql-np.cos(cc.phi0)/cc.absQc)*cc.Ql))**2
        dQi_absQc = (-np.cos(cc.phi0)/((1./cc.Ql-np.cos(cc.phi0)/cc.absQc)**2 *
                     cc.absQc**2))
        dQi_phi0 = (cc.Ql**2 * cc.absQc*np.sin(cc.phi0) /
                    (cc.absQc - cc.Ql*np.cos(cc.phi0))**2)
        err_diag = (dQi_Ql**2 * Ql_err**2 + dQi_absQc**2 * absQc_err**2 +
                    dQi_phi0**2 * phi0_err**2)
        err_corr = (2*dQi_Ql*dQi_absQc*cov[3, 4] + 2*dQi_Ql*dQi_phi0 *
                    cov[3, 5] + 2*dQi_absQc*dQi_phi0 * cov[4, 5])
        Qi_err = np.sqrt(err_diag + err_corr)
        dQc_real_absQc = 1./np.cos(cc.phi0)
        dQc_real_phi0 = -1*np.sin(cc.phi0)*cc.absQc/(np.cos(cc.phi0)**2)
        err_diag = (dQc_real_absQc**2*absQc_err**2 + dQc_real_phi0**2 *
                    phi0_err**2)
        err_corr = 2*dQc_real_absQc*dQc_real_phi0*cov[4, 5]
        Qc_real_err = np.sqrt(err_diag + err_corr)
        
        # Return of the errors
        errors = np.array([Ql_err, Qc_real_err, Qi_err, fr_err, phi0_err])
        errors_full_model = np.array([a_err, alpha_err, delay_err, Ql_err,
                                      absQc_err, phi0_err, fr_err])

        return errors, errors_full_model, [sigma_real, sigma_imag], res

    else:
        print('Error calculation failed!')
        return np.zeros(5), np.zeros(7), [0, 0], None


