# -*- coding: utf-8 -*-

"""
Created on Mon Feb 15 11:15:53 2021
@author: JC

Last version: 2021-03-19

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.interpolate import interp1d
from scipy.constants import Planck, Boltzmann
import cvxpy as cp
import sympy as sy
from scipy import optimize
from scipy.optimize import minimize
from scipy.linalg import block_diag
import itertools

# Plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.titlesize'] = 23
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15



#%%

#---------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
#---------------------------------------------------------------------------------------------------------------------------------

# From Shan

def entanglement_criteriaN( vectors, cov_matrix, cov_matrix_error ):
    dimension = len( vectors )
    if dimension % 2 != 0:
        print( 'vector length is not even!' )
    h_vec = vectors[ :dimension // 2 ]
    g_vec = vectors[ dimension // 2: ]
    
    h_matrix = np.outer( h_vec, h_vec )
    h_matrix2 = np.outer( h_vec**2, h_vec**2 )
    
    g_matrix = np.outer( g_vec, g_vec )
    g_matrix2 = np.outer( g_vec**2, g_vec**2 )
    
    v_i_quant = cov_matrix[0::2, 0::2]
    v_q_quant = cov_matrix[1::2, 1::2]

    witness = np.trace( v_i_quant.dot(h_matrix) ) + np.trace( v_q_quant.dot(g_matrix) )
    
    vi_error2 = ( cov_matrix_error[0::2, 0::2] )**2
    vq_error2 = ( cov_matrix_error[1::2, 1::2] )**2
  
    witness_error = np.sqrt( np.trace( vi_error2.dot(h_matrix2) ) + np.trace( vq_error2.dot(g_matrix2) ) ) 

    compare = np.sum( np.abs(h_vec) * np.abs(g_vec) )
    
    # Slightly different signs, which allow me to minimize this function to find best entanglement witness
    # I suspect I should have a factor of 2, since my vacuum is normalized to the identity I and not 0.5I as in the paper
    quantity = ( witness - 2 * compare ) / witness_error
    
    return quantity



def entanglement_criteriaN_( vectors, cov_matrix, cov_matrix_error ):
    dimension = len( vectors )
    if dimension % 2 != 0:
        print( 'vector length is not even!' )
    h_vec = vectors[ :dimension // 2 ]
    g_vec = vectors[ dimension // 2:]
    
    h_matrix = np.outer( h_vec, h_vec )
    h_matrix2 = np.outer( h_vec**2, h_vec**2 )
    
    g_matrix = np.outer( g_vec, g_vec )
    g_matrix2 = np.outer( g_vec**2, g_vec**2 )
    
    v_i_quant = cov_matrix[0::2, 0::2]
    v_q_quant = cov_matrix[1::2, 1::2]

    witness = np.trace( v_i_quant.dot(h_matrix) ) + np.trace( v_q_quant.dot(g_matrix) )
    
    vi_error2 = ( cov_matrix_error[0::2, 0::2] )**2
    vq_error2 = ( cov_matrix_error[1::2, 1::2] )**2
    
    witness_error = np.sqrt( np.trace( vi_error2.dot(h_matrix2) ) + np.trace( vq_error2.dot(g_matrix2) ) )

    compare = np.sum( np.abs(h_vec) * np.abs(g_vec) )
    
    denominator = ( witness - 2 * compare )
    numerator = witness_error
    
    return denominator, numerator



def rotation_cov_matrix( cov_matrix, phi ):
    n_modes = len(cov_matrix[0]) // 2
    
    if len(phi) == n_modes:
        rot_matrix_list = np.zeros((n_modes, 2, 2))
    
        for ii in range(n_modes):
            rot_matrix = np.zeros((2, 2))
            rot_matrix[0, 0], rot_matrix[1, 1] = np.ones(2) * np.cos( phi[ii] )
            rot_matrix[0, 1] = -np.sin( phi[ii] ) 
            rot_matrix[1, 0] = np.sin( phi[ii] )
            rot_matrix_list[ii] = rot_matrix
    
        comp_rot_matrix = block_diag(*rot_matrix_list)
        rot_cov_matrix = comp_rot_matrix.dot( cov_matrix ).dot( np.transpose(comp_rot_matrix) )
    
    else:
        print('WARNING: number of angles do not correspond to the number of modes')
    
    return rot_cov_matrix



def anti_diagonal( arr ):
    return np.fliplr(arr).diagonal()



def decouple_IQ( cov_matrix_quantum, phi_array ):
    rot_cov = rotation_cov_matrix( cov_matrix_quantum, phi_array )
    anti_diagonal_rot_cov = np.zeros( 2 * len(phi_array) - 1)
    for ii in range(len(phi_array)):
        partial_cov_1 = rot_cov[0:2*(ii+1), 0:2*(ii+1)]
        partial_cov_2 = rot_cov[-2*(ii+1):, -2*(ii+1):]
        anti_diagonal_rot_cov[ii] = np.linalg.norm( anti_diagonal(partial_cov_1) )
        anti_diagonal_rot_cov[-ii-1] = np.linalg.norm( anti_diagonal(partial_cov_2) )
    
    quantity_minimize = np.sum( np.abs(anti_diagonal_rot_cov) )
    return 10 * np.log10( quantity_minimize )



def entanglement_criteria_arbitraryPartitioning( vectors, list_of_partitions, cov_matrix, cov_matrix_error ):
    dimension = len(vectors)
    if dimension%2 != 0:
        print('vector length is not even!!')
    h_vec = vectors[ :dimension // 2 ]
    g_vec = vectors[ dimension // 2: ]
    
    h_matrix = np.outer( h_vec, h_vec )
    h_matrix2 = np.outer( h_vec**2, h_vec**2 )
    
    g_matrix = np.outer( g_vec, g_vec )
    g_matrix2 = np.outer( g_vec**2, g_vec**2 )
    
    v_i_quant = cov_matrix[0::2, 0::2]
    v_q_quant = cov_matrix[1::2, 1::2]

    witness = np.trace( v_i_quant.dot(h_matrix) ) + np.trace( v_q_quant.dot(g_matrix) ) 
    
    vi_error2 = ( cov_matrix_error[0::2, 0::2] )**2
    vq_error2 = ( cov_matrix_error[1::2, 1::2] )**2
    
    witness_error = np.sqrt( np.trace( vi_error2.dot(h_matrix2) ) + np.trace( vq_error2.dot(g_matrix2) ) )
    
    n_partitions = len(list_of_partitions)
    comparitor = np.zeros(n_partitions)
    for ni in range(n_partitions):
        partition = list_of_partitions[ni]
        hJ = h_vec[partition]
        gJ = g_vec[partition]
        if len(partition)>1:
            comparitor[ni] = np.abs( np.inner(hJ,gJ) )
        else:
            comparitor[ni] = np.abs( hJ * gJ )
        
    compare = np.sum( comparitor )

    quantity = (witness - 2 * compare) / witness_error
    
    return quantity



def entanglement_criteria_evaluation( vectors, list_of_partitions, cov_matrix, cov_matrix_error ):
    dimension = len(vectors)
    if dimension%2 != 0:
        print('vector length is not even!!')
    h_vec = vectors[:dimension//2]
    g_vec = vectors[dimension//2:]
    
    h_matrix = np.outer( h_vec, h_vec )
    h_matrix2 = np.outer( h_vec**2, h_vec**2 )
    
    g_matrix = np.outer( g_vec, g_vec )
    g_matrix2 = np.outer( g_vec**2, g_vec**2 )
    
    v_i_quant = cov_matrix[0::2, 0::2]
    v_q_quant = cov_matrix[1::2, 1::2]

    witness = np.trace( v_i_quant.dot(h_matrix) ) + np.trace( v_q_quant.dot(g_matrix) ) 
    
    vi_error2 = ( cov_matrix_error[0::2, 0::2] )**2
    vq_error2 = ( cov_matrix_error[1::2, 1::2] )**2
    
    witness_error = np.sqrt( np.trace( vi_error2.dot(h_matrix2) ) + np.trace( vq_error2.dot(g_matrix2) ) )
    
    n_partitions = len(list_of_partitions)
    comparitor = np.zeros(n_partitions)
    for ni in range(n_partitions):
        partition = list_of_partitions[ni]
        hJ = h_vec[partition]
        gJ = g_vec[partition]
        if len(partition) > 1:
            comparitor[ni] = np.abs( np.inner(hJ,gJ) )
        else:
            comparitor[ni] = np.abs( hJ * gJ )
        
    compare = np.sum( comparitor )
    
    quantity = (witness - 2 * compare)
    
    return quantity, witness_error



def search_eval( partition_a, cov_matrix, cov_matrix_error, n_iter ):
    found_val = np.zeros(n_iter)
    found_err = np.zeros_like(found_val)
    
    for i in range(n_iter):
        bounds = [(-20, 20) for i in range( len(cov_matrix) ) ]
        min_res = optimize.differential_evolution( lambda vector_t: entanglement_criteria_arbitraryPartitioning( vector_t, partition_a, cov_matrix, cov_matrix_error ),
                                                  bounds,
                                                  maxiter = 100,
                                                  atol = 1,
                                                  )
        vector_op = min_res.x
        quant, quant_err = entanglement_criteria_evaluation( vector_op, partition_a, cov_matrix, cov_matrix_error )
        found_val[i] = quant
        found_err[i] = quant_err
    
    min_val = np.min(found_val)
    
    return min_val, found_err[found_val == min_val]



#%%

#---------------------------------------------------------------------------------------------------------------------------------
# DATA LOADING
#---------------------------------------------------------------------------------------------------------------------------------             

# Folder Path
# folder_high = os.path.join(r'D:\VivaceData\2020-10-22\23_39_30\fourpumps_ON_OFF_cycle_2khz_df_modejumping.hdf5')
folder_high = os.path.join(r'D:\VivaceData\2020-10-22\11_10_03\fourpumps_ON_OFF_cycle_2khz_df.hdf5')

hf = h5py.File(folder_high, 'r')

# 2020-10-22\23_39_30\fourpumps_ON_OFF_cycle_2khz_df_modejumping indices
# f1_len = 12
# p1_len = 6

# 2020-10-22\11_10_03\fourpumps_ON_OFF_cycle_2khz_df indices
f1_len = 8
p1_len = 6

# fourpumps_ON_OFF_cycle_2khz_df indices
# f1_ind = 6
# p1_ind = 1

# Covariance matrix chunks
part_len = 1

# Number of modes
# nmodes = np.arange( 0, 32, 2 )[6:-4]
nmodes = np.arange( 0, 32, 2 )[5:-5]

# Numerically controlled oscillator frequency
fNCO = 3.8e+9

# Gain and added noise from the calibration
npzfile = np.load(r'D:\VivaceData\Vivace-Calibration\Calib-Parameters-2021-01-Constrained-gi.npz')

gs = npzfile['gain_signal']
gs_std = npzfile['gain_signal_std']
gi = npzfile['gain_idler']
gi_std = npzfile['gain_idler_std']
ns = npzfile['ns']
ns_std = npzfile['ns_std']
sGsn = npzfile['sGsn']
freqs_calibration = npzfile['freq_comb']

sGsGi = np.zeros_like( gs )
sGin = np.zeros_like( gs )


# Experimental cov matrix
cov_matrix = np.zeros(( part_len, f1_len, p1_len, 2*len(nmodes), 2*len(nmodes) ))
cov_matrix_std = np.zeros_like(( cov_matrix ))

# Experimental cov matrix normalised
cov_matrix_norm = np.zeros_like(( cov_matrix ))
cov_matrix_norm_std = np.zeros_like(( cov_matrix ))

# Reconstruted cov matrix
cov_matrix_reconstructed = np.zeros_like(( cov_matrix ))
cov_matrix_reconstructed_std = np.zeros_like(( cov_matrix ))

# CVXPY quantum recosntructed cov matrix (closest physical cov matrix)
cov_matrix_quantum = np.zeros_like(( cov_matrix ))

# Quantum reconstructed cov matrix rotated (entanglement test improve when rotating the data)
cov_matrix_quantum_rot = np.zeros_like(( cov_matrix ))
cov_matrix_quantum_rot_err = np.zeros_like(( cov_matrix ))


# Reconstruction value (normalised to std)
optimal_value = np.zeros(( part_len, f1_len, p1_len ))

# Shchukin-Van Look entanglement criteria values
ent_test = np.zeros_like( optimal_value )
ent_error = np.zeros_like( optimal_value )

# Symplectic matrix in ordinary form
J_matrix = np.array([[0, 1], [-1, 0]])
symp = np.block( [[ (J_matrix if v==j else np.zeros_like(J_matrix, dtype = complex) )
                   for v in range( len(nmodes) ) ] for j  in range( len(nmodes) )] )

# number of k-1 partitions
n_kpartitions = int( sy.functions.combinatorial.numbers.stirling( len(nmodes), len(nmodes)-1, d=None, kind=2, signed=False ) )

# k-1 partitions entanglement test values
kpart_val = np.zeros(( part_len, f1_len, p1_len, n_kpartitions ))
kpart_err = np.zeros_like( kpart_val )

# number of bipartitions
n_bipartitions = int( sy.functions.combinatorial.numbers.stirling( len(nmodes), 2, d=None, kind=2, signed=False ) )

# bipartition entanglement test values
bipart_val = np.zeros(( part_len, f1_len, p1_len, n_bipartitions ))
bipart_err = np.zeros_like( bipart_val )

# PPT entanglement test values
min_eig_ppt = np.zeros_like( bipart_val )


for part_ind in range(part_len):
    for f1_ind in range(f1_len):
        for p1_ind in range(p1_len):
    
            #---------------------------------------------------------------------------------------------------------------------------------
            # EXPERIMENTAL COVARIANCE MATRIX
            #---------------------------------------------------------------------------------------------------------------------------------             
                  
            freq1_grp = hf.require_group( 'pump_freq_ind_' + str(f1_ind) )
            power1_grp = freq1_grp.require_group( 'pump_amp_ind' + str(p1_ind) )
            
            # Freq comb
            freqs_comb = np.array( freq1_grp['frequency comb'][nmodes] + fNCO )
                               
            # Uploading of the raw data
            on_raw = power1_grp['complex high sideband ON'][:]
            off_raw = power1_grp['complex high sideband OFF'][:]
            
            # Chuncking the measurement in several covariance matrices
            pts_chunck = len(on_raw) // part_len
            
            # COVARIANCE MATRIX
            partial_set_on = on_raw[ pts_chunck*part_ind:pts_chunck*(part_ind+1), nmodes ]
            real_part = np.real( partial_set_on )
            im_part = np.imag( partial_set_on )
            coord_array = [ ( real_part[:, v], im_part[:, v] ) for v in range( len(nmodes) ) ]
            coord_array = np.concatenate( coord_array, axis = 0 )
            cov_matrix[part_ind, f1_ind, p1_ind] = np.cov( coord_array ) * 6.33645961**2
             
            # ERRORS IN THE COVARIANCE MATRIX
            Zc = 50                                 # Transmission line impedance
            df = freq1_grp['df'][0]                 # Bandwidth
            conv_factor = 0.5 * Zc * df * Planck    # Conversion factor
            
            # We chop the cov matrix into several and do some statistics on them
            n_parts = 5
            part_of_cov = np.zeros(( n_parts, len(coord_array), len(coord_array) ))
            n_pts_keep = len(coord_array[0]) // n_parts
            for part_ind2 in range(n_parts):
                # Errors on each covariance matrix element
                part_of_cov[part_ind2, :, :] = np.cov( coord_array[:, n_pts_keep*part_ind2 : n_pts_keep*(part_ind2+1)] )
            
            # Errors on the experimental covariance matrix
            cov_matrix_std[part_ind, f1_ind, p1_ind] = np.std( part_of_cov[:, :, :], axis = 0, ddof = 1 ) / np.sqrt( n_parts ) * 6.33645961**2
            
            
            # Normalization of the experimental covariance matrix
            cov_matrix_exp = np.zeros(( 2*len(nmodes), 2*len(nmodes) ))
            cov_matrix_exp_std = np.zeros_like(( cov_matrix_exp ))
            for i in range( len(cov_matrix[part_ind, f1_ind, p1_ind]) ):
                for j in range( len(cov_matrix[part_ind, f1_ind, p1_ind]) ):
                        cov_matrix_exp[i,j] = cov_matrix[part_ind, f1_ind, p1_ind, i, j] / conv_factor / np.sqrt(freqs_comb[i//2]) / np.sqrt(freqs_comb[j//2])
                        cov_matrix_exp_std[i,j] = cov_matrix_std[part_ind, f1_ind, p1_ind, i, j] / conv_factor / np.sqrt(freqs_comb[i//2]) / np.sqrt(freqs_comb[j//2])
            
            
            #---------------------------------------------------------------------------------------------------------------------------------
            # INTERPOLATION OF GAIN AND ADDED NUMBER OF PHOTONS VALUES  
            #---------------------------------------------------------------------------------------------------------------------------------             
                
            # Since the frequencies where the gain and added n were extracted do not necessarily coincide with the freq. in our frequency comb.
            # We interpolate the gain and added noise to a polinomial to be able to get the values of the gain and n in the freq. of interest.
            
            # Interpolation functions
            f_gs = interp1d( freqs_calibration, gs,  kind='linear', fill_value="extrapolate" )
            f_gi = interp1d( freqs_calibration, gi,  kind='linear', fill_value="extrapolate" )
            f_n = interp1d( freqs_calibration, ns,  kind='linear', fill_value="extrapolate" )
            
            # Gain at the measured frequencies
            gs_comb = f_gs( freqs_comb )
            # Gain at the measured frequencies
            gi_comb = f_gi( freqs_comb )
            # ns at the measured frequencies
            ns_comb = f_n( freqs_comb )
        
            # Errors            
            # Gain of the signal error bars
            f_gain_up = interp1d( freqs_calibration, gs + gs_std, kind='linear', fill_value="extrapolate" )
            f_gain_down = interp1d( freqs_calibration, gs - gs_std, kind='linear', fill_value="extrapolate" )
            gain_comb_up = f_gain_up( freqs_comb )
            gain_comb_down = f_gain_down( freqs_comb )
            gs_std_comb = ( gain_comb_up - gain_comb_down) / 2
            
            # Gain of the idler error bars
            f_gain_id_up = interp1d( freqs_calibration, gi + gi_std, kind='linear', fill_value="extrapolate" )
            f_gain_id_down = interp1d( freqs_calibration, gi - gi_std, kind='linear', fill_value="extrapolate" )
            gain_id_comb_up = f_gain_id_up( freqs_comb )
            gain_id_comb_down = f_gain_id_down( freqs_comb )
            gi_std_comb = ( gain_id_comb_up - gain_id_comb_down) / 2
            
            # ns error bars
            f_ns_up = interp1d( freqs_calibration, ns + ns_std, kind='linear', fill_value="extrapolate" )
            f_ns_down = interp1d( freqs_calibration, ns - ns_std, kind='linear', fill_value="extrapolate" )
            ns_comb_up = f_ns_up( freqs_comb )
            ns_comb_down = f_ns_down( freqs_comb )
            ns_std_comb = ( ns_comb_up - ns_comb_down) / 2
            
            # Interpolation functions for the correlated errors
            f_sGsGi = interp1d( freqs_calibration, sGsGi,  kind='linear', fill_value="extrapolate" )
            f_sGsn = interp1d( freqs_calibration, sGsn,  kind='linear', fill_value="extrapolate" )
            f_sGin = interp1d( freqs_calibration, sGin,  kind='linear', fill_value="extrapolate" )
            
            # GsGi correlaion at the measured frequencies
            sGsGi_comb = f_sGsGi( freqs_comb )
            # Gsn correlaion at the measured frequencies
            sGsn_comb = f_sGsn( freqs_comb )
            # Gin correlaion at the measured frequencies
            sGin_comb = f_sGin( freqs_comb )
        
            
            #---------------------------------------------------------------------------------------------------------------------------------
            # RECONSTRUCTION OF THE COVARIANCE MATRIX
            #---------------------------------------------------------------------------------------------------------------------------------
            
            # Generation of the gain and ns matrices
            
            # Signal gain matrix
            gs_matrix = np.diag( [ ( 10**(gs_comb[ i//2 ]/10) ) for i in range( 2*len(gs_comb) ) ] )
            gs_std_matrix = np.diag( [ ( 10**(gs_std_comb[ i//2 ]/10) * np.log(10) / 10 ) for i in range( 2*len(gs_std_comb) ) ] )
            
            # Idler gain matrix
            gi_matrix = np.diag( [ ( 10**(gi_comb[ i//2 ]/10) ) for i in range( 2*len(gi_comb) ) ] )
            gi_std_matrix = np.diag( [ ( 10**(gi_std_comb[ i//2 ]/10) * np.log(10) / 10 ) for i in range( 2*len(gi_std_comb) ) ] )
            
            # n signal matrix
            ns_matrix = np.diag( [ ns_comb[i//2] for i in range( 2*len(ns_comb) ) ] )
            ns_std_matrix = np.diag( [ ns_std_comb[i//2] for i in range( 2*len(ns_std_comb) ) ] )
            
            # Corr matrices
            sGsGi_matrix = np.diag( [ sGsGi_comb[i//2] for i in range( 2*len(sGsGi_comb) ) ] )
            sGsn_matrix = np.diag( [ sGsn_comb[i//2] for i in range( 2*len(sGsn_comb) ) ] )
            sGin_matrix = np.diag( [ sGin_comb[i//2] for i in range( 2*len(sGin_comb) ) ] )
            
            # n idler matrix
            Tbase = 30e-3
            pump_freq = 6.05e9
            freqs_idler = 2 * pump_freq - freqs_comb
            ni_matrix = np.diag( [ ( 1 / np.tanh(Planck * freqs_idler[ i//2 ] / (2 * Boltzmann * Tbase)) - 1 ) / 2 for i in range( 2*len(freqs_comb) ) ] )
            
            # T matrix
            T = np.sqrt( gs_matrix )
            T_inv = np.linalg.inv( T )
            T_inv_std = np.diag( np.diag(gs_std_matrix) / ( 2 * np.sqrt( np.diag(gs_matrix) )**3 ) )
            
        
            # RECONSTRUCTION OF THE COVARIANCE MATRIX
            cov_matrix_reconstructed[part_ind, f1_ind, p1_ind] = T_inv.dot( cov_matrix_exp - 
                                                           (gs_matrix - np.identity(len(gs_matrix))) * (2*ns_matrix + np.identity(len(ns_matrix))) - 
                                                           (gi_matrix - np.identity(len(gi_matrix))) * (2*ni_matrix + np.identity(len(ni_matrix))) ).dot( T_inv )
            
            # ERRORS ON THE RECONSTRUCTED COVARIANCE MATRIX
            sA1 = np.diag( ( np.diag( cov_matrix_exp ) / np.diag( gs_matrix**2 ) * np.diag( gs_std_matrix ) )**2 + ( np.diag( cov_matrix_exp_std ) / np.diag( gs_matrix ) )**2 )
                
            sA2i = ( cov_matrix_exp / 2 / np.outer( np.diag( np.sqrt( gs_matrix**3 ) ), np.diag( np.sqrt( gs_matrix ) ) ) * np.diag( gs_std_matrix ) )**2
            sA2i = sA2i - np.diag( np.diag( sA2i ) )
            
            sA2ii = ( cov_matrix_exp / 2 / np.outer( np.diag( np.sqrt( gs_matrix ) ), np.diag( np.sqrt( gs_matrix**3 ) ) ) * np.diag( gs_std_matrix ) )**2
            sA2ii = sA2ii - np.diag( np.diag( sA2ii ) )
            
            sA2iii = ( cov_matrix_exp_std / np.outer( np.diag( np.sqrt( gs_matrix ) ), np.diag( np.sqrt( gs_matrix ) ) ) )**2
            sA2iii = sA2iii - np.diag( np.diag( sA2iii ) )
            
            sA2 = sA2i + sA2ii + sA2iii
            
            sB = ( 2 * ns_std_matrix )**2
            
            sC1 = np.diag( np.diag( gi_matrix ) * (2 * np.diag( ni_matrix + np.identity(len(ni_matrix)) ) ) / np.diag( gs_matrix )**2 * np.diag( gs_std_matrix ) )**2
            
            sC2 = np.diag( ( 2 * np.diag( ni_matrix ) + np.diag( np.identity(len(ni_matrix)) ) ) / np.diag( gs_matrix ) * np.diag( gi_std_matrix ) )**2
            
            corr_gsgi = 2 / np.diag( gs_matrix**3 ) * ( np.diag( cov_matrix_exp - gi_matrix * (2*ns_matrix + np.identity(len(ns_matrix))) ) ) * np.diag( 2*ni_matrix + np.identity(len(ni_matrix)) ) * sGsGi_matrix
            
            corr_gsn = 4 / np.diag( gs_matrix**2 ) * np.diag( cov_matrix_exp - gi_matrix * (2*ns_matrix + np.identity(len(ns_matrix))) ) * sGsn_matrix
            
            corr_gin = 4 / np.diag( gs_matrix ) * np.diag( 2*ni_matrix + np.identity(len(ni_matrix)) ) * sGin_matrix
            
            # Errors on the reconstructed cov matrix
            cov_matrix_reconstructed_std[part_ind, f1_ind, p1_ind] = np.sqrt ( sA1 + sA2 + sB + sC1 + sC2 + corr_gsgi + corr_gsn + corr_gin )
            
            
            #---------------------------------------------------------------------------------------------------------------------------------
            # CVXPY QUANTUM RECONSTRUCTION OF THE COVARIANCE MATRIX
            #---------------------------------------------------------------------------------------------------------------------------------
            
            # Find the closest physical covariance matrix to our reconstructed one
            
            # Define quantum matrix to identify
            quantum_cov = cp.Variable(( 2*len(nmodes), 2*len(nmodes) ), PSD = True)
            
            evaluation_block = cp.abs( (cov_matrix_reconstructed[part_ind, f1_ind, p1_ind] - quantum_cov) / cov_matrix_reconstructed_std[part_ind, f1_ind, p1_ind] )
            
            s = cp.max( evaluation_block )
            
            obj = cp.Minimize(s)
            
            # Constraints
            constraint = [ cp.lambda_min(quantum_cov - 1j*symp) >= 0, quantum_cov == cp.transpose(quantum_cov) ]
            
            prob_alt = cp.Problem( obj, constraint )
            prob_alt.solve(verbose = False)
            
            # Sanity check 
            if prob_alt.value is None:
                
                optimal_value[part_ind, f1_ind, p1_ind] = 100
                
                print("\nThe optimal value is", optimal_value[part_ind, f1_ind, p1_ind])
                
                ent_test[part_ind, f1_ind, p1_ind] = 1
                ent_error[part_ind, f1_ind, p1_ind] = 1
            
                print( "The entanglement criteria value is", 1 )
                
                continue
            
            # Reconstruction value (normalised to std)
            optimal_value[part_ind, f1_ind, p1_ind] = prob_alt.value
            
            # Print result
            print("\nThe optimal value is", optimal_value[part_ind, f1_ind, p1_ind])
            
            # QUANTUM RECONSTRUCTED COVARIANCE MATRIX
            cov_matrix_quantum[part_ind, f1_ind, p1_ind] = quantum_cov.value
            
            
            #---------------------------------------------------------------------------------------------------------------------------------
            # PPT ENTANGLEMENT CRITERIA
            #---------------------------------------------------------------------------------------------------------------------------------
        
            # l = np.arange( len(nmodes) )
            
            # # PPT criterion
            # flags = [False] * len(l)
            # ii = 0
            # while ii <= n_bipartitions:
            #     b1 = [ l[i] for i, flag in enumerate(flags) if not flag ]
                
            #     if ii != 0:
            #         b1 = np.array(b1)
            #         # only want to flip sign of p parts
            #         b1_m = 2 * b1 + 1
            #         diag_ones = np.ones( len(nmodes)*2 )
            #         diag_ones[b1_m] = -1
            #         ppt_matrix = np.diag( diag_ones )
                    
            #         ppt_cov_on = ppt_matrix.dot( cov_matrix_quantum[part_ind, f1_ind, p1_ind] ).dot( ppt_matrix )
            #         heisenberg_matrix_on = ppt_cov_on + 1j*symp
            #         heisenberg_eigvals_on = np.linalg.eigvals( heisenberg_matrix_on )
                    
            #         min_eig_ppt[part_ind, f1_ind, p1_ind, ii-1] = np.real( np.min(heisenberg_eigvals_on) )
                    
            #         if min_eig_ppt[part_ind, f1_ind, p1_ind, ii-1] > 1e2:
            #             min_eig_ppt[part_ind, f1_ind, p1_ind, ii-1] = 1
                    
            #     for i in range(len(l)):
            #         flags[i] = not flags[i]
            #         if flags[i]:
            #             break
            #     else:
            #         break
            #     ii += 1
            
            
            #---------------------------------------------------------------------------------------------------------------------------------
            # SHCHUKIN VAN LOOCK ENTANGLEMENT CRITERIA
            #---------------------------------------------------------------------------------------------------------------------------------
            
            x0 = np.random.rand( len(cov_matrix_quantum[part_ind, f1_ind, p1_ind]) // 2) * 2 * np.pi 
                
            res = minimize(lambda phi: decouple_IQ( cov_matrix_quantum[part_ind, f1_ind, p1_ind], phi), x0, method='nelder-mead', 
                           options={'maxiter': 80000, 'xtol': 1e-10, 'disp': False, 'ftol': 1e-10})
            x0 = res.x
            
            bounds = [(-10, 10) for i in range( len(cov_matrix_quantum[part_ind, f1_ind, p1_ind]) ) ]
            
            # Rotation of the data
            cov_matrix_quantum_rot[part_ind, f1_ind, p1_ind] = rotation_cov_matrix( cov_matrix_quantum[part_ind, f1_ind, p1_ind], x0 )
            cov_matrix_quantum_rot_err[part_ind, f1_ind, p1_ind] = rotation_cov_matrix( cov_matrix_reconstructed_std[part_ind, f1_ind, p1_ind], x0 )
            
            reconstructed_on_parsed = cov_matrix_quantum_rot[part_ind, f1_ind, p1_ind]
            reconstructed_on_parsed_err = cov_matrix_quantum_rot_err[part_ind, f1_ind, p1_ind]
            
            n_tries = 3
            ent_ratios = np.zeros( n_tries )
            ent_vectors = np.zeros(( n_tries, len( cov_matrix_quantum[part_ind, f1_ind, p1_ind] ) ))
            for n_try_ind in range( n_tries ):
                glob_res1 = optimize.differential_evolution( lambda vector_t: entanglement_criteriaN( vector_t, reconstructed_on_parsed, reconstructed_on_parsed_err ), 
                                                            bounds, 
                                                            maxiter = 1000 , 
                                                            atol = 1,
                                                            )
                ent_ratios[n_try_ind] = glob_res1.fun
                ent_vectors[n_try_ind] = glob_res1.x
            
            print( "The entanglement criteria value is", np.min(ent_ratios) )
                
            ent_test[part_ind, f1_ind, p1_ind], ent_error[part_ind, f1_ind, p1_ind] = entanglement_criteriaN_( ent_vectors[ent_ratios == np.min(ent_ratios)][0], reconstructed_on_parsed, reconstructed_on_parsed_err)
                        
            
            #---------------------------------------------------------------------------------------------------------------------------------
            # K-1 PARTITION ENTANGLEMENT CRITERIA
            #---------------------------------------------------------------------------------------------------------------------------------
            
            # test_list = np.arange( len(nmodes) )
            
            # ii = 0
            # for p in itertools.combinations( test_list, 2):
                
            #     grp_list = list(p)
            #     partition_list = [ test_list[ grp_list ] ]
            #     remaining_modes = np.delete( test_list , partition_list )
            #     [partition_list.append( [i] ) for i in remaining_modes]
                
            #     reconstructed_on_parsed = cov_matrix_quantum[part_ind, f1_ind, p1_ind]
            #     exp_quant_matrix_error_parsed = cov_matrix_reconstructed_std[part_ind, f1_ind, p1_ind]
            #     quant, quant_err = search_eval( partition_list, reconstructed_on_parsed, exp_quant_matrix_error_parsed, 2 )
            #     kpart_val[part_ind, f1_ind, p1_ind, ii-1] = quant
            #     kpart_err[part_ind, f1_ind, p1_ind, ii-1] = quant_err
                
            #     ii += 1
            
            # print( "The k-1 partition criteria value is", np.max( kpart_val[part_ind, f1_ind, p1_ind] / kpart_err[part_ind, f1_ind, p1_ind] ) )
            
            
            #---------------------------------------------------------------------------------------------------------------------------------
            # GENERAL BIPARTITION ENTANGLEMENT CRITERIA
            #---------------------------------------------------------------------------------------------------------------------------------    
        
            # l = np.arange( len(nmodes) )
            # flags = [False] * len(l)
            # ii = 0
            # while ii <= n_bipartitions:
            #     a1 = [l[i] for i, flag in enumerate(flags) if flag]
            #     b1 = [l[i] for i, flag in enumerate(flags) if not flag]
                
            #     if ii != 0:
            #         a1 = np.array(a1)
            #         b1 = np.array(b1)
                   
            #         partition_list = [a1, b1]
                    
            #         reconstructed_on_parsed = cov_matrix_quantum[part_ind, f1_ind, p1_ind]
            #         exp_quant_matrix_error_parsed = cov_matrix_reconstructed_std[part_ind, f1_ind, p1_ind]
            #         quant, quant_err = search_eval( partition_list, reconstructed_on_parsed, exp_quant_matrix_error_parsed, 3 )
                    
            #         bipart_val[part_ind, f1_ind, p1_ind, ii-1] = quant
            #         bipart_err[part_ind, f1_ind, p1_ind, ii-1] = quant_err
            
            #     for i in range(len(l)):
            #         flags[i] = not flags[i]
            #         if flags[i]:
            #             break
            #     else:
            #         break
                
            #     ii += 1
            
            # print( "The general bipartition criteria value is", np.max( bipart_val[part_ind, f1_ind, p1_ind] / bipart_err[part_ind, f1_ind, p1_ind]) )



#%%

#---------------------------------------------------------------------------------------------------------------------------------
# PLOTTING CALIBRATION
#---------------------------------------------------------------------------------------------------------------------------------

# Plotting gs, ns and gi
fig, ax = plt.subplots( 3, 1, sharex=True )

ax[0].errorbar( freqs_calibration, gs, yerr=gs_std, fmt ='.', label='Calibration' )
ax[0].errorbar( freqs_comb, gs_comb, yerr=gs_std_comb, fmt ='.', label='Measurement' )
ax[0].set_ylabel( 'Gain Signal [dB]' )

ax[1].errorbar( freqs_calibration, ns, yerr=ns_std, fmt ='.', label='Calibration' )
ax[1].errorbar( freqs_comb, ns_comb, yerr=ns_std_comb, fmt ='.', label='Measurement' )
ax[1].set_ylabel( 'n$_s$' )

ax[2].errorbar( freqs_calibration, gi, yerr=gi_std, fmt ='.', label='Calibration' )
ax[2].errorbar( freqs_comb, gi_comb, yerr=gi_std_comb, fmt ='.', label='Measurement' )
ax[2].set_ylabel( 'Gain Idler [dB]' )
ax[2].set_xlabel( 'Frequencies [Hz]' )
ax[0].legend()


# Plotting cross correlations
fig2, ax2 = plt.subplots( 3, 1, sharex=True )

ax2[0].plot( freqs_calibration, sGsGi, '.', label='Calibration' )
ax2[0].plot( freqs_comb, sGsGi_comb, '.', label='Measurement' )
ax2[0].set_ylabel( 'Cov(G$_s$,G$_i$)' )

ax2[1].plot( freqs_calibration, sGsn, '.', label='Calibration' )
ax2[1].plot( freqs_comb, sGsn_comb, '.', label='Measurement' )
ax2[1].set_ylabel( 'Cov(G$_s$,n)' )

ax2[2].plot( freqs_calibration, sGin, '.', label='Calibration' )
ax2[2].plot( freqs_comb, sGin_comb, '.', label='Measurement' )
ax2[2].set_ylabel( 'Cov(G$_i$,n)' )
ax2[2].set_xlabel( 'Frequencies [Hz]' )
ax2[0].legend()



#%%

#---------------------------------------------------------------------------------------------------------------------------------
# PLOTTING RECONSTRUCTION AND ENTANGLEMENT RESULTS
#---------------------------------------------------------------------------------------------------------------------------------

# Weighted Shchukin-Van Look entanglement test value
weight = 1 / ent_error**2
wavg = np.sum( weight * ent_test, axis=0 ) / np.sum( weight, axis=0 )
wavg_error = 1 / np.sqrt( np.sum( weight, axis=0 ) )
wavg_ratio = wavg / wavg_error


# 2D-plot of the reconstruction value for all frequency and amplitude settings
fig, ax = plt.subplots(1)
a = ax.pcolormesh( np.mean( optimal_value, axis=0), shading='nearest', cmap='RdBu_r', vmin = 0  )
ax.set_title( 'Reconstruction value', fontsize='large' )
fig.colorbar(a)


# 2D-plot of the Shchukin-van Look entanglement test for all frequency and amplitude settings
fig, ax = plt.subplots(1)
a = ax.pcolormesh( wavg_ratio, shading='nearest', cmap='RdBu_r', vmin = np.min(wavg_ratio)  )
ax.set_title( 'Reconstruction value', fontsize='large' )
fig.colorbar(a)



#%%

#---------------------------------------------------------------------------------------------------------------------------------
# PLOTTING COVARIANCE MATRICES
#---------------------------------------------------------------------------------------------------------------------------------

f1_ind = 0
p1_ind = 0


# Plotting the raw experimental covariance Matrix
listI = ['$_{'+str(nmodes[i]//2)+'}$' for i in range( len(nmodes) )]
labels = np.arange( nmodes[0], nmodes[-1]+2, 2)
x = np.arange( -1 + nmodes[0], nmodes[-1]+2, 1)
grid_arr = np.arange( -1 + nmodes[0], nmodes[-1]+2, 2)
zmax = np.max( np.mean( cov_matrix[:, f1_ind, p1_ind], axis=0 ) )
zmin = -zmax

fig1, ax1 = plt.subplots(1)
a = ax1.pcolormesh( x, x, np.flipud( np.mean( cov_matrix[:, f1_ind, p1_ind], axis=0 ) ), shading='nearest', cmap='RdBu_r', vmin=zmin, vmax=zmax )
ax1.set_title( 'Experimental Covariance Matrix', fontsize='large' )
fig1.colorbar(a)
plt.xticks( labels, listI, fontsize='large')
plt.yticks( labels, np.flipud(listI), fontsize='large')
ax1.grid(True, which='minor', axis='both', linestyle='-', color='w', linewidth=1.5)
ax1.set_xticks(grid_arr, minor=True)
ax1.set_yticks(grid_arr, minor=True)


# Plotting the reconstructed covariance Matrix
zmax = np.max( np.mean( cov_matrix_reconstructed[:, f1_ind, p1_ind], axis=0 ) )
zmin = -zmax

fig1, ax1 = plt.subplots(1)
a = ax1.pcolormesh( x, x, np.flipud( np.mean( cov_matrix_reconstructed[:, f1_ind, p1_ind], axis=0 ) ), shading='nearest', cmap='RdBu_r', vmin=zmin, vmax=zmax )
ax1.set_title( 'Reconstructed Covariance Matrix', fontsize='large' )
fig1.colorbar(a)
plt.xticks( labels, listI, fontsize='large')
plt.yticks( labels, np.flipud(listI), fontsize='large')
ax1.grid(True, which='minor', axis='both', linestyle='-', color='w', linewidth=1.5)
ax1.set_xticks(grid_arr, minor=True)
ax1.set_yticks(grid_arr, minor=True)


# Plotting the quantum covariance Matrix
zmax = np.max( np.mean( cov_matrix_quantum[:, f1_ind, p1_ind], axis=0 ) )
zmin = -zmax

fig1, ax1 = plt.subplots(1)
a = ax1.pcolormesh( x, x, np.flipud( np.mean( cov_matrix_quantum[:, f1_ind, p1_ind], axis=0 ) ), shading='nearest', cmap='RdBu_r', vmin=zmin, vmax=zmax )
ax1.set_title( 'CVXPY Covariance Matrix', fontsize='large' )
fig1.colorbar(a)
plt.xticks( labels, listI, fontsize='large')
plt.yticks( labels, np.flipud(listI), fontsize='large')
ax1.grid(True, which='minor', axis='both', linestyle='-', color='w', linewidth=1.5)
ax1.set_xticks(grid_arr, minor=True)
ax1.set_yticks(grid_arr, minor=True)
