# code for predicting the spectrum of a single star in normalized space. 
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from . import utils
import scipy

def sigmoid(z):
    '''
    This is the activation function used by default in all our neural networks. 
    You can experiment with using an ReLU instead, but I got worse results in 
    some simple tests. 
    '''
    return 1.0/(1.0 + np.exp(-z))
    
def get_spectrum_from_neural_net(scaled_labels, NN_coeffs, wavelength, obs_spec):
    '''
    Predict the rest-frame spectrum (normalized) of a single star. 
    We input the scaled stellar labels (not in the original unit). Each label ranges from -0.5 to 0.5
    '''
    
    # assuming your NN has two hidden layers. 
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs

    # the neural network architecture adopted in Ting+ 18, individual networks for individual pixels
    #inside = np.einsum('ijk,k->ij', w_array_0, scaled_labels) + b_array_0
    #outside = np.einsum('ik,ik->i', w_array_1, sigmoid(inside)) + b_array_1
    #spectrum = w_array_2*sigmoid(outside) + b_array_2

    # having a single large network seems for all pixels seems to work better
    # as it exploits the information between adjacent pixels
    inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
    outside = np.einsum('ij,j->i', w_array_1, sigmoid(inside)) + b_array_1
    spectrum = np.einsum('ij,j->i', w_array_2, sigmoid(outside)) + b_array_2

    print('kai2 spectrum and obs_spec:', scipy.stats.chisquare(f_obs=spectrum, f_exp=obs_spec))

    #renormalise spectrm from trianed payne model
    start_wave = [4700, 4740, 4770, 4800, 4840, 5650, 5700, 5750, 5800, 5840, 6450, 6530, 6600, 6640, 6680, 7680, 7760, 7820]
    end_wave = [4740, 4770, 4800, 4840, 4900, 5700, 5750, 5800, 5840, 5900, 6530, 6600, 6640, 6680, 6750, 7760, 7820, 7880]

    renormalise_spectrum = np.zeros(len(obs_spec))

    for i in range(len(start_wave)):
        
        spectrum_segment = spectrum[(start_wave[i] < wavelength) & (wavelength < end_wave[i])]
        obs_spec_segment = obs_spec[(start_wave[i] < wavelength) & (wavelength < end_wave[i])]
    
    #theil-sen estimator
        np.seterr(divide='ignore', invalid='ignore')
        fit = utils.theil_sen(wavelength[(start_wave[i] < wavelength) & (wavelength < end_wave[i])], spectrum_segment/obs_spec_segment)
        print('fit:', fit)
        median = wavelength[(start_wave[i] < wavelength) & (wavelength < end_wave[i])]*fit[0] + fit[1]
        spectrum_segment /= median
        renormalise_spectrum[(start_wave[i] < wavelength) & (wavelength < end_wave[i])] = spectrum_segment  
    print('kai2 renormalise_spectrum and obs_spec:', scipy.stats.chisquare(f_obs=renormalise_spectrum, f_exp=obs_spec))

    return renormalise_spectrum


    
