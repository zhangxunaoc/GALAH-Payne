# code for fitting spectra, using the models in spectral_model.py  
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy.optimize import curve_fit
from . import spectral_model
from . import utils

def fit_normalized_spectrum_single_star_model(norm_spec, spec_err, NN_coeffs, wavelength, mask, p0 = None):
    '''
    fit a single-star model to a single combined spectrum
    
    p0 is an initial guess for where to initialize the optimizer. Because 
        this is a simple model, having a good initial guess is usually not
        important. 
    
    labels = [Teff, Logg, Vturb [km/s],
              [C/H], [N/H], [O/H], [Na/H], [Mg/H],\
              [Al/H], [Si/H], [P/H], [S/H], [K/H],\
              [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H],\
              [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ge/H],\
              C12/C13, Vmacro [km/s], radial velocity
    
    returns:
        popt: the best-fit labels
        pcov: the covariance matrix, from which you can get formal fitting uncertainties
        model_spec: the model spectrum corresponding to popt 
    '''
    tol = 5e-4 # tolerance for when the optimizer should stop optimizing.

    # set infinity uncertainty to pixels that we want to omit
    spec_err[mask] = 999.

    # assuming your NN has two hidden layers. 
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs

    # number of labels
    num_labels = w_array_0.shape[-1]
    
 #   def get_logg(*labels):

 #       M_sun==1.00
#	Mbol_sun==4.75
#	logg_sun==4.44

#	Mbol = Ks+BC-5*np.log10(D/1000)
#	logg = np.log10(M/M_sun)+4*np.log10(label[0]/5770)+0.4*(Mbol-Mbol_sun)+np.log10(logg_sun)
 
#	return logg

    def fit_func(dummy_variable, *labels):                   ##label作用在dummy_variable上，生成曲线
        """
        labels = teff, feh, alpha, broad
        """
	
#	logg = get_logg(*labels)

#	labels = np.hstack([labels[0],logg,labels[1:]])

        norm_spec = spectral_model.get_spectrum_from_neural_net(scaled_labels = labels[:],
            NN_coeffs = NN_coeffs)
   
        return norm_spec
    
    # if no initial guess is supplied (in the scaled label space)
    if p0 is None:
        p0 = np.zeros(num_labels)
        
    # don't allow the minimimizer to go  outside the range of training set
    bounds = np.zeros((2,num_labels))
    bounds[0,:] = -0.5
    bounds[1,:] = 0.5

    # run the optimizer
    popt, pcov = curve_fit(fit_func, xdata=[], ydata = norm_spec, sigma = spec_err, p0 = p0,
                 ftol = tol, xtol = tol, absolute_sigma = True, method = 'trf')
    model_spec = fit_func([], *popt)

    # rescale the results back to normal unit
    popt[:] = (popt[:]+0.5)*(x_max-x_min) + x_min
    pcov[:,:] = pcov[:,:]*(x_max-x_min)
    return popt, pcov, model_spec
