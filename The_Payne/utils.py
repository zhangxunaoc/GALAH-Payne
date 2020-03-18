# a few low-level functions that are used throughout
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import os
import glob
from astropy.io import fits
import bottleneck
import itertools

def read_in_neural_network():
    '''
    read in the weights and biases parameterizing a particular neural network. 
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in. 
    '''

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'neural_nets/NN_normalized_spectra.npz')
    tmp = np.load(path)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return NN_coeffs

def create_wavelength_array(survey='galah'):
    '''
    create the wavelength array needed for galah
    '''
    if survey != 'galah':
        print('this function can only create the wavelength array for galah')
    else:
        ccd1=np.arange(4715.94,4896.00,0.046) # ab lines 4716.3 - 4892.3
        ccd2=np.arange(5650.06,5868.25,0.055) # ab lines 5646.0 - 5867.8
        ccd3=np.arange(6480.52,6733.92,0.064) # ab lines 6481.6 - 6733.4
        ccd4=np.arange(7693.50,7875.55,0.074) # ab lines 7691.2 - 7838.5
        wavelength = np.concatenate((ccd1, ccd2, ccd3, ccd4))
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/galah_wavelength.npz')
        np.savez(path, wavelength=wavelength)

def load_wavelength_array(survey='apogee'):
    '''
    read in the default wavelength grid onto which we interpolate all spectra
    the keyword 'survey' is 'apogee' by default, but we can also use 'galah' 
    '''
    if survey == 'apogee':
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/apogee_wavelength.npz')
    elif survey == 'galah':
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/galah_wavelength.npz')
        if not os.path.exists(path):
            create_wavelength_array(survey='galah')
    else:
        print('You should use either apogee or galah')
    tmp = np.load(path)
    wavelength = tmp['wavelength']
    tmp.close()
    return wavelength

def galah_mask(wavelength, sme_like=True, cores_out=True):
    """
    Deliver the appropriate GALAH wavelength range mask
    INPUT:
        wavelength -- array with wavelength entries
    OUTPUT:
        mask -- array with False (not masked) and True (masked)
    OPTIONS:
        sme_like -- if only use SME Sp pixels
        cores_out -- if Balmer line cores (+- 0.25 AA) should be masked
    """
    mask = np.zeros(wavelength.size, dtype=bool) # no masking
   
    if sme_like == True:
        # load segments from DR3
        sme_segments = np.loadtxt('/home/zhangxu/anaconda3/lib/python3.7/site-packages/The_Payne/other_data/DR3_Sp.dat',dtype=float,comments=';')
        # we will mask everything and then unmask the SME segments
        mask = np.ones(wavelength.size, dtype=bool) # all masked
        for l1, start, end, elem in sme_segments:
            mask[((wavelength > start) & (wavelength < end))] = False # Unmask from start to end of SME segment

        # Unmask Balmer Segments as well
        mask[((wavelength > 4840.51) & (wavelength < 4880.00))] = False # Unmask H_beta
        mask[((wavelength > 6530.01) & (wavelength < 6590.00))] = False # Unmask H_alpha
   
    if cores_out == True:
        mask[np.abs(wavelength-4861.3230)<0.25]=True
        mask[np.abs(wavelength-6562.7970)<0.25]=True
 
    return mask

def create_galah_mask():
    '''
    for now this will create an array with 14304 True entries
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/galah_mask.npz')
    galah_mask = np.ones(14304, dtype=bool)
    np.savez(path, galah_mask=galah_mask)

def load_galah_mask():
    '''
    read in the pixel mask with which we will omit bad pixels during spectral fitting
    in the future, this will help to omit bad pixels etc.
    for now, we will use all pixels during spectral fitting
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/galah_mask.npz')
    if not os.path.exists(path):
        create_galah_mask()
    tmp = np.load(path)
    mask = tmp['galah_mask']
    tmp.close()
    return mask

def get_model_grid(file_type='npz', fits2npz=False):
    '''
    Either read in the information from GALAH-spectra2.fits and save it into NPZ
    
    or
    
    read in the NPZ file
    '''
    
    if file_type=='fits':
        a1 = fits.getdata(os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/GALAH-spectra2.fits'),ext=1)[0]
        a2 = fits.getdata(os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/GALAH-spectra2.fits'),ext=2)
        
        model_grid=dict(
            wavelength = a1[0],
            teff = a1[1],
            logg = a1[2],
            feh = a1[3],
            alpha_fe = a1[4],
            c_fe = a1[5],
            smod = a2
        )

        if fits2npz==True:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/model_grid.npz')
            if not os.path.exists(path):
                np.savez(path, **model_grid)

            print('The model_grid has the following keywords:')
            print(model_grid.keys())
            
    if file_type=='npz':
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/GALAH_NordlanderGrid_GUESSnorm.npz')
        tmp = np.load(path)

        model_grid = dict(
            wavelength = tmp['wavelength'],
            teff = tmp['teff'],
            logg = tmp['logg'],
            feh = tmp['feh'],
            alpha_fe = tmp['alpha_fe'],
            broad = tmp['broad'],
            smod = tmp['smod_norm_guess']
            )
        print('The model_grid has the following keywords:')
        print(model_grid.keys())
        tmp.close()
        
    return(model_grid)

def load_training_data(survey='apogee', size=1000):
    '''
    read in the default Kurucz training spectra for APOGEE
    or
    read in the whole GALAH model grid and select a training set from it
    currently, 1000 random GALAH spectra will be chosen
    and from those we choose the first 800 to be part of the training set and the other 200 as validation
    '''
    
    if survey=='apogee':
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/kurucz_training_spectra.npz')
        tmp = np.load(path)
        training_labels = (tmp["labels"].T)[:800,:]
        training_spectra = tmp["spectra"][:800,:]
        validation_labels = (tmp["labels"].T)[800:,:]
        validation_spectra = tmp["spectra"][800:,:]
        tmp.close()
        
    if survey=='galah':
        # read in whole GALAH model grid
        model_grid = get_model_grid(file_type='npz', fits2npz=False)
        # make random choice reproducable
        np.random.seed(3)
        # now select 'size' entries from the model_grid, if the 'size' is not larger than the actual model grid
        if size <= np.shape(model_grid['teff'])[0]:
            training_array = np.random.choice(np.arange(np.shape(model_grid['teff'])[0]),size=size,replace=False)
        else:
            print('The model grid has only '+str(np.shape(model_grid['teff'])[0])+' entries')
            
        # select the labels and spectra from the chosen indices
        labels = np.array([model_grid[key][training_array] for key in ['teff','logg','feh','alpha_fe','broad']])
        spectra = model_grid['smod'][training_array]
        
        # we select/redefine 1/5 of the training set as validation set, the rest as training set
        fifth = int(size/5)
        print('Whole training set: '+str(size))
        print('Redefined training set ('+str(size-fifth)+') and validation set ('+str(fifth)+')')
        print(fifth)
        training_labels = (labels.T)[:size-fifth,:]
        training_spectra = spectra[:size-fifth,:]
        validation_labels = (labels.T)[size-fifth:,:]
        validation_spectra = spectra[size-fifth:,:]
        
    return training_labels, training_spectra, validation_labels, validation_spectra

def doppler_shift(wavelength, flux, dv):
    '''
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.
   
    This linear interpolation is actually not that accurate, but is fine if you 
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation. 
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c)) 
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux

#def renormalise_norm_spec_from_payne(wavelength, norm_spec, norm_spec_from_payne):

#    start_wave = [4700, 4740, 4770, 4800, 4840, 5650, 5700, 5750, 5800, 5840, 6450, 6530, 6600, 6640, 6680, 7680, 7760, 7820]
#    end_wave = [4740, 4770, 4800, 4840, 4900, 5700, 5750, 5800, 5840, 5900, 6530, 6600, 6640, 6680, 6750, 7760, 7820, 7880]

#    renormalise_norm_spec = np.zeros(len(norm_spec))

#    for i in range(len(start_wave)):
        
#        norm_spec_from_payne_segment = norm_spec_from_payne[(start_wave[i] < wavelength) & (wavelength < end_wave[i])]
        
#        norm_spec_segment = norm_spec[(start_wave[i] < wavelength) & (wavelength < end_wave[i])]
    
    #theil sen estimator
#        np.seterr(divide='ignore', invalid='ignore')
#        fit = theil_sen(wavelength[(start_wave[i] < wavelength) & (wavelength < end_wave[i])], norm_spec_from_payne_segment/  norm_spec_segment)
#        print('fit:',fit)
#        median = wavelength[(start_wave[i] < wavelength) & (wavelength < end_wave[i])]*fit[0] + fit[1]
    
#        norm_spec_from_payne_segment /= median

#        renormalise_norm_spec[(start_wave[i] < wavelength) & (wavelength < end_wave[i])] = norm_spec_from_payne_segment  

#    return renormalise_norm_spec


#    ccd1_segment1 = (wavelength > 4700) & (wavelength < 4740)
#    ccd1_segment2 = (wavelength > 4740) & (wavelength < 4770)
#    ccd1_segment3 = (wavelength > 4770) & (wavelength < 4800)
#    ccd1_segment4 = (wavelength > 4800) & (wavelength < 4840)
#    ccd1_segment5 = (wavelength > 4840) & (wavelength < 4900)
        
#    ccd2_segment1 = (wavelength > 5650) & (wavelength < 5700)
#    ccd2_segment2 = (wavelength > 5700) & (wavelength < 5750)
#    ccd2_segment3 = (wavelength > 5750) & (wavelength < 5800)
#    ccd2_segment4 = (wavelength > 5800) & (wavelength < 5840)
#    ccd2_segment5 = (wavelength > 5840) & (wavelength < 5900)
        
#    ccd3_segment1 = (wavelength > 6450) & (wavelength < 6530)
#    ccd3_segment2 = (wavelength > 6530) & (wavelength < 6600)
#    ccd3_segment3 = (wavelength > 6600) & (wavelength < 6640)
#    ccd3_segment4 = (wavelength > 6640) & (wavelength < 6680)
#    ccd3_segment5 = (wavelength > 6680) & (wavelength < 6750)
        
#    ccd4_segment1 = (wavelength > 7680) & (wavelength < 7760)
#    ccd4_segment2 = (wavelength > 7760) & (wavelength < 7820)
#    ccd4_segment3 = (wavelength > 7820) & (wavelength < 7880)

#    norm_spec_from_payne_ccd1_segment1= norm_spec_from_payne[ccd1_segment1]/np.median(norm_spec_from_payne[ccd1_segment1]/norm_spec[ccd1_segment1])
#    norm_spec_from_payne_ccd1_segment2= norm_spec_from_payne[ccd1_segment2]/np.median(norm_spec_from_payne[ccd1_segment2]/norm_spec[ccd1_segment2])
#    norm_spec_from_payne_ccd1_segment3= norm_spec_from_payne[ccd1_segment3]/np.median(norm_spec_from_payne[ccd1_segment3]/norm_spec[ccd1_segment3])
#    norm_spec_from_payne_ccd1_segment4= norm_spec_from_payne[ccd1_segment4]/np.median(norm_spec_from_payne[ccd1_segment4]/norm_spec[ccd1_segment4])
#    norm_spec_from_payne_ccd1_segment5= norm_spec_from_payne[ccd1_segment5]/np.median(norm_spec_from_payne[ccd1_segment5]/norm_spec[ccd1_segment5])
 
#    norm_spec_from_payne_ccd2_segment1= norm_spec_from_payne[ccd2_segment1]/np.median(norm_spec_from_payne[ccd2_segment1]/norm_spec[ccd2_segment1])
#    norm_spec_from_payne_ccd2_segment2= norm_spec_from_payne[ccd2_segment2]/np.median(norm_spec_from_payne[ccd2_segment2]/norm_spec[ccd2_segment2])
#    norm_spec_from_payne_ccd2_segment3= norm_spec_from_payne[ccd2_segment3]/np.median(norm_spec_from_payne[ccd2_segment3]/norm_spec[ccd2_segment3])
#    norm_spec_from_payne_ccd2_segment4= norm_spec_from_payne[ccd2_segment4]/np.median(norm_spec_from_payne[ccd2_segment4]/norm_spec[ccd2_segment4])
#    norm_spec_from_payne_ccd2_segment5= norm_spec_from_payne[ccd2_segment5]/np.median(norm_spec_from_payne[ccd2_segment5]/norm_spec[ccd2_segment5])

#    norm_spec_from_payne_ccd3_segment1= norm_spec_from_payne[ccd3_segment1]/np.median(norm_spec_from_payne[ccd3_segment1]/norm_spec[ccd3_segment1])
#    norm_spec_from_payne_ccd3_segment2= norm_spec_from_payne[ccd3_segment2]/np.median(norm_spec_from_payne[ccd3_segment2]/norm_spec[ccd3_segment2])
#    norm_spec_from_payne_ccd3_segment3= norm_spec_from_payne[ccd3_segment3]/np.median(norm_spec_from_payne[ccd3_segment3]/norm_spec[ccd3_segment3])
#    norm_spec_from_payne_ccd3_segment4= norm_spec_from_payne[ccd3_segment4]/np.median(norm_spec_from_payne[ccd3_segment4]/norm_spec[ccd3_segment4])
#    norm_spec_from_payne_ccd3_segment5= norm_spec_from_payne[ccd3_segment5]/np.median(norm_spec_from_payne[ccd3_segment5]/norm_spec[ccd3_segment5])

#    norm_spec_from_payne_ccd4_segment1= norm_spec_from_payne[ccd4_segment1]/np.median(norm_spec_from_payne[ccd4_segment1]/norm_spec[ccd4_segment1])
#    norm_spec_from_payne_ccd4_segment2= norm_spec_from_payne[ccd4_segment2]/np.median(norm_spec_from_payne[ccd4_segment2]/norm_spec[ccd4_segment2])
#    norm_spec_from_payne_ccd4_segment3= norm_spec_from_payne[ccd4_segment3]/np.median(norm_spec_from_payne[ccd4_segment3]/norm_spec[ccd4_segment3])

#    norm_spec_from_payne = np.concatenate((
#                norm_spec_from_payne_ccd1_segment1, norm_spec_from_payne_ccd1_segment2, norm_spec_from_payne_ccd1_segment3,
#                norm_spec_from_payne_ccd1_segment4, norm_spec_from_payne_ccd1_segment5, norm_spec_from_payne_ccd2_segment1,
#                norm_spec_from_payne_ccd2_segment2, norm_spec_from_payne_ccd2_segment3, norm_spec_from_payne_ccd2_segment4, 
#                norm_spec_from_payne_ccd2_segment5, norm_spec_from_payne_ccd3_segment1, norm_spec_from_payne_ccd3_segment2,
#                norm_spec_from_payne_ccd3_segment3, norm_spec_from_payne_ccd3_segment4, norm_spec_from_payne_ccd3_segment5,
#                norm_spec_from_payne_ccd4_segment1, norm_spec_from_payne_ccd4_segment2, norm_spec_from_payne_ccd4_segment3),  axis=0)

#    return norm_spec_from_payne

#from https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/Estimators/theil_sen.py
def theil_sen(x, y, sample= "auto", n_samples = 1e7):
    
    assert x.shape[0] == y.shape[0]
    
    n = x.shape[0]
    
    if n < 100 or not sample:
        ix = np.argsort( x )
        slopes = np.empty(int(n*(n-1)*0.5))
        for c, pair in enumerate(itertools.combinations(range(n), 2)):
            i,j = ix[pair[0]], ix[pair[1]]
            slopes[c] = slope(x[i], x[j], y[i], y[j])
    else:
        i1 = np.random.randint(int(0), int(n), int(n_samples))
        i2 = np.random.randint(int(0), int(n), int(n_samples))
        slopes = slope(x[i1], x[i2], y[i1], y[i2])

    slope_ = bottleneck.nanmedian(slopes)
    #find the optimal b as the median of y_i - slope*x_i
    intercepts = np.empty(n)
    for c in range(n):
        intercepts[c] = y[c] - slope_*x[c]
    intercept_ = bottleneck.median(intercepts)

    return np.array([slope_, intercept_])

def slope(x_1, x_2, y_1, y_2):
    return (1 - 2*(x_1>x_2))*((y_2 - y_1)/np.abs((x_2-x_1)))
    


