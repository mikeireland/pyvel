'''
Created on 4 Sep. 2017

@author: christoph
'''
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
from scipy import ndimage
import h5py
import scipy.sparse as sparse
#import logging
import time
from scipy.optimize import curve_fit
import collections
from scipy import special
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, Model
from astropy.io import ascii
from matplotlib.colors import LogNorm
from veloce_reduction.helper_functions import gaussian_with_offset, fibmodel_with_amp_and_offset, fibmodel_with_amp_and_offset_and_slope, fibmodel_with_amp



def find_stripes(flat, deg_polynomial=2, gauss_filter_sigma=5., min_peak=0.25, weighted_fits=1, timit=False, debug_level=0):
    """
    BASED ON JULIAN STUERMER'S MAROON_X PIPELINE:
    
    Locates and fits stripes (ie orders) in a flat field echelle spectrum.
    
    Starting in the central column, the algorithm identifies peaks and traces each stripe to the edge of the detector
    by following the brightest pixels along each order. It then fits a polynomial to each stripe.
    To improve algorithm stability, the image is first smoothed with a Gaussian filter. It not only eliminates noise, but
    also ensures that the cross-section profile of the flat becomes peaked in the middle, which helps to identify the
    center of each stripe. Choose gauss_filter accordingly.
    To avoid false positives, only peaks above a certain (relative) intensity threshold are used.
      
    :param flat: dark-corrected flat field spectrum
    :type flat: np.ndarray
    :param deg_polynomial: degree of the polynomial fit
    :type deg_polynomial: int
    :param gauss_filter_sigma: sigma of the gaussian filter used to smooth the image.
    :type gauss_filter_sigma: float
    :param min_peak: minimum relative peak height 
    :type min_peak: float
    :param debug_level: debug level flag
    :type debug_level: int
    :return: list of polynomial fits (np.poly1d)
    :rtype: list
    """
    
    if timit:
        start_time = time.time()
    
    #logging.info('Finding stripes...')
    print("Finding stripes...")
    ny, nx = flat.shape

    # smooth image slightly for noise reduction
    filtered_flat = ndimage.gaussian_filter(flat.astype(np.float), gauss_filter_sigma)
    
    # find peaks in center column
    data = filtered_flat[:, int(nx / 2)]
    peaks = np.r_[True, data[1:] > data[:-1]] & np.r_[data[:-1] > data[1:], True]

    if debug_level > 1:
        plt.figure()
        plt.title('Local maxima')
        plt.plot(data)
        plt.scatter(np.arange(ny)[peaks], data[peaks],s=25)
        plt.show()

    idx = np.logical_and(peaks, data > min_peak * np.max(data))
    maxima = np.arange(ny)[idx]

    # filter out maxima too close to the boundary to avoid problems
    maxima = maxima[maxima > 3]
    maxima = maxima[maxima < ny - 3]

    if debug_level > 1:
        #labels, n_labels = ndimage.measurements.label(data > min_peak * np.max(data))
        plt.figure()
        plt.title('Order peaks')
        plt.plot(data)
        plt.scatter(np.arange(ny)[maxima], data[maxima],s=25, c='red')
        #plt.plot((labels[0] > 0) * np.max(data))   #what does this do???
        plt.show()

    n_order = len(maxima)
    #logging.info('Number of stripes found: %d' % n_order)
    print('Number of stripes found: %d' % n_order)

    orders = np.zeros((n_order, nx))
    # because we only want to use good pixels in the fit later on
    mask = np.ones((n_order, nx), dtype=bool)

    # walk through to the left and right along the maximum of the order
    # loop over all orders:
    for m, row in enumerate(maxima):
        column = int(nx / 2)
        orders[m, column] = row
        start_row = row
        # walk right
        while (column + 1 < nx):
            column += 1
            args = np.array(np.linspace(max(1, start_row - 1), min(start_row + 1, ny - 1), 3), dtype=int)
            args = args[np.logical_and(args < ny, args > 0)]     #deal with potential edge effects
            p = filtered_flat[args, column]
            # new maximum (apply only when there are actually flux values in p, ie not when eg p=[0,0,0]), otherwise leave start_row unchanged
            if ~(p[0]==p[1] and p[0]==p[2]):
                start_row = args[np.argmax(p)]
            orders[m, column] = start_row
            if ((p < 10).all()) or ((column > 3500) and (mask[m,column-1]==False)) or (start_row in (0,4095)) or (m==42 and (p < 100).all()):
                mask[m,column] = False
        # walk left
        column = int(nx / 2)
        start_row = row
        while (column > 0):
            column -= 1
            args = np.array(np.linspace(max(1, start_row - 1), min(start_row + 1, ny - 1), 3), dtype=int)
            args = args[np.logical_and(args < ny, args > 0)]     #deal with potential edge effects
            p = filtered_flat[args, column]
            # new maximum
            start_row = args[np.argmax(p)]
            orders[m, column] = start_row
            if ((p < 10).all()) or ((column < 500) and (mask[m,column+1]==False)) or (start_row in (0,4095)) or (m==42 and (p < 100).all()):
                mask[m,column] = False

    # do Polynomial fit for each order
    #logging.info('Fit polynomial of order %d to each stripe' % deg_polynomial)
    print('Fit polynomial of order %d to each stripe...' % deg_polynomial)
    P = []
    xx = np.arange(nx)
    for i in range(len(orders)):
        if weighted_fits == 0:
            #unweighted
            p = np.poly1d(np.polyfit(xx[mask[i,:]], orders[i,mask[i,:]], deg_polynomial))
        else:
            #weighted
            filtered_flux_along_order = np.zeros(nx)
            for j in range(nx):
                #filtered_flux_along_order[j] = filtered_flat[o[j].astype(int),j]    #that was when the loop reas: "for o in orders:"
                filtered_flux_along_order[j] = filtered_flat[orders[i,j].astype(int),j]
            filtered_flux_along_order[filtered_flux_along_order < 1] = 1   
            #w = 1. / np.sqrt(filtered_flux_along_order)   this would weight the order centres less!!!
            w = np.sqrt(filtered_flux_along_order)
            p = np.poly1d(np.polyfit(xx[mask[i,:]], orders[i,mask[i,:]], deg_polynomial, w=w[mask[i,:]]))
        P.append(p)

    if debug_level > 0:
        plt.figure()
        plt.imshow(filtered_flat, interpolation='none', vmin=np.min(flat), vmax=0.9 * np.max(flat), cmap=plt.get_cmap('gray'))
        for p in P:
            plt.plot(xx, p(xx), 'g', alpha=1)
        plt.ylim((0, ny))
        plt.xlim((0, nx))
        plt.show()    
        
    if timit:
        print('Elapsed time: '+str(time.time() - start_time)+' seconds')

    return P,mask



def make_P_id_old(P):
    Ptemp = {}
    ordernames = []
    for i in range(1,10):
        ordernames.append('order_0%i' % i)
    for i in range(10,len(P)+1):
        ordernames.append('order_%i' % i)
    #the array parms comes from the "find_stripes" function
    for i in range(len(P)):
        Ptemp.update({ordernames[i]:P[i]})
    P_id = {'fibre_01': Ptemp}
     
    return P_id



def make_P_id(P):
    P_id = {}
    ordernames = []
    for i in range(1,10):
        ordernames.append('order_0%i' % i)
    for i in range(10,len(P)+1):
        ordernames.append('order_%i' % i)
    #the array parms comes from the "find_stripes" function
    for i in range(len(P)):
        P_id.update({ordernames[i]:P[i]})
    
    return P_id



def make_mask_dict(tempmask):
    mask = {}
    ordernames = []
    for i in range(1,len(tempmask)+1):
        ordernum = str(i).zfill(2)
        ordernames.append('order_'+ordernum)
    #the array parms comes from the "find_stripes" function
    for i in range(len(tempmask)):
        mask.update({ordernames[i]:tempmask[i,:]})
    
    return mask



def extract_single_stripe(img, p, slit_height=25, debug_level=0):
    """
    Extracts single stripe from 2d image.

    This function returns a sparse matrix containing all relevant pixel for a single stripe for a given polynomial p
    and a given slit height.

    :param img: 2d echelle spectrum
    :type img: np.ndarray
    :param P: polynomial coefficients
    :type P: np.ndarray
    :param slit_height: total slit height in pixel to be extracted
    :type slit_height: double
    :param debug_level: debug level
    :type debug_level: int
    :return: extracted spectrum
    :rtype: scipy.sparse.csc_matrix
    """
    
    #start_time = time.time()
    
    ny, nx = img.shape
    #xx = np.arange(nx, dtype=img.dtype)
    xx = np.arange(nx, dtype='f8')
    #yy = np.arange(ny, dtype=img.dtype)
    yy = np.arange(ny, dtype='f8')

    y = np.poly1d(p)(xx)
    x_grid, y_grid = np.meshgrid(xx, yy, copy=False)

    distance = y_grid - y.repeat(ny).reshape((nx, ny)).T
    indices = abs(distance) <= slit_height

    if debug_level > 2:
        plt.figure()
        plt.imshow(img)
        plt.imshow(indices, alpha=0.5)
        plt.show()

    mat = sparse.coo_matrix((img[indices], (y_grid[indices], x_grid[indices])), shape=(ny, nx))
    # return mat.tocsr()
    
    #print('Elapsed time: ',time.time() - start_time,' seconds')
    
    return mat.tocsc()



def extract_stripes(img, P_id, slit_height=25, output_file=None, timit=False, debug_level=0):
    """
    Extracts the stripes from the original 2D spectrum to a sparse array, containing only relevant pixels.
    
    This function marks all relevant pixels for extraction. Using the provided dictionary P_id it iterates over all
    stripes in the image and saves a sparse matrix for each stripe.
    
    :param img: 2d echelle spectrum
    :type img: np.ndarray
    :param P_id: dictionary of the form of {fiber_number:{order: np.poly1d, ...}, ...} (as returned by identify_stripes)
    or path to file
    :type P_id: dict or str
    :param slit_height: total slit height in px
    :type slit_height: float
    :param output_file: path to file where result is saved
    :type output_file: str
    :param debug_level: debug level
    :type debug_level: int
    :return: dictionary of the form {fiber_number:{order: scipy.sparse_matrix}}
    :rtype: dict
    """
    
    if timit:
        overall_start_time = time.time()
    
    #logging.info('Extract stripes...')
    print('Extracting stripes...')
    stripes = {}
#     if isinstance(P_id, str):
#         # get all fibers
#         P_id = utils.load_dict_from_hdf5(P_id, 'extraction_parameters/')
#         # copy extraction parameters to result file
#         if output_file is not None:
#             utils.save_dict_to_hdf5(P_id, output_file, 'extraction_parameters/')

    for o, p in P_id.items():
        stripe = extract_single_stripe(img, p, slit_height, debug_level)
#         if o in stripes:
        stripes.update({o: stripe})
#         else:
#              stripes = {o: stripe}

#     if output_file is not None:
#         for f in stripes.keys():
#             for o in stripes[f].keys():
#                 utils.store_sparse_mat(stripes[f][o], 'extracted_stripes/%s/%s' % (f, o), output_file)

    if timit:
        print('Total time taken for "EXTRACT_STRIPES": ',time.time() - overall_start_time,' seconds')

    return stripes



def flatten_single_stripe(stripe, slit_height=25, timit=False):
    """
    CMB 06/09/2017
    
    This function stores the non-zero values of the sparse matrix "stripe" in a rectangular array, ie
    take out the curvature of the order/stripe, potentially only useful for further processing.

    INPUT:
    "stripe": sparse 4096x4096 matrix, non-zero for just one stripe/order of user defined width (=2*slit_height in "extract_stripes")
    
    OUTPUT:
    "stripe_columns": dense rectangular matrix containing only the non-zero elements of "stripe". This has
                      dimensions of (2*slit_height, 4096)
    "stripe_rows":    row indices (ie rows being in dispersion direction) of the original image for the columns (ie the "cutouts")
    """
    #input is sth like this
    #stripe = stripes['fibre_01']['order_01']
    #TODO: sort the dictionary by order number...
    
    if timit:
        start_time = time.time()    
    
    ny, nx = stripe.todense().shape
    
    #find the non-zero parts of the sparse matrix
    #format is:
    #contents[0] = row indices
    #contents[1] = column indices
    #contents[2] = values
    contents = sparse.find(stripe)
    #the individual columns correspond to the unique values of the x-indices, stored in contents[1]    
    col_values, col_indices, counts = np.unique(contents[1], return_index=True, return_counts=True)     
    #stripe_columns = np.zeros((int(len(contents[0]) / len(col_indices)),len(col_indices)))
    #stripe_flux = np.zeros((2*slit_height, 4096))
    stripe_flux = np.zeros((2*slit_height, nx))
    #stripe_rows = np.zeros((int(len(contents[0]) / len(col_indices)),len(col_indices)))
    stripe_rows = np.zeros((2*slit_height, ny))
    
    #check if whole order falls on CCD in dispersion direction
    if len(col_indices) != nx:
        print('WARNING: Not the entire order falls on the CCD:'),    
        #parts of order missing on LEFT side of chip?
        if contents[1][0] != 0:
            print('some parts of the order are missing on LEFT side of chip...')
            #this way we also know how many pixels are defined (ie on the CCD) for each column of the stripe
            for i,coli,ct in zip(col_values,col_indices,counts):
                if i == np.max(col_values):
                    flux_temp = contents[2][coli:]         #flux
                    rownum_temp = contents[0][coli:]       #row number
                else:
                    #this is confusing, but: coli = col_indices[i-np.min(col_values)]
                    flux_temp = contents[2][col_indices[i-np.min(col_values)]:col_indices[i-np.min(col_values)+1]]         #flux
                    rownum_temp = contents[0][col_indices[i-np.min(col_values)]:col_indices[i-np.min(col_values)+1]]       #row number        
                #now, because the valid pixels are supposed to be at the bottom of the cutout, we need to roll them to the end
                if ct != (2*slit_height):
                    flux = np.zeros(2*slit_height) - 1.     #negative flux can be used to identify these pixels layer
                    flux[0:ct] = flux_temp
                    flux = np.roll(flux,2*slit_height - ct)
                    rownum = np.zeros(2*slit_height,dtype='int32')    #zeroes can be used to identify these pixels later
                    rownum[0:ct] = rownum_temp
                    rownum = np.roll(rownum,2*slit_height - ct)
                else:
                    flux = flux_temp
                    rownum = rownum_temp
                stripe_flux[:,i] = flux
                stripe_rows[:,i] = rownum
        
        
        
        #parts of order missing on RIGHT side of chip?
        elif contents[1][-1] != nx-1:
            print('some parts of the order are missing on RIGHT side of chip...NIGHTMARE, THIS HAS NOT BEEN IMPLEMENTED YET') 
            quit()
#             for i in col_values:
#                 #is there a value for all pixels across the i-th cutout?
#                 if i == np.max(col_values):
#                     flux = contents[2][col_indices[i]:]         #flux
#                     rownum = contents[0][col_indices[i]:]       #row number
#                 else:
#                     flux = contents[2][col_indices[i]:col_indices[i+1]]         #flux
#                     rownum = contents[0][col_indices[i]:col_indices[i+1]]       #row number
#                 stripe_flux[:,i] = flux
#                 stripe_rows[:,i] = rownum 
    
    
    #check if whole order falls on CCD in spatial direction
    elif ~(counts == 2*slit_height).all():
        print('WARNING: Not the entire order falls on the CCD:'),
        #parts of order missing at the top?
        if np.max(contents[0]) == ny-1:
            print('some parts of the order are missing on at the TOP of the chip...') 
            #this way we also know how many pixels are defined (ie on the CCD) for each column of the stripe
            for i,coli,ct in zip(col_values,col_indices,counts):
                
                if i == np.max(col_values):
                    flux_temp = contents[2][coli:]         #flux
                    rownum_temp = contents[0][coli:]       #row number
                else:
                    #this is confusing, but: coli = col_indices[i-np.min(col_values)]
                    flux_temp = contents[2][col_indices[i-np.min(col_values)]:col_indices[i-np.min(col_values)+1]]         #flux
                    rownum_temp = contents[0][col_indices[i-np.min(col_values)]:col_indices[i-np.min(col_values)+1]]       #row number        
                
                #now, because the valid pixels are supposed to be at the top of the cutout, we do NOT need to roll them to the end
                if ct != (2*slit_height):
                    flux = np.zeros(2*slit_height) - 1.         #negative flux can be used to identify these pixels layer
                    flux[0:ct] = flux_temp
                    #flux = np.roll(flux,2*slit_height - ct)
                    rownum = np.zeros(2*slit_height,dtype='int32')   #zeroes can be used to identify these pixels later
                    rownum[0:ct] = rownum_temp
                    #rownum = np.roll(rownum,2*slit_height - ct)
                else:
                    flux = flux_temp
                    rownum = rownum_temp
                stripe_flux[:,i] = flux
                stripe_rows[:,i] = rownum
        
        #parts of order missing at the bottom?
        elif np.min(contents[0]) == 0:
            print('some parts of the order are missing on at the BOTTOM of the chip...NIGHTMARE, THIS HAS NOT BEEN IMPLEMENTED YET') 
            quit()
        
    else:    
        #this is the "normal", easy part, where all (2*slit_height,4096) pixels of the stripe lie on the CCD 
        for i in range(len(col_indices)):
            #this is the cutout from the original image
            if i == len(col_indices)-1:
                flux = contents[2][col_indices[i]:]         #flux
                rownum = contents[0][col_indices[i]:]       #row number
            else:
                flux = contents[2][col_indices[i]:col_indices[i+1]]         #flux
                rownum = contents[0][col_indices[i]:col_indices[i+1]]       #row number
            stripe_flux[:,i] = flux
            stripe_rows[:,i] = rownum
    
    if timit:
        delta_t = time.time() - start_time
        print('Time taken for "flattening" stripe: '+str(delta_t)+' seconds...')
            
    return stripe_flux,stripe_rows.astype(int)
    


def flatten_stripes(stripes,slit_height=25):
    """
    CMB 27/09/2017
    
    For each stripe (ie order), this function stores the non-zero values of the sparse matrix "stripe" ("stripes" contains one "stripe" for each order)
    in a rectangular array, ie take out the curvature of the order/stripe, potentially only useful for further processing.

    INPUT:
    "stripes": dictionary containing the output from "extract_stripes", ie a sparse matrix for each stripe containing the respective non-zero elements
    
    OUTPUT:
    """

    order_boxes = {}
    
    # loop over all orders
    for ord in stripes.keys():
        #print(ord)
        stripe = stripes[ord]
        sc,sr = flatten_single_stripe(stripe,slit_height=slit_height)
        order_boxes[ord] = {'rows':sr, 'cols':sc}
    
    return order_boxes
        
        

def find_maxima(data, gauss_filter_sigma=0., min_peak=0.1, return_values=0):
    # smooth image slightly for noise reduction
    smooth_data = ndimage.gaussian_filter(data, gauss_filter_sigma)
    # find all local maxima
    peaks = np.r_[True, smooth_data[1:] > smooth_data[:-1]] & np.r_[smooth_data[:-1] > smooth_data[1:], True]
    # only use peaks higher than a certain threshold
    idx = np.logical_and(peaks, smooth_data > min_peak * np.max(smooth_data))
    maxix = np.arange(len(data))[idx]
    maxima = data[maxix]
    
    if return_values != 0:
        return maxix,maxima
    else:
        return maxix



def fit_single_fibre_profile(grid, data, pos=None, osf=1, fix_posns=False, method='leastsq', offset=False, debug_level=0, timit=False):
    #print('OK, pos: ',pos)
    if timit:
        start_time = time.time()
    
    if pos == None:
        #initial guess for the locations of the individual fibre profiles from location of maxima
        #maxix,maxval = find_maxima(data, return_values=1)
        maxix = np.where(data == np.max(data))[0]
        maxval = data[maxix]
    else:
        maxval = data[np.int(np.rint(pos-grid[0]))]
    
    # go to oversampled grid (the number of grid points should be: n_os = ((n_orig-1)*osf)+1
    if osf != 1:
        os_grid = np.linspace(grid[0],grid[-1],osf * (len(grid)-1) + 1)
        os_data = np.interp(os_grid, grid, data)
        grid = os_grid
        data = os_data
    
    #initial guesses in the right format for function "fibmodel_with_amp"
    if pos == None:
        guess = np.array([maxix[0]+grid[0], .7, maxval[0], 2.]).flatten()
    else:
        guess = np.array([pos, .7, maxval, 2.]).flatten()
    if offset:
        guess = np.append(guess,np.median(data))
       
    #create model function for fitting with LMFIT
    #model = Model(nineteen_fib_model_explicit)
    if not offset:
        model = Model(fibmodel_with_amp)
        guessmodel = fibmodel_with_amp(grid,*guess)
    else:
        model = Model(fibmodel_with_amp_and_offset)
        guessmodel = fibmodel_with_amp_and_offset(grid,*guess)
    
    #create instance of Parameters-class needed for fitting with LMFIT
    parms = Parameters()
    if fix_posns:
        parms.add('mu', guess[0], vary=False)
    else:
        parms.add('mu', guess[0], min=guess[0]-3*osf, max=guess[0]+3*osf)
    parms.add('sigma', guess[1], min=0.5, max=1.)
    parms.add('amp', guess[2], min=0.)
    parms.add('beta', guess[3], min=1., max=4.)
    if offset:
        parms.add('offset',guess[4], min=0., max=65535.)
    #parms.add('offset', guess[4], min=0.)
    #parms.add('slope', guess[5], min=-0.5, max=0.5)
    
    #perform fit
    result = model.fit(data,parms,xarr=grid,method=method)
    
    if debug_level >= 1:
        plot_osf = 10
        plot_os_grid = np.linspace(grid[0],grid[-1],plot_osf * (len(grid)-1) + 1)
        #plot_os_data = np.interp(plot_os_grid, grid, data)
        guessmodel = fibmodel_with_amp(plot_os_grid,*guess)
        bestparms = np.array([result.best_values['mu'], result.best_values['sigma'], result.best_values['amp'], result.best_values['beta']])
        bestmodel = fibmodel_with_amp(plot_os_grid,*bestparms)
        plt.plot(grid, data, 'bx')
        plt.plot(plot_os_grid, guessmodel, 'r--')
        plt.plot(plot_os_grid, bestmodel, 'g-')     
    
    if timit:
        print(time.time() - start_time, ' seconds')
    
    return result



def fitfib_single_cutout(grid,data,osf=1,method='leastsq',debug_level=0):
    
    #timing test
    start_time = time.time()
    
    nfib = 19
    
    #initial guess for the locations of the individual fibre profiles from location of maxima
    #NAIVE METHOD:
    maxix,maxval = find_maxima(data, return_values=1)
    
    if len(maxix) == len(maxix):
    #if len(maxix) != nfib:
        # smooth image slightly for noise reduction
        filtered_data = ndimage.gaussian_filter(data, 2.)
        #filtered_data = ndimage.gaussian_filter(data, 3.)
        maxval = np.r_[[.3]*6,[.7]*3,1,[.7]*3,[.3]*6] * np.max(maxval)
        
        #ALTERNATIVE 1: FIND HIGHEST MAXIMUM AND GO 2.5 PIXELS TO EACH SIDE FOR NEIGHBOURING FIBRES
        print('Did not find exactly 19 peaks! Using smoothed data to determine starting values for peak locations...')
        #top_max = np.where(filtered_data == np.max(filtered_data))[0]
        top_max = np.where(filtered_data == np.max(filtered_data))[0]
        ##top_max = np.where(data == np.max(data))[0]
        
#         #ALTERNATIVE 2:
#         #do a cross-correlation between the data and a "standard profile model" (with highest peak at grid[len(grid)/2] ) to find the rough position of the central peak, then go either side to get locations of other peaks
#         print('Did not find exactly 19 peaks! Performing a cross-correlation to determine starting values for peak locations...')
#         stdguess = np.append((np.array([np.arange(-22.5,22.51,2.5)+grid[len(grid)/2], maxval]).flatten()),[.7,2.])
#         stdmod = nineteen_fib_model_explicit_onesig_onebeta(grid,*stdguess)
#         xc = np.correlate(data,stdmod,mode='same')
#         #now fit gaussian + offset to CCF
#         pguess = [grid[len(grid)/2], 17., 2.7e7, 0.]
#         popt,pcov = curve_fit(gaussian_with_offset,grid,xc,p0=pguess)
#         ccf_fit = gaussian_with_offset(grid, *popt)
#         shift = grid[len(grid)/2] - popt[0]
#         
#         if debug_level > 0:
#             plt.figure()
#             plt.title('CCF')
#             plt.plot(grid,xc)
#             plt.plot(grid,gaussian_with_offset(grid,*pguess),'r--')
#             plt.plot(grid,gaussian_with_offset(grid,*popt),'g-')
#             plt.show()
#         
#         top_max = popt[0]
        
        
        #for ALTERNATIVE 1:
        maxix = np.arange(-22.5,22.51,2.5) + top_max
        #for ALTERNATIVE 2:
        #maxix = np.arange(-22.5,22.51,2.5) + top_max - grid[0]
        #maxval = np.r_[[.3]*6,[.7]*3,1,[.7]*3,[.3]*6] * filtered_data[top_max]     #already done above now
    
    # go to oversampled grid (the number of grid points should be: n_os = ((n_orig-1)*osf)+1
    if osf != 1:
        os_grid = np.linspace(grid[0],grid[-1],osf * (len(grid)-1) + 1)
        os_data = np.interp(os_grid, grid, data)
        grid = os_grid
        data = os_data
    
    #nur zur Erinnerung...
    #def fibmodel(xarr, mu, fwhm, beta=2, alpha=0, norm=0):

#     os_prof = np.zeros((len(maxix),len(os_grid)))
#     for i in range(len(maxix)):
#         os_prof[i,:] = fibmodel(os_grid, maxix[i]+col_rows[0], sigma=1.5, norm=1) * maxval[i]
#     os_prof_sum = np.sum(os_prof, axis=0)

    #initial guesses in the right format for function "multi_fib_model"
    #guess = np.array([maxix+grid[0], [0.7]*nfib, maxval, [2.]*nfib]).flatten()
    guess = np.append((np.array([maxix+grid[0], maxval]).flatten()),[.7,2.])
       
    #create model function for fitting with LMFIT
    #model = Model(nineteen_fib_model_explicit)
    model = Model(nineteen_fib_model_explicit_onesig_onebeta)
    #create instance of Parameters-class needed for fitting with LMFIT
    parms = Parameters()
    for i in range(nfib):
        parms.add('mu'+str(i+1), guess[i], min=guess[i]-3*osf, max=guess[i]+3*osf)
        #parms.add('sigma'+str(i+1), guess[nfib+i], min=0.)
        parms.add('amp'+str(i+1), guess[nfib+i], min=0.)
        #parms.add('beta'+str(i+1), guess[3*nfib+i], min=1.5, max=3.)
    parms.add('sigma', guess[2*nfib], min=0.)
    parms.add('beta', guess[2*nfib+1], min=1.5, max=3.)
    #perform fit
    result = model.fit(data,parms,x=grid,method=method)
    
    print(time.time() - start_time, ' seconds')

    return result



def multi_fib_model_star(x,*p):
    #determine number of fibres
    #nfib = len(p)/4
    nfib = 1
    
    #fill input-arrays for function "fibmodel"
    if nfib == 1:
        mu = p[0]
        sigma = p[1]
        amp = p[2]
        beta = p[3]
    else:
        mu = p[:nfib]
        sigma = p[nfib:nfib*2]
        amp = p[nfib*2:nfib*3]    
        beta = p[nfib*3:]
    
    if nfib == 1:
        model = fibmodel(x, mu, sigma, beta=beta, norm=1) * amp
    else:
        single_models = np.zeros((nfib, len(x)))
        for i in range(nfib):
            single_models[i,:] = fibmodel(x, mu[i], sigma[i], beta=beta[i], norm=1) * amp[i]
        model = np.sum(single_models, axis=0)
      
    return model
    
def multi_fib_model(x,p):
    #determine number of fibres
    nfib = len(p)/4
    #nfib = 19
    
    #fill input-arrays for function "fibmodel"
    if nfib == 1:
        mu = p[0]
        sigma = p[1]
        amp = p[2]
        beta = p[3]
    else:
        mu = p[:nfib]
        sigma = p[nfib:nfib*2]
        amp = p[nfib*2:nfib*3]    
        beta = p[nfib*3:]
    
#     print('nfib = ',nfib)
#     print('mu = ',mu)
#     print('sigma = ',sigma)
#     print('amp = ',amp)
#     print('beta = ',beta)
#     return 1
#    print('beta = ',beta)
#     #check if all the arrays provided have the same length
#     if len(fwhm) != len(mu):
#         print('ERROR: "fwhm" and "mu" must have the same length!!!')
#         quit()
#     if len(amp) != len(mu):
#         print('ERROR: "amp" and "mu" must have the same length!!!')
#         quit()
#     if len(fwhm) != len(amp):
#         print('ERROR: "fwhm" and "amp" must have the same length!!!')
#         quit()
    
#     #determine number of fibres
#     if isinstance(mu, collections.Sequence):
#         nfib = len(mu)
#     else:
#         nfib = 1
    
    if nfib == 1:
        model = fibmodel(x, mu, sigma, beta=beta, norm=1) * amp
    else:
        single_models = np.zeros((nfib, len(x)))
        for i in range(nfib):
            single_models[i,:] = fibmodel(x, mu[i], sigma[i], beta=beta[i], norm=1) * amp[i]
        model = np.sum(single_models, axis=0)
      
    return model
    
def nineteen_fib_model_explicit(x, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15, mu16, mu17, mu18, mu19,
                                sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9, sigma10, sigma11, sigma12, sigma13, sigma14, sigma15, sigma16, sigma17, sigma18, sigma19,
                                amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8, amp9, amp10, amp11, amp12, amp13, amp14, amp15, amp16, amp17, amp18, amp19,
                                beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8, beta9, beta10, beta11, beta12, beta13, beta14, beta15, beta16, beta17, beta18, beta19):
    
    nfib=19
    
    mu = np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15, mu16, mu17, mu18, mu19])
    sigma= np.array([sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9, sigma10, sigma11, sigma12, sigma13, sigma14, sigma15, sigma16, sigma17, sigma18, sigma19])
    amp = np.array([amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8, amp9, amp10, amp11, amp12, amp13, amp14, amp15, amp16, amp17, amp18, amp19])
    beta = np.array([beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8, beta9, beta10, beta11, beta12, beta13, beta14, beta15, beta16, beta17, beta18, beta19])
    
    single_models = np.zeros((nfib, len(x)))
    for i in range(nfib):
        single_models[i,:] = fibmodel(x, mu[i], sigma[i], beta=beta[i], norm=0) * amp[i]
    model = np.sum(single_models, axis=0)
      
    return model
   
def nineteen_fib_model_explicit_onesig_onebeta(x, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15, mu16, mu17, mu18, mu19,
                                               amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8, amp9, amp10, amp11, amp12, amp13, amp14, amp15, amp16, amp17, amp18, amp19, sigma, beta):
    
    nfib=19
    
    mu = np.array([mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9, mu10, mu11, mu12, mu13, mu14, mu15, mu16, mu17, mu18, mu19])
    #sigma= np.array([sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8, sigma9, sigma10, sigma11, sigma12, sigma13, sigma14, sigma15, sigma16, sigma17, sigma18, sigma19])
    amp = np.array([amp1, amp2, amp3, amp4, amp5, amp6, amp7, amp8, amp9, amp10, amp11, amp12, amp13, amp14, amp15, amp16, amp17, amp18, amp19])
    #beta = np.array([beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8, beta9, beta10, beta11, beta12, beta13, beta14, beta15, beta16, beta17, beta18, beta19])
    
    single_models = np.zeros((nfib, len(x)))
    for i in range(nfib):
        single_models[i,:] = fibmodel(x, mu[i], sigma, beta=beta, norm=0) * amp[i]
    model = np.sum(single_models, axis=0)
      
    return model



def fit_profiles_single_order(stripe_rows, stripe_columns, ordpol, osf=1, method='leastsq', offset=False, timit=False, silent=False):
    
    if not silent:
        choice = None
        while choice is None:
            choice = raw_input("WARNING: Fitting the fibre profiles for an entire order currently takes more than 2 hours. Do you want to continue? [y/n]: ")
            if choice not in ('y','n'):
                print('Invalid input! Please try again...')
                choice = None
    else:
        choice = 'y'
            
    if choice == 'n':
        print('OK, stopping script...')
        quit()
    else:         
        
        if timit:
            start_time = time.time()
        print('Fitting fibre profiles for one order...') 
        #loop over all columns for one order and do the profile fitting
        #'colfits' is a dictionary, that has 4096 keys. Each key is an instance of the 'ModelResult'-class from the 'lmfit' package
        colfits = {}
        npix = sc.shape[1]
        fiblocs = np.poly1d(ordpol)(np.arange(npix))
        #starting_values = None
        for i in range(npix):
            #fit_result = fitfib_single_cutout(stripe_rows[:,i], stripe_columns[:,i], osf=osf, method=method)
            if not silent:
                print('i = ',str(i))
            fu = 0
            #check if that particular cutout falls fully onto CCD
            checkprod = np.product(stripe_rows[1:,i])    #exclude the first row number, as that can legitimately be zero
            if checkprod == 0:
                fu = 1
                checksum = np.sum(stripe_rows[:,i])
                if checksum == 0:
                    print('WARNING: the entire cutout lies outside the chip!!!')
                    #fit_result = fit_single_fibre_profile(np.arange(len(stripe_rows[:,i])),stripe_columns[:,i],guess=None)
                    best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
                else:
                    print('WARNING: parts of the cutout lie outside the chip!!!')
                    #fit_result = fit_single_fibre_profile(np.arange(len(stripe_rows[:,i])),stripe_columns[:,i],guess=None)
                    best_values = {'amp':-1., 'beta':-1., 'mu':-1., 'sigma':-1.}
            else:    
                #fit_result = fit_single_fibre_profile(stripe_rows[:,i],stripe_columns[:,i],guess=None,timit=1)
                fit_result = fit_single_fibre_profile(stripe_rows[:,i],stripe_columns[:,i]-1.,pos=fiblocs[i],offset=offset,timit=0)     #the minus 1 is necessary because we added an offset to start with, due to stripes needing that
            #starting_values = fit_result.best_values
            #colfits['col_'+str(i)] = fit_result
            if fu == 1:
                for keyname in best_values.keys():
                    try:
                        colfits[keyname].append(best_values[keyname])
                    except KeyError:
                        colfits[keyname] = [best_values[keyname]]
            else:
                for keyname in fit_result.best_values.keys():
                    try:
                        colfits[keyname].append(fit_result.best_values[keyname])
                    except KeyError:
                        colfits[keyname] = [fit_result.best_values[keyname]]
    
    if timit:
        print('Elapsed time for fitting profiles to a single order: '+str(time.time() - start_time)+' seconds...')
    
    return colfits



def fit_profiles(P_id, stripes, timit=False):
    
    print('Fitting fibre profiles...')
    
    if timit:
        start_time = time.time()
        
    #create "global" parameter dictionary for entire chip
    fibre_profiles = {}
    #loop over all orders
    for ord in sorted(P_id.iterkeys()):
        print('OK, now processing '+str(ord))
        ordpol = P_id[ord]
        
        # define stripe
        stripe = stripes[ord]
        # find the "order-box"
        sc,sr = flatten_single_stripe(stripe,slit_height=10,timit=False)
        # fit profile for single order and save result in "global" parameter dictionary for entire chip
        colfits = fit_profiles_single_order(sr,sc,ordpol,osf=1,silent=True,timit=timit)
        fibre_profiles[ord] = colfits
    
    if timit:
        print('Time elapsed: '+str(int(time.time() - start_time))+' seconds...')  
          
    return fibre_profiles



def test():    
    
    

# np.save('/Users/christoph/UNSW/fibre_profiles/masks/mask_01.npy',mask)
    
#     stripe = stripes_03[ord]
#     ordpol = P_id_03[ord]
#     sc,sr = flatten_single_stripe(stripe,slit_height=10,timit=False)
#     colfits = fit_profiles_single_order(sr,sc,ordpol,osf=1,silent=1,timit=timit)
#     fibre_profiles_03 = {}
#     fibre_profiles_03[ord] = colfits
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_03_temp.npy', fibre_profiles_03) 
#     
#     stripe = stripes_21[ord]
#     ordpol = P_id_21[ord]
#     sc,sr = flatten_single_stripe(stripe,slit_height=10,timit=False)
#     colfits = fit_profiles_single_order(sr,sc,ordpol,osf=1,silent=1,timit=timit)
#     fibre_profiles_21 = {}
#     fibre_profiles_21[ord] = colfits
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_21_temp.npy', fibre_profiles_21)
#     
#     stripe = stripes_22[ord]
#     ordpol = P_id_22[ord]
#     sc,sr = flatten_single_stripe(stripe,slit_height=10,timit=False)
#     colfits = fit_profiles_single_order(sr,sc,ordpol,osf=1,silent=1,timit=timit)
#     fibre_profiles_22 = {}
#     fibre_profiles_22[ord] = colfits
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_22_temp.npy', fibre_profiles_22)
    
    
#     fibre_profiles_02 = fit_profiles(img02, P_id_02, stripes_02)
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_02.npy', fibre_profiles_02) 
#     fibre_profiles_03 = fit_profiles(img03, P_id_03, stripes_03)
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_03.npy', fibre_profiles_03) 
#     fibre_profiles_21 = fit_profiles(img21, P_id_21, stripes_21)
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_21.npy', fibre_profiles_21) 
#     fibre_profiles_22 = fit_profiles(img22, P_id_22, stripes_22)
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_22.npy', fibre_profiles_22) 
#     
#     fibre_profiles_02 = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_02.npy').item()
#     fibre_profiles_03 = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_03.npy').item()
#     mu02 = np.array(fibre_profiles_02['order_10']['mu'])
#     mu03 = np.array(fibre_profiles_03['order_10']['mu'])
#     amp02 = np.array(fibre_profiles_02['order_10']['amp'])
#     amp03 = np.array(fibre_profiles_03['order_10']['amp'])
#     chi2red_02 = np.array(fibre_profiles_02['order_10']['chi2red'])
#     chi2red_03 = np.array(fibre_profiles_03['order_10']['chi2red'])
    
#     flat01name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib01.fit'
#     flat02name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib02.fit'
#     flat03name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib03.fit'
#     flat04name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib04.fit'
#     flat05name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib05.fit'
#     flat06name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib06.fit'
#     flat07name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib07.fit'
#     flat08name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib08.fit'
#     flat09name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib09.fit'
#     flat01 = pyfits.getdata(flat01name)
#     flat02 = pyfits.getdata(flat02name)
#     flat03 = pyfits.getdata(flat03name)
#     flat04 = pyfits.getdata(flat04name)
#     flat05 = pyfits.getdata(flat05name)
#     flat06 = pyfits.getdata(flat06name)
#     flat07 = pyfits.getdata(flat07name)
#     flat08 = pyfits.getdata(flat08name)
#     flat09 = pyfits.getdata(flat09name)
#     img01 = flat01 + 1.
#     img02 = flat02 + 1.
#     img03 = flat03 + 1.
#     img04 = flat04 + 1.
#     img05 = flat05 + 1.
#     img06 = flat06 + 1.
#     img07 = flat07 + 1.
#     img08 = flat08 + 1.  
#     img09 = flat09 + 1.  
    
    
# for i in range(28):
#     print(i)
#     
#     flatname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib'+str(i+1).zfill(2)+'.fit'
#     flat = pyfits.getdata(flatname)
#     img = flat + 1.
#       
#     P,tempmask = find_stripes(flat, deg_polynomial=2)
#     P_id = make_P_id(P)
#     mask = make_mask_dict(tempmask)
#     np.save('/Users/christoph/UNSW/fibre_profiles/masks/mask_'+str(i+1).zfill(2)+'.npy', mask)
    
#     stripes = extract_stripes(img, P_id, slit_height=10)
#     fibre_profiles = fit_profiles(P_id, stripes)   
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_'+str(i+1).zfill(2)+'.npy', fibre_profiles)
     
     
     
     
     
#     fibre_profiles_02 = fit_profiles(img02, P_id_02, stripes_02)
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_02.npy', fibre_profiles_02) 
#     fibre_profiles_03 = fit_profiles(img03, P_id_03, stripes_03)
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_03.npy', fibre_profiles_03) 
#     
#     fibre_profiles_21 = fit_profiles(img21, P_id_21, stripes_21)
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_21.npy', fibre_profiles_21) 
#     fibre_profiles_22 = fit_profiles(img22, P_id_22, stripes_22)
#     np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_22.npy', fibre_profiles_22) 

    
#     # do Polynomial fit 
#     xx = np.arange(4096)
#     
# #     p2 = np.poly1d(np.polyfit(xx, mu03, 2))
# #     p3 = np.poly1d(np.polyfit(xx, mu03, 3))
# #     p4 = np.poly1d(np.polyfit(xx, mu03, 4))
# #     p5 = np.poly1d(np.polyfit(xx, mu03, 5))
# #     res2 = mu03 - p2(xx)
# #     res3 = mu03 - p3(xx)
# #     res4 = mu03 - p4(xx)
# #     res5 = mu03 - p5(xx)
#     w = np.sqrt(amp)
#     p = np.poly1d(np.polyfit(xx, mu03, 5, w=w))
#     res = p(xx) - mu03
#     rms = np.sqrt( np.sum(res*res)/len(res) )
#     
#     
#     #mu02 = fibre_profiles_02['order_10']['mu']
#     #mu03 = fibre_profiles_03['order_10']['mu']
#     #mu21 = fibre_profiles_21['order_10']['mu']
#     #mu22 = fibre_profiles_22['order_10']['mu']
#     
#     mu21 = mu03.copy() - (18*diff) + np.random.normal(0,rms,4096)
#     mu22 = mu03.copy() - (19*diff) + np.random.normal(0,rms,4096)
    return 1


def find_tramlines_single_order(uu, ul, lu, ll, mask_uu, mask_ul, mask_lu, mask_ll):
    
    #make sure they're all the same length
    if (len(uu) != len(ul)) or (len(uu) != len(lu)) or (len(uu) != len(ll)):
        print('ERROR: Dimensions of input arrays do not agree!!!')
        quit()
        
    #fit 5th-order polynomial to each peak-array, with the weights of the fit being the RMS of the initial fit to the fibre profiles!?!?!?
    #WARNING: do unweighted for now...
    xx = np.arange(len(uu))
    
    # for the subsequent fit we only want to take the parts of the order, for which all the order traces lie on the chip
    # (this works b/c all fit parameters are set to -1 if any part of the cutout lies outside the chip; this is done in "fit_profiles_single_order")
    #good = np.logical_and(np.logical_and(uu>=0, ul>=0), np.logical_and(lu>=0, ll>=0))
    upper_good = np.logical_and(uu>=0, ul>=0)     #this removes the parts of the order where the fitted mu-values of any of the two upper fibres fall outside the chip
    lower_good = np.logical_and(lu>=0, ll>=0)     #this removes the parts of the order where the fitted mu-values of any of the two lower fibres fall outside the chip
    upper_mask = np.logical_and(upper_good, np.logical_and(mask_uu,mask_ul))     #this uses the mask from "find_stripes", ie also removes the low-flux regions at either side of some orders
    lower_mask = np.logical_and(lower_good, np.logical_and(mask_lu,mask_ll))     #this uses the mask from "find_stripes", ie also removes the low-flux regions at either side of some orders
    
    #w = np.sqrt(amp)
    #p = np.poly1d(np.polyfit(xx, mu03, 5,w=w))
    p_uu = np.poly1d(np.polyfit(xx[upper_mask], uu[upper_mask], 5))
    p_ul = np.poly1d(np.polyfit(xx[upper_mask], ul[upper_mask], 5))
    p_lu = np.poly1d(np.polyfit(xx[lower_mask], lu[lower_mask], 5))
    p_ll = np.poly1d(np.polyfit(xx[lower_mask], ll[lower_mask], 5))
    
#     res02 = p02(xx) - mu02
#     res03 = p03(xx) - mu03
#     res21 = p21(xx) - mu21
#     res22 = p22(xx) - mu22
#      
#     rms02 = np.sqrt( np.sum(res02*res02)/len(res02) )
#     rms03 = np.sqrt( np.sum(res03*res03)/len(res03) )
#     rms21 = np.sqrt( np.sum(res21*res21)/len(res21) )
#     rms22 = np.sqrt( np.sum(res22*res22)/len(res22) )
    
    #but define the boundaries for the entire order, ie also for the "bad" parts
    upper_boundary = 0.5 * (p_uu(xx) + p_ul(xx))
    lower_boundary = 0.5 * (p_lu(xx) + p_ll(xx))
    
    
    return upper_boundary, lower_boundary



def find_tramlines(fp_uu, fp_ul, fp_lu, fp_ll, mask_uu, mask_ul, mask_lu, mask_ll, debug_level=0, timit=False):
    '''
    INPUT: 
    P_id
    four single-fibre fibre-profiles-dictionaries, from fitting the single fibres for each order and cutout
    '''
    
    if timit:
        start_time = time.time()
    
    #make sure they're all the same length
    if (len(fp_uu) != len(fp_ul)) or (len(fp_uu) != len(fp_lu)) or (len(fp_uu) != len(fp_ll)):
        print('ERROR: Dimensions of input dictionaries do not agree!!!')
        quit()
    
    tramlines = {}
    
    for ord in sorted(fp_uu.iterkeys()):
        uu = np.array(fp_uu[ord]['mu'])
        ul = np.array(fp_ul[ord]['mu'])
        lu = np.array(fp_lu[ord]['mu'])
        ll = np.array(fp_ll[ord]['mu'])
        upper_boundary, lower_boundary = find_tramlines_single_order(uu, ul, lu, ll, mask_uu[ord], mask_ul[ord], mask_lu[ord], mask_ll[ord])
        tramlines[ord] = {'upper_boundary':upper_boundary, 'lower_boundary':lower_boundary}
    
    if debug_level >= 1:
        xx = np.arange(4096)
        plt.figure()
        plt.imshow(img, origin='lower', norm=LogNorm())
        for ord in tramlines.keys():
            plt.plot(xx, tramlines[ord]['upper_boundary'],'y-')
            plt.plot(xx, tramlines[ord]['lower_boundary'],'r-')
    
    if timit:
        print('Time taken for finding extraction tramlines: '+str(time.time() - start_time)+' seconds...')    
    
    return tramlines



def find_laser_tramlines_single_order(mu, mask):
        
    #fit 5th-order polynomial to each peak-array, with the weights of the fit being the RMS of the initial fit to the fibre profiles!?!?!?
    #WARNING: do unweighted for now...
    xx = np.arange(len(mu))
    
    # for the subsequent fit we only want to take the parts of the order, for which all the order traces lie on the chip
    # (this works b/c all fit parameters are set to -1 if any part of the cutout lies outside the chip; this is done in "fit_profiles_single_order")
    #good = np.logical_and(np.logical_and(uu>=0, ul>=0), np.logical_and(lu>=0, ll>=0))
    good = np.logical_and(mu>=0, mask)     #this removes the parts of the order where the fitted mu-values of any of the two upper fibres fall outside the chip
    
    #w = np.sqrt(amp)
    #p = np.poly1d(np.polyfit(xx, mu03, 5,w=w))
    p_mu = np.poly1d(np.polyfit(xx[good], mu[good], 5))
    
#     res02 = p02(xx) - mu02
#     res03 = p03(xx) - mu03
#     res21 = p21(xx) - mu21
#     res22 = p22(xx) - mu22
#      
#     rms02 = np.sqrt( np.sum(res02*res02)/len(res02) )
#     rms03 = np.sqrt( np.sum(res03*res03)/len(res03) )
#     rms21 = np.sqrt( np.sum(res21*res21)/len(res21) )
#     rms22 = np.sqrt( np.sum(res22*res22)/len(res22) )
    
    #but define the boundaries for the entire order, ie also for the "bad" parts
    upper_boundary = p_mu(xx) + 3
    lower_boundary = p_mu(xx) - 3
    
    
    return upper_boundary, lower_boundary



def find_laser_tramlines(fp, mask, debug_level=0, timit=False):
    
    if timit:
        start_time = time.time()
    
    tramlines = {}
    
    for ord in sorted(fp.iterkeys()):
        mu = np.array(fp[ord]['mu'])
        upper_boundary, lower_boundary = find_laser_tramlines_single_order(mu, mask[ord])
        tramlines[ord] = {'upper_boundary':upper_boundary, 'lower_boundary':lower_boundary}
    
    if debug_level >= 1:
        xx = np.arange(4096)
        plt.figure()
        plt.imshow(img, origin='lower', norm=LogNorm())
        for ord in tramlines.keys():
            plt.plot(xx, tramlines[ord]['upper_boundary'],'y-')
            plt.plot(xx, tramlines[ord]['lower_boundary'],'r-')
    
    if timit:
        print('Time taken for finding extraction tramlines: '+str(time.time() - start_time)+' seconds...')    
    
    return tramlines



def WIP():
# parms = Parameters()
# for i in range(76):
#     parms.add('p'+str(i), value=guess[i])
# 
# def multi_fib_model_res(p,x,y):
#     model = multi_fib_model(x,p)
#     return model - data

    








    return 1
