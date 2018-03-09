'''
Created on 21 Nov. 2017

@author: christoph
'''

import numpy as np
from veloce_reduction.helper_functions import fibmodel, fibmodel_with_amp_and_offset,\
    blaze
from veloce_reduction.order_tracing import *

#read in polynomial coefficients of best-fit individual-fibre-profile parameters
fibparms = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibparms_by_ord.npy').item()



def make_norm_profiles(x, ord, col, fibparms, slope=False, offset=False):  
    
    xx = np.arange(4096)
    
    #same number of fibres for every order, of course
    nfib = len(fibparms['order_02'])   
    
    #do we want to include extra "fibres" to take care of slope and/or offset? Default is NO for both (as this should be already taken care of globally)
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),nfib))
    
    for k,fib in enumerate(sorted(fibparms[ord].iterkeys())):
        mu = fibparms[ord][fib]['mu_fit'](col)
        sigma = fibparms[ord][fib]['sigma_fit'](col)
        beta = fibparms[ord][fib]['beta_fit'](col)
        phi[:,k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized profiles
    return phi/np.sum(phi,axis=0)


def make_norm_profiles_temp(x, ord, col, fibparms, slope=False, offset=False):  
    
    xx = np.arange(4096)
    
    #same number of fibres for every order, of course
    nfib = 19
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),nfib))
    
    
    mu = fibparms[ord]['fibre_03']['mu_fit'](col)
    sigma = fibparms[ord]['fibre_03']['sigma_fit'](col)
    beta = fibparms[ord]['fibre_03']['beta_fit'](col)
    for k in range(nfib):
        phi[:,k] = fibmodel(x, mu-k*1.98, sigma, beta=beta, alpha=0, norm=0)
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized_profiles
    return phi/np.sum(phi,axis=0)



def make_norm_single_profile_temp(x, ord, col, fibparms, slope=False, offset=False):  

    
    #same number of fibres for every order, of course
    nfib = 1
    if offset:
        nfib += 1
    if slope:
        nfib += 1
    
    phi = np.zeros((len(x),nfib))
    
    
    mu = fibparms[ord]['mu'][col]
    sigma = fibparms[ord]['sigma'][col]
    beta = fibparms[ord]['beta'][col]
    for k in range(nfib):
        phi[:,k] = fibmodel(x, mu, sigma, beta=beta, alpha=0, norm=0)
    
    if offset:
        phi[:,-2] = 1.
    if slope:
        phi[:,-1] = x - x[0]
    
    #return normalized_profiles
    return phi/np.sum(phi,axis=0)



def optimal_extraction(img, P_id, stripes, stripe_indices, gain=1., RON=4., onthefly=False, timit=False, simu=False, individual_fibres=False, debug_level=0):
    
    if timit:
        start_time = time.time()
    
    #read in polynomial coefficients of best-fit individual-fibre-profile parameters
    #fibparms = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibparms_by_ord.npy').item()
    #fibparms = np.load('/Users/christoph/UNSW/fibre_profiles/real/first_real_veloce_test_fps.npy').item()
    fibparms = np.load('/Users/christoph/UNSW/fibre_profiles/real/from_master_white.npy').item()
    #nfib = len(fibparms['order_01'])
    nfib = 19
    
    flux = {}
    err = {}
    pixnum = {}
    
#     for ord in sorted(P_id.iterkeys()):
    for ord in sorted(P_id.iterkeys()):
        if debug_level > 0:
            print('Processing order '+ord)
        if timit:
            order_start_time = time.time()
        
        
        #order number
        ordnum = ord[-2:]
        print('OK, now processing order: '+ordnum)
        ordpol = P_id[ord]
        
        # define stripe
        stripe = stripes[ord]
        indices = stripe_indices[ord]
        # find the "order-box"
        #sc,sr = flatten_single_stripe(stripe,slit_height=25,timit=False)
        sc,sr = flatten_single_stripe_from_indices(img,indices,slit_height=25,timit=False)
        
        npix = sc.shape[1]
        pix=[]
        if individual_fibres:
            f_ord = np.zeros((nfib,npix))
            e_ord = np.zeros((nfib,npix))
        else:
            f_ord = np.zeros(npix)
            e_ord = np.zeros(npix)
        
        goodrange = np.arange(npix)
        if simu and ord == 'order_01':
            #goodrange = goodrange[fibparms[ord]['fibre_21']['onchip']]
            goodrange = np.arange(1200,4096)
            for j in range(1200):
                pix.append(ordnum+str(j+1).zfill(4))
                    
        for i in goodrange:
            if debug_level > 0:
                print(str(i+1)+'/'+str(npix))
            pix.append(ordnum+str(i+1).zfill(4))
            z = sc[:,i]
            if simu:
                z -= -1.     #note the minus 1 is because we added 1 artificially at the beginning in order for extract_stripes to work properly
            pixerr = np.sqrt( RON*RON + np.abs(z) )
            #assign weights for flux (and take care of NaNs and INFs)
            pix_w = 1./(pixerr*pixerr)
            
            if onthefly:
                quickfit = fit_single_fibre_profile(sr[:,i],z)
                bestparms = np.array([quickfit.best_values['mu'], quickfit.best_values['sigma'], quickfit.best_values['amp'], quickfit.best_values['beta']])
                phi = fibmodel_with_amp(sr[:,i],*bestparms)
                phi /= np.max([np.sum(phi),0.001])   #we can do this because grid-stepsize = 1; also make sure that we do not divide by zero
                phi = phi.reshape(len(phi),1)   #stupid python...
            else:
                #get normalized profiles for all fibres for this cutout
                #phi = make_norm_profiles(sr[:,i], ord, i, fibparms)
                #phi = make_norm_profiles_temp(sr[:,i], ord, i, fibparms)
                phi = make_norm_single_profile_temp(sr[:,i], ord, i, fibparms)
            
#             print('WARNING: TEMPORARY offset correction is not commented out!!!')
#             #subtract the median as the offset if BG is not properly corrected for
#             z -= np.median(z)
            
            #do the optimal extraction
            if np.sum(phi)==0:
                f,v = (0.,np.sqrt(len(phi)*RON*RON))
            else:
                f,v = linalg_extract_column(z, pix_w, phi, altvar=1)
            #e = np.sqrt(v)
            #model = np.sum(f*phi,axis=1)
        
            #UNLESS YOU WANT TO EXTRACT THE SPECTRUM FOR INDIVIDUAL FIBRES!!!
            if not onthefly:
                if individual_fibres:   
                    #there should not be negative values!!!
                    f[f<0] = 0.
                    f_ord[:,i] = f
                    #not sure if this is the proper way to do this, but we can't have negative variance
                    v[v<0] = 1.
                    e_ord[:,i] = np.sqrt(v)
                else:
                    #there should not be negative values!!!
                    f[f<0] = 0.
                    f_ord[i] = np.sum(f)
                    #not sure if this is the proper way to do this, but we can't have negative variance
                    v[v<0] = 1.
                    e_ord[i] = np.sqrt(np.sum(v))
            else:
                f_ord[i] = np.max([f,0.])
                if f_ord[i] == 0 or v <= 0:
                    e_ord[i] = np.sqrt(len(phi)*RON*RON)
                else:
                    e_ord[i] = np.sqrt(v)
         
            
        flux[ord] = f_ord
        err[ord] = e_ord
        pixnum[ord] = pix        
        
        if timit:
            print('Time taken for extraction of '+ord+': '+str(time.time() - order_start_time)+' seconds')
                
                
    if timit:
        print('Time elapsed for optimal extraction of entire spectrum: '+str(time.time() - start_time)+' seconds...')  

    return pixnum,flux,err



def linalg_extract_column(z, w, phi, gain=1., RON=4.,altvar=0):
    
#     #create phi-array (ie the array containing the individual fibre profiles (=phi_i))
#     phi = np.zeros((len(y), nfib))
#     #loop over all fibres
#     for i in range(nfib):
#         phi[:,i] = fibmodel([pos[i], fwhm[i], beta], y)

#     #assign artificial "uncertainties" as sqrt(flux)+RON (not realistic!!!)
#     err = 10. + np.sqrt(z)
#     #assign weights for flux (and take care of NaNs and INFs)
#     w = err**(-2)
#     w[np.isinf(w)]=0
#     w[np.isnan(w)]=0

    #create diagonal matrix for weights
    ### XXX maybe use sparse matrix here instead to save memory / speed things up
    w_mat = np.diag(w)

    #create the cross-talk matrix
    #C = np.transpose(phi) @ w_mat @ phi
    C = np.matmul( np.matmul(np.transpose(phi),w_mat), phi )
    

    #compute b
    btemp = phi * np.transpose(np.array([z]))     
    #this reshaping is necessary so that z has shape (4096,1) instead of (4096,), which is needed
    # for the broadcasting (ie singelton expansion) functionality
    b = sum( btemp * np.transpose(np.array([w])) )
    #alternatively: b = sum( btemp / (np.transpose(np.array([err**2]))) )
    

    #compute eta (ie the array of the fibre-intensities (or amplitudes)
    ### XXX check if that can be made more efficient
    C_inv = np.linalg.inv(C)
    #eta =  C_inv @ b
    eta = np.matmul(C_inv,b)
    #the following line does the same thing, but contrary to EVERY piece of literature I find it is NOT faster
    #np.linalg.solve(C, b)
    
    #retrieved = eta * phi
    
    
    
    if altvar == 1:
        C_prime = np.matmul(np.transpose(phi),phi)
        C_prime_inv = np.linalg.inv(C_prime)
        b_prime = sum( phi * np.transpose(np.array([z-RON*RON])) )     #that z here should be the variance actually, but input weights should be from error array that leaves error bars high when cosmics are removed
        var = np.matmul(C_prime_inv,b_prime)
    elif altvar == 0:
        #convert error from variance to proper error
        eta_err = np.sqrt(abs(eta))
        var = eta_err ** 2
        
    
    
    #return np.array([eta, eta_err, retrieved])   
    return np.array([eta, var])   




