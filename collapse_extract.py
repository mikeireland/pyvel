'''
Created on 31 Oct. 2017

@author: christoph
'''

import numpy as np
import time
from veloce_reduction.order_tracing import flatten_single_stripe

#needs "stripes" and "fibre_profiles_**" as an input (from order_tracing.py)

def collapse_extract_single_cutout(cutout, top, bottom, RON=4., gain=1.):
    
    x = np.arange(len(cutout))
    inner_range = np.logical_and(x >= np.ceil(bottom), x <= np.floor(top))
    top_frac = top - np.floor(top)
    bottom_frac = np.ceil(bottom) - bottom
    flux = gain * ( np.sum(cutout[inner_range]) + top_frac * cutout[int(np.ceil(top))] + bottom_frac * cutout[int(np.floor(bottom))] )
    n = np.sum(inner_range)     # as in my thesis; sum is fine because inner_range is boolean
    w = n + 2                   # as in my thesis
    err = np.sqrt(flux + w*RON*RON)
    
    return flux, err



def collapse_extract_order(ordnum, data, row_ix, upper_boundary, lower_boundary, RON=4., gain=1.):
    
    flux,err = (np.zeros(len(upper_boundary)),np.zeros(len(upper_boundary)))
    pixnum = []
    
    for i in range(data.shape[1]):
        pixnum.append(ordnum+str(i+1).zfill(4))
        cutout = data[:,i]
        top = upper_boundary[i] - row_ix[0,i]
        bottom = lower_boundary[i] - row_ix[0,i]
        if top>=0 and bottom>=0:
            if top<=data.shape[0] and bottom <=data.shape[0] and top>bottom:
                # this is the normal case, where the entire cutout lies on the chip
                f,e = collapse_extract_single_cutout(cutout, top, bottom, RON=RON, gain=gain)
            else:
                print('ERROR: Tramlines are not properly defined!!!')
                quit()
        else:
            # just output zero if any bit of the cutout falls outside the chip
            f,e = (0.,0.)
        flux[i] = f
        err[i] = e
    
    return pixnum, flux, err



def collapse_extract(stripes, tramlines, RON, gain, laser=False, verbose=False, timit=False):
    
    if verbose:
        print('Collapsing and extracting orders...')
    
    if timit:    
        start_time = time.time()
    
    flux = {}
    err = {}
    pixnum = {}
    
#     for ord in stripes.keys():
    for ord in tramlines.keys():
        if verbose:
            print('OK, now processing order '+str(ord)+'...')
        if timit:
            order_start_time = time.time()
        
        #order number
        ordnum = ord[-2:]
        
        # define stripe
        stripe = stripes[ord]
        # find the "order-box"
        if laser:
            sc,sr = flatten_single_stripe(stripe,slit_height=10,timit=False)
        else:
            sc,sr = flatten_single_stripe(stripe,slit_height=25,timit=False)
        # define upper and lower extraction boundaries
        upper_boundary = tramlines[ord]['upper_boundary']
        lower_boundary = tramlines[ord]['lower_boundary']
        # call extraction routine for this order (NOTE the sc-1 is becasue we added 1 artificially at the beginning in order for extract_stripes tow work properly)
        pix,f,e = collapse_extract_order(ordnum, sc-1, sr, upper_boundary, lower_boundary, RON=RON, gain=gain)
        
        flux[ord] = f
        err[ord] = e
        pixnum[ord] = pix
    
        if timit:
            print('Time taken for extraction of '+ord+': '+str(time.time() - order_start_time)+' seconds')
    
    
    if timit:
        print('Time taken for extraction of spectrum: '+str(time.time() - start_time)+' seconds')
    
    if verbose:
        print('Extraction complete! Have a coffee now...')
    
    return pixnum,flux,err


