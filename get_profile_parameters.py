'''
Created on 21 Nov. 2017

@author: christoph
'''

import glob, os
import astropy.io.fits as pyfits
from veloce_reduction.helper_functions import *
from veloce_reduction.order_tracing import *


def make_fibparms_by_fib(savefile=True):

    path = '/Users/christoph/UNSW/fibre_profiles/'
    fp_files = glob.glob(path+"sim/"+"fibre_profiles*.npy")
    #mask_files = glob.glob(path+"masks/"+"mask*.npy")
    
    fibparms = {}
    
    for file in fp_files:
        fibnum = file[-6:-4]
        fib = 'fibre_'+fibnum
        fp = np.load(file).item()
        mask = np.load(path+"masks/"+"mask_"+fibnum+".npy").item()
        flatname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib'+fibnum+'.fit'
        flat = pyfits.getdata(flatname)
        img = flat + 1.
    
        #we need the flux later only to weight the fits
        P,tempmask = find_stripes(flat, deg_polynomial=2)
        P_id = make_P_id(P)
        #mask = make_mask_dict(tempmask)
        stripes = extract_stripes(img, P_id, slit_height=10)
    
        fibparms[fib] = {}
        for ord in sorted(fp.iterkeys()):
            
            fibparms[fib][ord] = {}
            
            mu = np.array(fp[ord]['mu'])
            #amp = np.array(fp[ord]['amp'])
            sigma = np.array(fp[ord]['sigma'])
            beta = np.array(fp[ord]['beta'])
            #offset = np.array(fp[ord]['offset'])
            #slope = np.array(fp[ord]['slope'])
            
            onchip = mu>=0
            good = np.logical_and(mu>=0, mask[ord])
            
            stripe = stripes[ord]
            sc,sr = flatten_single_stripe(stripe,slit_height=10,timit=False)
            fluxsum = np.sum(sc,axis=0)
            w = np.sqrt(fluxsum)   #use relative error so that the order centres receive the largest weight-contribution
            
            xx = np.arange(len(mu))
            mu_fit = np.poly1d(np.polyfit(xx[good], mu[good], 5, w=w[good]))
            # xarr = xx*4*np.pi/np.max(xx) - 2*np.pi
            # amp_popt,amp_pcov = curve_fit(blaze,xarr,amp,sigma=w,p0=(0.2,np.max(amp),0.))
            sigma_fit = np.poly1d(np.polyfit(xx[good], sigma[good], 2, w=w[good]))
            beta_fit = np.poly1d(np.polyfit(xx[good], beta[good], 2, w=w[good]))
            #offset_fit = np.poly1d(np.polyfit(xx[good], offset[good], 0, w=w[good]))
            #offset_fit = np.average(offset, weights=w)
        
            fibparms[fib][ord]['mu_fit'] = mu_fit
            fibparms[fib][ord]['sigma_fit'] = sigma_fit
            fibparms[fib][ord]['beta_fit'] = beta_fit
            #fibparms[fib][ord]['offset_fit'] = offset_fit
            fibparms[fib][ord]['onchip'] = onchip
    
    if savefile:
        np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibparms_by_fib.npy', fibparms) 
    
    return fibparms



def make_fibparms_by_ord(by_fib, savefile=True):
    
    by_orders = {}
    for fibkey in by_fib.keys():
        for ord in by_fib[fibkey].keys():
            by_orders[ord] = {}
    
    for ord in by_orders.keys():
        for fibkey in by_fib.keys():
                    
            dum = by_fib[fibkey][ord]
            by_orders[ord][fibkey] = dum
    
    if savefile:
        np.save('/Users/christoph/UNSW/fibre_profiles/sim/fibparms_by_ord.npy', by_orders) 
    
    return by_orders
    
    
    
    
    
    
    
    