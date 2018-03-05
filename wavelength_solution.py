'''
Created on 29 Nov. 2017

@author: christoph
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy import ndimage
import lmfit
from lmfit import parameter, minimizer
from lmfit.models import LinearModel, GaussianModel
import time
from astropy.io import ascii

from veloce_reduction.helper_functions import CMB_pure_gaussian, CMB_multi_gaussian, fibmodel, fibmodel_with_amp, multi_fibmodel_with_amp
from readcol import *


thardata = np.load('/Users/christoph/UNSW/rvtest/thardata.npy').item()
laserdata = np.load('/Users/christoph/UNSW/rvtest/laserdata.npy').item()

# thdata = thardata['flux']['order_01']
# ldata = laserdata['flux']['order_01']




def find_suitable_peaks(data, thresh = 5000., bgthresh = 2000., gauss_filter_sigma=1., maxthresh = None, debug_level=0, return_masks=False, timit=False):
    
    #this routine is extremely fast, no need to optimise for speed
    if timit:
        start_time = time.time()
    
    xx = np.arange(len(data))
    
    #smooth data to make sure we are not finding noise peaks, and add tiny slope to make sure peaks are found even when pixel-values are like [...,3,6,18,41,41,21,11,4,...]
    filtered_data = ndimage.gaussian_filter(data.astype(np.float), gauss_filter_sigma) + xx*1e-4
    
    #find all local maxima in smoothed data (and exclude the leftmost and rightmost maxima to avoid nasty edge effects...)
    allpeaks = signal.argrelextrema(filtered_data, np.greater)[0]
    
    ### this alternative version of finding extrema is completely equivalent
    #testix = np.r_[True, filtered_data[1:] > filtered_data[:-1]] & np.r_[filtered_data[:-1] > filtered_data[1:], True]
    #testpeaks = np.arange(4096)[testix]
    
    allpeaks = allpeaks[1:-1]
    
    #make first mask to determine which peaks from linelist to use in wavelength solution
    first_mask = np.ones(len(allpeaks), dtype='bool')
    
    #remove shallow noise peaks
    first_mask[filtered_data[allpeaks] < bgthresh] = False
    mostpeaks = allpeaks[first_mask]
    
    #make mask which we need later to determine which peaks from linelist to use in wavelength solution
    second_mask = np.ones(len(mostpeaks), dtype='bool')
    
    #remove saturated lines
    if maxthresh is not None:
        second_mask[filtered_data[mostpeaks] > maxthresh] = False
        mostpeaks = mostpeaks[second_mask]
     
    #make mask which we need later to determine which peaks from linelist to use in wavelength solution    
    third_mask = np.ones(len(mostpeaks), dtype='bool')
    
    #only select good peaks higher than a certain threshold
    third_mask[filtered_data[mostpeaks] < thresh] = False
    goodpeaks = mostpeaks[third_mask]           #ie goodpeaks = allpeaks[first_mask][second_mask][third_mask]
    
    #for testing and debugging...
    if debug_level >= 1:
        print('Total number of peaks found: '+str(len(allpeaks)))
        print('Number of peaks found that are higher than '+str(int(thresh))+' counts: '+str(len(goodpeaks)))
        plt.figure()
        plt.plot(data)
        plt.plot(filtered_data)
        plt.scatter(goodpeaks, data[goodpeaks], marker='x', color='r', s=40)
        plt.plot((0,4096),(5e3,5e3),'r--')
        #plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.show()
    
    if timit:
        delta_t = time.time() - start_time
        print('Time elapsed: '+str(round(delta_t,5))+' seconds')
    
    if return_masks:
        return goodpeaks, mostpeaks, allpeaks, first_mask, second_mask, third_mask
    else:
        return goodpeaks, mostpeaks, allpeaks



def fit_emission_lines(data, fitwidth=4, laser=False, varbeta=True, timit=False, verbose=False, return_all_pars=False, return_qualflag=False):
    
    if timit:
        start_time = time.time()
    
    xx = np.arange(len(data))
    
    #find rough peak locations
    goodpeaks,mostpeaks,allpeaks = find_suitable_peaks(data)    
    
    if verbose:
        print('Fitting '+str(len(goodpeaks))+' emission lines...')
    
    line_pos_fitted = []
    if return_all_pars:
        line_amp_fitted = []
        line_sigma_fitted = []
        if varbeta:
            line_beta_fitted = []
    if return_qualflag:
        qualflag = []
        
    for xguess in goodpeaks:
#         if verbose:
#             print('xguess = ',xguess)
        ################################################################################################################################################################################
        #METHOD 1 (using curve_fit; slightly faster than method 2, but IDK how to make sure the fit converged (as with .ier below))
        
        if not laser:
            #check if there are any other peaks in the vicinity of the peak in question (exclude the peak itself)
            checkrange = np.r_[xx[xguess - 2*fitwidth : xguess], xx[xguess+1 : xguess + 2*fitwidth+1]]
            peaks = np.r_[xguess]
            while len((set(checkrange) & set(allpeaks))) > 0:
                #where are the other peaks?
                other_peaks = np.intersect1d(checkrange, allpeaks)
                peaks = np.sort(np.r_[peaks, other_peaks])
                #define new checkrange
                checkrange = xx[peaks[0] - 2*fitwidth : peaks[-1] + 2*fitwidth + 1]
                dum = np.in1d(checkrange, peaks)
                checkrange = checkrange[~dum]
                
            npeaks = len(peaks)      
        else:
            peaks = np.r_[xguess]
            
        npeaks = len(peaks)        
        xrange = xx[peaks[0] - fitwidth : peaks[-1] + fitwidth + 1]      #this should satisfy: len(xrange) == len(checkrange) + -2*fitwidth + len(peaks)
        
        if npeaks == 1:
            if varbeta:
                guess = np.array([xguess, 1., data[xguess], 2.])
                popt, pcov = op.curve_fit(fibmodel_with_amp, xrange, data[xrange], p0=guess, bounds=([xguess-2,0,0,1],[xguess+2,np.inf,np.inf,4]))
            else:
                guess = np.array([xguess, 1., data[xguess]])
                popt, pcov = op.curve_fit(CMB_pure_gaussian, xrange, data[xrange], p0=guess, bounds=([xguess-2,0,0],[xguess+2,np.inf,np.inf]))
            fitted_pos = popt[0]
            if return_all_pars:
                fitted_sigma = popt[1]
                fitted_amp = popt[2]
                if varbeta:
                    fitted_beta = popt[3]
        else:
            guess = []
            lower_bounds = []
            upper_bounds = []
            for i in range(npeaks):
                if varbeta:
                    guess.append(np.array([peaks[i], 1., data[peaks[i]], 2.]))
                    lower_bounds.append([peaks[i]-2,0,0,1])
                    upper_bounds.append([peaks[i]+2,np.inf,np.inf,4])
                else:
                    guess.append(np.array([peaks[i], 1., data[peaks[i]]]))
                    lower_bounds.append([peaks[i]-2,0,0])
                    upper_bounds.append([peaks[i]+2,np.inf,np.inf])
            guess = np.array(guess).flatten()
            lower_bounds = np.array(lower_bounds).flatten()
            upper_bounds = np.array(upper_bounds).flatten()
            if varbeta:
                popt, pcov = op.curve_fit(multi_fibmodel_with_amp, xrange, data[xrange], p0=guess, bounds=(lower_bounds,upper_bounds))
            else:
                popt, pcov = op.curve_fit(CMB_multi_gaussian, xrange, data[xrange], p0=guess, bounds=(lower_bounds,upper_bounds))           
            
            #now figure out which peak is the one we wanted originally
            q = np.argwhere(peaks==xguess)[0]
            if varbeta:
                fitted_pos = popt[q*4]
                if return_all_pars:
                    fitted_sigma = popt[q*4+1]
                    fitted_amp = popt[q*4+2]
                    fitted_beta = popt[q*4+3]
            else:
                fitted_pos = popt[q*3]
                if return_all_pars:
                    fitted_sigma = popt[q*3+1]
                    fitted_amp = popt[q*3+2]
                
        
        
        #make sure we actually found a good peak
        if abs(fitted_pos - xguess) >= 2.:
            line_pos_fitted.append(xguess)
            if return_qualflag:
                qualflag.append(0)
            if return_all_pars:
                line_sigma_fitted.append(fitted_sigma)
                line_amp_fitted.append(fitted_amp)
                if varbeta:
                    line_beta_fitted.append(fitted_beta)
        else:
            line_pos_fitted.append(fitted_pos)
            if return_qualflag:
                qualflag.append(1)
            if return_all_pars:
                line_sigma_fitted.append(fitted_sigma)
                line_amp_fitted.append(fitted_amp)
                if varbeta:
                    line_beta_fitted.append(fitted_beta)
        ################################################################################################################################################################################
        
#         ################################################################################################################################################################################
#         #METHOD 2 (using lmfit) (NOTE THAT THE TWO METHODS HAVE different amplitudes for the Gaussian b/c of different normalization, but we are only interested in the position)
#         #xguess = int(xguess)
#         gm = GaussianModel()
#         gm_pars = gm.guess(data[xguess - fitwidth:xguess + fitwidth], xx[xguess - fitwidth:xguess + fitwidth])
#         gm_fit_result = gm.fit(data[xguess - fitwidth:xguess + fitwidth], gm_pars, x=xx[xguess - fitwidth:xguess + fitwidth])
#         
#         #make sure we actually found the correct peak
#         if gm_fit_result.ier not in (1,2,3,4):     #if this is any other value it means the fit did not converge
#         #if gm_fit_result.ier > 4:   
#             # gm_fit_result.plot()
#             # plt.show()
#             thar_pos_fitted.append(xguess)
#         elif abs(gm_fit_result.best_values['center'] - xguess) > 2.:
#             thar_pos_fitted.append(xguess)
#         else:
#             thar_pos_fitted.append(gm_fit_result.best_values['center'])
#         ################################################################################################################################################################################
    
    if verbose:    
        plt.figure()
        plt.plot(xx,data)
        #plt.vlines(thar_pos_guess, 0, np.max(data))
        plt.vlines(line_pos_fitted, 0, np.max(data), color='g', linestyles='dotted')
        plt.show()
    
    if timit:
        print('Time taken for fitting emission lines: '+str(time.time() - start_time)+' seconds...')
    
    if return_all_pars:
        if varbeta:
            if return_qualflag:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(line_beta_fitted), np.array(qualflag)
            else:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(line_beta_fitted)
        else:
            if return_qualflag:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted), np.array(qualflag)
            else:
                return np.array(line_pos_fitted), np.array(line_sigma_fitted), np.array(line_amp_fitted)
    else:
        if return_qualflag:
            return np.array(line_pos_fitted), np.array(qualflag)
        else:
            return np.array(line_pos_fitted)



###########################################

###########################################
thar_refwlord01, thar_relintord01, flag = readcol('/Users/christoph/UNSW/linelists/test_thar_list_order_01.dat',fsep=';',twod=False)
thar_refwlord01 *= 1e3
refdata = {}
refdata['order_01'] = {}
refdata['order_01']['wl'] = thar_refwlord01[np.argwhere(flag == ' resolved')][::-1]          #note the array is turned around to match other arrays
refdata['order_01']['relint'] = thar_relintord01[np.argwhere(flag == ' resolved')][::-1]     #note the array is turned around to match other arrays
###########################################



def get_dispsol_from_thar(thardata, refdata, deg_polynomial=5, timit=False, verbose=False):

    if timit:
        start_time = time.time()

    thar_dispsol = {}
    
    #loop over all orders
    #for ord in sorted(thardata['flux'].iterkeys()):
    for ord in ['order_01']:
    
        if verbose:
            print('Finding wavelength solution for '+str(ord))
    
        #find fitted x-positions of ThAr peaks
        fitted_thar_pos, thar_qualflag = fit_emission_lines(thardata['flux'][ord], return_all_pars=False, return_qualflag=True, varbeta=False)
        x = fitted_thar_pos.copy()
        
        #these are the theoretical wavelengths from the NIST linelists
        lam = (refdata[ord]['wl']).flatten()
        
        #exclude some peaks as they are a blend of multiple lines: TODO: clean up
        filt = np.ones(len(fitted_thar_pos),dtype='bool')
        filt[[33,40,42,58,60]] = False
        x = x[filt]
        
        #fit polynomial to lambda as a function of x
        thar_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
        #save to output dictionary
        thar_dispsol[ord] = thar_fit

    if timit:
        print('Time taken for finding ThAr wavelength solution: '+str(time.time() - start_time)+' seconds...')

    return thar_dispsol


'''xxx'''
#################################################################
# the following is needed as input for "get_dispsol_from_laser" #
#################################################################
laser_ref_wl,laser_relint = readcol('/Users/christoph/UNSW/linelists/laser_linelist_25GHz.dat',fsep=';',twod=False)
laser_ref_wl *= 1e3

#wavelength solution from HDF file
#read dispersion solution from file
dispsol = np.load('/Users/christoph/UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
#read extracted spectrum from files (obviously this needs to be improved)
xx = np.arange(4096)
#this is so as to match the order number with the physical order number (66 <= m <= 108)
# order01 corresponds to m=66
# order43 corresponds to m=108
wl = {}
for ord in dispsol.keys():
    m = ord[5:]
    ordnum = str(int(m)-65).zfill(2)
    wl['order_'+ordnum] = dispsol['order'+m]['model'](xx)
    


def get_dispsol_from_laser(laserdata, laser_ref_wl, deg_polynomial=5, timit=False, verbose=False, return_stats=False, varbeta=False):
    
    if timit:
        start_time = time.time()

    if return_stats:
        stats = {}

    #read in mask for fibre_01 (ie the Laser-comb fibre) from order_tracing as a first step in excluding low-flux regions
    mask_01 = np.load('/Users/christoph/UNSW/fibre_profiles/masks/mask_01.npy').item()

    laser_dispsol = {}
    
    #loop over all orders
    #order 43 does not work properly, as some laser peaks are missing!!!
    for ord in sorted(laserdata['flux'].iterkeys())[:-1]:

        if verbose:
            print('Finding wavelength solution for '+str(ord))
        
        #find fitted x-positions of ThAr peaks
        data = laserdata['flux'][ord] * mask_01[ord]
        goodpeaks,mostpeaks,allpeaks,first_mask,second_mask,third_mask = find_suitable_peaks(data,return_masks=True)    #from this we just want the masks this time (should be very fast)
        #fitted_laser_pos, laser_qualflag = fit_emission_lines(data, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
        if varbeta:
            fitted_laser_pos, fitted_laser_sigma, fitted_laser_amp, fitted_laser_beta = fit_emission_lines(data, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta, timit=timit, verbose=verbose)
        else:
            fitted_laser_pos, fitted_laser_sigma, fitted_laser_amp = fit_emission_lines(data, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta, timit=timit, verbose=verbose)
        x = fitted_laser_pos.copy()
        #exclude the leftmost and rightmost peaks (nasty edge effects...)
#         blue_cutoff = int(np.round((x[-1]+x[-2])/2.,0))
#         red_cutoff = int(np.round((x[0]+x[1])/2.,0))
        blue_cutoff = int(np.round(allpeaks[-1]+((allpeaks[-1] - allpeaks[-2])/2),0))
        red_cutoff = int(np.round(allpeaks[0]-((allpeaks[1] - allpeaks[0])/2),0))
        cond1 = (laser_ref_wl >= wl[ord][blue_cutoff])
        cond2 = (laser_ref_wl <= wl[ord][red_cutoff])
        #these are the theoretical wavelengths from the NIST linelists
        lam = laser_ref_wl[np.logical_and(cond1,cond2)][::-1]
        lam = lam[first_mask][second_mask][third_mask]
        
        #check if the number of lines found equals the number of lines from the line list
#         if verbose:
#             print(len(x),len(lam))
        if len(x) != len(lam):
            print('fuganda')
            return 'fuganda'
        
        #fit polynomial to lambda as a function of x
        laser_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
        
        if return_stats:
            stats[ord] = {}
            resid = laser_fit(x) - lam
            stats[ord]['resids'] = resid
            #mean error in RV for a single line = c * (stddev(resid) / mean(lambda))
            stats[ord]['single_rverr'] = 3e8 * (np.std(resid) / np.mean(lam))
            stats[ord]['rverr'] = 3e8 * (np.std(resid) / np.mean(lam)) / np.sqrt(len(lam))
            stats[ord]['n_lines'] = len(lam)
            
        #save to output dictionary
        laser_dispsol[ord] = laser_fit

    
    #let's do order 43 differently because it has the stupid gap in the middle
    #find fitted x-positions of ThAr peaks
    ord = 'order_43'
    if verbose:
            print('Finding wavelength solution for '+str(ord))
    data = laserdata['flux'][ord] * mask_01[ord]
    data1 = data[:2500]
    data2 = data[2500:]
    goodpeaks1,mostpeaks1,allpeaks1,first_mask1,second_mask1,third_mask1 = find_suitable_peaks(data1,return_masks=True)    #from this we just want use_mask this time (should be very fast)
    goodpeaks2,mostpeaks2,allpeaks2,first_mask2,second_mask2,third_mask2 = find_suitable_peaks(data2,return_masks=True)    #from this we just want use_mask this time (should be very fast)
    #fitted_laser_pos1, laser_qualflag1 = fit_emission_lines(data1, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
    #fitted_laser_pos2, laser_qualflag2 = fit_emission_lines(data2, laser=True, return_all_pars=False, return_qualflag=True, varbeta=varbeta)
    if varbeta:
        fitted_laser_pos1, fitted_laser_sigma1, fitted_laser_amp1, fitted_laser_beta1 = fit_emission_lines(data1, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
        fitted_laser_pos2, fitted_laser_sigma2, fitted_laser_amp2, fitted_laser_beta2 = fit_emission_lines(data2, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
    else:
        fitted_laser_pos1, fitted_laser_sigma1, fitted_laser_amp1 = fit_emission_lines(data1, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
        fitted_laser_pos2, fitted_laser_sigma2, fitted_laser_amp2 = fit_emission_lines(data2, laser=True, return_all_pars=True, return_qualflag=False, varbeta=varbeta)
    x1 = fitted_laser_pos1.copy()
    x2 = fitted_laser_pos2.copy() + 2500
    #exclude the leftmost and rightmost peaks (nasty edge effects...)
#         blue_cutoff = int(np.round((x[-1]+x[-2])/2.,0))
#         red_cutoff = int(np.round((x[0]+x[1])/2.,0))
    blue_cutoff1 = int(np.round(allpeaks1[-1]+((allpeaks1[-1] - allpeaks1[-2])/2),0))
    blue_cutoff2 = int(np.round(allpeaks2[-1]+((allpeaks2[-1] - allpeaks2[-2])/2)+2500,0))
    red_cutoff1 = int(np.round(allpeaks1[0]-((allpeaks1[1] - allpeaks1[0])/2),0))
    red_cutoff2 = int(np.round(allpeaks2[0]-((allpeaks2[1] - allpeaks2[0])/2)+2500,0))
    cond1_1 = (laser_ref_wl >= wl[ord][blue_cutoff1])
    cond1_2 = (laser_ref_wl >= wl[ord][blue_cutoff2])
    cond2_1 = (laser_ref_wl <= wl[ord][red_cutoff1])
    cond2_2 = (laser_ref_wl <= wl[ord][red_cutoff2])
    #these are the theoretical wavelengths from the NIST linelists
    lam1 = laser_ref_wl[np.logical_and(cond1_1,cond2_1)][::-1]
    lam2 = laser_ref_wl[np.logical_and(cond1_2,cond2_2)][::-1]
    lam1 = lam1[first_mask1][second_mask1][third_mask1]
    lam2 = lam2[first_mask2][second_mask2][third_mask2]
    
    x = np.r_[x1,x2]
    lam = np.r_[lam1,lam2]
    
    #check if the number of lines found equals the number of lines from the line list
    if verbose:
        print(len(x),len(lam))
    if len(x) != len(lam):
        print('fuganda')
        return 'fuganda'
    
    #fit polynomial to lambda as a function of x
    laser_fit = np.poly1d(np.polyfit(x, lam, deg_polynomial))
    
    if return_stats:
        stats[ord] = {}
        resid = laser_fit(x) - lam
        stats[ord]['resids'] = resid
        #mean error in RV for a single line = c * (stddev(resid) / mean(lambda))
        stats[ord]['single_rverr'] = 3e8 * (np.std(resid) / np.mean(lam))
        stats[ord]['rverr'] = 3e8 * (np.std(resid) / np.mean(lam)) / np.sqrt(len(lam))
        stats[ord]['n_lines'] = len(lam)
    
    #save to output dictionary
    laser_dispsol[ord] = laser_fit

    if timit:
        print('Time taken for finding Laser-comb wavelength solution: '+str(time.time() - start_time)+' seconds...')

    if return_stats:
        return laser_dispsol, stats 
    else:
        return laser_dispsol
        




laser_dispsol2,stats2 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=2)
laser_dispsol3,stats3 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=3)
laser_dispsol5,stats5 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=5)
laser_dispsol11,stats11 = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=11)







