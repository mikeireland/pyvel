'''
Created on 25 Oct. 2017

@author: christoph
'''

import time
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from mpl_toolkits.mplot3d import Axes3D
from astropy.modeling import models, fitting
from astropy.modeling.models import Gaussian2D
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
#from astropy.modeling.functional_models import Gaussian2D
from scipy.optimize import curve_fit



def gauss2D(xytuple, amp, x0, y0, x_sig, y_sig, theta):
    x,y = xytuple
    a = ((np.cos(theta))**2 / (2*x_sig**2)) + ((np.sin(theta))**2 / (2*y_sig**2))
    b = -np.sin(2*theta)/(4*x_sig**2) + np.sin(2*theta)/(4*y_sig**2)
    c = ((np.sin(theta))**2 / (2*x_sig**2)) + ((np.cos(theta))**2 / (2*y_sig**2))

    return amp * np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2) ) 



#some parameters
RON = 4.                 # my guess for read-out noise
bg_thresh = 1            # background threshold
count_thresh = 2000.     # threshold for selection of peaks
xsigma = 1.1             # sigma of 2D-Gaussian PSF in dispersion direction
ysigma = 0.7             # sigma of 2D-Gaussian PSF in cross-dispersion direction
boxwidth = 11            # size of box to use for the 2D Gaussian peak fitting


imgname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_laser_comb.fit'
img = pyfits.getdata(imgname) + 1
testimg = img[250:450,250:450]

def find_laser_peaks_2D(img, boxwidth=11, bg_thresh=1., count_thresh=2000., RON=4., xsigma=1.1, ysigma=0.7, smooth=False, timit=False):

    if timit:
        start_time = time.time()
    
    # define an 8-connected neighbourhood
    neighbourhood = generate_binary_structure(2,2)
    
    ### smooth image slightly for noise reduction
    if smooth:
        filtered_img = ndimage.gaussian_filter(img, 1.)   
    else:
        filtered_img = img.copy()
        
#     #WARNING - THIS INTRODUCES ONE EXTRA MAXIMUM at (4095,4095)!!!
#     #add tiny slope in x- and y-directions to make sure neighbouring pixels cannot have identical count values
#     dumxr = np.arange(filtered_img.shape[0]) / (filtered_img.shape[0]*10.)
#     dumyr = np.arange(filtered_img.shape[1]) / (filtered_img.shape[1]*10.)
#     dumxx,dumyy = np.meshgrid(dumxr,dumyr)
#     filtered_img = filtered_img + dumxx + dumyy
    
    #apply the local maximum filter; all pixel of max. value in their neighbourhood are set to 1
    local_max = ndimage.maximum_filter(filtered_img, footprint=neighbourhood)==filtered_img
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.
    
    #we create the mask of the background
    background = (filtered_img <= bg_thresh)
    
    #a little technicality: we must erode the background in order to 
    #successfully subtract it from local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighbourhood, border_value=1)
    
    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask 
    peaks = local_max ^ eroded_background     # XOR operator
    
    labeled, n_peaks = ndimage.label(peaks)
    
#     if n_peaks != np.sum(peaks):
#         print('ERROR: number of peaks found is inconsistent!')
#         quit()
    
    xy = np.array(ndimage.center_of_mass(filtered_img, labeled, range(1, n_peaks+1)))
        
    #boxwidth = 11     #should probably be an odd number
    padsize = boxwidth / 2
    
    peakshapes = {'number':[], 'x0':[], 'y0':[], 'amp':[], 'x_sigma':[], 'y_sigma':[], 'theta':[], 'chi2red':[]}
                  
    #LOOP OVER ALL LABELED MAXIMA...
    for i in range(n_peaks):
        
        #print(i, xy[i,:])
        
        xr = np.arange(int(round(xy[i,0])) - padsize, int(round(xy[i,0])) + padsize +1)
        yr = np.arange(int(round(xy[i,1])) - padsize, int(round(xy[i,1])) + padsize +1)
        xx,yy = np.meshgrid(xr,yr)
        z = img[xr[0]:xr[-1]+1, yr[0]:yr[-1]+1] 
        
        #don't use peaks that are too close to the edge of the chip or that have less than 2000 counts:
        if (z.size == boxwidth**2) and (np.max(z) >= count_thresh):  
            
            # # METHOD 1: ASTROPY (CANNOT GET IT TO WORK...)
            # g2d_init = models.Gaussian2D (amplitude=5.e4, x_mean=28., y_mean=975., x_stddev=2.5, y_stddev=1.5, theta=0, cov_matrix=None)
            # guess = g2d_init.evaluate(xx, yy, 5e4, 28, 975, 2.5, 1.5, 0)
            # fit_p = fitting.LevMarLSQFitter()
            # g2d = fit_p(g2d_init, xx, yy, z)
            
            #METHOD 2: 
            pguess = (np.max(z), xy[i,0], xy[i,1], xsigma, ysigma, 0.)
            guessmodel = gauss2D((xx,yy), *pguess)
            xflat = xx.flatten()
            yflat = yy.flatten()
            zflat = z.flatten()
            popt,pcov = curve_fit(gauss2D, (xflat,yflat), zflat, p0=pguess)
            bestmodel = gauss2D((xx,yy), *popt)
            res = z - bestmodel
            err = np.sqrt( z + RON**2 )
            chisqarr = (res/err)**2
            chisq = np.sum(chisqarr)
            dof = z.size - len(pguess)
            chi2red = chisq / dof
            
            peakshapes['number'].append(i+1)
            peakshapes['x0'].append(popt[1])
            peakshapes['y0'].append(popt[2])
            peakshapes['amp'].append(popt[0])
            peakshapes['x_sigma'].append(popt[3])
            peakshapes['y_sigma'].append(popt[4])
            peakshapes['theta'].append(popt[5])
            peakshapes['chi2red'].append(chi2red)
        
    if timit:
        print('Elapsed time for finding peaks: ',time.time() - start_time,' seconds')
    
    return peakshapes







