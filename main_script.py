'''
Created on 10 Nov. 2017

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
from mpl_toolkits.mplot3d import Axes3D

from veloce_reduction.helper_functions import gaussian_with_offset, fibmodel_with_amp_and_offset
from veloce_reduction.order_tracing import *
from veloce_reduction.collapse_extract import *
from veloce_reduction.optimal_extraction import *




#imgname = '/Users/christoph/UNSW/simulated_spectra/blue_ghost_spectrum_20170803.fits'
#imgname = '/Users/christoph/UNSW/simulated_spectra/blue_ghost_spectrum_nothar_highsnr_20170906.fits'
#imgname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib01.fit'
#imgname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_high_SNR_solar_template.fit'
imgname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_full_solar_image_with_2calibs.fit'
imgname2 = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_solar_red100ms.fit'
imgname3 = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_full_solar_image_with_2calibs_red1000ms.fit'
#gflatname = '/Users/christoph/UNSW/simulated_spectra/blue_ghost_flat_20170905.fits'
#flatname = '/Users/christoph/UNSW/simulated_spectra/veloce_flat_highsn2.fit'
#flatname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib01.fit'
flatname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_nfib19.fit'
flat02name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib02.fit'
flat03name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib03.fit'
flat21name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib21.fit'
flat22name = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_flat_t70000_single_fib22.fit'
img = pyfits.getdata(imgname) + 1.
img2 = pyfits.getdata(imgname2) + 1.
img3 = pyfits.getdata(imgname3) + 1.
flat = pyfits.getdata(flatname)
flat02 = pyfits.getdata(flat02name)
flat03 = pyfits.getdata(flat03name)
flat21 = pyfits.getdata(flat21name)
flat22 = pyfits.getdata(flat22name)
img02 = flat02 + 1.
img03 = flat03 + 1.
img21 = flat21 + 1.
img22 = flat22 + 1.
flatimg = flat + 1.


# (0) identify whites, thoriums, stellar etc based on information in header




# (1) find orders roughly
P,tempmask = find_stripes(flat, deg_polynomial=2)

# (2) assign physical diffraction order numbers (this is only a dummy function for now) to order-fit polynomials and bad-region masks
P_id = make_P_id(P)
mask = make_mask_dict(tempmask)

# (3) extract stripes of user-defined width from the science image, centred on the polynomial fits defined in step (1)
stripes = extract_stripes(img, P_id, slit_height=25)

# (4) extract and fit background
bg = extract_background(img, P_id, slit_height=25)

####################################################################################################################################
# (4) prepare the four fibres that are used for the tramline definition (only once, then saved to file for the simulations anyway) #
# P_**,tempmask_** = find_stripes(flat**, deg_polynomial=2)                                                                        #
# P_id_** = make_P_id(P_**)                                                                                                        #
# mask_** = make_mask_dict(tempmask_**)                                                                                            #
# stripes_** = extract_stripes(img**, P_id_**, slit_height=10)                                                                     #
#                                                                                                                                  #
# This is only needed once, ie for fitting the fibre profiles, then written to file as it takes ages...                            #
# fibre_profiles_** = fit_profiles(img**, P_id_**, stripes_**)                                                                     #
####################################################################################################################################

# (5) identify tramlines for extraction
fibre_profiles_01 = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_01.npy').item()
laser_tramlines = find_laser_tramlines(fibre_profiles_01, mask_01)
#-----------------------------------------------------------------------------------------------------
fibre_profiles_02 = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_02.npy').item()
fibre_profiles_03 = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_03.npy').item()
fibre_profiles_21 = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_21.npy').item()
fibre_profiles_22 = np.load('/Users/christoph/UNSW/fibre_profiles/sim/fibre_profiles_22.npy').item()
mask_02 = np.load('/Users/christoph/UNSW/fibre_profiles/masks/mask_02.npy').item()
mask_03 = np.load('/Users/christoph/UNSW/fibre_profiles/masks/mask_03.npy').item()
mask_21 = np.load('/Users/christoph/UNSW/fibre_profiles/masks/mask_21.npy').item()
mask_22 = np.load('/Users/christoph/UNSW/fibre_profiles/masks/mask_22.npy').item()
#-----------------------------------------------------------------------------------------------------
tramlines = find_tramlines(fibre_profiles_02, fibre_profiles_03, fibre_profiles_21, fibre_profiles_22, mask_02, mask_03, mask_21, mask_22)

# (6) extract one-dimensional spectrum via (a) tramline extraction, or (b) optimal extraction
# for laser: pix,flux,err = collapse_extract(stripes, laser_tramlines, laser=True, RON=4., gain=1., timit=True)
pix,flux,err = collapse_extract(stripes, tramlines, RON=4., gain=1., timit=True)
pix2,flux2,err2 = optimal_extraction(P_id, stripes, RON=4., gain=1., timit=True, individual_fibres=False)

#### (7) read dispersion solution from file (obviously this is only a temporary crutch)
####dispsol = np.load('/Users/christoph/UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
# (7) get dispersion solution from laser frequency comb
#first read extracted laser comb spectrum
laserdata = np.load('/Users/christoph/UNSW/rvtest/laserdata.npy').item()
# now read laser_linelist
laser_ref_wl,laser_relint = readcol('/Users/christoph/UNSW/linelists/laser_linelist_25GHz.dat',fsep=';',twod=False)
laser_ref_wl *= 1e3
#ZEMAX wavelength solution from HDF file (only needed to determine the region from line list to be used in the wavelength solution)
dispsol = np.load('/Users/christoph/UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy').item()
xx = np.arange(4096)
#this is so as to match the order number with the physical order number (66 <= m <= 108)
# order01 corresponds to m=66
# order43 corresponds to m=108
wl = {}
for ord in dispsol.keys():
    m = ord[5:]
    ordnum = str(int(m)-65).zfill(2)
    wl['order_'+ordnum] = dispsol['order'+m]['model'](xx)
#finally, we are ready to call the wavelength solution routine
laser_dispsol,stats = get_dispsol_from_laser(laserdata, laser_ref_wl, verbose=True, timit=True, return_stats=True, deg_polynomial=5)


# (8) get radial velocities



# (9) get barycentric corrections, e.g.:
result = get_BC_vel(JDUTC=JDUTC,hip_id=8102,lat=-31.2755,longi=149.0673,alt=1165.0,ephemeris='de430',zmeas=0.0)
result2  = get_BC_vel(JDUTC=JDUTC,hip_id=8102,obsname='AAO',ephemeris='de430')

# (10) write output file(s)!?!?!?



