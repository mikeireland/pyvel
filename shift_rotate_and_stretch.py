'''
Created on 30 Oct. 2017

@author: christoph
'''

import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import scipy.ndimage as ndimage
import cv2     #this only works with Python 3.6!!!!! (IDK Y)
from matplotlib.colors import LogNorm

imgname = '/Users/christoph/UNSW/simulated_spectra/ES/veloce_laser_comb.fit'
img_tmp = pyfits.getdata(imgname) + 1
img = img_tmp[2000:2500,2000:2500]

#using Pillow
im = cv2.imread("/Users/christoph/UNSW/shapes.jpeg")


def shift_image(img, xshift, yshift):
    
#     test1 = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#     
#     height, width = img.shape[:2]
#     test2 = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
    shifted = img.copy()     
    return shifted



def rotate_image(img, theta):
    rotated = img.copy()
    return rotated



def stretch_image(img, fac):
    stretched = img.copy()
    return stretched