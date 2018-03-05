'''
Created on 5 Feb. 2018

@author: christoph
'''

import numpy as np
from veloce_reduction.helper_functions import polyfit2d, polyval2d


#make simulated background
nx, ny = 4096, 4096
x = np.repeat(np.arange(4096) - 2048,nx)
y = np.tile(np.arange(4096) - 2048,ny)
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))
#parms = [90., 0., 0., 1.5e-9, 0., 0., 0., 0., -4e-9, 0., 0., 0., 1e-9, 0., 0., 0.]
parms = np.array([1000., 0., -5.e-5, 0., 0., 0., 0., 0., -5.e-5, 0., 0., 0., 0., 0., 0., 0.])
zz = polyval2d(xx, yy, parms)
#add white noise
noise = np.resize(np.random.normal(0, 1, 4096**2),(4096,4096))
scaled_noise = noise * np.sqrt(zz)
zz += scaled_noise
imgtest = img + zz



def extract_background(img, P_id, slit_height=25, output_file=None, timit=False):
    """
    Extracts the background (ie everything outside the stripes) from the original 2D spectrum to a sparse array, containing only relevant pixels.
    
    This function marks all relevant pixels for extraction. Using the provided dictionary P_id it iterates over all order-stripes
    in the image and saves a sparse matrix for each background-stripe.
    
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
    
    #logging.info('Extracting background...')
    print('Extracting background...')

    ny, nx = img.shape
    xx = np.arange(nx, dtype='f8')
    yy = np.arange(ny, dtype='f8')
    x_grid, y_grid = np.meshgrid(xx, yy, copy=False)
    bg_mask = np.ones((ny, nx), dtype=bool)

    for o, p in P_id.items():
        #xx = np.arange(nx, dtype=img.dtype)
        xx = np.arange(nx, dtype='f8')
        #yy = np.arange(ny, dtype=img.dtype)
        yy = np.arange(ny, dtype='f8')
    
        y = np.poly1d(p)(xx)
        x_grid, y_grid = np.meshgrid(xx, yy, copy=False)
    
        distance = y_grid - y.repeat(ny).reshape((nx, ny)).T
        indices = abs(distance) > slit_height
        
        bg_mask *= indices
        

    mat = sparse.coo_matrix((img[bg_mask], (y_grid[bg_mask], x_grid[bg_mask])), shape=(ny, nx))
    # return mat.tocsr()
    
    #print('Elapsed time: ',time.time() - start_time,' seconds')
    
    return mat.tocsc()



def fit_background(bg, deg=3):
    #find the non-zero parts of the sparse matrix
    #format is:
    #contents[0] = row indices
    #contents[1] = column indices
    #contents[2] = values
    contents = sparse.find(bg)
    
    ny, nx = bg.todense().shape
    
    # Fit a 3rd order, 2d polynomial
    #m = polyfit2d(contents[0]-2048, contents[1]-2048, contents[2], order=3)
    m = polyfit2d(contents[0]-int(ny/2), contents[1]-int(nx/2), contents[2], order=deg)
    
    # Evaluate it on a grid...
    #nx, ny = 4096, 4096
    x = np.repeat(np.arange(nx) - int(nx/2),nx)
    y = np.tile(np.arange(ny) - int(ny/2),ny)
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))
    bkgd_img = polyval2d(xx, yy, m)
    
    return bkgd_img
    
 




