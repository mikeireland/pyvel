'''
Created on 11 Aug. 2017

@author: christoph
'''

""" This assumes that some input preparation has been done, ie see "extract_1dim_from_pymfe", but this is supposed to be a cleaned up version """

from veloce_reduction.helper_functions import spectral_format, spectral_format_with_matrix

filename = '/Users/christoph/UNSW/simulated_spectra/blue_ghost_spectrum_20170803.fits'
data = pyfits.getdata(filename).T
#data_trans = fits.getdata(filename).T

#some required settings
#create instance of Arm (this could be removed and settings could live in standalone global variables
#sim = ghost.Arm('blue','high')
sim = ghost.Arm(spect='ghost', arm='blue', mode='high')
sim.gain = 1.0
sim.badpixmask=[]
#call spectral format creator
sim.x_map,sim.w_map,sim.blaze,sim.matrices = spectral_format_with_matrix(sim)

#define fluxes (done in ghost but not ghostsim...)
fluxes = np.zeros((sim.nl, 3))
fluxes[2:21, 0] = 0.37
fluxes[8:15, 0] = 0.78
fluxes[11, 0] = 1.0
# NB if on the following line, fluxes[2:,1]=1.0 is set, sky will be
# subtracted automatically.
fluxes[2 + 19:, 1] = 1.0
fluxes[0, 2] = 1.0

#I don't understand the calling syntax for this one: self.define_profile(sim.fluxes)
#profile_func = define_profile(sim,fluxes)
#so do it here instead of in a separate function within a class
sim.square_profile = np.empty( (fluxes.shape[0]*2, fluxes.shape[1]) )
sim.sim_profile = np.empty( (sim.im_slit_sz, fluxes.shape[1]) )
for i in range(fluxes.shape[1]):
    sim.square_profile[:,i] = np.array(fluxes[:,i]).repeat(2)
    im_slit=make_lenslets(sim,fluxes=fluxes[:,i])
    sim.sim_profile[:,i] = np.sum(im_slit, axis=0)

#Set some default pixel offsets for each lenslet, as used for a square lenslet profile
ny = sim.x_map.shape[1]
nm = sim.x_map.shape[0]
pix_offset_ix = np.append(np.append([0],np.arange(1,sim.nl).repeat(2)),sim.nl)

# The [0,0] component of "matrices" measures the size of a detector pixel in the 
# simulated slit image space. i.e. slitmicrons/detpix.
# sim.square_offsets = np.empty( (2*sim.nl,nm) )
# for i in range(sim.nl):
#     sim.square_offsets[:,i] = (pix_offset_ix - sim.nl/2.0) * sim.lenslet_width / sim.matrices[i,sim.x_map.shape[1]//2,0,0]

#Create an array of slit positions in microns. !!! Add an optional offset to this, i.e. a 1D offset !!!
sim.sim_offsets = np.empty( (sim.im_slit_sz,nm) )
im_slit_pix_in_microns = (np.arange(sim.im_slit_sz) - sim.im_slit_sz/2.0) * sim.microns_pix
for i in range(nm):
    sim.sim_offsets[:,i] = im_slit_pix_in_microns / sim.matrices[i,sim.x_map.shape[1]//2,0,0]
    
    
    


def extract_veloce_1dim(data, sim, lenslet_profile='simu', rnoise=4.0, cmb=0, altvar=0, indiv_fib=0, simprof=0, debug_mode=0):
    """ Extract flux by integrating down columns (the "y" direction), using an
    optimal extraction method.
        
    Given that some of this code is in common with two_d_extract, the routines could
    easily be merged... however that would make one_d_extract less readable.
    
    Parameters
    ----------
    data: numpy array (optional) 
        Image data, transposed so that dispersion is in the "y" direction. Note that
        this is the transpose of a conventional echellogram. Either data or file
        must be given
        
    file: string (optional)
        A fits file with conventional row/column directions containing the data to be
        extracted.
        
    lenslet_profile: 'square' or 'sim'
        Shape of the profile of each fiber as used in the extraction. For a final
        implementation, 'measured' should be a possibility. 'square' assigns each
        pixel uniquely to a single lenslet. For testing only
        
    badpix: (float array, float array)
        Output of e.g. np.where giving the bad pixel coordinates.
        
    rnoise: float
        The assumed readout noise.
        
    WARNING: Binning not implemented yet"""
    
        
    start_time = time.time()
    ex_time = 0.
        
    #ny = sim.x_map.shape[1]
    #nm = sim.x_map.shape[0]
    #nx = sim.szx
    
    #number of columns
    nx = data.shape[0]
    #number of rows
    ny = data.shape[1]
    #number of spectral orders
    nm = sim.x_map.shape[0]
    #number of fibres
    nfib = sim.nl
    #number of "objects" (ie star, sky, calibration)
    no = 3
    if indiv_fib == 1:
        no = 28
    
    #load the exact individual-fibre profile that I have reverse-engineered from M.I.'s simulator code
    if simprof == 1:
        standard_phi_os = np.loadtxt('/Users/christoph/UNSW/simulated_spectra/simulated_fibre_profile_N2048.dat')
        #phi = np.loadtxt('/Users/christoph/UNSW/simulated_spectra/phi_N88.txt')
    
    #these are the exact locations of the individual fibres in the 2048-element array used in M.I.'s simulator code
    if indiv_fib == 1:
        #peaklocs = np.arange(316,1378+59,59) that's the stellar ones only
        peaklocs = np.arange(198,1791+59,59)
        peaks = np.zeros(2048)
        peaks[peaklocs] = 1
        
#     #Number of "objects" (are the "objects" star, sky and calibration???)
#     no = sim.square_profile.shape[1]
#     extracted_flux = np.zeros( (nm,ny,no) )
#     extracted_var = np.zeros( (nm,ny,no) )
    
    #Number of "objects" (are the "objects" star, sky and calibration???)
    #no = sim.square_profile.shape[1]
    extracted_flux = np.zeros( (nm,ny,no) )
    extracted_var = np.zeros( (nm,ny,no) )
    
    #Assuming that the data are in photo-electrons, construct a simple model for the
    #pixel inverse variance.
    #pixel_inv_var = 1.0/(np.maximum(data,0)/sim.gain + rnoise**2)
#     pixel_inv_var[sim.badpixmask]=0.0
    w = 1.0 / (np.maximum(data,0)/sim.gain + rnoise*rnoise)
            
    #Loop through all orders then through all y pixels.
    for i in range(nm):
    #for i in range(1):
        print("Extracting order: {0:d}".format(i))
        #Based on the profile we're using, create the local offsets and profile vectors
        if lenslet_profile == 'square':
            offsets = sim.square_offsets[:,i]
            profile = sim.square_profile
        elif lenslet_profile == 'simu':
            offsets = sim.sim_offsets[:,i]
            profile = sim.sim_profile
        nx_cutout = 2*int( (np.max(offsets) - np.min(offsets))/2 ) + 2
        if simprof == 0:
            phi = np.empty( (nx_cutout,no) )
        
        #timing tests
        print("Time taken until loop over pixel columns starts: ",time.time()-start_time, 'seconds')
        
        for j in range(ny):
            #Check for NaNs
            if sim.x_map[i,j] != sim.x_map[i,j]:
                extracted_var[i,j,:] = np.nan
                continue
            #Create our column cutout for the data and the PSF. !!! Is "round" correct on the next line???
            #x_map gives the central locations of each order 
            x_ix = int(np.round(sim.x_map[i,j])) - nx_cutout//2 + np.arange(nx_cutout,dtype=int) + nx//2
            
            if indiv_fib == 0:
                #CMB 09/08/2017 - added check that the new x-coordinate sequence is increasing (otherwise it returns rubbish)
                if np.all(np.diff(offsets) > 0):
                    for k in range(no):   
                        phi[:,k] = np.interp(x_ix - sim.x_map[i,j] - nx//2, offsets, profile[:,k])
                        phi[:,k] /= np.sum(phi[:,k])     #this only works because step size is equal to 1
                else:
                    #print 'ERROR: x-coordinate sequence not in increasing order!!!'
                    quit()
            elif indiv_fib == 1:
                
                #timing tests
                if i == 0 and j == 0:
                    fibtime = 0.
                fibreftime = time.time()
                 
                #CMB 09/08/2017 - added check that the new x-coordinate sequence is increasing (otherwise it returns rubbish)
                if np.all(np.diff(offsets) > 0):
                    xgrid = x_ix - sim.x_map[i,j] - nx//2
                    offsets_peaklocs = offsets[peaklocs]
                    phi_os = np.zeros((len(offsets),no))
                    phi = np.zeros((len(xgrid),no))
                    testphi=phi
                    for k in range(no):                                       
                        if simprof == 0:
                            phi_os[:,k] = fibmodel(offsets,offsets_peaklocs[k],2,norm=1)
                        elif simprof == 1:
                            phi_os[:,k] = np.roll(standard_phi_os, peaklocs[k]-1024)
                        phi[:,k] = np.interp(xgrid, offsets, phi_os[:,k])
                        #TESTING
                        #intfunc = scipy.interpolate.interp1d(offsets, standard_phi_os)
                        #testphi[:,k] = intfunc(xgrid)
                        #print("phi == testphi ???   :  ", (phi==testphi).all() )
                        phi[:,k] /= np.sum(phi[:,k])     #this only works because step size is equal to 1
                        #TODO: 3 fibres are zero in the ghost configuration!!!
                    
                else:
                    #print 'ERROR: x-coordinate sequence not in increasing order!!!'
                    quit()
                     
                #timing tests
                fib_delta_t = time.time() - fibreftime
                fibtime += fib_delta_t 
                  
                    
            #Deal with edge effects...
            ww = np.where( (x_ix >= nx) | (x_ix < 0) )[0]
            x_ix[ww]=0
            phi[ww,:]=0.0
            
            #Stop here. 
#                if i==10:
#                    pdb.set_trace()
        
            #Cut out our data and inverse variance.
            col_data = data[j,x_ix]
            col_w = w[j,x_ix]
            
            #timing test
            if i == 0 and j == 0:
                ex_time = 0.
            ref_time = time.time()
                            
            if cmb == 1:
                eta,var = linalg_extract_column(col_data, col_w, phi, altvar=1)
                extracted_flux[i,j,:] = eta
                extracted_var[i,j,:] = var
            elif cmb == 0:
                #this is Mike Ireland's original version
                #Fill in the "c" matrix and "b" vector from Sharp and Birchall equation 9
                #Simplify things by writing the sum in the computation of "b" as a matrix
                #multiplication. We can do this because we're content to invert the 
                #(small) matrix "c" here. Equation 17 from Sharp and Birchall 
                #doesn't make a lot of sense... so lets just calculate the variance in the
                #simple explicit way.
                col_w_mat = np.reshape(col_w.repeat(no), (nx_cutout,no) )     #why do the weights have to be the same for every "object"?
                b_mat = phi * col_w_mat
                c_mat = np.dot(phi.T,phi * col_w_mat)
                pixel_weights = np.dot(b_mat,np.linalg.inv(c_mat))   #pixel weights are the z_ki in M.I.'s description
                extracted_flux[i,j,:] = np.dot(col_data,pixel_weights)   #these are the etas
                extracted_var[i,j,:] = np.dot(1.0/np.maximum(col_w,1e-12),pixel_weights**2)
                #if ((i % 5)==1) & (j==ny//2):
                #if (i%5==1) & (j==ny//2):
                #if (j==ny//2):
                #    pdb.set_trace()
            
            delta_t = time.time() - ref_time
            ex_time += delta_t
            
    print('Total time: ', time.time() - start_time, 'seconds')
    if indiv_fib == 1:
        print('Time for creation of phi: ',fibtime, 'seconds')
    print('Time for extraction: ',ex_time, 'seconds')
                
    return extracted_flux, extracted_var