'''
Created on 11 Aug. 2017

@author: christoph
'''

#how to measure program run-time quick-and-dirty style
start_time = time.time()
i=1
x=123456.789
while i < 10000000:
    np.float_power(x,2)
    i += 1
print(time.time() - start_time)


#locations for individual peaks
peaklocs = np.arange(316,1378+59,59)
peaks = np.zeros(2048)
peaks[peaklocs]=1
xgrid = x_ix - sim.x_map[i,j] - nx//2

offsets_peaklocs = offsets[peaklocs]

g=[]
for ii in range(19):
    g.append(fibmodel(offsets,offsets_peaklocs[ii],2,norm=0))
g = np.array(g)

ginterp = []
for ii in range(19):
    ginterp.append(np.interp(xgrid, offsets, g[ii,:]))
myphi = np.array(ginterp).T

result = eta * myphi
result_sum = np.sum(result,axis=1)

#then just plug into linalg_extract_column




#how I created the temporary P_id dictionary for JS's code
P_id = {}
Ptemp = {}
ordernames = []
for i in range(1,10):
    ordernames.append('order_0%i' % i)
for i in range(10,34):
    ordernames.append('order_%i' % i)
#the array parms comes from the "find_stripes" function
for i in range(33):
    Ptemp.update({ordernames[i]:parms[i]})
P_id = {'fibre_01': Ptemp}



#timing tests
start_time = time.time()
for i in range(10):
    mask2 = stripe.copy()
    mask2[:,:] = 1
print(time.time() - start_time)






#from POLYSPECT
def spectral_format_with_matrix_rechts(input_arm):
        """Create a spectral format, including a detector to slit matrix at
           every point.

        Returns
        -------
        x: (nm, ny) float array
            The x-direction pixel co-ordinate corresponding to each y-pixel
            and each order (m).
        w: (nm, ny) float array
            The wavelength co-ordinate corresponding to each y-pixel and each
            order (m).
        blaze: (nm, ny) float array
            The blaze function (pixel flux divided by order center flux)
            corresponding to each y-pixel and each order (m).
        matrices: (nm, ny, 2, 2) float array
            2x2 slit rotation matrices, mapping output co-ordinates back
            to the slit.
        """
        x, w, b, ccd_centre = spectral_format(input_arm)
        matrices = np.zeros((x.shape[0], x.shape[1], 2, 2))
        amat = np.zeros((2, 2))

        for i in range(x.shape[0]):  # i is the order
            for j in range(x.shape[1]):
                # Create a matrix where we map input angles to output
                # coordinates.
                slit_microns_per_det_pix = input_arm.slit_microns_per_det_pix_first + \
                    float(i) / x.shape[0] * (input_arm.slit_microns_per_det_pix_last - \
                                             input_arm.slit_microns_per_det_pix_first)
                amat[0, 0] = 1.0 / slit_microns_per_det_pix
                amat[0, 1] = 0
                amat[1, 0] = 0
                amat[1, 1] = 1.0 / slit_microns_per_det_pix
                # Apply an additional rotation matrix. If the simulation was
                # complete, this wouldn't be required.
                r_rad = np.radians(input_arm.extra_rot)
                dy_frac = (j - x.shape[1] / 2.0) / (x.shape[1] / 2.0)
                extra_rot_mat = np.array([[np.cos(r_rad * dy_frac),
                                           np.sin(r_rad * dy_frac)],
                                          [-np.sin(r_rad * dy_frac),
                                           np.cos(r_rad * dy_frac)]])
                amat = np.dot(extra_rot_mat, amat)
                # We actually want the inverse of this (mapping output
                # coordinates back onto the slit.
                matrices[i, j, :, :] = np.linalg.inv(amat)
        return x, w, b, matrices




#from GHOSTSIM
def spectral_format_with_matrix_links(input_arm):
    """Create a spectral format, including a detector to slit matrix at every point.
    
    Returns
    -------
    x: (nm, ny) float array
        The x-direction pixel co-ordinate corresponding to each y-pixel and each
        order (m).    
    w: (nm, ny) float array
        The wavelength co-ordinate corresponding to each y-pixel and each
        order (m).
    blaze: (nm, ny) float array
        The blaze function (pixel flux divided by order center flux) corresponding
        to each y-pixel and each order (m).
    matrices: (nm, ny, 2, 2) float array
        2x2 slit rotation matrices.
    """        
    x,w,b,ccd_centre = spectral_format(input_arm)
    x_xp,w_xp,b_xp,dummy = spectral_format(input_arm,xoff=-1e-3,ccd_centre=ccd_centre)
    x_yp,w_yp,b_yp,dummy = spectral_format(input_arm,yoff=-1e-3,ccd_centre=ccd_centre)
    dy_dyoff = np.zeros(x.shape)
    dy_dxoff = np.zeros(x.shape)
    #For the y coordinate, spectral_format output the wavelength at fixed pixel, not 
    #the pixel at fixed wavelength. This means we need to interpolate to find the 
    #slit to detector transform.
    isbad = w*w_xp*w_yp == 0
    for i in range(x.shape[0]):
        ww = np.where(isbad[i,:] == False)[0]
        dy_dyoff[i,ww] =     np.interp(w_yp[i,ww],w[i,ww],np.arange(len(ww))) - np.arange(len(ww))
        dy_dxoff[i,ww] =     np.interp(w_xp[i,ww],w[i,ww],np.arange(len(ww))) - np.arange(len(ww))
        #Interpolation won't work beyond the end, so extrapolate manually (why isn't this a numpy
        #option???)
        dy_dyoff[i,ww[-1]] = dy_dyoff[i,ww[-2]]
        dy_dxoff[i,ww[-1]] = dy_dxoff[i,ww[-2]]
                
    #For dx, no interpolation is needed so the numerical derivative is trivial...
    dx_dxoff = x_xp - x
    dx_dyoff = x_yp - x

    #flag bad data...
    x[isbad] = np.nan
    w[isbad] = np.nan
    b[isbad] = np.nan
    dy_dyoff[isbad] = np.nan
    dy_dxoff[isbad] = np.nan
    dx_dyoff[isbad] = np.nan
    dx_dxoff[isbad] = np.nan
    matrices = np.zeros( (x.shape[0],x.shape[1],2,2) )
    amat = np.zeros((2,2))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ## Create a matrix where we map input angles to output coordinates.
            amat[0,0] = dx_dxoff[i,j]
            amat[0,1] = dx_dyoff[i,j]
            amat[1,0] = dy_dxoff[i,j]
            amat[1,1] = dy_dyoff[i,j]
            ## Apply an additional rotation matrix. If the simulation was complete,
            ## this wouldn't be required.
            r_rad = np.radians(input_arm.extra_rot)
            dy_frac = (j - x.shape[1]/2.0)/(x.shape[1]/2.0)
            extra_rot_mat = np.array([[np.cos(r_rad*dy_frac),np.sin(r_rad*dy_frac)],[-np.sin(r_rad*dy_frac),np.cos(r_rad*dy_frac)]])
            amat = np.dot(extra_rot_mat,amat)
            ## We actually want the inverse of this (mapping output coordinates back
            ## onto the slit.
            matrices[i,j,:,:] =  np.linalg.inv(amat)
    return x,w,b,matrices



#finding files (needs "import glob, os")
path = '/Users/christoph/UNSW/simulated_spectra/ES/'
files = glob.glob(path+"veloce_flat_t70000_single*.fit")
images={}
allfibflat = np.zeros((4096,4096))
for i,file in enumerate(files):
    fibname = file[-9:-4]
    images[fibname] = pyfits.getdata(file)
    allfibflat += images[fibname]
#read one dummy header
h = pyfits.getheader(files[0])
#write output file
pyfits.writeto(path+'veloce_flat_t70000_nfib19.fit', allfibflat, h)



#read table from file
import re
non_decimal = re.compile(r'[^\d.]+')
#f = open('/Users/christoph/UNSW/linelists/ThXe_linelist_raw2.txt', 'r')
f = open('/Users/christoph/UNSW/linelists/ThAr_linelist_raw2.txt', 'r')
species = []
wl = []
relint = []
for i,line in enumerate(f):
    print('Line ',str(i+1),': ',repr(line))
    line = line.strip()
    cols = line.split("|")
    if (cols[0].strip() != '') and (cols[1].strip() != '') :
        species.append(cols[0].strip())
        wl.append(float(cols[1].strip()))
        if non_decimal.sub('', cols[2].strip()) == '':
            relint.append(0)
        else:
            relint.append(float(non_decimal.sub('', cols[2].strip())))
f.close()
linelist = np.array([species,wl,relint])
#np.savetxt('/Users/christoph/UNSW/linelists/ThXe_linelist.dat', np.c_[wl, relint], delimiter=';', fmt='%10.8f; %i')
np.savetxt('/Users/christoph/UNSW/linelists/ThAr_linelist.dat', np.c_[wl, relint], delimiter=';', fmt='%10.8f; %i')


#make fake laser-comb line-list
delta_f = 2.5E10    #final frequency spacing is 25 Ghz
c = 2.99792458E8    #speed of light in m/s
f0 = c / 1E-6           #do for a wavelength range of 550nm-1000nm, ie 1000nm is ~300Thz
fmax = c / 5.5E-7
ct=0
frange = np.arange(9812) * delta_f + f0
wl = np.flip((c / frange) / 1e-6, axis=0)     #wavelength in microns (from shortest to longest wavelength)
relint = [1]*len(wl)
np.savetxt('/Users/christoph/UNSW/linelists/laser_linelist_25GHz.dat', np.c_[wl, relint], delimiter=';', fmt='%10.8f; %i')




#write arrays to output file
outfn = open('/Users/christoph/UNSW/rvtest/solar_0ms.txt', 'w')
for ord in sorted(pix.iterkeys()):
    for i in range(4096):
        outfn.write("%s     %f     %f     %f\n" %(pix[ord][i],wl[ord][i],flux[ord][i],err[ord][i]))

outfn.close()




#timing tests for matrix inversion
start_time = time.time()
for i in range(10000):
    C_inv = np.linalg.inv(C)
    eta = np.matmul(C_inv,b)
    
    #np.linalg.solve(C, b)
print(time.time() - start_time,' seconds')


#finding peaks when
for ord in sorted(laserdata['flux'].iterkeys())[:-1]:
    data = laserdata['flux'][ord].copy()
    #data[::2] += 1e-3
    filtered_data = ndimage.gaussian_filter(data.astype(np.float), 1)
    filtered_data2 = ndimage.gaussian_filter(data.astype(np.float), 2)
    allpeaks = signal.argrelextrema(filtered_data, np.greater)[0]
    allpeaks2 = signal.argrelextrema(filtered_data2, np.greater)[0]
    print(ord,' : ',len(allpeaks),' | ',len(allpeaks2))
    if len(allpeaks) != len(allpeaks2):
        print('fuganda')
        
        
        
        
# co-add spectra
path = '/Users/christoph/UNSW/veloce_spectra/Mar01/'
for n,file in enumerate(glob.glob(path+'*white*')):
    img = pyfits.getdata(file)
    if n==0:
        h = pyfits.getheader(file)
        master = img.copy().astype(float)
    else:
        master += img
pyfits.writeto(path+'master_white.fits', master, h)
        

