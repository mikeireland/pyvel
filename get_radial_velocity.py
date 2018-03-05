'''
Created on 9 Nov. 2017

@author: christoph
'''

import h5py
from astropy.io import ascii
import scipy.interpolate as interp
import scipy.optimize as op
from veloce_reduction.helper_functions import blaze, fibmodel, gausslike_with_amp_and_offset_and_slope


#speed of light in m/s
c = 2.99792458e8
#oversampling factor for logarithmic wavelength rebinning
osf = 2

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
    

#re-bin into logarithmic wavelength space for cross-correlation, b/c delta_log_wl = c * delta_v
refdata = ascii.read('/Users/christoph/UNSW/rvtest/high_SNR_solar_template.txt', names=('pixnum','wl','flux','err'))
# flux1 = data1['flux']
# err1 = data1['err']
#data2 = ascii.read('/Users/christoph/UNSW/rvtest/solar_red100ms.txt', names=('pixnum','wl','flux','err'))
# flux2 = data2['flux']
# err2 = data2['err']
# obsdata = ascii.read('/Users/christoph/UNSW/rvtest/solar_0ms.txt', names=('pixnum','wl','flux','err'))
# obsdata = ascii.read('/Users/christoph/UNSW/rvtest/solar_red100ms.txt', names=('pixnum','wl','flux','err'))
obsdata = ascii.read('/Users/christoph/UNSW/rvtest/solar_red1000ms.txt', names=('pixnum','wl','flux','err'))
# flux3 = data3['flux']
# err3 = data3['err']
flatdata = ascii.read('/Users/christoph/UNSW/rvtest/tramline_extracted_flat.txt', names=('pixnum','wl','flux','err'))
# flatflux = flatdata['flux']
# flaterr = flatdata['err']
#wl1 = data1['wl']
#wl2 = data2['wl']
# wl = data1['wl']
# pixnum = data1['pixnum']



# testwl = np.array(data1['wl'][4096:8192])[::-1]            #need to turn around otherwise np.interp fails; units do not matter, as log(x) - log(y) = log(ax)-log(ay) = log(a) + log(x) - log(a) - log(y)
# testf1 = np.array(data1['flux'][4096:8192])[::-1]          #need to turn around otherwise np.interp fails
# testerr1 = np.array(data1['err'][4096:8192])[::-1]          #need to turn around otherwise np.interp fails
# testf2 = np.array(data2['flux'][4096:8192])[::-1]          #need to turn around otherwise np.interp fails
# testerr2 = np.array(data2['err'][4096:8192])[::-1]          #need to turn around otherwise np.interp fails
# testf3 = np.array(data3['flux'][4096:8192])[::-1]          #need to turn around otherwise np.interp fails
# testerr3 = np.array(data3['err'][4096:8192])[::-1]          #need to turn around otherwise np.interp fails
# testflat = np.array(flatdata['flux'][4096:8192])[::-1]     #need to turn around otherwise np.interp fails


# f0 = template
# f = observation
f0 = {}
err0 = {}
f = {}
err = {}
f_flat = {}
err_flat = {}
ref_wl = {}
obs_wl = {}
flat_wl = {}
for m in range(43):
    ord = 'order_'+str(m+1).zfill(2)
    f0[ord] = np.array(refdata['flux'][m*4096:(m+1)*4096])[::-1]              # need to turn around otherwise np.interp fails
    f[ord] = np.array(obsdata['flux'][m*4096:(m+1)*4096])[::-1]
    f_flat[ord] = np.array(flatdata['flux'][m*4096:(m+1)*4096])[::-1]
    err0[ord] = np.array(refdata['err'][m*4096:(m+1)*4096])[::-1]
    err[ord] = np.array(obsdata['err'][m*4096:(m+1)*4096])[::-1]
    err_flat[ord] = np.array(flatdata['err'][m*4096:(m+1)*4096])[::-1]
    ref_wl[ord] = np.array(refdata['wl'][m*4096:(m+1)*4096])[::-1]            # units do not matter, as log(x) - log(y) = log(ax)-log(ay) = log(a) + log(x) - log(a) - log(y)
    obs_wl[ord] = np.array(obsdata['wl'][m*4096:(m+1)*4096])[::-1]
    flat_wl[ord] = np.array(flatdata['wl'][m*4096:(m+1)*4096])[::-1]
    
    
    
###COMMENT: the template should be stored in an unblazed way already here
#loop over orders
rv = {}
rverr = {}

for ord in sorted(f.iterkeys()):
    
    bad_threshold = 0.02
        
    filtered_flat = ndimage.gaussian_filter(f_flat[ord], 10.)    #edge effects!!!
    normflat = filtered_flat / np.max(filtered_flat)
    normflat[normflat <= 0] = 1e-6
    
    mask = np.ones(len(normflat), dtype = bool)
    if np.min(normflat) < bad_threshold:
        mask[normflat < bad_threshold] = False
        #once the blaze function falls below a certain value, exlcude what's outside of that pixel column, even if it's above the threshold again, ie we want to only use a single consecutive region
        leftmask = mask[: len(mask)//2]
        leftkill_index = [i for i,x in enumerate(leftmask) if not x]
        try:
            mask[: leftkill_index[0]] = False
        except:
            pass
        rightmask = mask[len(mask)//2 :]
        rightkill_index = [i for i,x in enumerate(rightmask) if not x]
        if ord == 'order_01':
            try:
                mask[len(mask)//2 + rightkill_index[0] - 100 :] = False
            except:
                pass
        else:
            try:
                mask[len(mask)//2 + rightkill_index[-1] + 1 :] = False
            except:
                pass
    #this is dodgy, but the gaussian filtering above introduces edge effects due to the flux in the flats dropping sharply in some orders (IDKY)...
    mask[:30] = False
    mask[-30:] = False

    f0_unblazed = f0[ord] / np.max(f0[ord]) / normflat   #see COMMENT above
    f_unblazed = f[ord] / np.max(f[ord]) / normflat
    
    logwl = np.log(ref_wl[ord])
    logwlgrid = np.linspace(np.min(logwl[mask]), np.max(logwl[mask]), osf*np.sum(mask))
    delta_log_wl = logwlgrid[1] - logwlgrid[0]
    
#     rebinned_f0 = np.interp(logwlgrid,logwl,f0_unblazed)
#     rebinned_f = np.interp(logwlgrid,logwl,f_unblazed)
    spl_ref_f0 = interp.InterpolatedUnivariateSpline(logwl[mask], f0_unblazed[mask], k=3)    #slower for linear, but best performance for cubic spline
    rebinned_f0 = spl_ref_f0(logwlgrid)
    spl_ref_f = interp.InterpolatedUnivariateSpline(logwl[mask], f_unblazed[mask], k=3)    #slower for linear, but best performance for cubic spline
    rebinned_f = spl_ref_f(logwlgrid)

    # do we want to cross-correlate the entire order???
    #xc = np.correlate(rebinned_f0 - np.median(rebinned_f0), rebinned_f - np.median(rebinned_f), mode='same')
#     #now this is slightly dodgy, but cutting off the edges works better because the division by the normflat introduces artefacts there
#     if ord == 'order_01':
#         xcorr_region = np.arange(2500,16000,1)
#     else:
#         xcorr_region = np.arange(2500,17500,1)
    
    xc = np.correlate(rebinned_f0 - np.median(rebinned_f0), rebinned_f - np.median(rebinned_f), mode='same')
    #now fit Gaussian to central section of CCF
    fitrangesize = osf*6    #this factor was simply eye-balled
    xrange = np.arange(np.argmax(xc)-fitrangesize, np.argmax(xc)+fitrangesize+1, 1)
    guess = np.array((np.argmax(xc), 10., (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize])/np.max(xc), 2., xc[np.argmax(xc)-fitrangesize]/np.max(xc), 0.))
    #popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange, xc[np.argmax(xc)-fitrangesize : np.argmax(xc)+fitrangesize+1]/np.max(xc[xrange]), p0=guess)
    popt, pcov = op.curve_fit(gausslike_with_amp_and_offset_and_slope, xrange, xc[np.argmax(xc)-fitrangesize : np.argmax(xc)+fitrangesize+1]/np.max(xc[xrange]), p0=guess)
    shift = popt[0]
    shift_err = pcov[0,0]
    rv[ord] = c * (shift - (len(xc)//2)) * delta_log_wl
    rverr[ord] = c * shift_err * delta_log_wl
    
    
    
testrv = np.array([value for (key, value) in sorted(rv.items())])
testerr = np.array([value for (key, value) in sorted(rverr.items())])  
testw = 1./(testerr**2)
print(np.average(testrv, weights=testw))
    
    
 
#############################################################################################################################    
    
    
    
# or do we maybe want to cut it up into chunks, and determine a RV for every chunk???
dum1 = (rebinned_flux1 - np.median(rebinned_flux1)).reshape((osf*16,256))
dum3 = (rebinned_flux3 - np.median(rebinned_flux3)).reshape((osf*16,256))
dumwl = logwlgrid.reshape((osf*16,256))   
rv = [] 
for i in range(len(dum1)):
    ref = dum1[i,:]
    flux = dum3[i,:]
    xc = np.correlate(ref, flux, mode='same')
    #now fit Gaussian to central section of CCF
    fitrangesize = 9
    xrange = np.arange(np.argmax(xc)-fitrangesize, np.argmax(xc)+fitrangesize+1, 1)
    guess = np.array((np.argmax(xc), 3., (xc[np.argmax(xc)]-xc[np.argmax(xc)-fitrangesize])/np.max(xc), xc[np.argmax(xc)-fitrangesize]/np.max(xc), 0.))
    #maybe use a different model, ie include a varying beta-parameter???
    popt, pcov = op.curve_fit(gaussian_with_offset_and_slope, xrange, xc[np.argmax(xc)-fitrangesize : np.argmax(xc)+fitrangesize+1]/np.max(xc), p0=guess)
    shift = popt[0]
    rv.append(c * (shift - (len(xc)//2)) * delta_log_wl)
    
    
# start_time = time.time()
# for i in range(1000):
#     spl_ref1 = interp.InterpolatedUnivariateSpline(logwl, f1_unblazed, k=1)
#     rebinned_f1_xxx = spl_ref1(logwlgrid)
#     #rebinned_flux1 = np.interp(logwlgrid,logwl,f1_unblazed)
# print(str(time.time() - start_time), 'seconds')





#########################################################################################
### CMB - 15/11/2017                                                                  ###
### The following functions are based on the RV parts from Mike Ireland's pymf        ###
### I also made them standalone routines rather than part of an object-oriented class ###
#########################################################################################    
def rv_shift_resid(parms, wave, spect, spect_sdev, spline_ref, return_spect=False):
    """Find the residuals to a fit of a (subsampled) reference spectrum to an 
    observed spectrum. 
    
    The function for parameters p[0] through p[3] is:
    
    .. math::
        y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
    
    Parameters
    ----------        
    params: array-like
    wave: float array
        Wavelengths for the observed spectrum.        
    spect: float array
        The observed spectra     
    spect_sdev: float array
        standard deviation of the input spectra.        
    spline_ref: InterpolatedUnivariateSpline instance
        For interpolating the reference spectrum
    return_spect: boolean
        Whether to return the fitted spectrum or the residuals.
    wave_ref: float array
        The wavelengths of the reference spectrum
    ref: float array
        The reference spectrum
    
    Returns
    -------
    resid: float array
        The fit residuals
    """
    
    ny = len(spect)
    # CMB change: necessary to make xx go smoothly from -0.5 to 0.5, rather than a step function (step at ny//2) from -1.0 to 0.0      
    #xx = (np.arange(ny)-ny//2)/ny
    xx = (np.arange(ny)-ny//2)/float(ny)
    norm = np.exp(parms[1]*xx*xx + parms[2]*xx + parms[3])     #CMB change for speed (was *xx**2)
    # Lets get this sign correct. A redshift (positive velocity) means that
    # a given wavelength for the reference corresponds to a longer  
    # wavelength for the target, which in turn means that the target 
    # wavelength has to be interpolated onto shorter wavelengths for the 
    # reference.
    #fitted_spect = spline_ref(wave*(1.0 - parms[0]/const.c.si.value))*norm
    # CMB change: just declared c above
    fitted_spect = spline_ref(wave * (1.0 - parms[0]/c)) * norm
    
    if return_spect:
        return fitted_spect
    else:
        return (fitted_spect - spect)/spect_sdev



def rv_shift_chi2(parms, wave, spect, spect_sdev, spline_ref):
    """Find the chi-squared for an RV fit. Just a wrapper for rv_shift_resid,
    so the docstring is cut and paste!
    
    The function for parameters p[0] through p[3] is:
    
    .. math::
        y(x) = Ref[ wave(x) * (1 - p[0]/c) ] * exp(p[1] * x^2 + p[2] * x + p[3])
    
    Here "Ref" is a function f(wave)
     
    Parameters
    ----------
    
    params: 
        ...
    wave: float array
        Wavelengths for the observed spectrum.
    spect: float array
        The observed spectrum
    spect_sdev: 
        ...
    spline_ref: 
        ...
    return_spect: boolean
        Whether to return the fitted spectrum or the 
        
    wave_ref: float array
        The wavelengths of the reference spectrum
    ref: float array
        The reference spectrum
    
    Returns
    -------
    chi2:
        The fit chi-squared
    """
    return np.sum(rv_shift_resid(parms, wave, spect, spect_sdev, spline_ref)**2)



def rv_shift_jac(parms, wave, spect, spect_sdev, spline_ref):
    """Explicit Jacobian function for rv_shift_resid. 
    
    This is not a completely analytic solution, but without it there seems to be 
    numerical instability.
    
    The key equations are:
    
    .. math:: f(x) = R( \lambda(x)  (1 - p_0/c) ) \times \exp(p_1 x^2 + p_2 x + p_3)
    
       g(x) = (f(x) - d(x))/\sigma(x)
    
       \frac{dg}{dp_0}(x) \approx  [f(x + 1 m/s) -f(x) ]/\sigma(x)
    
       \frac{dg}{dp_1}(x) = x^2 f(x) / \sigma(x)
    
       \frac{dg}{dp_2}(x) = x f(x) / \sigma(x)
    
       \frac{dg}{dp_3}(x) = f(x) / \sigma(x)
    
    Parameters
    ----------
    
    params: float array
    wave: float array
        Wavelengths for the observed spectrum.
    spect: float array
        The observed spectrum
    spect_sdev: 
        ...
    spline_ref: 
        ...
        
    Returns
    -------
    jac: 
        The Jacobian.
    """
    
    ny = len(spect)
    # CMB change: necessary to make xx go smoothly from -0.5 to 0.5, rather than a step function (step at ny//2) from -1.0 to 0.0      
    #xx = (np.arange(ny)-ny//2)/ny
    xx = (np.arange(ny)-ny//2)/float(ny)
    norm = np.exp(parms[1]*xx*xx + parms[2]*xx + parms[3])     #CMB change for speed (was *xx**2)
    #fitted_spect = spline_ref(wave*(1.0 - parms[0]/const.c.si.value))*norm
    fitted_spect = spline_ref(wave * (1.0 - parms[0]/c)) * norm
    
    #The Jacobian is the derivative of fitted_spect/spect_sdev with respect to p[0] through p[3]
    jac = np.empty((ny,4))
    jac[:,3] = fitted_spect / spect_sdev
    jac[:,2] = fitted_spect*xx / spect_sdev
    jac[:,1] = fitted_spect*xx*xx / spect_sdev     #CMB change for speed (was *xx**2)
    #jac[:,0] = (spline_ref(wave*(1.0 - (parms[0] + 1.0)/const.c.si.value))*
    #            norm - fitted_spect)/spect_sdev
    jac[:,0] = ((spline_ref(wave * (1.0 - (parms[0] + 1.0)/c)) * norm) - fitted_spect) / spect_sdev
    
    return jac



def calculate_rv_shift(wave_ref, ref_spect, fluxes, vars, bcors, wave, return_fitted_spects=False, bad_threshold=10):
    """Calculates the Radial Velocity of each spectrum
    
    The radial velocity shift of the reference spectrum required
    to match the flux in each order in each input spectrum is calculated.
    
    The input fluxes to this method are flat-fielded data, which are then fitted with 
    a barycentrically corrected reference spectrum :math:`R(\lambda)`, according to 
    the following equation:

    .. math::
        f(x) = R( \lambda(x)  (1 - p_0/c) ) \\times \exp(p_1 x^2 + p_2 x + p_3)

    The first term in this equation is simply the velocity corrected spectrum, based on a 
    the arc-lamp derived reference wavelength scale :math:`\lambda(x)` for pixels coordinates x.
    The second term in the equation is a continuum normalisation - a shifted Gaussian was 
    chosen as a function that is non-zero everywhere. The scipy.optimize.leastsq function is used
    to find the best fitting set fof parameters :math:`p_0` through to :math`p_3`. 

    The reference spectrum function :math:`R(\lambda)` is created using a wavelength grid 
    which is over-sampled with respect to the data by a factor of 2. Individual fitted 
    wavelengths are then found by cubic spline interpolation on this :math:`R_j(\lambda_j)` 
    discrete grid.
    
    Parameters
    ----------
    wave_ref: 2D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel*2+2), 
        where the wavelength scale has been interpolated.
    ref_spect: 2D np.array(float)
        Reference spectrum of form (Order, Flux/pixel*2+2), 
        where the flux scale has been interpolated.
    fluxes: 3D np.array(float)
        Fluxes of form (Observation, Order, Flux/pixel)
    vars: 3D np.array(float)
        Variance of form (Observation, Order, Variance/pixel)    
    bcors: 1D np.array(float)
        Barycentric correction for each observation.
    wave: 2D np.array(float)
        Wavelength coordinate map of form (Order, Wavelength/pixel)

    Returns
    -------
    rvs: 2D np.array(float)
        Radial velocities of format (Observation, Order)
    rv_sigs: 2D np.array(float)
        Radial velocity sigmas of format (Observation, Order)
    """
    nm = fluxes.shape[1]
    ny = fluxes.shape[2]
    nf = fluxes.shape[0]
    
    # initialise output arrays
    rvs = np.zeros( (nf,nm) )
    rv_sigs = np.zeros( (nf,nm) )
    initp = np.zeros(4)
    initp[3]=0.5
    initp[0]=0.0
    spect_sdev = np.sqrt(vars)
    fitted_spects = np.empty(fluxes.shape)
    
    #loop over all fibres(?)
    for i in range(nf):
        # Start with initial guess of no intrinsic RV for the target.
        initp[0] = -bcors[i] #!!! New Change 
        nbad=0
        #loop over all orders(?)
        for j in range(nm):
            # This is the *only* non-linear interpolation function that 
            # doesn't take forever
            spl_ref = interp.InterpolatedUnivariateSpline(wave_ref[j,::-1], ref_spect[j,::-1])
            args = (wave[j,:], fluxes[i,j,:], spect_sdev[i,j,:], spl_ref)
            
            # Remove edge effects in a slightly dodgy way. 
            # 20 pixels is about 30km/s. 
            args[2][:20] = np.inf
            args[2][-20:] = np.inf
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1],Dfun=rv_shift_jac, full_output=True)
            #the_fit = op.leastsq(self.rv_shift_resid, initp, args=args,diag=[1e3,1e-6,1e-3,1], full_output=True,epsfcn=1e-9)
            
            #The following line also doesn't work "out of the box".
            #the_fit = op.minimize(self.rv_shift_chi2,initp,args=args)
            #pdb.set_trace()
            #Remove bad points...
            resid = rv_shift_resid( the_fit[0], *args)
            wbad = np.where( np.abs(resid) > bad_threshold)[0]
            nbad += len(wbad)
            #15 bad pixels in a single order is *crazy*
            if len(wbad)>20:
                fitted_spect = rv_shift_resid(the_fit[0], *args, return_spect=True)
                plt.clf()
                plt.plot(args[0], args[1])
                plt.plot(args[0][wbad], args[1][wbad],'o')
                plt.plot(args[0], fitted_spect)
                plt.xlabel("Wavelength")
                plt.ylabel("Flux")
                #print("Lots of 'bad' pixels. Type c to continue if not a problem")
                #pdb.set_trace()

            args[2][wbad] = np.inf
            the_fit = op.leastsq(rv_shift_resid, initp, args=args, diag=[1e3,1,1,1], Dfun=rv_shift_jac, full_output=True)
            #the_fit = op.leastsq(self.rv_shift_resid, initp,args=args, diag=[1e3,1e-6,1e-3,1], full_output=True, epsfcn=1e-9)
            
            #Some outputs for testing
            fitted_spects[i,j] = rv_shift_resid(the_fit[0], *args, return_spect=True)
            #the_fit[0][0] is the RV shift
            if ( np.abs(the_fit[0][0] - bcors[i]) < 1e-4 ):
                #pdb.set_trace() #This shouldn't happen, and indicates a problem with the fit.
                pass
            #Save the fit and the uncertainty.
            rvs[i,j] = the_fit[0][0]
            try:
                rv_sigs[i,j] = np.sqrt(the_fit[1][0,0])
            except:
                rv_sigs[i,j] = np.NaN
        print("Done file {0:d}. Bad spectral pixels: {1:d}".format(i,nbad))
    if return_fitted_spects:
        return rvs, rv_sigs, fitted_spects
    else:
        return rvs, rv_sigs
 
#########################################################################################
#########################################################################################
#########################################################################################











