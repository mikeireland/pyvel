'''
Created on 18 Oct. 2017

@author: christoph
'''

C = []
I = []
O = []
tot = []
seeing = np.arange(.5,5,.1)
# for fwhm in seeing:
#     fluxes = flux_ratios_from_seeing(fwhm, return_values=True)
#     C.append(fluxes[1])
#     I.append(fluxes[2])
#     O.append(fluxes[3])
#     tot.append(fluxes[0])



def flux_ratios_from_seeing(seeing, return_values=False):
    '''
    FWHM - the FWHM of the seeing disk (approximated as a 2-dim Gaussian)
    '''
    
    #stupid python...
    seeing = np.array(seeing)
    
    #inner-radius r of hexagonal fibre is 0.26", therefore outer-radius R is (2/sqrt(3))*0.26" = 0.30"
    #what we really want is the "effective" radius though, for a regular hexagon that comes from A = 3*r*R = 2*sqrt(3)*r = pi * r_eff**2
    #ie r_eff = sqrt( (2*sqrt(3)) / pi )
    fac = np.sqrt((2.*np.sqrt(3.)) / np.pi)
    rc = 0.26
    Rc = rc * 2. / np.sqrt(3.) 
    ri = 0.78
    ro = 1.30
    reff = rc * fac
    di = 0.52
    do1 = 1.04
    xo2 = di + rc
    yo2 = 1.5 * Rc
    do2 = 3. * (2./np.sqrt(3.)) * 0.26
    
    x = np.arange(-20,20,.01)
    y = np.arange(-20,20,.01)
    xx, yy = np.meshgrid(x, y)
    #define constant (FWHM vs sigma, because FWHM = sigma * 2*sqrt(2*log(2))
    cons = 2*np.sqrt(np.log(2)) 
    
    #central = (np.sqrt(xx*xx + yy*yy) <= rc)
    #inner = np.logical_and(np.sqrt(xx*xx + yy*yy) <= ri, ~central)
    #outer = np.logical_and(np.sqrt(xx*xx + yy*yy) <= ro, np.logical_and(~central, ~inner))
    #ifu = (np.sqrt(xx*xx + yy*yy) <= ro)
    m1 = 1./np.sqrt(3.)  #slope1
    m2 = -m1              #slope2
    central = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx) <= rc, yy <= m1*xx + Rc), yy >= m1*xx - Rc), yy <= m2*xx + Rc), yy >= m2*xx - Rc)
    inner = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-di) <= rc, yy <= m1*(xx-di) + Rc), yy >= m1*(xx-di) - Rc), yy <= m2*(xx-di) + Rc), yy >= m2*(xx-di) - Rc), ~central)
    outer1 = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-do1) <= rc, yy <= m1*(xx-do1) + Rc), yy >= m1*(xx-do1) - Rc), yy <= m2*(xx-do1) + Rc), yy >= m2*(xx-do1) - Rc), ~inner)
    outer2 = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.abs(xx-xo2) <= rc, yy <= m1*(xx-xo2) + Rc - yo2), yy >= m1*(xx-xo2) - Rc - yo2), yy <= m2*(xx-xo2) + Rc - yo2), yy >= m2*(xx-xo2) - Rc - yo2), ~inner), ~outer1)
#     central = (np.sqrt(xx*xx + yy*yy) <= reff)
#     inner = np.sqrt((xx-di)*(xx-di) + yy*yy) <= reff
#     outer1 = np.sqrt((xx-do1)*(xx-do1) + yy*yy) <= reff
#     outer2 = np.sqrt((xx-do2)*(xx-do2) + yy*yy) <= reff
    #ifu = (np.sqrt(xx*xx + yy*yy) <= ro*fac)
    
    if len(np.atleast_1d(seeing)) > 1:
        
        frac_ifu = []
        renorm_frac_c = []
        renorm_frac_i = []
        renorm_frac_o = []
        
        for fwhm in seeing:
        
            if not return_values:
                print('Simulating '+str(fwhm)+'" seeing...')
            
            #calculate 2-dim Gaussian flux distribution as function of input FWHM
            fx = np.exp(-(np.absolute(xx) * cons / fwhm) ** 2.0)
            fy = np.exp(-(np.absolute(yy) * cons / fwhm) ** 2.0)
            f = fx * fy * 1e6
            
            frac_c = np.sum(f[central]) / np.sum(f)
            frac_i = 6. * np.sum(f[inner]) / np.sum(f)
            frac_o = 6. * ( np.sum(f[outer1]) + np.sum(f[outer2]) ) / np.sum(f)
            ifu_frac = frac_c + frac_i + frac_o     #this is slightly overestimated (<1%) because they are not actually circular fibres
            frac_ifu.append(ifu_frac)
            
            rfc = frac_c / ifu_frac
            rfi = frac_i / ifu_frac
            rfo = frac_o / ifu_frac
            
            renorm_frac_c.append(rfc)
            renorm_frac_i.append(rfi)
            renorm_frac_o.append(rfo)
            
            if not return_values:
                print('Total fraction of flux captured by IFU: '+str(np.round(ifu_frac * 100,1))+'%')
                print('----------------------------------------------')
                print('Contribution from central fibre: '+str(np.round(rfc * 100,1))+'%')
                print('Contribution from inner-ring fibres: '+str(np.round(rfi * 100,1))+'%')
                print('Contribution from outer-ring fibres: '+str(np.round(rfo * 100,1))+'%')
                print
    
    else:
        
        fwhm = seeing
        if not return_values:
                print('Simulating '+str(fwhm)+'" seeing...')
        #calculate 2-dim Gaussian flux distribution as function of input FWHM
        fx = np.exp(-(np.absolute(xx) * cons / fwhm) ** 2.0)
        fy = np.exp(-(np.absolute(yy) * cons / fwhm) ** 2.0)
        f = fx * fy * 1e6
        frac_c = np.sum(f[central]) / np.sum(f)
        frac_i = 6. * np.sum(f[inner]) / np.sum(f)
        frac_o = 6. * ( np.sum(f[outer1]) + np.sum(f[outer2]) ) / np.sum(f)
        frac_ifu = frac_c + frac_i + frac_o     #this is slightly overestimated (<1%) because they are not actually circular fibres
        renorm_frac_c = frac_c / frac_ifu
        renorm_frac_i = frac_i / frac_ifu
        renorm_frac_o = frac_o / frac_ifu
        if not return_values:
                print('Total fraction of flux captured by IFU: '+str(np.round(frac_ifu * 100,1))+'%')
                print('----------------------------------------------')
                print('Contribution from central fibre: '+str(np.round(renorm_frac_c * 100,1))+'%')
                print('Contribution from inner-ring fibres: '+str(np.round(renorm_frac_i * 100,1))+'%')
                print('Contribution from outer-ring fibres: '+str(np.round(renorm_frac_o * 100,1))+'%')
                print
    
    if return_values:
        return frac_ifu, renorm_frac_c, renorm_frac_i, renorm_frac_o
    else:
        return 1
    
    
    
    
    