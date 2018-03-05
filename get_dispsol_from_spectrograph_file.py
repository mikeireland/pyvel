'''
Created on 9 Nov. 2017

@author: christoph
'''

import numpy as np
import h5py
import pandas
import math


def compose_matrix(sx,sy,shear,rot,tx,ty):
    m = np.zeros((3,3))
    m[0,0] = sx * math.cos(rot)
    m[1,0] = sx * math.sin(rot)
    m[0,1] = -sy * math.sin(rot+shear)
    m[1,1] = sy * math.cos(rot+shear)
    m[0,2] = tx
    m[1,2] = ty
    m[2,2] = 1
    return m



def center(df, width, height):
    m = compose_matrix(df['scale_x'], df['scale_y'], df['shear'], df['rotation'], df['translation_x'], df['translation_y'])
    xy = m.dot([width, height, 1])
    return xy[0], xy[1]



def get_dispsol_from_spectrograph_file(spectrograph = '/Users/christoph/UNSW/EchelleSimulator/data/spectrographs/VeloceRosso.hdf',outfn = '/Users/christoph/UNSW/dispsol/dispsol_by_fibres_from_zemax.npy',savefile=False):
    
    # overview = pandas.read_hdf(spectrograph)
    # data = pandas.read_hdf(specfn, "fiber_1/order100")
    
    f = h5py.File(spectrograph, 'r+')
    
    dispsol = {}
    
    #first and second keys ('CCD' and 'spectrograph') are excluded as they are empty
    for fibkey in list(f.keys())[2:]:
        dispsol[fibkey] = {}
        for ord in list(f[fibkey].keys())[0:43]:
#             tx = np.array(f[fibkey][ord]['translation_x'])
#             ty = np.array(f[fibkey][ord]['translation_y'])
#             sx = np.array(f[fibkey][ord]['scale_x'])
#             sy = np.array(f[fibkey][ord]['scale_y'])
#             shear = np.array(f[fibkey][ord]['shear'])
#             rot = np.array(f[fibkey][ord]['rotation'])
#             wl = np.array(f[fibkey][ord]['wavelength']) * 1e3        #wavelength in nm
            
            #read from HDF file        
            data = pandas.read_hdf('/Users/christoph/UNSW/EchelleSimulator/data/spectrographs/VeloceRosso.hdf', fibkey+"/"+ord).sort_values("wavelength")
            dum = data.apply(center, axis=1, args=(54, 54))
            dum = np.array(list(map(list, dum)))
            X = dum[:,0]
            Y = dum[:,1]
            
            wl = data['wavelength'] * 1e3
            fitparms = np.poly1d(np.polyfit(X, wl, 5))
            
            dispsol[fibkey][ord] = {'x':X, 'y':Y, 'wl':wl, 'fitparms':fitparms}
            
    if savefile:
        np.save(outfn, dispsol)         
            
    return dispsol

    
    
def make_dispsol_by_orders(dispsol,outfn = '/Users/christoph/UNSW/dispsol/dispsol_by_orders_from_zemax.npy',savefile=False):
    by_orders = {}
    for fibkey in dispsol.keys():
        for ord in dispsol[fibkey].keys():
            by_orders[ord] = {}
    
    for ord in by_orders.keys():
        for fibkey in dispsol.keys():
                    
            dum = dispsol[fibkey][ord]
            by_orders[ord][fibkey] = dum
     
    if savefile:
        np.save(outfn, by_orders)         
            
    return by_orders    
        
        
        
def make_final_dispsol(savefile=False,outfn = '/Users/christoph/UNSW/dispsol/mean_dispsol_by_orders_from_zemax.npy'):        
    #dbf = np.load('/Users/christoph/UNSW/dispsol/dispsol_by_fibres_from_zemax.npy').item()
    dbo = np.load('/Users/christoph/UNSW/dispsol/dispsol_by_orders_from_zemax.npy').item()
    
    xx = np.arange(4096)
    
    # mean_x = np.zeros((len(dbo),len(dbo['order66']['fiber_1']['x'])))
    # mean_y = np.zeros((len(dbo),len(dbo['order66']['fiber_1']['y'])))
    mean_x = {}
    mean_y = {}
    
    for ord in dbo.keys():
        mean_x[ord] = []
        mean_y[ord] = []
        for i in range(len(dbo[ord]['fiber_1']['x'])):
            xarr = []
            yarr = []
            for fibkey in sorted(dbo[ord].iterkeys()):
                #only use fibres 3-21 (19 stellar fibres in current pseudo-slit layout)
                dum = fibkey[-2:]
                try:
                    fibnum = int(dum)
                except:
                    fibnum = int(dum[-1])
                if fibnum in range(3,22):
                    xarr.append(dbo[ord][fibkey]['x'][i])
                    yarr.append(dbo[ord][fibkey]['y'][i])
            mean_x[ord].append(np.mean(xarr))
            mean_y[ord].append(np.mean(yarr))
    
            
    mean_dispsol = {}
    
    for ord in dbo.keys(): 
        fitparms = np.poly1d(np.polyfit(mean_x[ord], dbo[ord]['fiber_1']['wl'], 5))     #this works because all fibres have the same 'wavelength' array
        x = np.array(mean_x[ord])
        y = np.array(mean_y[ord])
        wl = np.array(dbo[ord]['fiber_1']['wl'])
        sortix = np.array(x).argsort()
        x = x[sortix]
        y = y[sortix]
        wl = wl[sortix]
        mean_dispsol[ord] = {'x':x, 'y':y, 'wl':wl, 'model':fitparms}
        
    if savefile:
        np.save(outfn, mean_dispsol)     
           
    return mean_dispsol    
        
        
      