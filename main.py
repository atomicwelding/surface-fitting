import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from tifffile import imread

import pandas as pd


# images
taus = pd.read_csv('output.tif_delays.dat')['Real_Time']
ddm = imread('output.tif')


# init
_, leny, lenx = ddm.shape

""" We may want to use only a portion of the image,
So let's define an area of interest"""
size_aoi = 11
aoim = int(leny//2 - size_aoi/2)
aoip = int(leny//2 + size_aoi/2)

print(f"aoim={aoim},aoip={aoip}")

raw_frequencies = np.empty((size_aoi, size_aoi))
As = np.empty((size_aoi,size_aoi))
Bs = np.empty((size_aoi,size_aoi))

# utils
exp_to_fit = lambda tau, A, f, B: A * (1-np.exp(-f*tau)) + B

# fitting raw frequencies
for y in range(size_aoi):
    print(f"Computing ... {(y+1)/size_aoi :.2f}")
    for x in range(size_aoi):
        fitted = curve_fit(exp_to_fit, taus, ddm[:,aoim+y,aoim+x], p0=(ddm[-1,aoim+y,aoim+x], 10, 0))[0]
        As[y,x] = fitted[0]
        raw_frequencies[y,x] = fitted[1]
        Bs[y,x] = fitted[2]

"""We now want to smooth surface"""
di = 1
size_smoothing = size_aoi - 2*di
decay_frequencies = np.empty((size_smoothing, size_smoothing))

surffit = lambda r, f0, xx, yy, xy, x, y: xx*r[0,:]**2 + xy*r[1,:]**2 + yy*r[1,:]**2 + x*r[0,:] + y*r[1,:] + f0 

ivals = np.arange(-di, di+1)
Y,X = np.meshgrid(ivals, ivals, indexing="ij")
r = np.stack((X,Y), axis=-1).reshape(((2*di+1)**2,2)).transpose()

for y in range(size_smoothing):
    for x in range(size_smoothing):
        local_frequencies = raw_frequencies[y:2*di+y+1, x:2*di+x+1]
        decay_frequencies[y,x] = curve_fit(surffit, r, local_frequencies.flatten(),
                                            p0=(np.mean(local_frequencies), 0,0,0,0,0))[0][0]

plt.plot(taus, ddm[:,aoim+di,aoim+di], 'r')

exp = np.vectorize(exp_to_fit)
raw = exp(taus, As[di,di], raw_frequencies[di,di], Bs[di,di])
smoothed = exp(taus, As[di,di], decay_frequencies[0,0], Bs[di,di])

plt.plot(taus, raw, 'k--') 
plt.plot(taus, smoothed, 'k')

plt.legend(['raw','fit','smoothed'])
plt.show()
        



#to_plot = np.vectorize(exp_to_fit) 
#plt.plot(taus, to_plot(taus, As[5,5], raw_frequencies[5,5], Bs[5,5]), 'k')
#plt.plot(taus, ddm[:,aoim+5, aoim+5], 'r')

#plt.xlabel('$tau$')
#plt.ylabel('relaxation frequencies')
#plt.show()


        
