import glob
import numpy as np
import re
from scipy.interpolate import interp1d
from scipy.optimize import ridder

import astropy.units as u
from astropy.constants import c

from colossus.cosmology import cosmology

params = {'flat': True, 'H0': 67.66, 'Om0': 0.3111, 'Ob0': 0.02242/(0.6766**2), 'sigma8': 0.8102, 'ns': 0.9665}
cosmo = cosmology.setCosmology('myCosmo', params)

h = 0.6766

def read_xps(files):
    xps_list = glob.glob(files)
    zlist = np.array([float(re.findall('\d\d\d\.\d\d', f)[0]) for f in xps_list])
    
    output = []
    for i,key in enumerate(np.argsort(zlist)):
        fname = xps_list[key]
        data = np.genfromtxt(fname)
        xps = data[:,3]
        output.append(xps)
        if i ==0:
            klist = data[:,0]/h
    
    zlist = np.sort(zlist)

    output = np.array(output)
    return zlist, klist, output

def find_ztran(zlist, klist, xps):
    ztran = []
    for k in range(len(klist)):
        this_xps = xps[:,k]
        
        xps_fn_z = interp1d(zlist, this_xps)
        try:
            sol = ridder(xps_fn_z, 9, 12.3)
        except:
            sol = np.nan
        this_ztran = float(sol)
        ztran.append(this_ztran)

    return klist, np.array(ztran)

def gen_halo_catalog(fin, fout, n, Mcut=None):
    t = np.genfromtxt(fin)

    if Mcut is not None:
        keys = np.where(t[:,0] > Mcut)[0]
        t = t[keys]

    poslist = t[:,1:4]
    pos_in_n = (poslist*n).astype('int')

    ng = np.histogramdd(pos_in_n, bins=(n,n,n))[0]
    aven = np.mean(ng)
    ng = np.divide(np.subtract(ng, aven), aven)

    ng = ng.astype('f4')
    ng.tofile(fout)

def read_file(fname, npix, reshape=True):
    data = np.fromfile(fname, dtype='f4')
    assert len(data) == npix**3, "Data is not right length for number of specified pixels"

    if reshape:
        return np.reshape(data, (npix, npix, npix))
    else:
        return data

def compute_power_spectrum(field, Lbox, npix, kmin=None, kmax=None, deltak=1.4):
    if kmin is None:
        kmin=2.*np.pi/Lbox
    if kmax is None:
        kmax=kmin*npix

    bins = [kmin]
    while bins[-1] <= kmax:
        bins.append(bins[-1]*deltak)

    volume = Lbox**3
    tot_npix = npix**3

    field_ft = np.fft.fftn(field)
    k1d = np.fft.fftfreq(npix, d=Lbox/npix)

    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    kmag = np.sqrt(np.square(kx) + np.square(ky) + np.square(kz))

    kmag_1d = np.reshape(kmag, npix**3)
    field_ft_1d = np.reshape(field_ft, npix**3)

    Nlist = np.histogram(kmag_1d, bins)[0]
    klist = np.histogram(kmag_1d, bins, weights=kmag_1d)[0]/Nlist
    Pklist = np.histogram(kmag_1d, bins, weights=np.square(np.absolute(field_ft_1d))/volume)[0]/Nlist

    return klist, Pklist

def Halpha_intensity(z):
    def psi(zp):
        ans = 0.015
        ans *= (1+zp)**(2.7)
        ans /= 1 + ((1+zp)/2.9)**(5.6)
        return ans

    psi_at_z = psi(z) * u.Msun/u.year/(u.Mpc**3)
    L0 = 3.29E7 * u.Lsun / (u.Msun/u.year)
    Hz = cosmo.Hz(z) * u.km/u.s/u.Mpc
    nurest_Ha = c/(656.28 * u.nm)

    ans = (L0*psi_at_z / (4.*np.pi*nurest_Ha)) * (c/Hz)
    return ans

