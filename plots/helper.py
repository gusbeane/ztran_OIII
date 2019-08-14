import glob
import numpy as np
import re
from scipy.interpolate import interp1d
from scipy.optimize import ridder

import astropy.units as u
from astropy.constants import c
from scipy.interpolate import interp2d

from colossus.cosmology import cosmology

params = {'flat': True, 'H0': 67.66, 'Om0': 0.3111, 'Ob0': 0.02242/(0.6766**2), 'sigma8': 0.8102, 'ns': 0.9665}
cosmo = cosmology.setCosmology('myCosmo', params)

h = 0.6766

def read_xps(files, return_auto=False):
    xps_list = glob.glob(files)
    zlist = np.array([float(re.findall('\d\d\d\.\d\d', f)[0]) for f in xps_list])
    
    output = []
    output_auto1 = []
    output_auto2 = []
    for i,key in enumerate(np.argsort(zlist)):
        fname = xps_list[key]
        data = np.genfromtxt(fname)
        xps = data[:,3]
        auto1 = data[:,1]
        auto2 = data[:,2]
        output.append(xps)
        output_auto1.append(auto1)
        output_auto2.append(auto2)
        if i ==0:
            klist = data[:,0]/h
    
    zlist = np.sort(zlist)

    output = np.array(output)
    output_auto1 = np.array(output_auto1)
    output_auto2 = np.array(output_auto2)
    if return_auto:
        return zlist, klist, output, output_auto1, output_auto2
    else:
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
    return ans.to_value(u.Jy)

def calc_vpix(wave_obs, pix_length_in_arcsecond=1, R=300, wave_emit=0.65628):
    dlambda = wave_obs/R
    wave_upper = wave_obs + dlambda/2.0
    wave_lower = wave_obs - dlambda/2.0

    z_upper = (wave_upper-wave_emit)/wave_emit
    z_lower = (wave_lower-wave_emit)/wave_emit

    d_upper = cosmo.comovingDistance(z_max=z_upper)
    d_lower = cosmo.comovingDistance(z_max=z_lower)
    volume = (4.*np.pi/3.) * (d_upper**3 - d_lower**3)

    Apix = (pix_length_in_arcsecond * u.arcsecond)**2
    Apix = Apix.to_value(u.deg**2)
    Apix_fraction = Apix/41253

    return Apix_fraction * volume

def Nmodes(k, deltak, z, deltaz, Asurv=31.1):
    z_upper = z + deltaz/2.0
    z_lower = z - deltaz/2.0
    dlos = cosmo.comovingDistance(z_lower, z_upper)

    # convert A surv to on sky width
    Lindeg = np.sqrt(Asurv)
    Lperp = cosmo.angularDiameterDistance(z) * Lindeg * (np.pi/180.)

    Vfund = (2.*np.pi)**3 / (Lperp**2 * dlos)

    return 4.*np.pi*k**2 * deltak / Vfund

def _calc_snr_(zst, P21, Pdelta, P21delta_deriv, N21, Nmodes, b, I, sigma=4E4, wave_emit=0.65628):
    wave_obs = (1.+zst)*wave_emit
    Vpix = calc_vpix(wave_obs)
    NHa = sigma**2 * Vpix

    Pi = b**2 * I**2 * Pdelta
    xps = np.sqrt(Pi*P21)

    P21tot = P21 + N21
    Pitot = Pi + NHa


    varP21i = xps**2 + P21tot*Pitot

    P21i_deriv = b * I * P21delta_deriv

    varz = varP21i / (P21i_deriv**2)
    return np.sqrt(varz/Nmodes)

def snr_wrapper(zst, zlist, klist, xps, pdelta, p21, kmin, kmax, deltaz, b=4):
    keval = (kmin+kmax)/2.0
    deltak = kmax - kmin

    fn = interp2d(zlist, klist, np.transpose(xps))
    fn_pdelta = interp2d(zlist, klist, np.transpose(pdelta))
    fn_p21 = interp2d(zlist, klist, np.transpose(p21))

    I = Halpha_intensity(zst)

    # i know im numerically differentiating an interpolation
    # i know im a bad person, pls do not write me angry emails about it
    # or do it might be kind of entertaining
    Pup = fn(zst+0.01, keval)
    Pdown = fn(zst-0.01, keval)
    Pderiv = (Pup-Pdown)/0.02

    P21 = fn_p21(zst, keval)
    Pdelta = fn_pdelta(zst, keval)
    N21 = fn_p21(zst, 0.1)

    Nm = Nmodes(keval, deltak, zst, deltaz)
    print(Nm)

    sigmazst = _calc_snr_(zst, P21, Pdelta, Pderiv, N21, Nm, b, I)
    return float(sigmazst)

def add_snr_in_quadrature(zst, zlist, klist, xps, pdelta, p21, kmin, kmax, Nk, deltaz, b=4):
    kbins = np.linspace(kmin, kmax, Nk)
    sigmaz_tot = 0
    for i in range(len(kbins)-1):
        this_kmin = kbins[i]
        this_kmax = kbins[i+1]
        this_sigmaz = snr_wrapper(zst, zlist, klist, xps, pdelta, p21, this_kmin, this_kmax, deltaz, b=4)
        sigmaz_tot += this_sigmaz**(-2) # add them in inverse quadrature
    sigmaz_tot = sigmaz_tot**(-0.5)
    return sigmaz_tot


if __name__ == '__main__':
    directory = '/Users/abeane/scratch/ztran_OIII_sims/v1.2/256Mpch/256/fid/MyOutput'
    zlist, klist, xps, pdelta, p21 = read_xps(directory+'/xps*', return_auto=True)

    sigmazst = add_snr_in_quadrature(10, zlist, klist, xps, pdelta, p21, 0.1, 1.0, 10, 0.1)
