import glob
import numpy as np
import re
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import ridder

import astropy.units as u
from astropy.constants import c
from scipy.interpolate import interp2d

from colossus.cosmology import cosmology
from helper import read_xps

params = {'flat': True, 'H0': 67.66, 'Om0': 0.3111, 'Ob0': 0.02242/(0.6766**2), 'sigma8': 0.8102, 'ns': 0.9665}
cosmo = cosmology.setCosmology('myCosmo', params)

h = 0.6766

def line_intensity(z, line):
    if line == 'Halpha':
        wave = 656.45377 * u.nm
        L0 = 3.29E7 * u.Lsun / (u.Msun/u.year)
    elif line == 'Hbeta':
        wave = 486.13615 * u.nm
        L0 = 0.35 * 3.29E7 * u.Lsun / (u.Msun/u.year)
    elif line == 'Lyalpha':
        wave = 121.567 * u.nm
        L0 = 2.859E8 * u.Lsun / (u.Msun/u.year)       

    def psi(zp):
        ans = 0.015
        ans *= (1+zp)**(2.7)
        ans /= 1 + ((1+zp)/2.9)**(5.6)
        return ans

    psi_at_z = psi(z) * u.Msun/u.year/(u.Mpc**3)
    Hz = cosmo.Hz(z) * u.km/u.s/u.Mpc
    nurest = c/wave

    ans = (L0*psi_at_z / (4.*np.pi*nurest)) * (c/Hz)
    ans = ans.to_value(u.Jy)
    return ans

def Halpha_intensity(z, Hbeta=False):
    if Hbeta:
        wave = 486.13615 * u.nm
        L0 = 0.35 * 3.29E7 * u.Lsun / (u.Msun/u.year)
    else:
        wave = 656.45377 * u.nm
        L0 = 3.29E7 * u.Lsun / (u.Msun/u.year)

    def psi(zp):
        ans = 0.015
        ans *= (1+zp)**(2.7)
        ans /= 1 + ((1+zp)/2.9)**(5.6)
        return ans

    psi_at_z = psi(z) * u.Msun/u.year/(u.Mpc**3)
    Hz = cosmo.Hz(z) * u.km/u.s/u.Mpc
    nurest = c/wave

    ans = (L0*psi_at_z / (4.*np.pi*nurest)) * (c/Hz)
    ans = ans.to_value(u.Jy)
    return ans

def compute_Nx(wave_obs, sigma=4E4, wave_emit=0.65628):
    vpix = calc_vpix(wave_obs, wave_emit=wave_emit)
    ans = sigma**2 * vpix
    return ans

def Nmodes(kmin, kmax, z, deltaz, Asurv=31.1):
    z_upper = z + deltaz/2.0
    z_lower = z - deltaz/2.0
    dlos = cosmo.comovingDistance(z_lower, z_upper)/cosmo.h

    # convert A surv to on sky width
    Lindeg = np.sqrt(Asurv)
    Lperp = cosmo.angularDiameterDistance(z)/cosmo.h * Lindeg * (np.pi/180.)

    Vfund = (2.*np.pi)**3 / (Lperp**2 * dlos)
    Vkspace = (4*np.pi/3.) * (kmax**3 - kmin**3)

    return Vkspace / Vfund

def calc_vpix(wave_obs, pix_length_in_arcsecond=1, R=300, wave_emit=0.65628):
    # compute the upper and lower redshift from the spectrograph resolution
    dlambda = wave_obs/R
    wave_upper = wave_obs + dlambda/2.0
    wave_lower = wave_obs - dlambda/2.0

    z_upper = (wave_upper-wave_emit)/wave_emit
    z_lower = (wave_lower-wave_emit)/wave_emit

    d_upper = cosmo.comovingDistance(z_max=z_upper)/cosmo.h
    d_lower = cosmo.comovingDistance(z_max=z_lower)/cosmo.h
    volume = (4.*np.pi/3.) * (d_upper**3 - d_lower**3)

    Apix = (pix_length_in_arcsecond * u.arcsecond)**2
    Apix = Apix.to_value(u.deg**2)
    tot_sky_in_degree = (4.*np.pi * u.steradian).to_value(u.deg**2)
    Apix_fraction = Apix / tot_sky_in_degree

    return Apix_fraction * volume

def construct_interpolators(zlist, klist, xps, pdelta, p21):
    fn_xps = interp2d(zlist, klist, np.transpose(xps))
    fn_pdelta = interp2d(zlist, klist, np.transpose(pdelta))
    fn_p21 = interp2d(zlist, klist, np.transpose(p21))

    return fn_xps, fn_pdelta, fn_p21

def varp21x(P21, N21, Px, Nx):
    xps = np.sqrt(P21*Px)
    P21tot = P21 + N21
    Pxtot = Px + Nx
    return xps**2 + P21tot*Pxtot

def varp21x_wrapper(z, deltaz, kmin, kmax, zlist, klist, xps, pdelta, p21, b=4, sigma=4E4, wave_emit_x=0.65628):
    fn_xps, fn_pdelta, fn_p21 = construct_interpolators(zlist, klist, xps, pdelta, p21)

    kcen = (kmin+kmax)/2.0
    deltak = kmax-kmin

    P21 = fn_p21(z, kcen)

    Ix = Halpha_intensity(z)
    Px = (b*Ix)**2 * fn_pdelta(z, kcen)

    wave_obs_x = wave_emit_x*(1.+z)

    N21 = fn_p21(z, 0.1)
    Nx = compute_Nx(wave_obs_x, sigma=sigma)

    varxps = varp21x(P21, N21, Px, Nx)
    Nm = Nmodes(kmin, kmax, z, deltaz)
    xps = np.sqrt(P21*Px)

    return float(xps), float(np.sqrt(varxps/Nm))


if __name__ == '__main__':
    directory = '/Users/abeane/scratch/ztran_OIII_sims/v1.2/256Mpch/256/fid/MyOutput'
    zlist, klist, xps, pdelta, p21 = read_xps(directory+'/xps*', return_auto=True)
