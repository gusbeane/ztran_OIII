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

def Nmodes(k, deltak, z, deltaz, Asurv=31.1):
    z_upper = z + deltaz/2.0
    z_lower = z - deltaz/2.0
    dlos = cosmo.comovingDistance(z_lower, z_upper)/cosmo.h
    print(z, deltaz, z_lower, z_upper, dlos)

    # convert A surv to on sky width
    Lindeg = np.sqrt(Asurv)
    Lperp = cosmo.angularDiameterDistance(z)/cosmo.h * Lindeg * (np.pi/180.)

    Vfund = (2.*np.pi)**3 / (Lperp**2 * dlos)

    return 4.*np.pi*k**2 * deltak / Vfund
