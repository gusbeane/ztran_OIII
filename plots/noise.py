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

