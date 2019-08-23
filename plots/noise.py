import numpy as np

import astropy.units as u
from astropy.constants import c

from colossus.cosmology import cosmology

params = {'flat': True, 'H0': 67.66, 'Om0': 0.3111, 'Ob0': 0.02242/(0.6766**2), 'sigma8': 0.8102, 'ns': 0.9665}
cosmo = cosmology.setCosmology('myCosmo', params)

line_wavelength = {'Halpha': 656.45377 * u.nm,
                   'Hbeta': 486.13615 * u.nm,
                   'Lyalpha': 121.567 * u.nm}

line_L0 = {'Halpha': 3.29E7 * u.Lsun / (u.Msun/u.year),
           'Hbeta': 0.35 * 3.29E7 * u.Lsun / (u.Msun/u.year),
           'Lyalpha': 2.859E8 * u.Lsun / (u.Msun/u.year)}

def psi(zp):
    ans = 0.015
    ans *= (1+zp)**(2.7)
    ans /= 1 + ((1+zp)/2.9)**(5.6)
    return ans * u.Msun/(u.year * u.Mpc**3)

def line_intensity(z, line, surface_brightness=True):
    assert isinstance(line, str), "line must be a string!"
    assert line in line_wavelength.keys(), "line: "+line+" not recognized!"

    wave = line_wavelength[line]
    L0 = line_L0[line]

    psi_at_z = psi(z)
    Hz = cosmo.Hz(z) * u.km/u.s/u.Mpc
    nurest = c/wave
    nuobs = nurest / (1. + z)

    ans = (L0*psi_at_z / (4.*np.pi*nurest)) * (c/Hz) / (1. * u.sr)
    
    if surface_brightness:
        ans *= nuobs
        ans = ans.to(u.erg/u.cm**2/u.s/u.sr)
    else:
        ans = ans.to(u.erg/u.cm**2/u.s/u.sr/u.Hz)

    return ans

