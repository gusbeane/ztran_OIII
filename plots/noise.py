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

def intensity_power_spectrum(z, k, I=None, line=None, b=4, dimensionless=True, surface_brightness=True):
    Pden = cosmo.matterPowerSpectrum(k.to_value(1/u.Mpc)/cosmo.h, z)/(cosmo.h**3)
    Pden *= u.Mpc**3

    if I is None:
        assert line is not None, "If I is not specified you must specify the line"
        I = line_intensity(z, line, surface_brightness=surface_brightness)

    ans = (b*I)**2 * Pden

    if dimensionless:
        ans *= k**3 / (2.*np.pi**2)

    return ans

def calc_vpix(wave_obs, wave_emit, pix_length_in_arcsecond=1, R=300):
    # compute the upper and lower redshift from the spectrograph resolution
    dlambda = wave_obs/R
    wave_upper = wave_obs + dlambda/2.0
    wave_lower = wave_obs - dlambda/2.0

    z_upper = ((wave_upper-wave_emit)/wave_emit).to_value(u.dimensionless_unscaled)
    z_lower = (wave_lower-wave_emit)/wave_emit.to_value(u.dimenisonless_unscaled)

    d_upper = cosmo.comovingDistance(z_max=z_upper)/cosmo.h
    d_lower = cosmo.comovingDistance(z_max=z_lower)/cosmo.h
    volume = (4.*np.pi/3.) * (d_upper**3 - d_lower**3)

    Apix = (pix_length_in_arcsecond * u.arcsecond)**2
    Apix = Apix.to_value(u.deg**2)
    tot_sky_in_degree = (4.*np.pi * u.steradian).to_value(u.deg**2)
    Apix_fraction = Apix / tot_sky_in_degree

    return Apix_fraction * volume

def calc_Nmodes(kmin, kmax, z, deltaz, Asurv=31.1):
    z_upper = z + deltaz/2.0
    z_lower = z - deltaz/2.0
    dlos = cosmo.comovingDistance(z_lower, z_upper)/cosmo.h

    # convert A surv to on sky width
    Lindeg = np.sqrt(Asurv)
    Lperp = cosmo.angularDiameterDistance(z)/cosmo.h * Lindeg * (np.pi/180.)

    Vfund = (2.*np.pi)**3 / (Lperp**2 * dlos)
    Vkspace = (4*np.pi/3.) * (kmax**3 - kmin**3)

    return Vkspace / Vfund

def compute_Nx(wave_obs, wave_emit, sigma):
    vpix = calc_vpix(wave_obs, wave_emit)
    ans = sigma**2 * vpix
    return ans

def _var_auto_singlemode(P, N):
    Ptot = P + N
    return np.square(Ptot)
