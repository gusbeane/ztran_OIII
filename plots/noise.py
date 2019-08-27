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

def sigmaN(lam, IZL, tobs, epsilon, R=300, D=84*u.cm, Omegapix=(1*u.arcsecond)**2):
    sigmaN0 = 1.37 * u.nW/u.m**2/u.sr
    fac = (1 * u.micron)/lam
    fac *= R/100
    fac *= IZL/(1E3*u.nW/u.m**2/u.sr)
    fac *= 0.126 * u.m**2/(np.pi*D**2)
    fac *= 8.5E-10 * u.sr / (Omegapix)
    fac *= 1E5 * u.s / (tobs)
    fac *= 1./np.sqrt(epsilon)
    fac = np.sqrt(fac).to_value(u.dimensionless_unscaled)

    sigmaN0 *= fac
    obs_freq = c/lam
    sigmaN0 /= obs_freq

    return sigmaN0.to(u.erg/u.s/u.cm**2/u.Hz/u.sr)
    

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
    z_lower = ((wave_lower-wave_emit)/wave_emit).to_value(u.dimensionless_unscaled)
    z = (wave_obs/wave_emit).to_value(u.dimensionless_unscaled) - 1

    d_upper = cosmo.comovingDistance(z_max=z_upper)/cosmo.h * u.Mpc
    d_lower = cosmo.comovingDistance(z_max=z_lower)/cosmo.h * u.Mpc
    dpar = d_upper - d_lower
    print(dpar, cosmo.comovingDistance(z_lower, z_upper)/cosmo.h * u.Mpc)

    xpix = (pix_length_in_arcsecond * u.arcsecond).to_value(u.radian)
    dperp = xpix * cosmo.comovingDistance(0, z)/cosmo.h * u.Mpc
    print(dperp)

    return dpar * dperp**2

def calc_Nmodes(kmin, kmax, z, deltaz, Asurv=31.1):
    z_upper = z + deltaz/2.0
    z_lower = z - deltaz/2.0
    dlos = cosmo.comovingDistance(z_lower, z_upper)/cosmo.h

    kmin = kmin.to_value(1/u.Mpc)
    kmax = kmax.to_value(1/u.Mpc)

    # convert A surv to on sky width
    Lindeg = np.sqrt(Asurv)
    Lperp = cosmo.angularDiameterDistance(z)/cosmo.h * Lindeg * (np.pi/180.)

    Vfund = (2.*np.pi)**3 / (Lperp**2 * dlos)
    Vkspace = (4*np.pi/3.) * (kmax**3 - kmin**3)

    return Vkspace / Vfund

def compute_Nx(wave_obs, wave_emit, sigma, surface_brightness=True):
    vpix = calc_vpix(wave_obs, wave_emit)
    ans = sigma**2 * vpix
    if surface_brightness:
        ans *= (c/wave_obs)**2
    return ans

def _var_auto_singlemode(P, N):
    Ptot = P + N
    return np.square(Ptot)

def var_auto(line, z, deltaz, kmin, kmax, sigma, Asurv, b=4, return_signal=True, 
             surface_brightness=True, dimensionless=False):
    Nmodes = calc_Nmodes(kmin, kmax, z, deltaz, Asurv)

    kcen = (kmin + kmax)/2.0

    wave_emit = line_wavelength[line]
    wave_obs = wave_emit * (1. + z)
    freq_obs = c/wave_obs

    P = intensity_power_spectrum(z, kcen, line=line, b=b, dimensionless=False, 
                                 surface_brightness=surface_brightness)
    N = compute_Nx(wave_obs, wave_emit, sigma, surface_brightness=surface_brightness)
    if dimensionless:
        P *= (kcen**3)/(2.*np.pi**2)
        N *= (kcen**3)/(2.*np.pi**2)

    print(P, N)
    var_singlemode = _var_auto_singlemode(P, N)
    var = var_singlemode/Nmodes
    print(var)

    if return_signal:
        # return var.to((u.erg/u.s/u.cm**2/u.sr)**4), P.to((u.erg/u.s/u.cm**2/u.sr)**2)
        return var, P
    else:
        return var.to((u.erg/u.s/u.cm**2/u.sr)**4)
