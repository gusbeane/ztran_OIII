import sys
sys.path.append('../')
from helper import read_xps, find_ztran

import numpy as np
import emcee
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt
import corner

zmin = 7
zmax = 15

def lnprior(theta):
    m, z0 = theta
    if m > 0.0 and zmin < z0 < zmax:
        return 0.0
    return -np.inf

def lnlike(theta, zlist, Plist, sigma):
    m, z0 = theta
    model = m*(zlist-z0)
    return -0.5 * np.sum(np.divide(np.square(np.subtract(Plist, model)), np.square(sigma)))

def lnprob(theta, zlist, Plist, sigma):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, zlist, Plist, sigma)

directory = '/Users/abeane/scratch/ztran_OIII_sims/v1.2/256Mpch/256/fid/MyOutput'
zlist, klist, xps, pdelta, p21 = read_xps(directory+'/xps*', return_auto=True)
ztran_klist, ztran = find_ztran(zlist, klist, xps)
our_ztran = ztran[5]
print(our_ztran)

fn_xps = interp2d(zlist, klist, np.transpose(xps))

zlist = np.arange(zmin, zmax, 0.4)
truth = fn_xps(zlist, 0.1)
data = np.random.normal(loc=truth, scale=4*np.abs(truth[-1]))

sigma = 5 * np.abs(truth[-1])
sigma = np.full(len(truth), sigma)

ndim, nwalkers = 2, 100
mlist = np.abs(2000.0*np.random.rand(nwalkers))
this_zlist = np.random.normal(our_ztran, 1, nwalkers)
pos = np.transpose([mlist, this_zlist])

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(zlist, data, sigma))
sampler.run_mcmc(pos, 10000)

samples = sampler.chain[:, 2000:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=['m', 'z0'], truths=[2000, our_ztran])
fig.savefig("test.png")

plt.close()

plt.plot(zlist, truth)
plt.scatter(zlist, data)

plt.xlabel('z')
plt.ylabel('xps (21 and delta)')

plt.savefig('signal_data.png')
