import numpy as np 
import glob

import sys
sys.path.append('../')
from helper import *

import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath} \usepackage{bm}']

textwidth = 7.10000594991
columnwidth = 3.35224200913

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

base_dir = '/Users/abeane/scratch/ztran_OIII_sims/v1.2/256Mpch/256'
fid_dir = base_dir + '/fid'
cold_dir = base_dir + '/cold'
hot_dir = base_dir + '/hot'
dir_list = [hot_dir, fid_dir, cold_dir]
name_list = [r'\texttt{hot}', r'\texttt{fid}', r'\texttt{cold}']
fname_list = ['hot', 'fid', 'cold']
color_list = [tb_c[1], tb_c[-1], tb_c[3]]

fig, ax = plt.subplots(3, 1, figsize=(textwidth, 1.5*1.2*columnwidth), sharex=True)

for directory, name, fname, c in zip(dir_list, name_list, fname_list, color_list):
    try:
        zlist = np.load('zlist_'+fname+'.npy')
        nflist = np.load('nflist_'+fname+'.npy')
        aveTb_list = np.load('Tblist_'+fname+'.npy')
        ztran = np.load('ztran_'+fname+'.npy')
        xps_to_plot = np.load('xps_to_plot_'+fname+'.npy')
    except:
        dTfiles = glob.glob(directory+'/Boxes/delta_T*')
        zlist = np.array([float(re.findall('z\d\d\d\.\d\d', f)[0][1:]) for f in dTfiles])
        nflist = np.array([float(re.findall('nf\d.\d\d\d\d\d\d', f)[0][2:]) for f in dTfiles])

        aveTb_list = []
        for k in np.argsort(zlist):
            Tb = np.fromfile(dTfiles[k], dtype='f4')
            aveTb_list.append(np.mean(Tb))
        aveTb_list = np.array(aveTb_list)
    
        this_zlist, klist, xps = read_xps(directory+'/MyOutput/xps*')
        klist, ztran = find_ztran(this_zlist, klist, xps)
        # ztran = ztran[5]
        xps_to_plot = klist[3]**3 * xps[:,3]/(2.*np.pi**2)
        print('k=', klist[3])
    
        keys = np.argsort(zlist)
        nflist = nflist[keys]
        zlist = np.sort(zlist)

        np.save('zlist_'+fname+'.npy', zlist)
        np.save('nflist_'+fname+'.npy', nflist)
        np.save('Tblist_'+fname+'.npy', aveTb_list)
        np.save('ztran_'+fname+'.npy', ztran)
        np.save('xps_to_plot_'+fname+'.npy', xps_to_plot)

    ax[0].plot(zlist, 1.-nflist, c=c, label=name)
    ax[1].plot(zlist, aveTb_list, label=name, c=c)
    for x in ax:
        x.axvline(ztran[3], c=c, ls='dashed')
    ax[2].plot(zlist, xps_to_plot, label=name, c=c)

ax[0].set_yscale('log')
ax[2].set_yscale('symlog', linthreshy=0.0001)
ax[0].set_ylim(0.0001, 1)
ax[1].set_xlim(26, 6)
ax[1].set_ylim(-200, 50)

ax[0].set_ylabel(r'$\left<x_{\text{H~\textsc{ii}}}\right>$')
ax[2].set_xlabel(r'$z$')
ax[1].set_ylabel(r'$\left<\delta T_b\right>\,[\,\text{mK}\,]$')
ax[2].set_ylabel(r'$\Delta^2_{21,\delta}$')

ax[2].text(25.5, 0.1, r'$k = 0.082\,\text{Mpc}^{-1}$')

ax[0].legend(title='model', frameon=False)
fig.tight_layout()

fig.savefig('aveTb_nf.pdf')

