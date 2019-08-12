import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from helper import *

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
fid_dir = base_dir + '/fid' + '/MyOutput'
cold_dir = base_dir + '/cold' + '/MyOutput'
hot_dir = base_dir + '/hot' + '/MyOutput'
dir_list = [hot_dir, fid_dir, cold_dir]
name_list = [r'\texttt{hot}', r'\texttt{fid}', r'\texttt{cold}']
color_list = [tb_c[1], tb_c[-1], tb_c[3]]

alpha_fid = 1
sigma_fid = 0.4
alpha_fid_label = 'alpha'+"{:.3f}".format(alpha_fid)
sigma_fid_label = 'sigma'+"{:.3f}".format(sigma_fid)

alpha_list = np.linspace(0.5, 1.5, 11)
sigma_list = np.linspace(0.0, 0.8, 11)
alpha_label_list = ['alpha'+"{:.3f}".format(alpha) for alpha in alpha_list]
sigma_label_list = ['sigma'+"{:.3f}".format(sigma) for sigma in sigma_list]

fig, ax = plt.subplots(1, 2, figsize=(textwidth, columnwidth), sharey=True)

for name, directory, c in zip(name_list, dir_list, color_list):
    
    ztran_list = []
    for alpha_label in alpha_label_list:
        # print(directory+'/IM_xps/*'+alpha_label+'_'+sigma_fid_label)
        zlist, klist, xps = read_xps(directory+'/IM_xps/*'+alpha_label+'_'+sigma_fid_label)
        klist, ztran = find_ztran(zlist, klist, xps)
        ztran_list.append(ztran[5])
        # print(klist[5], ztran[5])

    print(name, 'alpha variation:', 100.0*(np.max(ztran_list) - np.min(ztran_list))/np.mean(ztran_list))
    ax[0].plot(alpha_list, ztran_list, label=name, c=c)

    ztran_list = []
    for sigma_label in sigma_label_list:
        zlist, klist, xps = read_xps(directory+'/IM_xps/*'+alpha_fid_label+'_'+sigma_label)
        klist, ztran = find_ztran(zlist, klist, xps)
        ztran_list.append(ztran[5])
        # print(klist[5], ztran[5])

    print(name, 'sigma variation:', 100.0*(np.max(ztran_list) - np.min(ztran_list))/np.mean(ztran_list))
    ax[1].plot(sigma_list, ztran_list, c=c)


ax[0].set_xlim(0.5, 1.5)
ax[1].set_xlim(0, 0.8)
ax[0].set_ylim(9.5, 14)

ax[0].set_xlabel(r'$\alpha$')
ax[1].set_xlabel(r'$\sigma$')
ax[0].set_ylabel(r'$z_{\star}$')

ax[0].legend(title='model', frameon=False)
fig.tight_layout()

fig.savefig('ztran_vs_alpha_sigma.pdf')
