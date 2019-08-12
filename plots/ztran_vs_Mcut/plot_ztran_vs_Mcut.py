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

Mcut_list = np.logspace(9.7, 11, 10)
Mcut_list_label =['Mcut'+"{:2.3e}".format(Mcut) for Mcut in Mcut_list]

fig, ax = plt.subplots(1, 1, figsize=(columnwidth, columnwidth))

for name, directory, c in zip(name_list, dir_list, color_list):
    
    ztran_list = []
    for Mcut, Mcut_label in zip(Mcut_list, Mcut_list_label):
        zlist, klist, xps = read_xps(directory+'/Halos_xps/*'+Mcut_label+'.txt')
        klist, ztran = find_ztran(zlist, klist, xps)
        ztran_list.append(ztran[5])
        print(klist[5], ztran[5])

    ax.plot(Mcut_list, ztran_list, label=name, c=c)

ax.set_xscale('log')

ax.set_xlim(10.**(9.7), 1e11)
ax.set_ylim(9.5, 14)

ax.set_xlabel(r'$M_{\text{cut}}\,[\,M_{\odot}\,]$')
ax.set_ylabel(r'$z_{\star}$')

ax.legend(title='model', frameon=False)
fig.tight_layout()

fig.savefig('ztran_vs_Mcut.pdf')
