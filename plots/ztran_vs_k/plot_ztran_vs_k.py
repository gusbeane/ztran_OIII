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

fig, ax = plt.subplots(1, 1, figsize=(columnwidth, columnwidth))

for name, directory, c in zip(name_list, dir_list, color_list):
    zlist, klist, xps = read_xps(directory+'/xps*')
    klist, ztran = find_ztran(zlist, klist, xps)

    print(ztran)

    ax.plot(klist, ztran, label=name, c=c)

ax.set_xscale('log')

ax.set_xlim(0.1, 1.0)
ax.set_ylim(9, 12)

ax.set_xlabel(r'$k\,[\,text{Mpc}^{-1}\,]$')
ax.set_ylabel(r'$z_{\star}$')

# ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

ax.legend(title='model', frameon=False)
fig.tight_layout()

fig.savefig('ztran_vs_k.pdf')
