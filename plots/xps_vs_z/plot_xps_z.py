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

#Mcut_list = [1E9, 1E10, 1E11]
Mcut_list = [1E9]
Mcut_list_label =['Mcut5.012e+09']
ls_list = ['solid']

fig, ax = plt.subplots(1, 1, figsize=(textwidth, columnwidth))

for name, directory, c in zip(name_list, dir_list, color_list):
    for i,(label, ls) in enumerate(zip(Mcut_list_label, ls_list)):
        zlist, klist, xps = read_xps(directory+'/xps*.txt')
        klist, ztran = find_ztran(zlist, klist, xps)
        if i==0:
            ax.plot(zlist, xps[:,3], label=name, c=c, ls=ls)
        else:    
            ax.plot(zlist, xps[:,3], c=c, ls=ls)

# ax.set_xscale('log')
ax.set_yscale('symlog')
#ax.set_xscale('log')

ax.set_xlim(8, 25)
ax.set_ylim(-1e6, 1e6)

ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\Delta_{21,\delta}^2\,[\,\text{mK}\,]$')

#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

ax.legend(title='model', frameon=False)
fig.tight_layout()

fig.savefig('xps_vs_z.pdf')
