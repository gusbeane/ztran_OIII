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
fid_dir = base_dir + '/fid' + '/Boxes'
cold_dir = base_dir + '/cold' + '/Boxes'
hot_dir = base_dir + '/hot' + '/Boxes'
dir_list = [hot_dir, fid_dir, cold_dir]
name_list = [r'\texttt{hot}', r'\texttt{fid}', r'\texttt{cold}']
color_list = [tb_c[1], tb_c[-1], tb_c[3]]

fig, ax = plt.subplots(1, 1, figsize=(columnwidth, columnwidth))

for directory, name, c in zip(dir_list, name_list, color_list):
    dTfiles = glob.glob(directory+'/delta_T*')
    zlist = np.array([float(re.findall('z\d\d\d\.\d\d', f)[0][1:]) for f in dTfiles])

    aveTb_list = []
    for k in np.argsort(zlist):
        Tb = np.fromfile(dTfiles[k], dtype='f4')
        aveTb_list.append(np.mean(Tb))
    aveTb_list = np.array(aveTb_list)

    zlist = np.sort(zlist)

    ax.plot(zlist, aveTb_list, label=name, c=c)

ax.set_xlim(8, 14)
ax.set_ylim(-50, 50)

ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\delta T_b\,[\,\text{mK}\,]$')

ax.legend(title='model', frameon=False)
fig.tight_layout()

fig.savefig('aveTb.pdf')

