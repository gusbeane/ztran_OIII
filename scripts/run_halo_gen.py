import numpy as np
import glob
from tqdm import tqdm

import sys
sys.path.append('../plots/')
from helper import *

basedir = '/Users/abeane/scratch/ztran_OIII_sims/v1.2/256Mpch/256'
fid_dir = basedir + '/fid'
cold_dir = basedir + '/cold'
hot_dir = basedir + '/hot'
dir_list = [hot_dir, fid_dir, cold_dir]

Mcut_list = [1E9, 1E10, 1E11]
Mcut_list_label =['Mcut1e9','Mcut1e10','Mcut1e11']

for directory in dir_list:
    fin_directory = directory + '/Output_files/Halo_lists'
    fout_directory = directory + '/MyOutput'

    for fin in tqdm(glob.glob(fin_directory+'/halos*')):

        z = float(re.findall('\d?\d\.\d\d', fin)[0])
        zlabel = "{:06.2f}".format(z)

        for Mcut,label in zip(Mcut_list,Mcut_list_label):
            fout = fout_directory+'/halos_z'+zlabel+'_'+label
            gen_halo_catalog(fin, fout, 256, Mcut=Mcut)
