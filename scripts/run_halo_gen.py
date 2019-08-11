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
#dir_list = [hot_dir, fid_dir, cold_dir]
dir_list = [fid_dir]

Mcut_list = np.logspace(9.7, 11, 10)
Mcut_list_label =['Mcut'+"{:2.3e}".format(Mcut) for Mcut in Mcut_list]

for directory in dir_list:
    fin_directory = directory + '/Output_files/Halo_lists'
    fout_directory = directory + '/MyOutput/Halo_fields'

    for fin in tqdm(glob.glob(fin_directory+'/updated_halos*')):

        z = float(re.findall('\d?\d\.\d\d', fin)[0])
        zlabel = "{:06.2f}".format(z)

        for Mcut,label in zip(Mcut_list,Mcut_list_label):
            fout = fout_directory+'/updated_halos_z'+zlabel+'_'+label
            gen_halo_catalog(fin, fout, 256, Mcut=Mcut)
