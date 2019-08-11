import glob
import numpy as np
import re
from scipy.interpolate import interp1d
from scipy.optimize import ridder

h = 0.6766

def read_xps(directory):
    directory = directory+'/xps*'
    xps_list = glob.glob(directory)
    print(directory)
    zlist = np.array([float(re.findall('\d\d\d\.\d\d', f)[0]) for f in xps_list])
    
    output = []
    for i,key in enumerate(np.argsort(zlist)):
        fname = xps_list[key]
        data = np.genfromtxt(fname)
        xps = data[:,3]
        output.append(xps)
        if i ==0:
            klist = data[:,0]/h
    
    zlist = np.sort(zlist)

    output = np.array(output)
    return zlist, klist, output

def find_ztran(zlist, klist, xps):
    ztran = []
    for k in range(len(klist)):
        this_xps = xps[:,k]
        
        xps_fn_z = interp1d(zlist, this_xps)
        try:
            sol = ridder(xps_fn_z, 9, 16)
        except:
            sol = np.nan
        print(sol)
        this_ztran = float(sol)
        ztran.append(this_ztran)

    return klist, np.array(ztran)

