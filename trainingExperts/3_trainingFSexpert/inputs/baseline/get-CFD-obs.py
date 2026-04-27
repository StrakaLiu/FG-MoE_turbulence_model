import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from collections import OrderedDict
import os

def getLatestTime(Dir):
    import os
    # get the latest time
    t = os.listdir(Dir)
    for i in range(len(t)):
        t[i] = float(t[i])
    t_latest = np.array(t).max()
    return int(t_latest)

def count_folders(directory):
    contents = os.listdir(directory)
    folders = [item for item in contents if os.path.isdir(os.path.join(directory, item))]
    return len(folders)



expDict = '../../inputs/data/'
baseDict = '../../inputs/baseline/'
nnDict = './'

tlast = getLatestTime(nnDict + 'postProcessing/sampleDict/')


x_exp = np.loadtxt(expDict+'center.dat', skiprows = 1)[:,0]
U_exp = np.loadtxt(expDict+'center.dat', skiprows = 1)[:,2]


file1 = baseDict + 'postProcessing/sampleDict/1/line_center_U.xy'
x_base = np.loadtxt(file1)[:,0]
U_base = np.loadtxt(file1)[:,1]

file1 = nnDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_center_U.xy'
x = np.loadtxt(file1)[:,0]
U = np.loadtxt(file1)[:,1]

U_jet = 171.0

f = interpolate.interp1d(x/0.0508, U/U_jet)
U_obs_mapped = f(x_exp)
U_exp_array = []
U_obs_array = []
U_exp_array = np.append(U_exp_array, U_exp)
U_obs_array = np.append(U_obs_array, U_obs_mapped)
np.savetxt(nnDict + 'postProcessing/sampleDict/' + str(tlast)+ '/Array-exp.dat', U_exp_array)
np.savetxt(nnDict + 'postProcessing/sampleDict/' + str(tlast)+ '/Array-obs.dat', U_obs_array)
misfit = np.sum(np.abs(U_exp_array - U_obs_array))
np.savetxt(nnDict + 'postProcessing/sampleDict/' + str(tlast)+ '/misfit.dat', np.array([misfit]))
print('time=', tlast, 'misfit=', misfit)
file = open('./misfit.dat', "a")
num = count_folders(nnDict + 'postProcessing/sampleDict')
file.write(str(num-1) + ', ' + str(misfit) + '\n')
file.close()
