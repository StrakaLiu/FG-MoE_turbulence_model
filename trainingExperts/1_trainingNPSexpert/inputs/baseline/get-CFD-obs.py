import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from collections import OrderedDict


def getLatestTime(Dir):
    import os
    # get the latest time
    t = os.listdir(Dir)
    for i in range(len(t)):
        t[i] = int(t[i])
    t_latest = np.array(t).max()
    #print('latest time', t_latest)
    return t_latest

expDict = '../../inputs/data/'
baseDict = '../../inputs/baseline/'
nnDict = './'

tlast = getLatestTime(nnDict + 'postProcessing/sample_down/')

file = expDict + 'exp_cf.dat'
x_exp1 = np.loadtxt(file, skiprows = 5)[:,0]
Cf_exp = np.loadtxt(file, skiprows = 5)[:,1]

file = expDict + 'exp_cp.dat'
x_exp2 = np.loadtxt(file, skiprows = 4)[:,0]
Cp_exp = -np.loadtxt(file, skiprows = 4)[:,1]

nz = 1; nx = 192
file1 = baseDict + 'postProcessing/sample_down/1/wallShearStress_down.raw'
file2 = baseDict + 'postProcessing/sample_down/1/p_down.raw'
x = np.loadtxt(file1, skiprows = 2, max_rows=nx)[:,0]
xHump = x
yHump = np.loadtxt(file1, skiprows = 2, max_rows=nx)[:,1]
wallShearStress_x = x*0.0
wallShearStress_y = x*0.0
p = x*0.0
for iz in range(nz):
    wallShearStress_x += np.loadtxt(file1, skiprows = 2+iz*nx, max_rows=nx)[:,3]/nz
    wallShearStress_y += np.loadtxt(file1, skiprows = 2+iz*nx, max_rows=nx)[:,4]/nz
    p += np.loadtxt(file2, skiprows = 2+iz*nx, max_rows=nx)[:,3]/nz
wallShearStress = np.sqrt(wallShearStress_x**2 + wallShearStress_y**2) * np.sign(wallShearStress_x)

nz = 1; nx = 192
file1 = nnDict + 'postProcessing/sample_down/' + str(tlast)+ '/wallShearStress_down.raw'
file2 = nnDict + 'postProcessing/sample_down/' + str(tlast)+ '/p_down.raw'
x_obs = np.loadtxt(file1, skiprows = 2, max_rows=nx)[:,0]/0.42
wallShearStress_x = x*0.0
wallShearStress_y = x*0.0
p = x*0.0
for iz in range(nz):
    wallShearStress_x += np.loadtxt(file1, skiprows = 2+iz*nx, max_rows=nx)[:,3]/nz
    wallShearStress_y += np.loadtxt(file1, skiprows = 2+iz*nx, max_rows=nx)[:,4]/nz
    p += np.loadtxt(file2, skiprows = 2+iz*nx, max_rows=nx)[:,3]/nz
wallShearStress = np.sqrt(wallShearStress_x**2 + wallShearStress_y**2) * np.sign(wallShearStress_x)
Cf_obs = -wallShearStress/(0.5*34.6**2)
Cp_obs = -p/(0.5*34.6**2)

file1 = expDict + 'Ulines/'
xline = [0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
for i in range(7):
    y = np.loadtxt(file1 + 'line' +str(i) + '.dat', skiprows = 2)[:,1]
    U = np.loadtxt(file1 + 'line' +str(i) + '.dat', skiprows = 2)[:,2]
    V = np.loadtxt(file1 + 'line' +str(i) + '.dat', skiprows = 2)[:,3]

file1 = baseDict + 'postProcessing/sample_lines_U/1/'
xline = [0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
for i in range(7):
    y = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,0]/0.42
    U = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,1]/34.6
    V = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,2]/34.6

file1 = nnDict + 'postProcessing/sample_lines_U/' + str(tlast)+ '/'
xline = [0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
for i in range(7):
    y = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,0]/0.42
    U = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,1]/34.6
    V = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,2]/34.6


file2 = nnDict + 'postProcessing/sample_lines_C/' + str(tlast)+ '/'
A_reguMax = 0.01
step_to_Max = 5
A_regu = min(A_reguMax/step_to_Max*(tlast-1), A_reguMax)

U_exp_array = []
U_obs_array = []
for i in range(1, 7):
    y = np.loadtxt(expDict + 'Ulines/' + 'line' +str(i) + '.dat', skiprows = 2)[:,1]
    U = np.loadtxt(expDict + 'Ulines/' + 'line' +str(i) + '.dat', skiprows = 2)[:,2]
    V = np.loadtxt(expDict + 'Ulines/' + 'line' +str(i) + '.dat', skiprows = 2)[:,3]
    y_obs = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,0]/0.42
    U_obs = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,1]/34.6
    V_obs = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,2]/34.6
    f = interpolate.interp1d(y_obs, U_obs)
    U_obs_mapped = f(y)
    f = interpolate.interp1d(y_obs, V_obs)
    V_obs_mapped = f(y)
    c1_obs = np.loadtxt(file2 + 'line' +str(i) + '_c1__c2__g3_.xy')[:,1]/1.8 - 1
    c2_obs = np.loadtxt(file2 + 'line' +str(i) + '_c1__c2__g3_.xy')[:,2]/0.555555 -1
    g3_obs = np.loadtxt(file2 + 'line' +str(i) + '_c1__c2__g3_.xy')[:,3]
    c1_exp = c1_obs*0.0
    c2_exp = c2_obs*0.0
    g3_exp = g3_obs*0.0
    U_exp_array = np.append(U_exp_array, c1_exp*A_regu)
    U_exp_array = np.append(U_exp_array, c2_exp*A_regu)
    U_exp_array = np.append(U_exp_array, g3_exp*A_regu)
    U_obs_array = np.append(U_obs_array, c1_obs*A_regu)
    U_obs_array = np.append(U_obs_array, c2_obs*A_regu)
    U_obs_array = np.append(U_obs_array, g3_obs*A_regu)
U_exp_array = np.append(U_exp_array, Cf_exp/0.006*20.0)
f = interpolate.interp1d(x_obs, Cf_obs)
Cf_obs_mapped = f(x_exp1)
f = interpolate.interp1d(x_obs, Cp_obs)
Cp_obs_mapped = f(x_exp2)
U_obs_array = np.append(U_obs_array, Cf_obs_mapped/0.006*20.0)
np.savetxt(nnDict + 'postProcessing/sample_lines_U/' + str(tlast)+ '/Array-exp.dat', U_exp_array)
np.savetxt(nnDict + 'postProcessing/sample_lines_U/' + str(tlast)+ '/Array-obs.dat', U_obs_array)
misfit = np.sum(np.abs(U_exp_array - U_obs_array))
np.savetxt(nnDict + 'postProcessing/sample_lines_U/' + str(tlast)+ '/misfit.dat', np.array([misfit]))
print('time=', tlast, 'misfit=', misfit)
file = open('./misfit.dat', "a")
file.write(str(tlast) + ', ' + str(misfit) + '\n')
file.close()