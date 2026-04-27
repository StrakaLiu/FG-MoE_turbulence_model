import os
import numpy as np
import scipy
from scipy import interpolate
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.ticker import AutoMinorLocator


def getLatestTime(Dir):
    import os
    t = os.listdir(Dir)
    for i in range(len(t)):
        t[i] = int(t[i])
    t_latest = np.array(t).max()
    return t_latest

def getLatestTimeFloat(Dir):
    import os
    t = os.listdir(Dir)
    for i in range(len(t)):
        t[i] = float(t[i])
    t_latest = np.array(t).max()
    return t_latest

def yPlus_UPlus(y, U, nu):
    yNearWall = y[0:4]
    UNearWall = U[0:4]
    xx,xy=0,0
    for i in range(len(yNearWall)):
        xx+=yNearWall[i]**2
        xy+=yNearWall[i]*UNearWall[i]
    dUdy = xy/xx
    utau = np.sqrt(nu * dUdy)
    Retau = utau / nu
    print(utau, Retau)
    yPlus = utau * y / nu
    UPlus = U / utau
    return(yPlus, UPlus, utau)

def getInterpolate2D(xyCFD, xyDNS, uDNS):
    interp = interpolate.LinearNDInterpolator(xyDNS, uDNS)
    uCFD = interp(xyCFD[:,0], xyCFD[:,1])
    return uCFD

def halfWidth(y, U, Um, U1):
    N = len(U)
    for i in range(N-1):
        if ((U[i]-U1)/(Um-U1)>0.5 and (U[i+1]-U1)/(Um-U1)<0.5):
            d = y[i]
    return d


with open('caseDir.txt', 'r') as file:
    caseDir = file.read()

file = open(caseDir + '/modelErr.txt', 'w')
file.write('case, err\n')
#-----------------------channel--------------------------#
scale_ux = 1.0
U_ref = 1.0
postPath = caseDir + '/01_channel/postProcessing/sample_in'
t_latest = getLatestTime(postPath)
filePath = os.path.join(postPath, f'{t_latest}', 'U_left.raw')
U_prof_cfd = np.loadtxt(filePath, skiprows = 2)
Ux_obs_array = U_prof_cfd[:, 3] * scale_ux / U_ref
U_obs = Ux_obs_array
# load DNS data
filePath = os.path.join(
    '../refData/truthData/channel', 'Re=5200.dat')
U_prof_dns = np.loadtxt(filePath, skiprows = 1)
utauDNS = 4.14872e-02
y_CFD = U_prof_cfd[:, 1]
y_DNS = U_prof_dns[:, 0]
ux_dns_mapped = np.interp(y_CFD, y_DNS, U_prof_dns[:,2])
Ux_dns_array = ux_dns_mapped * utauDNS
dy_CFD = y_CFD.copy()
for i in range(1, len(y_CFD)):
    dy_CFD[i] = y_CFD[i] - y_CFD[i-1]
U_dns = Ux_dns_array.copy()
misfit_CFD = np.sum(np.abs(U_obs - U_dns)*dy_CFD)
truthSum = np.sum(np.abs(U_dns)*dy_CFD)
print('channel,', f"{misfit_CFD/truthSum:.6g}")
file.write(f"channel, {misfit_CFD/truthSum:.6g}\n")

#-------------------------------ZPG plate-------------------------------#
moeDict = caseDir + '/02_ZPGPlate/'
tlast = getLatestTime(caseDir + '/02_ZPGPlate/postProcessing/sample_down')
file1 = moeDict + 'postProcessing/sample_down/'+ str(tlast) + '/wallShearStress_down.raw'
x = np.loadtxt(file1, skiprows = 2)[:,0]
x_data = 69.4*x/1.388e-5 * 1e-7
Cf_exp = 0.288*(np.log10(69.4*x/1.388e-5))**(-2.45) *1e3
wallShearStress_x = np.loadtxt(file1, skiprows = 2)[:,3]
Cf_CFD = -wallShearStress_x/(0.5*69.4**2) *1e3
dx_data = x_data.copy()
for i in range(1, len(x_data)):
    dx_data[i] = x_data[i] - x_data[i-1]
mask = x_data > 0.1
misfit_CFD = np.sum(np.abs(Cf_CFD[mask] - Cf_exp[mask])*dx_data[mask])
truthSum = np.sum(np.abs(Cf_exp[mask])*dx_data[mask])
print('ZPG plate:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"ZPG plate, {misfit_CFD/truthSum:.6g}\n")

#----------------------------------plane jet--------------------------------#
expDict = '../refData/truthData/planeJet/'
moeDict = caseDir + '/03_planeJet/'
tlast = getLatestTime(moeDict + 'postProcessing/sampleDict/')
y_exp = np.loadtxt(expDict+'exp.dat', skiprows = 1)[:,0]
U_exp = np.loadtxt(expDict+'exp.dat', skiprows = 1)[:,1]
U_exp_array = [np.concatenate([U_exp,U_exp,U_exp])]
U_obs_array = []
xpos = [30, 60, 100]
for i in range(3):
    file1 = moeDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_'+str(xpos[i])+'_U.xy'
    y = np.loadtxt(file1)[:,0]
    U = np.loadtxt(file1)[:,1]
    Um = U[0]
    U1 = U[-1]
    d = halfWidth(y, U, Um, U1)
    f = interpolate.interp1d(y/d, (U-U1)/(Um-U1))
    U_mapped = f(y_exp)
    U_obs_array = np.append(U_obs_array, U_mapped)
misfit_CFD = np.sum(np.abs(U_obs_array - U_exp_array))
truthSum = np.sum(np.abs(U_exp_array))
print('PSJ:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"PSJ, {misfit_CFD/truthSum:.6g}\n")


#--------------------------------------------NACA0012-------------------------------#
CLexp10 = 1.0809
CLexp15 = 1.5169
moeDict = caseDir + '/4.1_NACA0012_AOA10/'
tlast = getLatestTime(moeDict + 'postProcessing/forceCoeffs/')
file1 = moeDict + 'postProcessing/forceCoeffs/' + str(tlast)+ '/'
CL10_obs = np.loadtxt(file1 + 'coefficient.dat', skiprows = 13)[-1,4]

moeDict = caseDir + '/4_NACA0012_AOA15/'
tlast = getLatestTime(moeDict + 'postProcessing/forceCoeffs/')
file1 = moeDict + 'postProcessing/forceCoeffs/' + str(tlast)+ '/'
CL15_obs = np.loadtxt(file1 + 'coefficient.dat', skiprows = 13)[-1,4]
misfit_CFD = np.abs(CL10_obs - CLexp10) + np.abs(CL15_obs - CLexp15)
truthSum = CLexp10 + CLexp15
print('NACA0012:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"NACA0012, {misfit_CFD/truthSum:.6g}\n")

#----------------------------2D-hump--------------------------------------#
expDict = '../refData/truthData/NASAhump/'
moeDict = caseDir + '/1_nasaHump/'
tlast = getLatestTime(moeDict + 'postProcessing/sample_lines_U/')
U_exp_array = []
U_obs_array = []
V_exp_array = []
V_obs_array = []
for i in range(1, 7):
    y = np.loadtxt(expDict + 'Ulines/' + 'line' +str(i) + '.dat', skiprows = 2)[:,1]
    U = np.loadtxt(expDict + 'Ulines/' + 'line' +str(i) + '.dat', skiprows = 2)[:,2]
    V = np.loadtxt(expDict + 'Ulines/' + 'line' +str(i) + '.dat', skiprows = 2)[:,3]
    U_exp_array = np.append(U_exp_array, U)
    V_exp_array = np.append(V_exp_array, V)
    file1 = moeDict + 'postProcessing/sample_lines_U/' + str(tlast)+ '/'
    y_obs = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,0]/0.42
    U_obs = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,1]/34.6
    V_obs = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,2]/34.6
    f = interpolate.interp1d(y_obs, U_obs)
    U_obs_mapped = f(y)
    f = interpolate.interp1d(y_obs, V_obs)
    V_obs_mapped = f(y)
    U_obs_array = np.append(U_obs_array, U_obs_mapped)
    V_obs_array = np.append(V_obs_array, V_obs_mapped)
misfit_CFD = np.sum(np.abs(U_obs_array - U_exp_array) + np.abs(V_obs_array - V_exp_array))
truthSum = np.sum(np.abs(U_exp_array) + np.abs(V_exp_array))
print('2D-hump:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"2D-hump, {misfit_CFD/truthSum:.6g}\n")

#-----------------------------------------2D bump-----------------------------------#
expDict = '../refData/truthData/bump/'
moeDict = caseDir + '/5_bump/'
tlast = getLatestTime(moeDict + 'postProcessing/sample_lines_U9/')
U_exp_array = []
U_moe_array = []
dy_array = []
Nx = 9
xline = np.arange(-0.2, 1.401, 0.2)
file1 = expDict + 'postProcessing/sample_lines_U9/1.136592/'
file3 = moeDict + 'postProcessing/sample_lines_U9/' + str(tlast) + '/'
for i in range(Nx):
    y = np.loadtxt(file1 + 'line_' +str(i) + '_avgUx_avgUy.xy')[:,0]/0.305
    U = np.loadtxt(file1 + 'line_' +str(i) + '_avgUx_avgUy.xy')[:,1]/16.7
    V = np.loadtxt(file1 + 'line_' +str(i) + '_avgUx_avgUy.xy')[:,2]/16.7
    y_moe = np.loadtxt(file3 + 'line_' +str(i) + '_U.xy')[:,0]/0.305
    U_moe = np.loadtxt(file3 + 'line_' +str(i) + '_U.xy')[:,1]/16.7
    V_moe = np.loadtxt(file3 + 'line_' +str(i) + '_U.xy')[:,2]/16.7
    if(xline[i] > 0.5):
        y_map = np.linspace(max(y_moe.min(), y.min()), min(y_moe.max(), y.max()), 201)
        f = interpolate.interp1d(y, U)
        U_exp_mapped = f(y_map)
        f = interpolate.interp1d(y, V)
        V_exp_mapped = f(y_map)
        f = interpolate.interp1d(y_moe, U_moe)
        U_moe_mapped = f(y_map)
        f = interpolate.interp1d(y_moe, V_moe)
        V_moe_mapped = f(y_map)
        U_exp_array = np.append(U_exp_array, U_exp_mapped)
        U_exp_array = np.append(U_exp_array, V_exp_mapped)
        U_moe_array = np.append(U_moe_array, U_moe_mapped)
        U_moe_array = np.append(U_moe_array, V_moe_mapped)
misfit_CFD = np.sum(np.abs(U_moe_array - U_exp_array))
truthSum = np.sum(np.abs(U_exp_array))
print('2D-bump:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"2D-bump, {misfit_CFD/truthSum:.6g}\n")

#----------------------------------periodic hill--------------------------------------#
U_ref = 0.028
expDict = '../refData/truthData/pehill/'
moeDict = caseDir + '/6_pehill/'
# load velocity profile
xpos = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
postPath = moeDict + 'postProcessing/sampleDict'
t_latest = getLatestTime(postPath)
U_prof_cfd = []
for p in xpos:
    filePath = os.path.join(postPath, f'{t_latest}', f'line_x{p}_U.xy')
    U_prof_cfd.append(np.loadtxt(filePath, usecols=[0, 1, 2]))
# load DNS data
U_prof_dns = []
for p in xpos:
    filePath = os.path.join(
        expDict + 'postProcessing/sampleDict/0',
        f'line_x{p}_UDNS.xy')
    U_prof_dns.append(np.loadtxt(filePath, usecols=[0, 1, 2]))
x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
U_exp_array = []
U_moe_array = []
dy_array = []
for i in range(len(U_prof_cfd)):
    U_exp_array = np.append(U_exp_array, U_prof_dns[i][:, 1])
    U_exp_array = np.append(U_exp_array, U_prof_dns[i][:, 2])
    U_moe_array = np.append(U_moe_array, U_prof_cfd[i][:, 1])
    U_moe_array = np.append(U_moe_array, U_prof_cfd[i][:, 2])
    y_CFD = U_prof_dns[i][:, 0].copy()
    dy = U_prof_dns[i][:, 0].copy()
    for j in range(1, len(dy)):
        dy[j] = y_CFD[j] - y_CFD[j-1]
    dy_array = np.append(dy_array, dy)
    dy_array = np.append(dy_array, dy)
misfit_CFD = np.sum(np.abs(U_moe_array - U_exp_array)*dy_array)
truthSum = np.sum(np.abs(U_exp_array)*dy_array)
print('2D-peHills:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"2D-peHills, {misfit_CFD/truthSum:.6g}\n")

#-------------------------------------------FAITH hill----------------------------------#
L_ref = 0.2286
# load DNS data
postPath = '../refData/truthData/FAITHhill/Centerline_FAITH_2Hz_4000samps_scalar'
xy_DNS = np.loadtxt(postPath+'/U_mean_axis00.dat', skiprows = 9)[:, 0:2]/1000
Ux_DNS_array = np.loadtxt(postPath+'/U_mean_axis00.dat', skiprows = 9)[:,2]
Uy_DNS_array = np.loadtxt(postPath+'/V_mean_axis00.dat', skiprows = 9)[:,2]
# load moe data
postPath = caseDir + '/7_FAITHhill/postProcessing'
t_moe = getLatestTime(postPath+'/sample_plane_z0')
filePath = os.path.join(postPath+'/sample_plane_z0', f'{t_moe}', 'U_z0.raw')
xy_CFD = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_obs_array, Uy_obs_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]
filePath = os.path.join(postPath+'/sample_plane_z0', f'{t_moe}', 'V_z0.raw')
cellV_CFD = np.loadtxt(filePath, skiprows = 2)[:,3]
x_cfd = xy_CFD[:,0]
y_cfd = xy_CFD[:,1]
mask = (x_cfd > 0) & (x_cfd < 2.5*L_ref) & (y_cfd < L_ref)
x_map = x_cfd[mask]
y_map = y_cfd[mask]
xy_map = np.zeros([len(x_map), 2])
xy_map[:,0] = x_map
xy_map[:,1] = y_map
cellV_map = cellV_CFD[mask]
Ux_obs_map = Ux_obs_array[mask]
Uy_obs_map = Uy_obs_array[mask]
Ux_dns_map = getInterpolate2D(xy_map, xy_DNS, Ux_DNS_array)
Uy_dns_map = getInterpolate2D(xy_map, xy_DNS, Uy_DNS_array)
misfit_CFD = np.sum(np.abs(Ux_obs_map - Ux_dns_map)*cellV_map)      + np.sum(np.abs(Uy_obs_map - Uy_dns_map)*cellV_map)
truthSum = np.sum(np.abs(Ux_dns_map)*cellV_map)                     + np.sum(np.abs(Uy_dns_map)*cellV_map)
print('3D-hill:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"3D-hill, {misfit_CFD/truthSum:.6g}\n")
#-------------------------------------------square duct----------------------------------#
postPath = caseDir + '/2_sqrDuct_Re=40000/postProcessing/sample_left/'
tlast = getLatestTime(postPath)
filePath = os.path.join(postPath, f'{tlast}', 'U_left.raw')
U_prof_cfd = np.loadtxt(filePath, skiprows = 2)
filePath1 = os.path.join(postPath, f'{tlast}', 'V_left.raw')
cellV_cfd = np.loadtxt(filePath1 , skiprows = 2)[:,3]*1e5
# load DNS data
filePath = os.path.join(
    '../refData/truthData/squareDuct_recDuct', 'DNS_Re=40000.dat')
U_prof_dns = np.loadtxt(filePath, skiprows = 20)
scale_ux = 1.0
scale_uy = 50.0
scale_uz = 50.0
U_ref = 1.0
Ux_obs_array = U_prof_cfd[:, 3] * scale_ux / U_ref
Uy_obs_array = U_prof_cfd[:, 4] * scale_uy / U_ref
Uz_obs_array = U_prof_cfd[:, 5] * scale_uz / U_ref
# ---------- interpolate the DNS data to the CFD mesh
# ---------- the DNS data is not strictly symmetry about y=z line
#----------- so use the transformed averaged uy and uz
yz_CFD = U_prof_cfd[:, 1:3]
zy_DNS = U_prof_dns[:, 0:2]
yz_DNS = zy_DNS[:, [1,0]]
ux_dns_mapped = getInterpolate2D(yz_CFD, yz_DNS, U_prof_dns[:,2])
uy_dns_mapped = getInterpolate2D(yz_CFD, yz_DNS, U_prof_dns[:,4])
uz_dns_mapped = getInterpolate2D(yz_CFD, yz_DNS, U_prof_dns[:,5])
uy_dns_mapped_trans = getInterpolate2D(yz_CFD, zy_DNS, U_prof_dns[:,4])
uz_dns_mapped_trans = getInterpolate2D(yz_CFD, zy_DNS, U_prof_dns[:,5])
Ux_dns_array = ux_dns_mapped * scale_ux / U_ref
Uy_dns_array = (uy_dns_mapped + uz_dns_mapped_trans)/2.0 * scale_uy / U_ref
Uz_dns_array = (uz_dns_mapped + uy_dns_mapped_trans)/2.0 * scale_uz / U_ref
U_obs = np.concatenate([Ux_obs_array*cellV_cfd, Uy_obs_array*cellV_cfd, Uz_obs_array*cellV_cfd])
U_dns = np.concatenate([Ux_dns_array*cellV_cfd, Uy_dns_array*cellV_cfd, Uz_dns_array*cellV_cfd])
misfit_CFD = np.sum(np.abs(U_obs - U_dns))
truthSum = np.sum(np.abs(U_dns))
print('SqrDuct:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"SqrDuct, {misfit_CFD/truthSum:.6g}\n")

#--------------------------------rec duct
# load DNS
filePath = '../refData/truthData/squareDuct_recDuct/'
Y_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_Y.dat', skiprows = 0)+1.0
Z_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_Z.dat', skiprows = 0)+3.0
U_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_U.dat', skiprows = 0)
V_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_V.dat', skiprows = 0)
W_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_W.dat', skiprows = 0)
Y_grid, Z_grid = np.meshgrid(Y_DNS, Z_DNS, indexing='ij')
# load CFD
postPath = caseDir + '/2.1_recDuct/postProcessing/extractPlane/'
t_latest = getLatestTime(postPath)
filePath = os.path.join(postPath, f'{t_latest}', 'U_s0.raw')
U_prof_cfd = np.loadtxt(filePath, skiprows = 2)
filePath1 = os.path.join(postPath, f'{t_latest}', 'V_s0.raw')
cellV_cfd = np.loadtxt(filePath1, skiprows = 2)[:, 3]
Ux_obs_array = U_prof_cfd[:, 3]
Uy_obs_array = U_prof_cfd[:, 4]
Uz_obs_array = U_prof_cfd[:, 5]
yz_CFD = U_prof_cfd[:, 1:3]
Z_DNS_f = Z_grid.flatten()
Y_DNS_f = Y_grid.flatten()
U_DNS_f = U_DNS.flatten()
V_DNS_f = V_DNS.flatten()
W_DNS_f = W_DNS.flatten()
yz_dns = np.zeros([len(Z_DNS_f), 2])
yz_dns[:,0] = Y_DNS_f
yz_dns[:,1] = Z_DNS_f
y_cfd = yz_CFD[:,0]
z_cfd = yz_CFD[:,1]
mask =  (y_cfd < 1) & (z_cfd < 3)
y_map = y_cfd[mask]
z_map = z_cfd[mask]
yz_map = np.zeros([len(y_map), 2])
yz_map[:,0] = y_map
yz_map[:,1] = z_map
cellV_map = cellV_cfd[mask]
Ux_obs_map = Ux_obs_array[mask]
Uy_obs_map = Uy_obs_array[mask]
Uz_obs_map = Uz_obs_array[mask]
Ux_dns_map = getInterpolate2D(yz_map, yz_dns, U_DNS_f)
Uy_dns_map = getInterpolate2D(yz_map, yz_dns, V_DNS_f)
Uz_dns_map = getInterpolate2D(yz_map, yz_dns, W_DNS_f)
misfit_CFD = np.sum(np.abs(Ux_obs_map - Ux_dns_map)*cellV_map)      + np.sum(np.abs(Uy_obs_map - Uy_dns_map)*cellV_map)*50  + np.sum(np.abs(Uz_obs_map - Uz_dns_map)*cellV_map*50)
truthSum = np.sum(np.abs(Ux_dns_map)*cellV_map)                       + np.sum(np.abs(Uy_dns_map)*cellV_map)*50                 + np.sum(np.abs(Uz_dns_map)*cellV_map*50)
print('RecDuct:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"RecDuct, {misfit_CFD/truthSum:.6g}\n")

#----------------------------ASJ-----------------------------
expDict = '../refData/truthData/ASJ_ANSJ/'
moeDict = caseDir + '/3_ASJ/'
tlast = getLatestTime(moeDict + 'postProcessing/sampleDict/')
x_exp = np.loadtxt(expDict+'center.dat', skiprows = 1)[:,0]
U_exp = np.loadtxt(expDict+'center.dat', skiprows = 1)[:,2]
file1 = moeDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_center_U.xy'
x = np.loadtxt(file1)[:,0]
U = np.loadtxt(file1)[:,1]
U_jet = 171.0
f = interpolate.interp1d(x/0.0508, U/U_jet)
U_obs_mapped = f(x_exp)
U_exp_array = []
U_obs_array = []
U_exp_array = np.append(U_exp_array, U_exp)
U_obs_array = np.append(U_obs_array, U_obs_mapped)

misfit_CFD = np.sum(np.abs(U_obs_array - U_exp_array))
truthSum = np.sum(np.abs(U_exp_array))
print('ASJ:',  f"{misfit_CFD/truthSum:.6g}")
file.write(f"ASJ, {misfit_CFD/truthSum:.6g}\n")

#--------------------------------ANSJ---------------------------
expDict = '../refData/truthData/ASJ_ANSJ/'
moeDict = caseDir + '/3.1_ANSJ/'
tlast = getLatestTime(moeDict + 'postProcessing/sampleDict/')
x_exp = np.loadtxt(expDict+'center_nearSonic.dat', skiprows = 1)[:,0]
U_exp = np.loadtxt(expDict+'center_nearSonic.dat', skiprows = 1)[:,2]
file1 = moeDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_center_U.xy'
x = np.loadtxt(file1)[:,0]
U = np.loadtxt(file1)[:,1]
U_jet = 310.0
f = interpolate.interp1d(x/0.0508, U/U_jet)
U_obs_mapped = f(x_exp)
U_exp_array = []
U_obs_array = []
U_exp_array = np.append(U_exp_array, U_exp)
U_obs_array = np.append(U_obs_array, U_obs_mapped)
misfit_CFD = np.sum(np.abs(U_obs_array - U_exp_array))
truthSum = np.sum(np.abs(U_exp_array))
print('ANSJ:', f"{misfit_CFD/truthSum:.6g}")
file.write(f"ANSJ, {misfit_CFD/truthSum:.6g}\n")



