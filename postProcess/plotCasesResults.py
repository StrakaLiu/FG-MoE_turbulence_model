import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator
import matplotlib
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.interpolate import griddata
from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import pdist, squareform
import matplotlib.gridspec as gridspec
import os
from scipy import interpolate
import matplotlib.patches as patches
from matplotlib import tri
import matplotlib.lines as mlines


matplotlib.use('Agg')  


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


def interpolate_to_structured_grid(X, Y, Gamma, xmin, xmax, ymin, ymax, Nx, Ny, method='linear', smooth_factor=None):
    xi = np.linspace(xmin, xmax, Nx)
    yi = np.linspace(ymin, ymax, Ny)
    Xi, Yi = np.meshgrid(xi, yi)
    
    points = np.column_stack((X, Y))
    values = Gamma
    
    Zi = griddata(points, values, (Xi, Yi), method=method)
    
    return Xi, Yi, Zi


def getInterpolate2D(xyCFD, xyDNS, uDNS):
    interp = interpolate.LinearNDInterpolator(xyDNS, uDNS)
    uCFD = interp(xyCFD[:,0], xyCFD[:,1])
    return uCFD

def mirrorAvgU(yz, ux, uy, uz):
    zy = yz[:, [1,0]]
    ux_mapped_trans = getInterpolate2D(zy, yz, ux)
    uy_mapped_trans = getInterpolate2D(zy, yz, uy)
    uz_mapped_trans = getInterpolate2D(zy, yz, uz)
    ux_avg = (ux + ux_mapped_trans)/2.0
    uy_avg = (uy + uz_mapped_trans)/2.0
    uz_avg = (uz + uy_mapped_trans)/2.0
    return ux_avg, uy_avg, uz_avg

def find_min_location_local(Xi, Yi, Phi, xmin, xmax, ymin, ymax):
    mask = np.ones_like(Xi, dtype=bool)
    mask &= (Xi >= xmin)
    mask &= (Xi <= xmax)
    mask &= (Yi >= ymin)
    mask &= (Yi <= ymax)
    Phi_masked = np.ma.masked_array(Phi, ~mask)
    min_idx = np.unravel_index(np.argmin(Phi_masked), Phi_masked.shape)
    min_coords = (Xi[min_idx], Yi[min_idx])
    min_value = Phi_masked[min_idx]
    
    return min_coords



def extract_contour(x_data, y_data, u_data, contour_level):
    fig1, ax = plt.subplots(figsize=(8, 6))
    cntr1 = ax.tricontour(x_data, y_data, u_data, levels=[contour_level])
    x_contour = []
    y_contour = []
    for collection in cntr1.collections:
        paths = collection.get_paths()
        for path in paths:
            xy_coords = path.vertices
            x_contour = np.concatenate([x_contour, xy_coords[:, 0]])
            y_contour = np.concatenate([y_contour, xy_coords[:, 1]])
    plt.close(fig1)
    return x_contour, y_contour


with open('caseDir.txt', 'r') as file:
    caseDir = file.read()


streamcolor = '#48c0aa'
markerfacecolor = 'none'
markercolor = '#456990'
lineCFDcolor = '#48c0aa'
lineBasecolor = '#ef767a'
lineSSTcolor = '#5b89d8'


fig = plt.figure(figsize=(12, 7))

gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1.5], width_ratios=[3,3])
ax11 = plt.subplot(gs[0, 0])
ax12 = plt.subplot(gs[0:2, 1])
ax21 = plt.subplot(gs[1, 0])
ax32 = plt.subplot(gs[2, 1])

ax11.set_aspect('equal')
ax12.set_aspect('equal')
ax21.set_aspect('equal')
ax32.set_aspect(10)


#---------------------------------------bump----------------------------------------#

nx = 175
tlast = getLatestTime(caseDir + '/5_bump/postProcessing/sample_lines_U9')
file1 = caseDir + '/5_bump/postProcessing/sample_down/' + str(tlast) + '/wallShearStress_down.raw'
xHump = np.loadtxt(file1, skiprows = 2, max_rows=nx)[:,0]
yHump = np.loadtxt(file1, skiprows = 2, max_rows=nx)[:,1]

# load DNS data
postPath = '../refData/truthData/bump/postProcessing'
filePath = os.path.join(postPath+'/sample_left_U/1.136592/avgUx_left.raw')
xy_DNS = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_DNS_array = np.loadtxt(filePath, skiprows = 2)[:,3]
filePath = os.path.join(postPath+'/sample_left_U/1.136592/avgUy_left.raw')
Uy_DNS_array = np.loadtxt(filePath, skiprows = 2)[:,3]

# load baseline data
postPath = '../refData/baselineData_EARSM05/5_bump/postProcessing'
filePath = os.path.join(postPath+'/sample_left_U/0.2/U_left.raw')
xy_base = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_base_array, Uy_base_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]

# load SST data
postPath = '../refData/baselineData_SST/5_bump/postProcessing'
filePath = os.path.join(postPath+'/sample_left_U/0.2/U_left.raw')
xy_SST = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_SST_array, Uy_SST_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]


# load CFD  data
postPath = caseDir + '/5_bump/postProcessing'
filePath = os.path.join(postPath+'/sample_left_U', f'{tlast}', 'U_left.raw')
xy_CFD = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_obs_array, Uy_obs_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]

scale_ux = 1.0
levels_ux = [0.01]

cntr1 = ax11.tricontour(xy_DNS[:,0]/0.305, xy_DNS[:,1]/0.305, Ux_DNS_array/16.3, levels=levels_ux,    colors = markercolor, extend = 'both', linewidths=2.0)
cntr2 = ax11.tricontour(xy_SST[:,0]/0.305, xy_SST[:,1]/0.305, Ux_SST_array/16.3, levels=levels_ux,    colors = lineSSTcolor, extend = 'both', linestyles = '-.', linewidths=2.0)
cntr3 = ax11.tricontour(xy_base[:,0]/0.305, xy_base[:,1]/0.305, Ux_base_array/16.3, levels=levels_ux, colors = lineBasecolor, extend = 'both', linestyles = '--', linewidths=2.0)
cntr4 = ax11.tricontour(xy_CFD[:,0]/0.305, xy_CFD[:,1]/0.305, Ux_obs_array/16.3, levels=levels_ux,    colors = lineCFDcolor, extend = 'both', linewidths=2.0)
handles = [mlines.Line2D([], [], color=markercolor, label='truth'),
            mlines.Line2D([], [], color=lineSSTcolor, linestyle='-.', label=r'$k-\omega$ SST'),
            mlines.Line2D([], [], color=lineBasecolor, linestyle='--', label='baseline'),
            mlines.Line2D([], [], color=lineCFDcolor, label='MoE')]
ax11.plot([0.095],[0.255],marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markersize=5)
ax11.plot([0.57],[0.19],'sr', markersize=5, markeredgecolor=lineCFDcolor,markerfacecolor=lineCFDcolor, markeredgewidth=1.0)
ax11.plot([0.57],[0.255],'^',  markersize=5, markeredgecolor=lineBasecolor,markerfacecolor=markerfacecolor, markeredgewidth=1.0)
ax11.legend(handles=handles, loc='upper left', ncol=2, fontsize=14, frameon=False)
ax11.set(xlim=(0, 1.3), ylim=(0, 0.3))
ax11.set_aspect('equal')
ax11.tick_params(labelsize=8)
ax11.set_xticks(np.arange(0, 1.301, 0.2))
ax11.set_yticks(np.arange(0, 0.301, 0.1))
ax11.plot( xHump/0.305, yHump/0.305,  linestyle = '-',color = 'black', linewidth = 1,zorder=4 )
ax11.fill_between(xHump/0.305, yHump/0.305, 0, facecolor='silver',zorder=3)
ax11.set_xlabel(r'$\it{x/C}$',fontsize=10, labelpad=1)
ax11.set_ylabel(r'$\it{y/C}$',fontsize=10, labelpad=1)


#---------------------------------------pehill---------------------------------------#

file1 = '../refData/truthData/pehill/postProcessing/sample_down/0/wallShearStress_down.raw'
xHump = np.loadtxt(file1, skiprows = 2)[:,0]
yHump = np.loadtxt(file1, skiprows = 2)[:,1]
xHump = np.insert(xHump, 0, 0)
xHump = np.append(xHump, 9)
yHump = np.insert(yHump, 0, 1)
yHump = np.append(yHump, 1)

# load DNS data
postPath = '../refData/truthData/pehill/postProcessing'
filePath = os.path.join(postPath+'/sample_left_U/0/U_left.raw')
xy_DNS = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_DNS_array, Uy_DNS_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]

# load baseline data
postPath = '../refData/baselineData_EARSM05/6_pehill/postProcessing'
filePath = os.path.join(postPath+'/sample_left_U/1/U_left.raw')
xy_base = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_base_array, Uy_base_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]

# load SST data
postPath = '../refData/baselineData_SST/6_pehill/postProcessing'
filePath = os.path.join(postPath+'/sample_left_U/1/U_left.raw')
xy_SST = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_SST_array, Uy_SST_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]


# load CFD data
tlast = getLatestTime(caseDir + '/6_pehill/postProcessing/sampleDict')
postPath = caseDir + '/6_pehill/postProcessing'
filePath = os.path.join(postPath+'/sample_left_U', f'{tlast}', 'U_left.raw')
xy_CFD = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_obs_array, Uy_obs_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]

levels_ux = [0.01]
U_ref = 0.028
cntr1 = ax21.tricontour(xy_DNS[:,0], xy_DNS[:,1], Ux_DNS_array/U_ref, levels=levels_ux, colors = markercolor, extend = 'both', linewidths=2.0)
cntr2 = ax21.tricontour(xy_SST[:,0], xy_SST[:,1], Ux_SST_array/U_ref, levels=levels_ux, colors = lineSSTcolor, extend = 'both', linestyles = '-.', linewidths=2.0)
cntr3 = ax21.tricontour(xy_base[:,0], xy_base[:,1], Ux_base_array/U_ref, levels=levels_ux, colors = lineBasecolor, extend = 'both', linestyles = '--', linewidths=2.0)
cntr4 = ax21.tricontour(xy_CFD[:,0], xy_CFD[:,1], Ux_obs_array/U_ref, levels=levels_ux, colors = lineCFDcolor, extend = 'both', linewidths=2.0)
ax21.set(xlim=(0, 9), ylim=(0, 2))
ax21.set_aspect('equal')
ax21.tick_params(labelsize=8)
ax21.set_xticks(np.arange(0, 9.01, 1))
ax21.set_yticks(np.arange(0, 2.01, 1))
ax21.plot( xHump, yHump,  linestyle = '-',color = 'black', linewidth = 1,zorder=4 )
ax21.fill_between(xHump, yHump, 0, facecolor='silver',zorder=1)
ax21.set_xlabel(r'$\it{x/H}$',fontsize=10, labelpad=1)
ax21.set_ylabel(r'$\it{y/H}$',fontsize=10, labelpad=1)


#--------------------------------------FAITH hill-----------------------------------#
U_ref = 50.292
L_ref = 0.2286
xHump = np.linspace(-50,50,501)
yHump = np.linspace(0,0,501)
mask = (xHump >= -9) & (xHump <= 9)
yHump[mask] = (3*np.cos(xHump[mask]*np.pi/9)+3)
xHump = xHump*0.0254/L_ref
yHump = yHump*0.0254/L_ref
# load DNS data
postPath = '../refData/truthData/FAITHhill/Centerline_FAITH_2Hz_4000samps_scalar'
xy_DNS = np.loadtxt(postPath+'/U_mean_axis00.dat', skiprows = 9)[:, 0:2]/1000
Ux_DNS_array = np.loadtxt(postPath+'/U_mean_axis00.dat', skiprows = 9)[:,2]
Uy_DNS_array = np.loadtxt(postPath+'/V_mean_axis00.dat', skiprows = 9)[:,2]
# load SST data
postPath = '../refData/baselineData_SST/7_FAITHhill/postProcessing'
filePath = os.path.join(postPath+'/sample_plane_z0/10000/U_z0.raw')
xy_SST = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_SST_array, Uy_SST_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]
# load baseline data
postPath = '../refData/baselineData_EARSM05/7_FAITHhill/postProcessing'
filePath = os.path.join(postPath+'/sample_plane_z0/8000/U_z0.raw')
xy_base = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_base_array, Uy_base_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]
# load CFD data
postPath = caseDir + '/7_FAITHhill/postProcessing'
t_CFD = getLatestTime(postPath+'/sample_plane_z0')
filePath = os.path.join(postPath+'/sample_plane_z0', f'{t_CFD}', 'U_z0.raw')
xy_CFD = np.loadtxt(filePath, skiprows = 2)[:, 0:2]
Ux_obs_array, Uy_obs_array = np.loadtxt(filePath, skiprows = 2)[:,3], np.loadtxt(filePath, skiprows = 2)[:,4]
filePath = os.path.join(postPath+'/sample_plane_z0', f'{t_CFD}', 'V_z0.raw')
cellV_CFD = np.loadtxt(filePath, skiprows = 2)[:,3]
levels_ux = [-0.01]



x = np.linspace(-1.2, 2, 300)
y = np.linspace(-1.2, 0, 100)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = (3 * np.cos(np.pi * R) + 3) / 9
mask = R > 1.0
Z[mask] = 0
ax_new = fig.add_axes([-0.1, -0.15, 0.7, 0.8], projection='3d')
surface = ax_new.plot_surface(X, Y, Z, color = 'silver', rstride=1, cstride=1, linewidth=0, antialiased=False)

f_wall = interp1d(xHump, yHump+0.01, kind='linear', fill_value='extrapolate')


mask = xy_DNS[:, 0] > 0
x_zero, z_zero = extract_contour(xy_DNS[mask,0]/L_ref, xy_DNS[mask,1]/L_ref, Ux_DNS_array[mask]/U_ref, -0.01)
mask = z_zero > f_wall(x_zero)
ax_new.plot(x_zero[mask], np.zeros_like(x_zero[mask]), z_zero[mask],  linestyle = '-',color = markercolor, linewidth = 2.0, zorder = 10)
mask = xy_SST[:, 0] > 0
x_zero, z_zero = extract_contour(xy_SST[mask,0]/L_ref, xy_SST[mask,1]/L_ref, Ux_SST_array[mask]/U_ref, -0.01)
mask = z_zero > f_wall(x_zero)
ax_new.plot(x_zero[mask], np.zeros_like(x_zero[mask]), z_zero[mask],  linestyle = '-.',color = lineSSTcolor, linewidth = 2.0, zorder = 10)
mask = xy_base[:, 0] > 0
x_zero, z_zero = extract_contour(xy_base[mask,0]/L_ref, xy_base[mask,1]/L_ref, Ux_base_array[mask]/U_ref, -0.01)
mask = z_zero > f_wall(x_zero)
ax_new.plot(x_zero[mask], np.zeros_like(x_zero[mask]), z_zero[mask],  linestyle = '--',color = lineBasecolor, linewidth = 2.0, zorder = 10)
mask = xy_CFD[:, 0] > 0
x_zero, z_zero = extract_contour(xy_CFD[mask,0]/L_ref, xy_CFD[mask,1]/L_ref, Ux_obs_array[mask]/U_ref, -0.01)
mask = z_zero > f_wall(x_zero)
ax_new.plot(x_zero[mask], np.zeros_like(x_zero[mask]), z_zero[mask],  linestyle = '-',color = lineCFDcolor, linewidth = 2.0, zorder = 10)

mask = (xHump > -1.2) & (xHump < 2)
ax_new.plot( xHump[mask], np.zeros_like(yHump[mask]), yHump[mask],  linestyle = '-',color = 'black', linewidth = 1,zorder=20 )
ax_new.plot( [-1.2, -1.2, 2, 2], [0,0,0,0], [0, 0.8, 0.8, 0],  linestyle = '-',color = 'black', linewidth = 1,zorder=20 )

ax_new.patch.set_alpha(0.0)
ax_new.set_xlabel('X/R')
ax_new.set_ylabel('Y/R')
ax_new.set_zlabel('Z/R')
ax_new.view_init(elev=30, azim=-110)  
ax_new.set(xlim=(-1.2, 2), ylim=(-1.2, 1.2), zlim=(0, 0.8))
ax_new.set_box_aspect([3.2, 2.4, 0.8])
ax_new.set_axis_off()
ax_new.text2D(0.65, 0.6, 'Symmetry plane',transform=ax_new.transAxes, fontsize=10, rotation=10)


#---------------------------------------recDuct--------------------------------------#

Nx = 600  
Ny = 200  

# load DNS
filePath = '../refData/truthData/squareDuct_recDuct/'
Y_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_Y.dat', skiprows = 0)+1.0
Z_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_Z.dat', skiprows = 0)+3.0
U_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_U.dat', skiprows = 0)
V_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_V.dat', skiprows = 0)
W_DNS = np.loadtxt(filePath + 'DNS_Re=5817_AR3_W.dat', skiprows = 0)
Y_grid, Z_grid = np.meshgrid(Y_DNS, Z_DNS, indexing='ij')
original_shape = W_DNS.shape
Z_flat = np.tile(Z_DNS, len(Y_DNS))  
Y_flat = np.repeat(Y_DNS, len(Z_DNS))  
W_flat = W_DNS.flatten()
V_flat = V_DNS.flatten()
Umag_dns = np.sqrt(W_flat**2 + V_flat**2)
yz_DNS = np.zeros([len(Z_flat),2])
yz_DNS[:,0] = Y_flat
yz_DNS[:,1] = Z_flat
Xi, Yi, Ui = interpolate_to_structured_grid(Z_flat, Y_flat, W_flat, 0,3,0,1, Nx, Ny, method='linear')
Xi, Yi, Vi = interpolate_to_structured_grid(Z_flat, Y_flat, V_flat, 0,3,0,1, Nx, Ny, method='linear')
vortex_coords_DNS1 = find_min_location_local(Xi, Yi, np.sqrt(Ui**2+Vi**2), 0.01, 0.5, 0.1, 0.9)
vortex_coords_DNS2 = find_min_location_local(Xi, Yi, np.sqrt(Ui**2+Vi**2), 0.5, 2, 0.1, 0.9)

# load baseline
t_latest = 6000
postPath = '../refData/baselineData_EARSM05/2.1_recDuct/postProcessing/extractPlane/'
filePath = os.path.join(postPath, f'{t_latest}', 'U_s0.raw')
U_prof_cfd = np.loadtxt(filePath, skiprows = 2)
Ux_obs_array = U_prof_cfd[:, 3]
Uy_obs_array = U_prof_cfd[:, 4]
Uz_obs_array = U_prof_cfd[:, 5]
yz_CFD = U_prof_cfd[:, 1:3]
Xi, Yi, Ui = interpolate_to_structured_grid(yz_CFD[:,1], yz_CFD[:,0], Uz_obs_array, 0,3,0,1, Nx, Ny, method='linear')
Xi, Yi, Vi = interpolate_to_structured_grid(yz_CFD[:,1], yz_CFD[:,0], Uy_obs_array, 0,3,0,1, Nx, Ny, method='linear')
vortex_coords_base1 = find_min_location_local(Xi, Yi, np.sqrt(Ui**2+Vi**2), 0.01, 0.5, 0.1, 0.9)
vortex_coords_base2 = find_min_location_local(Xi, Yi, np.sqrt(Ui**2+Vi**2), 0.5, 2, 0.1, 0.9)
Umag_base = np.sqrt(Uy_obs_array**2 + Uz_obs_array**2)

# load trained
tlast = getLatestTime(caseDir + '/2.1_recDuct/postProcessing/extractPlane')
postPath = caseDir + '/2.1_recDuct/postProcessing/extractPlane/'
filePath = os.path.join(postPath, f'{tlast}', 'U_s0.raw')
U_prof_cfd = np.loadtxt(filePath, skiprows = 2)
Ux_obs_array = U_prof_cfd[:, 3]
Uy_obs_array = U_prof_cfd[:, 4]
Uz_obs_array = U_prof_cfd[:, 5]
yz_CFD = U_prof_cfd[:, 1:3]
Umag_moe = np.sqrt(Uy_obs_array**2 + Uz_obs_array**2)
Xi, Yi, Ui = interpolate_to_structured_grid(yz_CFD[:,1], yz_CFD[:,0], Uz_obs_array, 0,3,0,1, Nx, Ny, method='linear')
Xi, Yi, Vi = interpolate_to_structured_grid(yz_CFD[:,1], yz_CFD[:,0], Uy_obs_array, 0,3,0,1, Nx, Ny, method='linear')
vortex_coords_moe1 = find_min_location_local(Xi, Yi, np.sqrt(Ui**2+Vi**2), 0.01, 0.5, 0.1, 0.9)
vortex_coords_moe2 = find_min_location_local(Xi, Yi, np.sqrt(Ui**2+Vi**2), 0.5, 2, 0.1, 0.9)

ax12.streamplot(Xi-3, Yi+0.1, Ui, Vi, color=lineCFDcolor, linewidth=1, density=1, arrowsize=0.8, arrowstyle = '->')
ax12.plot(vortex_coords_moe1[0]-3,  vortex_coords_moe1[1]+0.1, 'sr', markersize=8, markeredgecolor=lineCFDcolor,markerfacecolor=lineCFDcolor, markeredgewidth=1.5)
ax12.plot(vortex_coords_moe2[0]-3,  vortex_coords_moe2[1]+0.1, 'sr', markersize=8, markeredgecolor=lineCFDcolor,markerfacecolor=lineCFDcolor, markeredgewidth=1.5)
ax12.plot(vortex_coords_DNS1[0]-3,  vortex_coords_DNS1[1]+0.1, 'o',  markersize=8, markeredgecolor=markercolor,markerfacecolor='none', markeredgewidth=1.5)
ax12.plot(vortex_coords_DNS2[0]-3,  vortex_coords_DNS2[1]+0.1, 'o',  markersize=8, markeredgecolor=markercolor,markerfacecolor='none', markeredgewidth=1.5)
ax12.plot(vortex_coords_base1[0]-3, vortex_coords_base1[1]+0.1, '^', markersize=8, markeredgecolor=lineBasecolor,markerfacecolor='none', markeredgewidth=1.5)
ax12.plot(vortex_coords_base2[0]-3, vortex_coords_base2[1]+0.1, '^', markersize=8, markeredgecolor=lineBasecolor,markerfacecolor='none', markeredgewidth=1.5)




ny = 200
line1yz = np.zeros([ny, 2])
ymax = 1
ymin = 0
line1yz[:,0] = np.linspace(ymin, ymax, ny)
zsample = [ 0.1, 0.5, 1, 1.5, 2, 2.5]
nz = len(zsample)
Umag_dns_line1 = np.zeros([ny, nz])
Umag_base_line1 = np.zeros([ny, nz])
Umag_moe_line1 = np.zeros([ny, nz])
for iz in range(nz):
    line1yz[:,1] = zsample[iz]
    Umag_dns_line1[:, iz] = getInterpolate2D(line1yz, yz_DNS, (Umag_dns))
    Umag_base_line1[:, iz] = getInterpolate2D(line1yz, yz_CFD, (Umag_base))
    Umag_moe_line1[:, iz] = getInterpolate2D(line1yz, yz_CFD, (Umag_moe))
u_scale = 20

for iz in range(nz):
    ax12.plot( Umag_dns_line1[:, iz]*u_scale+zsample[iz]-3, line1yz[:,0]-1, marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=5, label=r'exp' )
    ax12.plot( Umag_dns_line1[:, iz]*0+zsample[iz]-3, line1yz[:,0]-1, linewidth=2.0, linestyle = '-.',color = lineSSTcolor,label=r'SST' )
    ax12.plot( Umag_base_line1[:, iz]*u_scale+zsample[iz]-3, line1yz[:,0]-1, linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    ax12.plot( Umag_moe_line1[:, iz]*u_scale+zsample[iz]-3, line1yz[:,0]-1, linewidth=2.0, linestyle = '-',color = lineCFDcolor,label=r'trained' )


rect1 = patches.Rectangle((-3.05, -1.05+1.1), 0.05, 1.05, linewidth=0, edgecolor='none', facecolor='silver', zorder=2, clip_on=False)
rect2 = patches.Rectangle((-3.05, -1.05+1.1), 3.05, 0.05, linewidth=0, edgecolor='none', facecolor='silver', zorder=2, clip_on=False)
rect3 = patches.Rectangle((-3, -1+1.1), 3, 1, linewidth=1, edgecolor='black', facecolor='none', zorder=3, clip_on=False)
ax12.add_patch(rect1)
ax12.add_patch(rect2)
ax12.add_patch(rect3)

#for spines in ax12.spines:
ax12.spines['left'].set_bounds(-1, 0)  
ax12.spines['right'].set_bounds(-1, 0)
ax12.spines['top'].set_position(('data', 0))
ax12.set_ylabel(r'$y$',fontsize=10)
ax12.set_xlabel(r'$z + 20\sqrt{V^2 + W^2}$',fontsize=10)
ax12.tick_params(labelsize=8)
ax12.set_yticks(np.arange(-1, 0.01, 0.2))
ax12.set_xlim(-3,0)
ax12.set_ylim(-1,1.1)

ax12.yaxis.set_label_coords(-0.06, 0.25) 


#---------------------------------ANSJ-----------------------------------#
expDict = '../refData/truthData/ASJ_ANSJ/'
baseDict = '../refData/baselineData_EARSM05/3.1_ANSJ/'
CFDDict = caseDir + '/3.1_ANSJ/'
SSTdict = '../refData/baselineData_SST/3.1_ANSJ/'
tlast = getLatestTime(CFDDict + 'postProcessing/sampleDict/')
x_exp = np.loadtxt(expDict+'center_nearSonic.dat', skiprows = 1)[:,0]
U_exp = np.loadtxt(expDict+'center_nearSonic.dat', skiprows = 1)[:,2]
file1 = baseDict + 'postProcessing/sampleDict/0.02/line_center_U.xy'
x_base = np.loadtxt(file1)[:,0]
U_base = np.loadtxt(file1)[:,1]
file1 = CFDDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_center_U.xy'
x = np.loadtxt(file1)[:,0]
U = np.loadtxt(file1)[:,1]
file1 = SSTdict + 'postProcessing/sampleDict/0.05/line_center_U.xy'
x_sst = np.loadtxt(file1)[:,0]
U_sst = np.loadtxt(file1)[:,1]
U_jet = 310.0
ax32.plot( x_exp, U_exp , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=5, label=r'exp'  )
ax32.plot( x_sst/0.0508, U_sst/U_jet , linewidth=2.0, linestyle = '-.',color = lineSSTcolor,label=r'SST')
ax32.plot( x_base/0.0508, U_base/U_jet , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
ax32.plot( x/0.0508, U/U_jet , linewidth=2.0, linestyle = '-',color = lineCFDcolor,label=r'trained' )
ax32.set_ylabel(r'$\it{U/U_j}$',fontsize=10)
ax32.set_xlabel(r'$\it{x/D}$',fontsize=10, labelpad=1)
ax32.tick_params(labelsize=8)
ax32.set_xlim(0,22)
ax32.set_ylim(0,1.1)
ax32.minorticks_on()
ax32.tick_params(direction='in')
ax32.tick_params(which="minor", direction='in')


ax11.text(-0.07, 0.33, '(b) 2D-bump', fontsize = 16)
ax21.text(-0.5, 2.2, '(c) 2D-peHill', fontsize = 16)
ax_new.text2D(0.03, 0.6, '(d) 3D-hill', transform=ax_new.transAxes, fontsize=16)
ax12.text(-3, 1.15, '(e) RecDuct', fontsize = 16)
ax32.text(-2, 1.15, '(f) ANSJ', fontsize = 16)


plt.tight_layout()
plt.savefig('moe_result_3D.png',dpi=200)
