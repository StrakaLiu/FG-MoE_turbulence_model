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


matplotlib.use('Agg')  




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


if __name__ == "__main__":

    streamcolor = '#48c0aa'
    markerfacecolor = 'none'
    markercolor = '#456990'
    lineNNcolor = '#48c0aa'
    lineBasecolor = '#ef767a'


    caseDir = '../refData/trainedExpertModelData/'

    fig = plt.figure(figsize=(11.5, 8))
    gs = gridspec.GridSpec(6, 3, height_ratios=[0.1,1,1,4,1,1], width_ratios=[4, 4, 3.5])
    ax01 = plt.subplot(gs[0,:])
    ax11 = plt.subplot(gs[1:3, 0])
    ax12 = plt.subplot(gs[3, 0])
    ax13 = plt.subplot(gs[4:6, 0])
    ax21 = plt.subplot(gs[1:3, 1])
    ax22 = plt.subplot(gs[3, 1])
    ax23 = plt.subplot(gs[4:6, 1])
    ax31 = plt.subplot(gs[1, 2])
    ax32 = plt.subplot(gs[3, 2])
    ax33 = plt.subplot(gs[4, 2])
    ax41 = plt.subplot(gs[2, 2])
    ax43 = plt.subplot(gs[5, 2])


    ax11.set_aspect('equal')
    ax12.set_aspect('equal')
    ax13.set_aspect('equal')
    ax22.set_aspect('equal')
    ax31.set_aspect('equal')
    ax32.set_aspect('equal')
    ax41.set_aspect('equal')

    ax01.axis('off')

    #-------------------------plot streamlines--------------------#

    #---------------hump-------------------
    dict1 = caseDir + '1_NASAhump/postProcessing/sample_left/101/'
    X = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:, 0]/0.42
    Y = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:, 1]/0.42
    U = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,3]
    V = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,4]
    dict2 = caseDir + '1_NASAhump/postProcessing/sample_down/101/'
    x_wall = np.loadtxt(dict2+'p_down.raw', skiprows = 2)[:,0]/0.42
    y_wall = np.loadtxt(dict2+'p_down.raw', skiprows = 2)[:,1]/0.42
    xmin, xmax = -0.2, 1.4
    ymin, ymax = 0, 0.8
    Nx, Ny = 500, 200
    Xi, Yi, Ui = interpolate_to_structured_grid(X, Y, U, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
    Xi, Yi, Vi = interpolate_to_structured_grid(X, Y, V, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')

    ax11.streamplot(Xi, Yi, Ui, Vi, color=streamcolor, linewidth=1, density=1, arrowsize=0.8, arrowstyle = '->', zorder=1)
    x_wall_new = np.linspace(xmin, xmax, 500)
    f_wall = interp1d(x_wall, y_wall, kind='linear', fill_value='extrapolate')
    y_wall_new = f_wall(x_wall_new)
    ax11.fill_between(x_wall_new, y_wall_new, -0.05*(ymax - ymin), color='silver', zorder=2)
    ax11.plot( x_wall_new, y_wall_new,  linestyle = '-',color = 'black', linewidth = 1 )
    ax11.plot( [xmin, xmin, xmax, xmax], [ymin, ymax, ymax, ymin] ,  linestyle = '-',color = 'black', linewidth = 1 )
    ax11.set(xlim=(xmin-0.05*(xmax - xmin), xmax+0.05*(xmax - xmin)), ylim=(ymin-0.05*(ymax - ymin), ymax+0.05*(ymax - ymin)))
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax11.set_xticklabels([])
    ax11.set_yticklabels([])
    for spine in ax11.spines.values():
        spine.set_visible(False)
    ax11.text(xmin-0.05*xmax, 1.05*ymax, '(a)', fontsize = 16)

        
    #---------------duct-------------------
    dict1 = caseDir + '2_sqrDuct/postProcessing/sample_left_theta/25/'
    X = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:, 2]
    Y = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:, 1]
    U = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,5]
    V = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,4]
    xmin, xmax = -1, 0
    ymin, ymax = -1, 0
    Nx, Ny = 200, 200
    Xi, Yi, Ui = interpolate_to_structured_grid(X, Y, U, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
    Xi, Yi, Vi = interpolate_to_structured_grid(X, Y, V, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')

    ax12.streamplot(Xi, Yi, Ui, Vi, color=streamcolor, linewidth=1, density=1, arrowsize=0.8, arrowstyle = '->', zorder=1)
    rect1 = patches.Rectangle((-1.05, -1.05), 0.05, 1.05, linewidth=1, edgecolor='none', facecolor='silver', zorder=2)
    rect2 = patches.Rectangle((-1.05, -1.05), 1.05, 0.05, linewidth=1, edgecolor='none', facecolor='silver', zorder=2)
    rect3 = patches.Rectangle((-1, -1), 1, 1, linewidth=1, edgecolor='black', facecolor='none', zorder=3)
    ax12.add_patch(rect1)
    ax12.add_patch(rect2)
    ax12.add_patch(rect3)
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax12.set_xticklabels([])
    ax12.set_yticklabels([])
    ax12.set_xlim([-1.05, 0.01])
    ax12.set_ylim([-1.05, 0.01])
    for spine in ax12.spines.values():
        spine.set_visible(False)
    ax12.text(-0.5, 0.02, 'Symmetry', ha='center', fontsize = 12)
    ax12.text(0.02, -0.5, 'Symmetry', va='center', fontsize = 12,rotation=90)
    ax12.text(-1.25, 0.05, '(b)', fontsize = 16)

    #---------------ASJ--------------------
    D = 0.0508
    dict1 = caseDir + '3_ASJ/postProcessing/sample_left/101/'
    X = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,0]/D
    Y = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,2]/D
    U = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,3]
    V = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,5]
    

    dict2 = caseDir + '3_ASJ/postProcessing/sample_wall/101/'
    x_wall1 = np.loadtxt(dict2+'p_wall1.raw', skiprows = 2)[:,0]/D
    y_wall1 = np.loadtxt(dict2+'p_wall1.raw', skiprows = 2)[:,2]/D
    x_wall2 = np.loadtxt(dict2+'p_wall2.raw', skiprows = 2)[:,0]/D
    y_wall2 = np.loadtxt(dict2+'p_wall2.raw', skiprows = 2)[:,2]/D

    # Set interpolation grid parameters
    xmin, xmax = -4, 12
    ymin, ymax = 0, 8
    Nx, Ny = 500, 200
    
    # Call interpolation function
    Xi, Yi, Ui = interpolate_to_structured_grid(X, Y, U, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
    Xi, Yi, Vi = interpolate_to_structured_grid(X, Y, V, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
        
    # Use fill_between to fill the area between walls
    # Sort wall data for plotting
    sorted_indices_wall1 = np.argsort(x_wall1)
    sorted_indices_wall2 = np.argsort(x_wall2)
    x_wall1_sorted = x_wall1[sorted_indices_wall1]
    y_wall1_sorted = y_wall1[sorted_indices_wall1]
    x_wall2_sorted = x_wall2[sorted_indices_wall2]
    y_wall2_sorted = y_wall2[sorted_indices_wall2]
    all_x = np.union1d(x_wall1_sorted, x_wall2_sorted)
    # If all_x range is too wide, limit to common range
    common_min = max(x_wall1_sorted.min(), x_wall2_sorted.min())
    common_max = min(x_wall1_sorted.max(), x_wall2_sorted.max())
    all_x = all_x[(all_x >= common_min) & (all_x <= common_max)]
    if len(all_x) == 0:
        x_common = np.linspace(common_min, common_max, max(len(x_wall1_sorted), len(x_wall2_sorted)))
    else:
        x_common = all_x
    f_wall1 = interp1d(x_wall1_sorted, y_wall1_sorted, kind='linear', fill_value='extrapolate')
    f_wall2 = interp1d(x_wall2_sorted, y_wall2_sorted, kind='linear', fill_value='extrapolate')
    y_wall1_interp = f_wall1(x_common)
    y_wall2_interp = f_wall2(x_common)
    Xwall1min = min(x_wall1_sorted)
    Xwall2min = min(x_wall2_sorted)


    ax13.streamplot(Xi, Yi, Ui, Vi, color=streamcolor, linewidth=1, density=1, arrowsize=0.8, arrowstyle = '->', zorder=1)
    ax13.fill_between(x_common, y_wall1_interp, y_wall2_interp, where=(y_wall1_interp <= y_wall2_interp), color='silver', zorder=3)
    #ax13.fill_between(np.array([Xwall1min, Xwall2min]), np.array([0.10794589, 0.10794589])/D, 0, color='silver', zorder=3)
    ax13.plot( x_wall1_sorted, y_wall1_sorted,  linestyle = '-',color = 'black', linewidth = 1, zorder=4 )
    ax13.plot( x_wall2_sorted, y_wall2_sorted,  linestyle = '-',color = 'black', linewidth = 1, zorder=4 )
    ax13.plot( np.array([Xwall1min, Xwall1min]), np.array([0, 0.076197099])/D,  linestyle = '-',color = 'black', linewidth = 1, zorder=4 )
    ax13.set(xlim=(Xwall1min, xmax), ylim=(ymin, ymax))
    ax13.set_xticks([])
    ax13.set_yticks([])
    ax13.set_xticklabels([])
    ax13.set_yticklabels([])
    ax13.text((xmax+Xwall1min)/2, -1, 'Axis', ha='center', fontsize = 12)
    ax13.text(xmin-0.05*xmax, 1.05*ymax, '(c)', fontsize = 16)

    

    #---------------------------hump lines-------------------------------------------#
    expDict = '../refData/truthData/NASAhump/'
    baseDict = '../refData/baselineData_EARSM05/1_nasaHump/'
    nnDict = caseDir + '1_NASAhump/'
    
    file = expDict + 'exp_cf.dat'
    x_exp1 = np.loadtxt(file, skiprows = 5)[:,0]
    Cf_exp = np.loadtxt(file, skiprows = 5)[:,1]
    ax21.plot( x_exp1, Cf_exp*1000 , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'truth' )
    
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
    ax21.plot( x/0.42, -wallShearStress/(0.5*34.6**2)*1000 , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    
    
    nz = 1; nx = 192
    file1 = nnDict + 'postProcessing/sample_down/101/wallShearStress_down.raw'
    file2 = nnDict + 'postProcessing/sample_down/101/p_down.raw'
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
    ax21.plot( x_obs,  Cf_obs*1000 , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    ax21.plot( [-2,10], [0,0] , linewidth=1, linestyle = '--',color = 'black' )
    ax21.set_ylabel(r'$\it{C_f}\times 10^3$',fontsize=10)
    ax21.set_xlabel(r'$\it{x/c}$',fontsize=10, labelpad=1)
    ax21.tick_params(labelsize=8)
    ax21.set_xlim(-0.2,1.6)
    ax21.minorticks_on()
    ax21.tick_params(direction='in')
    ax21.tick_params(which="minor", direction='in')
    
    ux_scale = 0.2
    uy_scale = -1.0
    file1 = expDict + 'Ulines/'
    xline = [0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    ishow = [ 1, 3, 5]
    for i in ishow:
        y = np.loadtxt(file1 + 'line' +str(i) + '.dat', skiprows = 2)[:,1]
        U = np.loadtxt(file1 + 'line' +str(i) + '.dat', skiprows = 2)[:,2]
        V = np.loadtxt(file1 + 'line' +str(i) + '.dat', skiprows = 2)[:,3]
        ax31.plot( xline[i] + ux_scale*U , y , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp' )
        ax41.plot( xline[i] + uy_scale*V , y , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp' )
    
    file1 = baseDict + 'postProcessing/sample_lines_U/1/'
    for i in ishow:
        y = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,0]/0.42
        U = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,1]/34.6
        V = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,2]/34.6
        ax31.plot( xline[i] + ux_scale*U , y, linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
        ax41.plot( xline[i] + uy_scale*V , y, linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    
    file1 = nnDict + 'postProcessing/sample_lines_U/101/'
    for i in ishow:
        y = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,0]/0.42
        U = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,1]/34.6
        V = np.loadtxt(file1 + 'line' +str(i) + '_U.xy')[:,2]/34.6
        ax31.plot( xline[i] + ux_scale*U , y, linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
        ax41.plot( xline[i] + uy_scale*V , y, linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    
    ax31.plot( xHump/0.42, yHump/0.42, linewidth=1, linestyle = '-',color = 'black' )
    ax31.fill_between(xHump/0.42, yHump/0.42, 0, facecolor='silver')
    ax41.plot( xHump/0.42, yHump/0.42, linewidth=1, linestyle = '-',color = 'black' )
    ax41.fill_between(xHump/0.42, yHump/0.42, 0, facecolor='silver')
    
    ax31.set_ylabel(r'$\it{y/c}$',fontsize=10, labelpad=-3)
    ax31.set_xlabel(r'$\it{x/c} + 0.2U/U_{0}$',fontsize=10, labelpad=1)
    ax31.tick_params(labelsize=8)
    ax31.set_xlim(0.55,1.45)
    ax31.set_ylim(0,0.2)
    ax31.minorticks_on()
    ax31.tick_params(direction='in')
    ax31.tick_params(which="minor", direction='in')
    
    ax41.set_ylabel(r'$\it{y/c}$',fontsize=10, labelpad=-3)
    ax41.set_xlabel(r'$\it{x/c} - V/U_{0}$',fontsize=10, labelpad=1)
    ax41.tick_params(labelsize=8)
    ax41.set_xlim(0.55,1.45)
    ax41.set_ylim(0,0.2)
    ax41.minorticks_on()
    ax41.tick_params(direction='in')
    ax41.tick_params(which="minor", direction='in')



    #---------------------------sqr lines-------------------------------------------#
    # load DNS data
    filePath = os.path.join('../refData/truthData/squareDuct_recDuct/', 'DNS_Re=40000.dat')
    U_prof_dns = np.loadtxt(filePath, skiprows = 20)
    zy_DNS = U_prof_dns[:, 0:2]
    yz_DNS = zy_DNS[:, [1,0]]
    Ux_dns_array, Uy_dns_array, Uz_dns_array = mirrorAvgU(yz_DNS, U_prof_dns[:,2], U_prof_dns[:,4], U_prof_dns[:,5])
    
    # load baseline 
    postPath = '../refData/baselineData_EARSM05/2_sqrDuct_Re=40000/postProcessing/sample_left/'
    t_latest = 1
    filePath = os.path.join(postPath, f'{t_latest}', 'U_left.raw')
    U_prof_base = np.loadtxt(filePath, skiprows = 2)
    yz_base = U_prof_base[:, 1:3]
    Ux_base_array, Uy_base_array, Uz_base_array = mirrorAvgU(yz_base, U_prof_base[:,3], U_prof_base[:,4], U_prof_base[:,5])
    
    # load NN data
    postPath = caseDir + '2_sqrDuct/postProcessing/sample_left/'
    t_latest = 24
    filePath = os.path.join(postPath, f'{t_latest}', 'U_left.raw')
    U_prof_cfd = np.loadtxt(filePath, skiprows = 2)
    yz_CFD = U_prof_cfd[:, 1:3]
    Ux_obs_array, Uy_obs_array, Uz_obs_array = mirrorAvgU(yz_CFD, U_prof_cfd[:,3], U_prof_cfd[:,4], U_prof_cfd[:,5])

    ny = 200
    line1yz = np.zeros([ny, 2])
    ymax = max(abs(yz_CFD[:,0]))
    ymin = min(abs(yz_CFD[:,0]))
    line1yz[:,0] = np.linspace(-ymax, -ymin, ny)
    zsample = [ -0.8,  -0.4, -0.02]
    nz = len(zsample)
    Ux_mean_dns_line1 = np.zeros([ny, nz])
    Uy_mean_dns_line1 = np.zeros([ny, nz])
    Ux_mean_base_line1 = np.zeros([ny, nz])
    Uy_mean_base_line1 = np.zeros([ny, nz])
    Ux_mean_nn_line1 = np.zeros([ny, nz])
    Uy_mean_nn_line1 = np.zeros([ny, nz])
    for iz in range(nz):
        line1yz[:,1] = zsample[iz]
        Ux_mean_dns_line1[:, iz] = getInterpolate2D(line1yz, yz_DNS, (Ux_dns_array))
        Uy_mean_dns_line1[:, iz] = getInterpolate2D(line1yz, yz_DNS, (Uy_dns_array))
        Ux_mean_base_line1[:, iz] = getInterpolate2D(line1yz, yz_base, (Ux_base_array))
        Uy_mean_base_line1[:, iz] = getInterpolate2D(line1yz, yz_base, (Uy_base_array))
        Ux_mean_nn_line1[:, iz] = getInterpolate2D(line1yz, yz_CFD, (Ux_obs_array))
        Uy_mean_nn_line1[:, iz] = getInterpolate2D(line1yz, yz_CFD, (Uy_obs_array))
    
    
    ux_scale = 0.3
    uy_scale = -20
    uu_scale = 1
    vv_scale = 3
    
    for iz in range(nz):
        ax22.plot( Ux_mean_dns_line1[:, iz]*ux_scale+zsample[iz], line1yz[:,0], marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp' )
        ax22.plot( Ux_mean_base_line1[:, iz]*ux_scale+zsample[iz], line1yz[:,0], linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
        ax22.plot( Ux_mean_nn_line1[:, iz]*ux_scale+zsample[iz], line1yz[:,0], linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
        ax32.plot( Uy_mean_dns_line1[:, iz]*uy_scale+zsample[iz], line1yz[:,0] , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp' )
        ax32.plot( Uy_mean_base_line1[:, iz]*uy_scale+zsample[iz], line1yz[:,0] , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
        ax32.plot( Uy_mean_nn_line1[:, iz]*uy_scale+zsample[iz], line1yz[:,0] , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    
    ax22.set_ylabel(r'$y$',fontsize=10, labelpad=-5)
    ax22.set_xlabel(r'$z + 0.3U$',fontsize=10, labelpad=1)
    ax22.set_xticks(np.arange(-1,0.21, 0.2))
    ax22.tick_params(labelsize=8)
    ax22.set_xlim(-0.8,0.4)
    ax22.set_ylim(-1,0)
    ax22.minorticks_on()
    ax32.set_ylabel(r'$y$',fontsize=10, labelpad=-5)
    ax32.set_xlabel(r'$z - 20V$',fontsize=10, labelpad=1)
    ax32.tick_params(labelsize=8)
    ax32.set_xlim(-1,0.2)
    ax32.set_ylim(-1,0)
    ax32.set_xticks(np.arange(-1,0.21, 0.2))
    ax32.minorticks_on()
    


    #---------------------------ASJ lines-------------------------------------------#
    U_jet = 171.0
    expDict = '../refData/truthData/ASJ_ANSJ/'
    baseDict = '../refData/baselineData_EARSM05/3_ASJ/'
    nnDict = caseDir + '3_ASJ/'
    tlast = 101
    x_exp = np.loadtxt(expDict+'center.dat', skiprows = 1)[:,0]
    U_exp = np.loadtxt(expDict+'center.dat', skiprows = 1)[:,2]
    file1 = baseDict + 'postProcessing/sampleDict/0.05/line_center_U.xy'
    x_base = np.loadtxt(file1)[:,0]
    U_base = np.loadtxt(file1)[:,1]
    file1 = nnDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_center_U.xy'
    x = np.loadtxt(file1)[:,0]
    U = np.loadtxt(file1)[:,1]
    ax23.plot( x_exp, U_exp , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp'  )
    ax23.plot( x_base/0.0508, U_base/U_jet , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    ax23.plot( x/0.0508, U/U_jet , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    ax23.set_ylabel(r'$\it{U}/U_j$',fontsize=10)
    ax23.set_xlabel(r'$\it{x/D_j}$',fontsize=10, labelpad=1)
    ax23.tick_params(labelsize=8)
    ax23.set_xlim(0,25)
    ax23.set_ylim(0,1.1)
    ax23.minorticks_on()
    ax23.tick_params(direction='in')
    ax23.tick_params(which="minor", direction='in')
    
    x = [2.0, 10.0, 20.0]
    z_exp2 = np.loadtxt(expDict+'x2.dat', skiprows = 1)[:,1]
    U_exp2 = np.loadtxt(expDict+'x2.dat', skiprows = 1)[:,2]
    V_exp2 = np.loadtxt(expDict+'x2.dat', skiprows = 1)[:,3]
    z_exp10 = np.loadtxt(expDict+'x10.dat', skiprows = 1)[:,1]
    U_exp10 = np.loadtxt(expDict+'x10.dat', skiprows = 1)[:,2]
    V_exp10 = np.loadtxt(expDict+'x10.dat', skiprows = 1)[:,3]
    z_exp20 = np.loadtxt(expDict+'x20.dat', skiprows = 1)[:,1]
    U_exp20 = np.loadtxt(expDict+'x20.dat', skiprows = 1)[:,2]
    V_exp20 = np.loadtxt(expDict+'x20.dat', skiprows = 1)[:,3]
    file1 = baseDict + 'postProcessing/sampleDict/0.05/line_x_2_U.xy'
    z_base2 = np.loadtxt(file1)[:,0]
    U_base2 = np.loadtxt(file1)[:,1]
    V_base2 = np.loadtxt(file1)[:,3]
    file1 = baseDict + 'postProcessing/sampleDict/0.05/line_x_10_U.xy'
    z_base10 = np.loadtxt(file1)[:,0]
    U_base10 = np.loadtxt(file1)[:,1]
    V_base10 = np.loadtxt(file1)[:,3]
    file1 = baseDict + 'postProcessing/sampleDict/0.05/line_x_20_U.xy'
    z_base20 = np.loadtxt(file1)[:,0]
    U_base20 = np.loadtxt(file1)[:,1]
    V_base20 = np.loadtxt(file1)[:,3]
    file1 = nnDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_x_2_U.xy'
    z2 = np.loadtxt(file1)[:,0]
    U2 = np.loadtxt(file1)[:,1]
    V2 = np.loadtxt(file1)[:,3]
    file1 = nnDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_x_10_U.xy'
    z10 = np.loadtxt(file1)[:,0]
    U10 = np.loadtxt(file1)[:,1]
    V10 = np.loadtxt(file1)[:,3]
    file1 = nnDict + 'postProcessing/sampleDict/' + str(tlast)+ '/line_x_20_U.xy'
    z20 = np.loadtxt(file1)[:,0]
    U20 = np.loadtxt(file1)[:,1]
    V20 = np.loadtxt(file1)[:,3]
    ax33.plot(x[0] + 10*U_exp2, z_exp2 , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp'  )
    ax33.plot(x[0] + 10*U_base2/U_jet, z_base2/0.0508 , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    ax33.plot(x[0] + 10*U2/U_jet, z2/0.0508 , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    ax33.plot(x[1] + 10*U_exp10, z_exp10 , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp'  )
    ax33.plot(x[1] + 10*U_base10/U_jet, z_base10/0.0508 , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    ax33.plot(x[1] + 10*U10/U_jet, z10/0.0508 , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    ax33.plot(x[2] + 10*U_exp20, z_exp20 , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp'  )
    ax33.plot(x[2] + 10*U_base20/U_jet, z_base20/0.0508 , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    ax33.plot(x[2] + 10*U20/U_jet, z20/0.0508 , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    
    ax43.plot(x[0] + 400*V_exp2, z_exp2 , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp'  )
    ax43.plot(x[0] + 400*V_base2/U_jet, z_base2/0.0508 , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    ax43.plot(x[0] + 400*V2/U_jet, z2/0.0508 , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    ax43.plot(x[1] + 400*V_exp10, z_exp10 , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp'  )
    ax43.plot(x[1] + 400*V_base10/U_jet, z_base10/0.0508 , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    ax43.plot(x[1] + 400*V10/U_jet, z10/0.0508 , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    ax43.plot(x[2] + 400*V_exp20, z_exp20 , marker='o',markeredgewidth =1.5, linestyle = 'none', color=markercolor,markerfacecolor=markerfacecolor, markevery = 0.03, markersize=4, label=r'exp'  )
    ax43.plot(x[2] + 400*V_base20/U_jet, z_base20/0.0508 , linewidth=2.0, linestyle = '--',color = lineBasecolor,label=r'baseline' )
    ax43.plot(x[2] + 400*V20/U_jet, z20/0.0508 , linewidth=2.0, linestyle = '-',color = lineNNcolor,label=r'trained' )
    
    ax33.set_ylim(0,1.5)
    ax33.minorticks_on()
    ax33.tick_params(direction='in')
    ax33.tick_params(which="minor", direction='in')
    ax33.tick_params(labelsize=8)
    
    ax43.set_ylim(0,1.5)
    ax43.minorticks_on()
    ax43.tick_params(direction='in')
    ax43.tick_params(which="minor", direction='in')
    ax43.tick_params(labelsize=8)
    
    ax33.set_ylabel(r'$y/D$',fontsize=10)
    ax33.set_xlabel(r'$x/D + 10U/U_j$',fontsize=10)
    ax43.set_ylabel(r'$y/D$',fontsize=10)
    ax43.set_xlabel(r'$x/D + 400V/U_j$',fontsize=10)


    ax_new = fig.add_axes([0, 0.9, 1.0, 0.1])
    ax_new.axis('off')
    handles, labels = ax21.get_legend_handles_labels()
    leg = ax_new.legend(handles, labels, 
                  loc=(0.4,0.1),
                  ncol=3,
                  frameon=False,
                  fontsize=16)


    plt.tight_layout()
    plt.savefig('expert_result.png',dpi=100)
    plt.savefig('expert_result.eps')