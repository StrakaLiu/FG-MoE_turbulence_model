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


matplotlib.use('Agg')  




def calTheta(theta_scaled):
    theta = np.sign(theta_scaled)*np.abs(theta_scaled)/(1.0 - np.abs(theta_scaled))
    return theta

def F_q(A, B, q):
    tempQ = np.clip(-A*(q - B), -100, 100)
    F = 1 / (1 + np.exp(tempQ))
    return F



def interpolate_to_structured_grid(X, Y, Gamma, xmin, xmax, ymin, ymax, Nx, Ny, method='linear', smooth_factor=None):
    xi = np.linspace(xmin, xmax, Nx)
    yi = np.linspace(ymin, ymax, Ny)
    Xi, Yi = np.meshgrid(xi, yi)
    
    points = np.column_stack((X, Y))
    values = Gamma
    
    if method == 'rbf':
        rbf = Rbf(X, Y, Gamma, function='multiquadric', smooth=smooth_factor or 0)
        Zi = rbf(Xi, Yi)
    elif method == 'idw':
        Zi = inverse_distance_weighting(X, Y, Gamma, Xi, Yi, power=2)
    else:
        Zi = griddata(points, values, (Xi, Yi), method=method)
    
    return Xi, Yi, Zi


def inverse_distance_weighting(x_obs, y_obs, z_obs, x_grid, y_grid, power=2):
    distances = np.sqrt((x_grid[:, :, np.newaxis] - x_obs[np.newaxis, np.newaxis, :])**2 + 
                        (y_grid[:, :, np.newaxis] - y_obs[np.newaxis, np.newaxis, :])**2)
    
    distances = np.where(distances == 0, 1e-9, distances)
    
    weights = 1 / distances**power
    
    numerator = np.sum(weights * z_obs[np.newaxis, np.newaxis, :], axis=2)
    denominator = np.sum(weights, axis=2)
    
    Zi = numerator / denominator
    
    return Zi

def adaptive_interpolation(X, Y, Gamma, xmin, xmax, ymin, ymax, Nx, Ny, method='linear'):
    xi = np.linspace(xmin, xmax, Nx)
    yi = np.linspace(ymin, ymax, Ny)
    Xi, Yi = np.meshgrid(xi, yi)
    
    grid_points = np.column_stack((Xi.ravel(), Yi.ravel()))
    data_points = np.column_stack((X, Y))
    
    from scipy.spatial.distance import cdist
    distances = cdist(grid_points, data_points)
    min_distances = np.min(distances, axis=1).reshape(Nx, Ny)
    
    points = np.column_stack((X, Y))
    values = Gamma
    
    Zi_standard = griddata(points, values, (Xi, Yi), method=method)
    
    distance_threshold = np.percentile(min_distances, 75)
    
    sparse_mask = min_distances > distance_threshold
    
    if np.any(sparse_mask):
        rbf = Rbf(X, Y, Gamma, function='multiquadric', smooth=0.1)
        Zi_rbf = rbf(Xi, Yi)
        
        Zi = Zi_standard.copy()
        Zi[sparse_mask] = Zi_rbf[sparse_mask]
    else:
        Zi = Zi_standard
    
    return Xi, Yi, Zi



if __name__ == "__main__":

    caseDir = '../refData/trainedExpertModelData/'

    fig = plt.figure(figsize=(9.3, 9))
    gs = gridspec.GridSpec(6, 3, height_ratios=[1,1,1,1,1,1], width_ratios=[2, 1, 0.1])
    ax11 = plt.subplot(gs[0, 0])
    ax12 = plt.subplot(gs[1, 0])
    ax13 = plt.subplot(gs[0:2, 1])
    ax21 = plt.subplot(gs[2, 0])
    ax22 = plt.subplot(gs[3, 0])
    ax23 = plt.subplot(gs[2:4, 1])
    ax1 = plt.subplot(gs[4, 0])
    ax2 = plt.subplot(gs[5, 0])
    ax3 = plt.subplot(gs[4:6, 1])
    ax0 = plt.subplot(gs[:, 2])
    ax11.set_aspect('equal')
    ax12.set_aspect('equal')
    ax13.set_aspect('equal')
    ax21.set_aspect('equal')
    ax22.set_aspect('equal')
    ax23.set_aspect('equal')    
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    #---------------hump-------------------
    dict1 = caseDir + '1_NASAhump/postProcessing/sample_left/101/'
    X = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:, 0]/0.42
    Y = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:, 1]/0.42
    U = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,3]
    V = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,4]
    Gamma = np.loadtxt(dict1+'Gamma_w_left.raw', skiprows = 2)[:,3]
    theta1 = calTheta(np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,3])
    theta2 = calTheta(np.loadtxt(dict1+'theta2_Scaled__left.raw', skiprows = 2)[:,3])
    theta3 = calTheta(np.loadtxt(dict1+'theta3_Scaled__left.raw', skiprows = 2)[:,3])
    theta4 = calTheta(np.loadtxt(dict1+'theta4_Scaled__left.raw', skiprows = 2)[:,3])
    q1 = np.abs(theta1 + theta2)
    q2 = np.abs(theta3) + np.abs(theta4)
    F_1 = F_q(40, 0.25, q1)
    F_2 = F_q(1000, 0.01, q2)
    F_3 = F_q(50, 0.2, Gamma)
    dict2 = caseDir + '1_NASAhump/postProcessing/sample_down/101/'
    x_wall = np.loadtxt(dict2+'p_down.raw', skiprows = 2)[:,0]/0.42
    y_wall = np.loadtxt(dict2+'p_down.raw', skiprows = 2)[:,1]/0.42
    xmin, xmax = 0, 1.6
    ymin, ymax = 0, 0.4
    Nx, Ny = 500, 200
    Xi, Yi, Ui = interpolate_to_structured_grid(X, Y, U, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
    Xi, Yi, Vi = interpolate_to_structured_grid(X, Y, V, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')

    contourf = ax11.tricontourf(X, Y, F_1, levels=np.linspace(0,1,51), cmap='coolwarm', extend='both')
    ax11.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax11.fill_between(x_wall, y_wall, 0, color='silver', zorder=2)
    ax11.plot( x_wall, y_wall,  linestyle = '-',color = 'black', linewidth = 1 )
    ax11.set(xlim=(0, 1.6), ylim=(0, 0.4))
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax11.set_xticklabels([])
    ax11.set_yticklabels([])

    contourf = ax21.tricontourf(X, Y, F_2, levels=np.linspace(0,1,51), cmap='coolwarm', extend='both')
    ax21.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax21.fill_between(x_wall, y_wall, 0, color='silver', zorder=2)
    ax21.plot( x_wall, y_wall,  linestyle = '-',color = 'black', linewidth = 1 )
    ax21.set(xlim=(0, 1.6), ylim=(0, 0.4))
    ax21.set_xticks([])
    ax21.set_yticks([])
    ax21.set_xticklabels([])
    ax21.set_yticklabels([])

    contourf = ax1.tricontourf(X, Y, F_3, levels=np.linspace(0,1,51), cmap='coolwarm', extend='both')
    ax1.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax1.fill_between(x_wall, y_wall, 0, color='silver', zorder=2)
    ax1.plot( x_wall, y_wall,  linestyle = '-',color = 'black', linewidth = 1 )
    ax1.set(xlim=(0, 1.6), ylim=(0, 0.4))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    #---------------ASJ--------------------
    D = 0.0508
    dict1 = caseDir + '3_ASJ/postProcessing/sample_left/101/'
    X = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,0]/D
    Y = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,2]/D
    U = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,3]
    V = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,5]
    Gamma = np.loadtxt(dict1+'Gamma_w_left.raw', skiprows = 2)[:,3]
    theta1 = calTheta(np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,3])
    theta2 = calTheta(np.loadtxt(dict1+'theta2_Scaled__left.raw', skiprows = 2)[:,3])
    theta3 = calTheta(np.loadtxt(dict1+'theta3_Scaled__left.raw', skiprows = 2)[:,3])
    theta4 = calTheta(np.loadtxt(dict1+'theta4_Scaled__left.raw', skiprows = 2)[:,3])
    q1 = np.abs(theta1 + theta2)
    q2 = np.abs(theta3) + np.abs(theta4)
    F_1 = F_q(40, 0.25, q1)
    F_2 = F_q(1000, 0.01, q2)
    F_3 = F_q(50, 0.2, Gamma)

    dict2 = caseDir + '3_ASJ/postProcessing/sample_wall/101/'
    x_wall1 = np.loadtxt(dict2+'p_wall1.raw', skiprows = 2)[:,0]/D
    y_wall1 = np.loadtxt(dict2+'p_wall1.raw', skiprows = 2)[:,2]/D
    x_wall2 = np.loadtxt(dict2+'p_wall2.raw', skiprows = 2)[:,0]/D
    y_wall2 = np.loadtxt(dict2+'p_wall2.raw', skiprows = 2)[:,2]/D

    # Set interpolation grid parameters
    xmin, xmax = -6, 8.5
    ymin, ymax = -0.01, 3
    Nx, Ny = 500, 200
    
    # Call interpolation function
    Xi, Yi, F1i = interpolate_to_structured_grid(X, Y, F_1, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
    Xi, Yi, F2i = interpolate_to_structured_grid(X, Y, F_2, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
    Xi, Yi, F3i = interpolate_to_structured_grid(X, Y, F_3, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
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


    contourf = ax12.contourf(Xi, Yi, F1i, levels=np.linspace(0,1,51), cmap='coolwarm')
    ax12.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax12.fill_between(x_common, y_wall1_interp, y_wall2_interp, where=(y_wall1_interp <= y_wall2_interp), color='silver', zorder=2)
    ax12.fill_between(np.array([Xwall1min, Xwall2min]), np.array([0.10794589, 0.10794589])/D, 0, color='silver')
    ax12.plot( x_wall1_sorted, y_wall1_sorted,  linestyle = '-',color = 'black', linewidth = 1 )
    ax12.plot( x_wall2_sorted, y_wall2_sorted,  linestyle = '-',color = 'black', linewidth = 1 )
    ax12.plot( np.array([Xwall1min, Xwall1min]), np.array([0, 0.076197099])/D,  linestyle = '-',color = 'black', linewidth = 1 )
    ax12.set(xlim=(-3.8, 8.2), ylim=(0, 3))
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax12.set_xticklabels([])
    ax12.set_yticklabels([])

    contourf = ax22.contourf(Xi, Yi, F2i, levels=np.linspace(0,1,51), cmap='coolwarm')
    ax22.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax22.fill_between(x_common, y_wall1_interp, y_wall2_interp, where=(y_wall1_interp <= y_wall2_interp), color='silver', zorder=2)
    ax22.fill_between(np.array([Xwall1min, Xwall2min]), np.array([0.10794589, 0.10794589])/D, 0, color='silver')
    ax22.plot( x_wall1_sorted, y_wall1_sorted,  linestyle = '-',color = 'black', linewidth = 1 )
    ax22.plot( x_wall2_sorted, y_wall2_sorted,  linestyle = '-',color = 'black', linewidth = 1 )
    ax22.plot( np.array([Xwall1min, Xwall1min]), np.array([0, 0.076197099])/D,  linestyle = '-',color = 'black', linewidth = 1 )
    ax22.set(xlim=(-3.8, 8.2), ylim=(0, 3))
    ax22.set_xticks([])
    ax22.set_yticks([])
    ax22.set_xticklabels([])
    ax22.set_yticklabels([])

    contourf = ax2.contourf(Xi, Yi, F3i, levels=np.linspace(0,1,51), cmap='coolwarm')
    ax2.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax2.fill_between(x_common, y_wall1_interp, y_wall2_interp, where=(y_wall1_interp <= y_wall2_interp), color='silver', zorder=2)
    ax2.fill_between(np.array([Xwall1min, Xwall2min]), np.array([0.10794589, 0.10794589])/D, 0, color='silver')
    ax2.plot( x_wall1_sorted, y_wall1_sorted,  linestyle = '-',color = 'black', linewidth = 1 )
    ax2.plot( x_wall2_sorted, y_wall2_sorted,  linestyle = '-',color = 'black', linewidth = 1 )
    ax2.plot( np.array([Xwall1min, Xwall1min]), np.array([0, 0.076197099])/D,  linestyle = '-',color = 'black', linewidth = 1 )
    ax2.set(xlim=(-3.8, 8.2), ylim=(0, 3))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])


    
    #---------------duct-------------------
    dict1 = caseDir + '2_sqrDuct/postProcessing/sample_left_theta/25/'
    X = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:, 2]
    Y = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:, 1]
    U = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,5]
    V = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,4]
    Gamma = np.loadtxt(dict1+'Gamma_w_left.raw', skiprows = 2)[:,3]
    theta1 = calTheta(np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,3])
    theta2 = calTheta(np.loadtxt(dict1+'theta2_Scaled__left.raw', skiprows = 2)[:,3])
    theta3 = calTheta(np.loadtxt(dict1+'theta3_Scaled__left.raw', skiprows = 2)[:,3])
    theta4 = calTheta(np.loadtxt(dict1+'theta4_Scaled__left.raw', skiprows = 2)[:,3])
    q1 = np.abs(theta1 + theta2)
    q2 = np.abs(theta3) + np.abs(theta4)
    F_1 = F_q(40, 0.25, q1)
    F_2 = F_q(1000, 0.01, q2)
    F_3 = F_q(50, 0.2, Gamma)
    xmin, xmax = -1, 0
    ymin, ymax = -1, 0
    Nx, Ny = 200, 200
    Xi, Yi, Ui = interpolate_to_structured_grid(X, Y, U, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')
    Xi, Yi, Vi = interpolate_to_structured_grid(X, Y, V, xmin, xmax, ymin, ymax, Nx, Ny, method='linear')

    contourf = ax13.tricontourf(X, Y, F_1, levels=np.linspace(0,1,51), cmap='coolwarm', extend='both')
    ax13.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax13.set_xticks([])
    ax13.set_yticks([])  # Clear y-axis ticks
    ax13.set_xticklabels([])  # Clear x-axis labels
    ax13.set_yticklabels([])  # Clear y-axis labels

    contourf = ax23.tricontourf(X, Y, F_2, levels=np.linspace(0,1,51), cmap='coolwarm', extend='both')
    ax23.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax23.set_xticks([])
    ax23.set_yticks([])
    ax23.set_xticklabels([])
    ax23.set_yticklabels([])

    contourf = ax3.tricontourf(X, Y, F_3, levels=np.linspace(0,1,51), cmap='coolwarm', extend='both')
    ax3.streamplot(Xi, Yi, Ui, Vi, color='#d2edf2', linewidth=1, density=0.5, arrowsize=0, arrowstyle = '-', zorder=1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    

    cbar = plt.colorbar(contourf, cax=ax0, extend='neither')
    cbar.set_ticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    plt.savefig('F_contour.png',dpi=170)






