import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from collections import OrderedDict
import os
import re
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

def getLatestTime(Dir):
    # get the latest time
    t = os.listdir(Dir)
    for i in range(len(t)):
        t[i] = int(t[i])
    t_latest = np.array(t).max()
    return t_latest

def count_folders(directory):
    contents = os.listdir(directory)
    folders = [item for item in contents if os.path.isdir(os.path.join(directory, item))]
    return len(folders)





def parse_tecplot_file(filename):
    zones = []
    variables = None
    current_zone_data = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue

            if line.startswith('VARIABLES'):
                var_line = line.split('=', 1)[1].strip()
                variables = re.findall(r'"([^"]+)"|(\S+)', var_line)
                variables = [name[0] if name[0] else name[1] for name in variables]
                n_vars = len(variables)
                continue

            if line.startswith('ZONE'):
                if current_zone_data:
                    zones.append(np.array(current_zone_data, dtype=float))
                    current_zone_data = []
                continue

            try:
                values = list(map(float, line.split()))
                if len(values) != n_vars:
                    print(f"Warning: skipping line with {len(values)} values (expected {n_vars}): {line}")
                    continue
                current_zone_data.append(values)
            except ValueError:
                continue

    if current_zone_data:
        zones.append(np.array(current_zone_data, dtype=float))

    return zones, variables


def interpolate_cp(data_xz_cp, xz_inter):
    """
    Using (x, z, Cp) scattered data, interpolate Cp values at given (x, z) positions.

    Parameters
    ----------
    data_xz_cp : array-like, shape (N, 3)
        Input data, columns are [x, z, Cp]
    xz_inter : array-like, shape (M, 2)
        Interpolation target points, columns are [x, z]

    Returns
    -------
    Cp_inter : ndarray, shape (M,)
        Interpolated Cp values. If linear interpolation is unavailable (e.g., outside convex hull), automatically use nearest neighbor.
    """
    data_xz_cp = np.asarray(data_xz_cp)
    xz_inter = np.asarray(xz_inter)

    points = data_xz_cp[:, :2]   # (x, z)
    values = data_xz_cp[:, 2]    # Cp

    # Linear interpolation (Delaunay triangulation)
    linear_interp = LinearNDInterpolator(points, values)
    Cp_linear = linear_interp(xz_inter)

    return Cp_linear


def plot_cp_distribution_fuselage(x_exp, cp_exp, x_inter, cp_base, cp_moe, x_max, x_min, figsize, title):
    # Define some color variables, you can adjust these values as needed
    streamcolor = '#48c0aa'
    markerfacecolor = 'none'
    markercolor = '#456990'
    linemoecolor = '#48c0aa'
    lineBasecolor = '#ef767a'
    lineSSTcolor = '#5b89d8'
    fig, ax = plt.subplots(figsize=figsize)
    # Experimental data points
    ax.plot((x_exp - x_min) / (x_max - x_min), -cp_exp,
        marker='o', markeredgewidth=3.0, linestyle='none',
        color=markercolor, markerfacecolor=markerfacecolor, markersize=8, markevery=0.03, linewidth=6,
        label=r'$-Cp$, exp'
    )
    # Baseline model prediction results
    ax.plot(
        (x_inter - x_min) / (x_max - x_min), -cp_base,
         linewidth=4.0, linestyle='--',
        color=lineBasecolor, label=r'$-Cp$, baseline'
    )
    # Post-training model prediction results
    ax.plot(
        (x_inter - x_min) / (x_max - x_min), -cp_moe,
         linewidth=4.0, dashes=(5, 2),
        color=linemoecolor, label=r'$-Cp$, MoE'
    )
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('white')

    ax.set_xlabel(r'$x/L$', fontsize=16, labelpad=-10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=16, width = 1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=200)
    plt.close()

def plot_cp_distribution_HT(x_exp, cp_exp, x_inter, cp_base, cp_moe, x_max, x_min, dCp, figsize, title, show_xlabel=True):
    # Define some color variables, you can adjust these values as needed
    streamcolor = '#48c0aa'
    markerfacecolor = 'none'
    markercolor = '#456990'
    linemoecolor = '#48c0aa'
    lineBasecolor = '#ef767a'
    lineSSTcolor = '#5b89d8'
    fig, ax = plt.subplots(figsize=figsize)
    # Experimental data points
    ax.plot((x_exp - x_min) / (x_max - x_min), -cp_exp,
        marker='o', markeredgewidth=3.0, linestyle='none',
        color=markercolor, markerfacecolor=markerfacecolor, markersize=8, markevery=0.03, linewidth=6,
        label=r'exp'
    )
    # Baseline model prediction results
    ax.plot(
        (x_inter - x_min) / (x_max - x_min), -cp_base,
         linewidth=4.0, linestyle='--',
        color=lineBasecolor, label=r'baseline'
    )
    # Post-training model prediction results
    ax.plot(
        (x_inter - x_min) / (x_max - x_min), -cp_moe,
         linewidth=4.0, dashes=(5, 2),
        color=linemoecolor, label=r'MoE'
    )

    fig.patch.set_alpha(0.0)
    ax.set_facecolor('white')

    # Set labels and ranges
    if show_xlabel:
        ax.set_xlabel(r'$x/C$', fontsize=16, labelpad = -8)
    else:
        # Hide labels but keep space: set empty label but maintain labelpad
        ax.set_xlabel('', fontsize=16, labelpad = -8)
    
    ax.set_xlim(-0.01, 1.01)
    ax.tick_params(labelsize=16, width = 1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Hide x-axis tick labels but don't change layout
    if not show_xlabel:
        ax.tick_params(labelbottom=False)  # Hide bottom tick labels, but keep space

    #plt.ylabel(r'$-C_p$', fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(dCp))
    #plt.ylim(ylim_min, ylim_max)
    # Use fixed subplot parameters instead of tight_layout()
    fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.17)
    plt.savefig(f'{title}.png', dpi=200)
    plt.close()

def plot_cp_distribution_VT(x_exp, cp_exp, x_inter, cp_base, cp_moe, x_max, x_min, figsize, title):
    # Define some color variables, you can adjust these values as needed
    streamcolor = '#48c0aa'
    markerfacecolor = 'none'
    markercolor = '#456990'
    linemoecolor = '#48c0aa'
    lineBasecolor = '#ef767a'
    lineSSTcolor = '#5b89d8'
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot((x_exp - x_min) / (x_max - x_min), -cp_exp,
            marker='o', markeredgewidth=3.0, linestyle='none',
            color=markercolor, markerfacecolor=markerfacecolor, markersize=8, markevery=0.03, linewidth=6,
            label=r'$-Cp$, exp')
    ax.plot((x_inter - x_min) / (x_max - x_min), -cp_base,
             linewidth=4.0, linestyle='--',
            color=lineBasecolor, label=r'$-Cp$, baseline')
    ax.plot((x_inter - x_min) / (x_max - x_min), -cp_moe,
             linewidth=4.0, dashes=(5, 2),
            color=linemoecolor, label=r'$-Cp$, MoE')

    fig.patch.set_alpha(0.0)
    ax.set_facecolor('white')

    ax.set_xlabel(r'$x/C$', fontsize=16, labelpad=-5)  # ← Adjustable distance
    ax.set_xlim(-0.01, 1.01)
    ax.tick_params(labelsize=16)
    ax.tick_params(labelsize=16, width = 1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.tight_layout()
    fig.savefig(f'{title}.png', dpi=200)
    plt.close(fig)

def plot_cp_distribution_2parts(x_exp1, cp_exp1, x_inter1, cp_base1, cp_moe1,
                                x_exp2, cp_exp2, x_inter2, cp_base2, cp_moe2,
                                x_max, x_min, figsize, title, ylim_min = -1.05, ylim_max = 5.05,
                                show_xlabel=True):
    # Color definitions (keep unchanged)
    streamcolor = '#48c0aa'
    markerfacecolor = 'none'
    markercolor = '#456990'
    linemoecolor = '#48c0aa'
    lineBasecolor = '#ef767a'
    lineSSTcolor = '#5b89d8'
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Experimental data points (exp2)
    ax.plot((x_exp2 - x_min) / (x_max - x_min), -cp_exp2,
            marker='o', markeredgewidth=3.0, linestyle='none',
            color=markercolor, markerfacecolor=markerfacecolor, markersize=8, markevery=0.03, linewidth=6,
            label=r'exp')
    
    # Baseline model prediction results (base2)
    ax.plot((x_inter2 - x_min) / (x_max - x_min), -cp_base2,
             linewidth=4.0, linestyle='--',
            color=lineBasecolor, label=r'baseline')
    
    # Post-training model prediction results (moe2)
    ax.plot((x_inter2 - x_min) / (x_max - x_min), -cp_moe2,
             linewidth=4.0, dashes=(5, 2),
            color=linemoecolor, label=r'MoE')
    
    # Experimental data points (exp1)
    ax.plot((x_exp1 - x_min) / (x_max - x_min), -cp_exp1,
            marker='o', markeredgewidth=3.0, linestyle='none',
            color=markercolor, markerfacecolor=markerfacecolor, markersize=8, markevery=0.03, linewidth=6,
            label=r'exp')
    
    # Baseline model prediction results (base1)
    ax.plot((x_inter1 - x_min) / (x_max - x_min), -cp_base1,
             linewidth=4.0, linestyle='--',
            color=lineBasecolor, label=r'baseline')
    
    # Post-training model prediction results (moe1)
    ax.plot((x_inter1 - x_min) / (x_max - x_min), -cp_moe1,
             linewidth=4.0, dashes=(5, 2),
            color=linemoecolor, label=r'MoE')
    
    # Set entire figure background transparency to 0, but drawing area (axes) to white background
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('white')
    
    # Set labels and ranges
    if show_xlabel:
        ax.set_xlabel(r'$x/C$', fontsize=16, labelpad = -8)
    else:
        # Hide labels but keep space: set empty label but maintain labelpad
        ax.set_xlabel('', fontsize=16, labelpad = -8)
    
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(ylim_min, ylim_max)
    ax.tick_params(labelsize=16)
    ax.tick_params(labelsize=16, width = 1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Hide x-axis tick labels but don't change layout
    if not show_xlabel:
        ax.tick_params(labelbottom=False)  # Hide bottom tick labels, but keep space
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.15)
    fig.savefig(f'{title}.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def plot_exp_cp_distribution_2parts(x_exp1, cp_exp1, x_exp2, cp_exp2, figsize, title, equal_aspect=False):
    # Define color variables
    streamcolor = '#48c0aa'
    markerfacecolor = 'none'
    markercolor = '#456990'
    linemoecolor = '#48c0aa'
    lineBasecolor = '#ef767a'
    lineSSTcolor = '#5b89d8'
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)  # Explicitly create axes object for setting aspect
    
    # Experimental data points
    ax.plot(x_exp1, -cp_exp1,
        marker='o', markeredgewidth=3.0, linestyle='-',
        color=markercolor, markerfacecolor=markerfacecolor, markersize=8, linewidth=6,
        label=r'exp'
    )
    ax.plot(x_exp2, -cp_exp2,
        marker='o', markeredgewidth=3.0, linestyle='-',
        color=markercolor, markerfacecolor=markerfacecolor, markersize=8, linewidth=6,
        label=r'exp'
    )
    
    ax.set_xlabel(r'$x/C$', fontsize=16)
    ax.tick_params(labelsize=16)
    ax.tick_params(labelsize=16, width = 1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')  # Or 'datalim', see below for explanation
    
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=100)
    plt.close()

def interpolate_on_surface_batch(points, values, query_points, k=4, p=2):
    """
    Perform inverse distance weighted (IDW) interpolation on multiple query points on discrete surfaces.
    
    Parameters:
        points       : (N, 3) array, known 3D points on the surface.
        values       : (N,)   array, scalar function values at those points.
        query_points : (M, 3) array, points to interpolate at.
        k            : int, number of nearest neighbors to use (default=5).
        p            : float, power for inverse distance weighting (default=2).
    
    Returns:
        interpolated_values : (M,) array of interpolated scalar values.
    """
    # Build KDTree once
    tree = cKDTree(points)
    
    # Query k nearest neighbors for all query points at once
    distances, indices = tree.query(query_points, k=k, workers=-1)  # workers=-1 uses all cores
    
    # Handle case where k == 1 (scipy returns 1D arrays instead of 2D)
    if k == 1:
        distances = distances[:, np.newaxis]  # shape (M, 1)
        indices = indices[:, np.newaxis]
    
    # Avoid division by zero: where distance is 0, use exact value
    # Create mask for zero distances
    zero_dist_mask = (distances == 0)
    
    # Initialize result array
    M = query_points.shape[0]
    interpolated_values = np.empty(M, dtype=values.dtype)
    
    # For points with at least one zero distance (exact match)
    has_exact = np.any(zero_dist_mask, axis=1)
    if np.any(has_exact):
        # Take the first exact match index
        first_zero_idx = np.argmax(zero_dist_mask[has_exact], axis=1)
        matched_indices = indices[has_exact, first_zero_idx]
        interpolated_values[has_exact] = values[matched_indices]
    
    # For the rest: perform IDW
    not_exact = ~has_exact
    if np.any(not_exact):
        d = distances[not_exact]          # (N_valid, k)
        idx = indices[not_exact]          # (N_valid, k)
        
        weights = 1.0 / (d ** p)          # Inverse distance weighting
        weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize per row
        
        interpolated_values[not_exact] = np.sum(weights * values[idx], axis=1)
    
    return interpolated_values



def interpolate_points_with_max_step(xyz_exp, dx):
    """
    Insert equidistant points between existing adjacent points so that the spacing between consecutive points does not exceed dx,
    while preserving all original points.

    Parameters:
        xyz_exp : (N, 3) array of original 3D points.
        dx      : float, maximum spacing between consecutive points.

    Returns:
        xyz_inter : (M, 3) array, M >= N, with original points included,
                    and new points inserted so that segment lengths <= dx.
    """
    if xyz_exp.shape[0] < 2:
        return xyz_exp.copy()

    segments = []
    
    for i in range(len(xyz_exp) - 1):
        p0 = xyz_exp[i]
        p1 = xyz_exp[i + 1]
        vec = p1 - p0
        dist = np.linalg.norm(vec)
        
        if dist == 0:
            # Skip zero-length segment (duplicate points)
            segments.append(p0[np.newaxis, :])
            continue
        
        # Number of intervals needed: ceil(dist / dx)
        n_intervals = int(np.ceil(dist / dx))
        # Avoid division by zero; n_intervals >= 1
        if n_intervals == 0:
            n_intervals = 1
        
        # Generate n_intervals + 1 points from p0 to p1 (inclusive)
        t = np.linspace(0, 1, n_intervals + 1)  # shape (n_intervals+1,)
        interpolated = p0 + np.outer(t, vec)    # shape (n_intervals+1, 3)
        
        # Append all but the last point to avoid duplication (last = next p0)
        segments.append(interpolated[:-1])
    
    # Append the very last point
    segments.append(xyz_exp[-1][np.newaxis, :])
    
    # Concatenate all segments
    xyz_inter = np.concatenate(segments, axis=0)
    return xyz_inter




dynP = 2312


expDict = '../refData/truthData/CRMHL/'
baseDict = '../refData/baselineData_EARSM05/8_CRMHL/'

with open('caseDir.txt', 'r') as file:
    caseDir = file.read()


expData, var_names = parse_tecplot_file(expDict+'TC2p2_pressure_model_v3.dat')
baseData = np.loadtxt(baseDict + 'postProcessing/sample_wall/12000/p_airfoil_wall.raw')
moeData = np.loadtxt(caseDir + '/8_CRMHL/postProcessing/sample_wall/8000/p_airfoil_wall.raw')


# ------------- fuselage A --------------------- #
i = 0
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp = CP_exp0[mask]
# mask the CFD data
y_min = 0; y_max = 120
z_min = 100; z_max = 350
mask = ((moeData[:, 2] > z_min) &
        (moeData[:, 2] < z_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) )
baseData_mask = baseData[mask]
moeData_mask = moeData[mask]
x_min = min(moeData_mask[:, 0]); x_max = max(moeData_mask[:, 0])
y_inter = 61.0
z_inter = (-110.602-0.865967*y_inter)/(-0.500102)
xyz_inter = np.zeros([200, 3])
xyz_inter[:,0] = np.linspace(x_min, x_max, 200)
xyz_inter[:,1] = y_inter
xyz_inter[:,2] = z_inter
Cp_base = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter, k=4, p=2)/dynP
Cp_moe = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter, k=4, p=2)/dynP
plot_cp_distribution_fuselage(CP_exp[:,0], CP_exp[:,3], xyz_inter[:,0], Cp_base, Cp_moe, x_max, x_min, (7,1.5), 'Cp_fuselage_A')
# ------------- fuselage B --------------------- #
i = 1
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp = CP_exp0[mask]
# mask the CFD data
y_min = 0; y_max = 170
z_min = 186; z_max = 232
xz_inter = np.zeros([200, 2])
xz_inter[:,0] = np.linspace(x_min, x_max, 200)
xz_inter[:,1] = 211.141
mask = ((moeData[:, 2] > z_min) &
        (moeData[:, 2] < z_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) )
baseData_mask = baseData[mask]
moeData_mask = moeData[mask]
x_min = min(moeData_mask[:, 0]); x_max = max(moeData_mask[:, 0])
Cp_base = interpolate_cp(baseData_mask[:, [0, 2,3]], xz_inter)/dynP
Cp_moe = interpolate_cp(moeData_mask[:, [0, 2,3]], xz_inter)/dynP
plot_cp_distribution_fuselage(CP_exp[:,0], CP_exp[:,3], xz_inter[:,0], Cp_base, Cp_moe, x_max, x_min, (8,1.7), 'Cp_fuselage_B')
# ------------- vertical tail z = 560.58661 --------------------- #
i = 2
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp = CP_exp0[mask]
x_min = 2310; x_max = 2520
z_min = 500; z_max = 600
xz_inter = np.zeros([1000, 2])
xz_inter[:,0] = np.linspace(x_min, x_max, 1000)
xz_inter[:,1] = 560.58661
mask = ((moeData[:, 2] > z_min) &
        (moeData[:, 2] < z_max))
baseData_mask = baseData[mask]
moeData_mask = moeData[mask]
Cp_base = interpolate_cp(baseData_mask[:, [0,2,3]], xz_inter)/dynP
Cp_moe = interpolate_cp(moeData_mask[:, [0,2,3]], xz_inter)/dynP
x_max = 2516; x_min = 2310.5
plot_cp_distribution_VT(CP_exp[:,0], CP_exp[:,3], xz_inter[:,0], Cp_base, Cp_moe, x_max, x_min, (4.5,2.5), 'Cp_VT')



# ------------- horizontal tail y = 84.426 --------------------- #
i = 3
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp = CP_exp0[mask]
#print(CP_exp[:,[0,2]])
x_mid1 = 2251.168; x_mid2 = 2471.949
z_mid1 = 257.461; z_mid2 = 264.478
# mask the CFD data
x_min = 2242; x_max = 2482
y_min = 70; y_max = 100
z_min = 280; z_max = 300
xy_inter = np.zeros([1000, 2])
xy_inter[:,0] = np.linspace(x_min, x_max, 1000)
xy_inter[:,1] = 84.426
mask = ((moeData[:, 0] > x_min) &
        (moeData[:, 0] < x_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) &
        ((x_mid2 - x_mid1) * (moeData[:, 2] - z_mid1) - (z_mid2 - z_mid1) * (moeData[:, 0] - x_mid1) > 0) )
baseData_mask_upper = baseData[mask]
moeData_mask_upper = moeData[mask]
mask = ((moeData[:, 0] > x_min) &
        (moeData[:, 0] < x_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) &
        ((x_mid2 - x_mid1) * (moeData[:, 2] - z_mid1) - (z_mid2 - z_mid1) * (moeData[:, 0] - x_mid1) < 0) )
baseData_mask_lower = baseData[mask]
moeData_mask_lower = moeData[mask]
Cp_base_upper = interpolate_cp(baseData_mask_upper[:, [0,1,3]], xy_inter)/dynP
Cp_moe_upper = interpolate_cp(moeData_mask_upper[:, [0,1,3]], xy_inter)/dynP
Cp_base_lower = interpolate_cp(baseData_mask_lower[:, [0,1,3]], xy_inter)/dynP
Cp_moe_lower = interpolate_cp(moeData_mask_lower[:, [0,1,3]], xy_inter)/dynP
Cp_base = np.concatenate((np.flip(Cp_base_upper), Cp_base_lower))
Cp_moe = np.concatenate((np.flip(Cp_moe_upper), Cp_moe_lower))
x_con = np.concatenate((np.flip(xy_inter[:,0]), xy_inter[:,0]))
mask = ~np.isnan(Cp_base)
Cp_base_clean = Cp_base[mask]
Cp_moe_clean = Cp_moe[mask]
x_con_clean = x_con[mask]
x_min = 2251.168; x_max = 2471.949
plot_cp_distribution_HT(CP_exp[:,0], CP_exp[:,3], x_con_clean, Cp_base_clean, Cp_moe_clean, x_max, x_min, 0.3, (4,2.2), 'Cp_HT_A', show_xlabel=True)
# ------------- horizontal tail y = 337.703 --------------------- #
i = 4
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp = CP_exp0[mask]
#print(CP_exp[:,[0,2]])
z_mid = 292.42
# mask the CFD data
x_min = 2456.93; x_max = 2598.15
y_min = 320; y_max = 350
z_min = 280; z_max = 300
xy_inter = np.zeros([1000, 2])
xy_inter[:,0] = np.linspace(x_min, x_max, 1000)
xy_inter[:,1] = 337.703
mask = ((moeData[:, 0] > x_min) &
        (moeData[:, 0] < x_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) &
        (moeData[:, 2] > z_mid) )
baseData_mask_upper = baseData[mask]
moeData_mask_upper = moeData[mask]
mask = ((moeData[:, 0] > x_min) &
        (moeData[:, 0] < x_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) &
        (moeData[:, 2] < z_mid) )
baseData_mask_lower = baseData[mask]
moeData_mask_lower = moeData[mask]
Cp_base_upper = interpolate_cp(baseData_mask_upper[:, [0,1,3]], xy_inter)/dynP
Cp_moe_upper = interpolate_cp(moeData_mask_upper[:, [0,1,3]], xy_inter)/dynP
Cp_base_lower = interpolate_cp(baseData_mask_lower[:, [0,1,3]], xy_inter)/dynP
Cp_moe_lower = interpolate_cp(moeData_mask_lower[:, [0,1,3]], xy_inter)/dynP
Cp_base = np.concatenate((np.flip(Cp_base_upper), Cp_base_lower))
Cp_moe = np.concatenate((np.flip(Cp_moe_upper), Cp_moe_lower))
x_con = np.concatenate((np.flip(xy_inter[:,0]), xy_inter[:,0]))
mask = ~np.isnan(Cp_base)
Cp_base_clean = Cp_base[mask]
Cp_moe_clean = Cp_moe[mask]
x_con_clean = x_con[mask]
x_min = 2466.93; x_max = 2588.15
plot_cp_distribution_HT(CP_exp[:,0], CP_exp[:,3], x_con_clean, Cp_base_clean, Cp_moe_clean, x_max, x_min, 1, (4,2.2), 'Cp_HT_C',show_xlabel=False)




# ------------- wing and slat A --------------------- #
#---wing
i = 7
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp2 = CP_exp0[mask]
xexp = CP_exp2[:, 0]
top2_indices = np.argsort(xexp)[-2:][::-1]
z_mid1 = np.mean(CP_exp2[top2_indices, 2]); x_mid1 = 1450
z_mid2 = 180; x_mid2 = 1075
mask = ((x_mid2 - x_mid1) * (CP_exp2[:,2] - z_mid1) - (z_mid2 - z_mid1) * (CP_exp2[:,0] - x_mid1) > 0) 
CP_exp2_upper = CP_exp2[mask]
mask = ((x_mid2 - x_mid1) * (CP_exp2[:,2] - z_mid1) - (z_mid2 - z_mid1) * (CP_exp2[:,0] - x_mid1) < 0) 
CP_exp2_lower = CP_exp2[mask]
sorted_indices = np.argsort(CP_exp2_upper[:, 0])[::-1]
CP_exp2_upper_sorted = CP_exp2_upper[sorted_indices]
sorted_indices = np.argsort(CP_exp2_lower[:, 0])
CP_exp2_lower_sorted = CP_exp2_lower[sorted_indices]
CP_exp2_sorted = np.vstack((CP_exp2_upper_sorted, CP_exp2_lower_sorted))
xyz_inter = interpolate_points_with_max_step(CP_exp2_sorted[:,:3], 2.0)
# mask the CFD data
x_min = 1000; x_max = 1890
y_min = 166; y_max = 1156
mask = ((moeData[:, 0] > x_min) &
        (moeData[:, 0] < x_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) )
baseData_mask= baseData[mask]
moeData_mask = moeData[mask]
baseCp_inter = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter, k=4, p=2)/dynP
moeCp_inter = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter, k=4, p=2)/dynP
#---slat
i = 6
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp1 = CP_exp0[mask]
CP_exp1 = np.vstack((CP_exp1, CP_exp1[0,:]))
xyz_inter1 = interpolate_points_with_max_step(CP_exp1[:,:3], 0.5)
baseCp_inter_slat = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter1, k=4, p=2)/dynP
moeCp_inter_slat = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter1, k=4, p=2)/dynP
#---plot
x_min = np.min(xyz_inter1[:,0])
x_max = np.max(xyz_inter[:,0])
plot_cp_distribution_2parts(CP_exp1[:,0], CP_exp1[:,3], xyz_inter1[:,0], baseCp_inter_slat, moeCp_inter_slat,
                            CP_exp2_sorted[:,0], CP_exp2_sorted[:,3], xyz_inter[:,0], baseCp_inter, moeCp_inter, 
                            x_max, x_min, (8,3), 'Cp_wing_A')

# ------------- wing and slat E --------------------- #
#---wing
i = 15
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp2 = CP_exp0[mask]
xexp = CP_exp2[:, 0]
top2_indices = np.argsort(xexp)[-2:][::-1]
z_mid1 = np.mean(CP_exp2[top2_indices, 2]); x_mid1 = 1580
z_mid2 = 215; x_mid2 = 1365
mask = ((x_mid2 - x_mid1) * (CP_exp2[:,2] - z_mid1) - (z_mid2 - z_mid1) * (CP_exp2[:,0] - x_mid1) > 0) 
CP_exp2_upper = CP_exp2[mask]
mask = ((x_mid2 - x_mid1) * (CP_exp2[:,2] - z_mid1) - (z_mid2 - z_mid1) * (CP_exp2[:,0] - x_mid1) < 0) 
CP_exp2_lower = CP_exp2[mask]
sorted_indices = np.argsort(CP_exp2_upper[:, 0])[::-1]
CP_exp2_upper_sorted = CP_exp2_upper[sorted_indices]
sorted_indices = np.argsort(CP_exp2_lower[:, 0])
CP_exp2_lower_sorted = CP_exp2_lower[sorted_indices]
CP_exp2_sorted = np.vstack((CP_exp2_upper_sorted, CP_exp2_lower_sorted))
xyz_inter = interpolate_points_with_max_step(CP_exp2_sorted[:,:3], 2.0)
# mask the CFD data
x_min = 1000; x_max = 1890
y_min = 166; y_max = 1156
mask = ((moeData[:, 0] > x_min) &
        (moeData[:, 0] < x_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) )
baseData_mask= baseData[mask]
moeData_mask = moeData[mask]
baseCp_inter = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter, k=4, p=2)/dynP
moeCp_inter = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter, k=4, p=2)/dynP
#---slat
i = 14
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp1 = CP_exp0[mask]
CP_exp1 = np.vstack((CP_exp1, CP_exp1[0,:]))
xyz_inter1 = interpolate_points_with_max_step(CP_exp1[:,:3], 0.5)
baseCp_inter_slat = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter1, k=4, p=2)/dynP
moeCp_inter_slat = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter1, k=4, p=2)/dynP
#---plot
x_min = np.min(xyz_inter1[:,0])
x_max = np.max(xyz_inter[:,0])
plot_cp_distribution_2parts(CP_exp1[:,0], CP_exp1[:,3], xyz_inter1[:,0], baseCp_inter_slat, moeCp_inter_slat,
                            CP_exp2_sorted[:,0], CP_exp2_sorted[:,3], xyz_inter[:,0], baseCp_inter, moeCp_inter, 
                            x_max, x_min, (8,3), 'Cp_wing_E', show_xlabel=False)


# ------------- wing and slat G --------------------- #
#---wing
i = 17
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp2 = CP_exp0[mask]
xexp = CP_exp2[:, 0]
top2_indices = np.argsort(xexp)[-2:][::-1]
z_mid1 = np.mean(CP_exp2[top2_indices, 2]); x_mid1 = 1695
z_mid2 = 233; x_mid2 = 1528
mask = ((x_mid2 - x_mid1) * (CP_exp2[:,2] - z_mid1) - (z_mid2 - z_mid1) * (CP_exp2[:,0] - x_mid1) > 0) 
CP_exp2_upper = CP_exp2[mask]
mask = ((x_mid2 - x_mid1) * (CP_exp2[:,2] - z_mid1) - (z_mid2 - z_mid1) * (CP_exp2[:,0] - x_mid1) < 0) 
CP_exp2_lower = CP_exp2[mask]
sorted_indices = np.argsort(CP_exp2_upper[:, 0])[::-1]
CP_exp2_upper_sorted = CP_exp2_upper[sorted_indices]
sorted_indices = np.argsort(CP_exp2_lower[:, 0])
CP_exp2_lower_sorted = CP_exp2_lower[sorted_indices]
CP_exp2_sorted = np.vstack((CP_exp2_upper_sorted, CP_exp2_lower_sorted))
xyz_inter =  np.abs(interpolate_points_with_max_step(CP_exp2_sorted[:,:3], 2.0))
# mask the CFD data
x_min = 1000; x_max = 1890
y_min = 166; y_max = 1156
mask = ((moeData[:, 0] > x_min) &
        (moeData[:, 0] < x_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) )
baseData_mask= baseData[mask]
moeData_mask = moeData[mask]
baseCp_inter = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter, k=4, p=2)/dynP
moeCp_inter = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter, k=4, p=2)/dynP
#---slat
i = 16
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp1 = CP_exp0[mask]
CP_exp1 = np.vstack((CP_exp1, CP_exp1[0,:]))
xyz_inter1 = np.abs(interpolate_points_with_max_step(CP_exp1[:,:3], 0.5))
baseCp_inter_slat = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter1, k=4, p=2)/dynP
moeCp_inter_slat = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter1, k=4, p=2)/dynP
#---plot
x_min = np.min(xyz_inter1[:,0])
x_max = np.max(xyz_inter[:,0])
plot_cp_distribution_2parts(CP_exp1[:,0], CP_exp1[:,3], xyz_inter1[:,0], baseCp_inter_slat, moeCp_inter_slat,
                            CP_exp2_sorted[:,0], CP_exp2_sorted[:,3], xyz_inter[:,0], baseCp_inter, moeCp_inter, 
                            x_max, x_min, (8,3), 'Cp_wing_G', show_xlabel=False)




# ------------- wing and slat I --------------------- #
#---wing
i = 19
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp2 = CP_exp0[mask]
xexp = CP_exp2[:, 0]
top2_indices = np.argsort(xexp)[-2:][::-1]
z_mid1 = np.mean(CP_exp2[top2_indices, 2]); x_mid1 = 1810
z_mid2 = 251; x_mid2 = 1690
mask = ((x_mid2 - x_mid1) * (CP_exp2[:,2] - z_mid1) - (z_mid2 - z_mid1) * (CP_exp2[:,0] - x_mid1) > 0) 
CP_exp2_upper = CP_exp2[mask]
mask = ((x_mid2 - x_mid1) * (CP_exp2[:,2] - z_mid1) - (z_mid2 - z_mid1) * (CP_exp2[:,0] - x_mid1) < 0) 
CP_exp2_lower = CP_exp2[mask]
sorted_indices = np.argsort(CP_exp2_upper[:, 0])[::-1]
CP_exp2_upper_sorted = CP_exp2_upper[sorted_indices]
sorted_indices = np.argsort(CP_exp2_lower[:, 0])
CP_exp2_lower_sorted = CP_exp2_lower[sorted_indices]
CP_exp2_sorted = np.vstack((CP_exp2_upper_sorted, CP_exp2_lower_sorted))
xyz_inter = interpolate_points_with_max_step(CP_exp2_sorted[:,:3], 2.0)
# mask the CFD data
x_min = 1000; x_max = 1890
y_min = 166; y_max = 1156
mask = ((moeData[:, 0] > x_min) &
        (moeData[:, 0] < x_max) &
        (moeData[:, 1] > y_min) &
        (moeData[:, 1] < y_max) )
baseData_mask= baseData[mask]
moeData_mask = moeData[mask]
baseCp_inter = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter, k=4, p=2)/dynP
moeCp_inter = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter, k=4, p=2)/dynP
#---slat
i = 18
CP_exp0 = expData[i][:, [8, 9, 10, 3]]
mask = (CP_exp0[:,3] < 900)
CP_exp1 = CP_exp0[mask]
CP_exp1 = np.vstack((CP_exp1, CP_exp1[0,:]))
xyz_inter1 = interpolate_points_with_max_step(CP_exp1[:,:3], 0.5)
baseCp_inter_slat = interpolate_on_surface_batch(baseData_mask[:,:3], baseData_mask[:,3], xyz_inter1, k=4, p=2)/dynP
moeCp_inter_slat = interpolate_on_surface_batch(moeData_mask[:,:3], moeData_mask[:,3], xyz_inter1, k=4, p=2)/dynP
#---plot
x_min = np.min(xyz_inter1[:,0])
x_max = np.max(xyz_inter[:,0])
plot_cp_distribution_2parts(CP_exp1[:,0], CP_exp1[:,3], xyz_inter1[:,0], baseCp_inter_slat, moeCp_inter_slat,
                            CP_exp2_sorted[:,0], CP_exp2_sorted[:,3], xyz_inter[:,0], baseCp_inter, moeCp_inter, 
                            x_max, x_min, (8,3), 'Cp_wing_I', show_xlabel=False)
