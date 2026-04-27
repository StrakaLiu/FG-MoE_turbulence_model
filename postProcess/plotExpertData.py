import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator
import matplotlib
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.interpolate import griddata


matplotlib.use('Agg')  




def calTheta(theta_scaled):
    theta = np.sign(theta_scaled)*np.abs(theta_scaled)/(1.0 - np.abs(theta_scaled))
    return theta

def interpolate_scattered_data(X, Y, U, x_min, x_max, y_min, y_max, n):
    points = np.column_stack((X, Y))  # shape (m, 2)
    interpolator = LinearNDInterpolator(points, U, fill_value=np.nan)

    np.random.seed(42)
    rng = np.random.default_rng()  
    x_rand = np.random.uniform(x_min, x_max, size=n)
    y_rand = np.random.uniform(y_min, y_max, size=n)
    query_points = np.column_stack((x_rand, y_rand))

    U_interp = interpolator(query_points)  

    valid_mask = ~np.isnan(U_interp)
    
    x_new = x_rand[valid_mask]
    y_new = y_rand[valid_mask]
    U_new = U_interp[valid_mask]

    return x_new, y_new, U_new

def deleteOutPoint(x, y, U, x_lower_curve, y_lower_curve, x_upper_curve, y_upper_curve):
    f_curve_lower = interp1d(x_lower_curve, y_lower_curve, kind='linear', bounds_error=False, fill_value=np.nan)
    f_curve_upper = interp1d(x_upper_curve, y_upper_curve, kind='linear', bounds_error=False, fill_value=np.nan)
    y_lower = f_curve_lower(x)  
    y_upper = f_curve_upper(x)
    valid_mask = np.logical_and(y > y_lower, y < y_upper)
    return x[valid_mask], y[valid_mask], U[valid_mask]


def F_q(A, B, q_sep):
    Fsep = 1 / (1 + np.exp(-A*(q_sep - B)))
    return Fsep


if __name__ == "__main__":

    linecolor = '#2f9411'
    color0 = '#5b89d8'
    edgecolor0 = '#1e4c9d'
    color1 = '#dc8192'
    edgecolor1 = '#b40426'

    caseDir = '../refData/trainedExpertModelData/'

    #-------------------------plot Q_NPS, F_NPS------------------------------#
    #--------hump
    nu_hump = 1.5526e-05
    file1 = caseDir + '1_NASAhump/postProcessing/sample_down/101/wallShearStress_down.raw'
    xHump = np.loadtxt(file1, skiprows = 2)[:,0]
    yHump = np.loadtxt(file1, skiprows = 2)[:,1]
    tauxHump = np.loadtxt(file1, skiprows = 2)[:,3]
    tauyHump = np.loadtxt(file1, skiprows = 2)[:,4]
    tau_wHump = np.sqrt(tauxHump**2 + tauyHump**2)
    u_tauHump = np.sqrt(tau_wHump)
    dict1 = caseDir + '1_NASAhump/postProcessing/sample_left/101/'
    X = np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,0]
    Y = np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,1]
    U = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,3]
    vorticity = np.loadtxt(dict1+'vorticity_left.raw', skiprows = 2)[:,5]
    magVor = np.abs(vorticity)
    Gamma = np.loadtxt(dict1+'Gamma_w_left.raw', skiprows = 2)[:,3]
    theta1_scaled = np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,3]
    theta2_scaled = np.loadtxt(dict1+'theta2_Scaled__left.raw', skiprows = 2)[:,3]
    theta3_scaled = np.loadtxt(dict1+'theta3_Scaled__left.raw', skiprows = 2)[:,3]
    theta4_scaled = np.loadtxt(dict1+'theta4_Scaled__left.raw', skiprows = 2)[:,3]
    dwall = np.loadtxt(dict1+'dwall_left.raw', skiprows = 2)[:,3]
    theta1 = calTheta(theta1_scaled)
    theta2 = calTheta(theta2_scaled)
    theta3 = calTheta(theta3_scaled)
    theta4 = calTheta(theta4_scaled)
    q_sep = np.abs(theta1 + theta2)
    q_3D = np.abs(theta3) + np.abs(theta4)
    xmin, xmax = 0.65*0.42, 1.1*0.42  
    points = np.column_stack((X, Y))  
    x_lines = np.linspace(xmin, xmax, 50)
    X_edge = []
    Y_edge = []
    X_shear = []
    Y_shear = []
    for x in x_lines:
        f_hump = interp1d(xHump, yHump, kind='linear', bounds_error=False, fill_value=np.nan)
        ymin = f_hump(x)+0.005
        ymax = 0.3*0.42
        y_vals = np.linspace(ymin, ymax, 100)
        query_points = np.column_stack((np.full_like(y_vals, x), y_vals))
        U_interp = griddata(points, U, query_points, method='linear', fill_value=np.nan)
        vor_interp = griddata(points, magVor, query_points, method='linear', fill_value=np.nan)
        valid = ~np.isnan(U_interp)
        y_valid = y_vals[valid]
        U_valid = U_interp[valid]
        vor_valid = vor_interp[valid]
        idx_max = np.argmax(U_valid)
        Umax = U_valid[idx_max]
        idx_min = np.argmin(U_valid)
        Umin = U_valid[idx_min]
        y_at_Umax = y_valid[idx_max]
        target_U = Umax - 0.1*(Umax-Umin)
        mask_lower = y_valid < y_at_Umax        
        y_lower = y_valid[mask_lower]
        U_lower = U_valid[mask_lower]
        U_curve = interp1d(U_lower, y_lower, kind='linear', bounds_error=False, fill_value=np.nan)
        y_edge = U_curve(target_U)
        X_edge.append(x)
        Y_edge.append(y_edge)
        idx_max = np.argmax(vor_valid)
        X_shear.append(x)
        Y_shear.append(y_valid[idx_max])
    X_edge = np.array(X_edge)
    Y_edge = np.array(Y_edge)
    X_shear = np.array(X_shear)
    Y_shear = np.array(Y_shear)
    
    x_min, x_max = 0.21, 0.63
    y_min, y_max = 0, 0.105
    n_samples = 10000
    x_new, y_new, q_sep_new = interpolate_scattered_data(X, Y, q_sep, x_min, x_max, y_min, y_max, n_samples)
    x_new, y_new, q_3D_new = interpolate_scattered_data(X, Y, q_3D, x_min, x_max, y_min, y_max, n_samples)
    x_new, y_new, dwall_new = interpolate_scattered_data(X, Y, dwall, x_min, x_max, y_min, y_max, n_samples)
    x_new, y_new, gamma_new = interpolate_scattered_data(X, Y, Gamma, x_min, x_max, y_min, y_max, n_samples)
    x_hump, y_hump, q_sep_hump = deleteOutPoint(x_new, y_new, q_sep_new, xHump, yHump, X_edge, Y_edge)
    x_hump, y_hump, q_3D_hump = deleteOutPoint(x_new, y_new, q_3D_new, xHump, yHump, X_edge, Y_edge)
    x_hump, y_hump, dwall_hump = deleteOutPoint(x_new, y_new, dwall_new, xHump, yHump, X_edge, Y_edge)
    x_hump, y_hump, Gamma_hump = deleteOutPoint(x_new, y_new, gamma_new, xHump, yHump, X_edge, Y_edge)

    #--------duct
    dict1 = caseDir + '2_sqrDuct/postProcessing/sample_left_theta/23/'
    X = np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,1]
    Y = np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,2]
    theta1_scaled = np.loadtxt(dict1+'theta1_Scaled__left.raw', skiprows = 2)[:,3]
    theta2_scaled = np.loadtxt(dict1+'theta2_Scaled__left.raw', skiprows = 2)[:,3]
    theta3_scaled = np.loadtxt(dict1+'theta3_Scaled__left.raw', skiprows = 2)[:,3]
    theta4_scaled = np.loadtxt(dict1+'theta4_Scaled__left.raw', skiprows = 2)[:,3]
    theta1 = calTheta(theta1_scaled)
    theta2 = calTheta(theta2_scaled)
    theta3 = calTheta(theta3_scaled)
    theta4 = calTheta(theta4_scaled)
    q_sep = np.abs(theta1 + theta2)
    q_3D = np.abs(theta3) + np.abs(theta4)
    x_min, x_max = -1.0, 0.0
    y_min, y_max = -1.0, 0.0
    n_samples = 10000
    x_duct, y_duct, q_sep_duct = interpolate_scattered_data(X, Y, q_sep, x_min, x_max, y_min, y_max, n_samples)
    x_duct, y_duct, q_3D_duct = interpolate_scattered_data(X, Y, q_3D, x_min, x_max, y_min, y_max, n_samples)
    dwall_duct = np.minimum(x_duct, y_duct)+1
    dict2 = caseDir + '2_sqrDuct/postProcessing/sample_down/24/'
    z_w = np.loadtxt(dict2+'wallShearStress_down.raw', skiprows = 2)[:,2]
    tau_w = np.loadtxt(dict2+'wallShearStress_down.raw', skiprows = 2)[:,3]
    f = interp1d(z_w, tau_w, kind='linear', bounds_error=False, fill_value=np.nan)
    tau_w_duct = np.abs(f(np.maximum(x_duct, y_duct)))
    utau_duct = np.sqrt(tau_w_duct)
    yplus_duct = utau_duct*dwall_duct/0.00005
    mask_vis = (yplus_duct < 5.0) & (~((x_duct < -0.97) & (y_duct < -0.97)))
    mask_out = (yplus_duct > 5.0) & (~((x_duct < -0.97) & (y_duct < -0.97)))
    q_3D_duct_vis = q_3D_duct[mask_vis]
    q_3D_duct_out = q_3D_duct[mask_out]
    dwall_duct_vis = dwall_duct[mask_vis]
    x_duct_vis = x_duct[mask_vis]+1
    y_duct_vis = y_duct[mask_vis]+1


    #------------ASJ
    dict1 = caseDir + '3_ASJ/postProcessing/sample_left/101/'
    X = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,0]
    Y = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,2]
    U = np.loadtxt(dict1+'U_left.raw', skiprows = 2)[:,3]
    Gamma = np.loadtxt(dict1+'Gamma_w_left.raw', skiprows = 2)[:,3]
    xmin, xmax = -0.19, 1  
    points = np.column_stack((X, Y))  
    x_lines = np.linspace(xmin, xmax, 200)
    X_edge = []
    Y_edge = []
    X_shear = []
    Y_shear = []
    for x in x_lines:
        X_edge.append(x)
        if x > 0:
            ymin = 0.0
            ymax = 0.15
            y_vals = np.linspace(ymin, ymax, 100)
            query_points = np.column_stack((np.full_like(y_vals, x), y_vals))
            U_interp = griddata(points, U, query_points, method='linear', fill_value=np.nan)
            valid = ~np.isnan(U_interp)
            y_valid = y_vals[valid]
            U_valid = U_interp[valid]
            Umax = U_valid[0]
            Umin = 3.4396
            y_at_Umax = 0.0
            target_U = Umax - 0.1*(Umax-Umin)
            U_curve = interp1d(U_valid, y_valid, kind='linear', bounds_error=False, fill_value=np.nan)
            y_edge = U_curve(target_U)
        else:
            y_edge = 0.0254
        Y_edge.append(y_edge)
    X_edge = np.array(X_edge)
    Y_edge = np.array(Y_edge)
    
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0, 0.2
    n_samples = 10000
    x_new, y_new, gamma_new = interpolate_scattered_data(X, Y, Gamma, x_min, x_max, y_min, y_max, n_samples)
    xSymm = np.linspace(x_min, x_max, 501)
    ySymm = 0.0*xSymm
    x_asj, y_asj, Gamma_asj = deleteOutPoint(x_new, y_new, gamma_new, xSymm, ySymm, X_edge, Y_edge)


    fig, (ax01, ax02, ax03) = plt.subplots(ncols = 1, nrows=3, figsize=(5, 10))
    ax012 = ax01.twinx()
    q_sample = np.linspace(0, 2, 501)
    ax012.plot(q_sample, F_q(40, 0.25, q_sample),  linestyle = '-', color = linecolor,label=r'$F_1$, $A=10$, $B=0.25$', linewidth = 2.0)
    bin_edges1 = np.arange(0, 1.01, 0.02)
    ax01.hist(q_sep_hump, bins=bin_edges1, density=True, alpha=1.0, color=color1, edgecolor=edgecolor1, label=r'2D-hump')
    counts, bin_edges = np.histogram(q_sep_duct, bins=bin_edges1, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax01.bar(bin_centers, -counts, width=np.diff(bin_edges), alpha=1.0, color=color0, edgecolor=edgecolor0, label=r'SqrDuct', align='center')
    ax01.set_xlabel(r'$\phi_1$', fontsize = 14)
    ax01.set_ylabel(r'PDF of $\phi_1$', fontsize = 14)
    ax01.tick_params(labelsize=12)
    ax012.set_ylabel(r'$F_1$', fontsize = 14, color = linecolor)
    ax012.tick_params(labelsize=12, color = linecolor, labelcolor=linecolor)
    ax012.spines['right'].set_color(linecolor) 
    ax012.yaxis.label.set_color(linecolor)
    ax01.set_xlim(0,1.0)
    ax01.set_ylim(-4,4)
    ax01.yaxis.set_major_locator(MultipleLocator(1.0))  
    def abs_formatter(x, pos):
        return f"{abs(x):g}"
    ax01.yaxis.set_major_formatter(FuncFormatter(abs_formatter))
    ax01.xaxis.set_major_locator(MultipleLocator(0.25))
    ax012.set_ylim(0,1.005)
    ax012.yaxis.set_major_locator(MultipleLocator(0.25))  
    handles1, labels1 = ax01.get_legend_handles_labels()
    handles2, labels2 = ax012.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax01.legend(handles, labels, loc='lower right', ncol=1, fontsize=10, frameon=False)    
    ax01.axhline(y=0, color='black', zorder=10, linewidth = 0.8)



    ax022 = ax02.twinx()
    q_sample = np.linspace(0, 0.1, 501)
    bin_edges2 = np.arange(0, 0.04001, 0.0008)
    ax02.hist(q_3D_duct_out, bins=bin_edges2, density=True, alpha=1.0, color=color1, edgecolor=edgecolor1, label=r'SqrDuct, vis. layer')
    counts, bin_edges = np.histogram(q_3D_duct_vis, bins=bin_edges2, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax02.bar(bin_centers, -counts, width=np.diff(bin_edges), alpha=1.0, color=color0, edgecolor=edgecolor0, label=r'SqrDuct, out of vis. layer', align='center')
    ax022.plot(q_sample, F_q(1000, 0.01, q_sample),  linestyle = '-', color = linecolor,label=r'$F_2$, $A=10$, $B=0.01$', linewidth = 2.0)
    ax02.set_xlabel(r'$\phi_2$', fontsize = 14)
    ax02.set_ylabel(r'PDF of $\phi_2$', fontsize = 14)
    ax02.tick_params(labelsize=12)
    ax022.set_ylabel(r'$F_2$', fontsize = 14, color = linecolor)
    ax022.tick_params(labelsize=12, color = linecolor, labelcolor=linecolor)
    ax022.spines['right'].set_color(linecolor) 
    ax022.yaxis.label.set_color(linecolor)
    ax02.set_xlim(0,0.04)
    ax02.set_ylim(-60,60)
    ax022.set_ylim(0,1.005)
    ax02.yaxis.set_major_locator(MultipleLocator(20.0))  
    ax02.yaxis.set_major_formatter(FuncFormatter(abs_formatter))
    ax022.yaxis.set_major_locator(MultipleLocator(0.25))  

    handles1, labels1 = ax02.get_legend_handles_labels()
    handles2, labels2 = ax022.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax02.legend(handles, labels, loc='lower right', ncol=1, fontsize=10, frameon=False)    
    ax02.axhline(y=0, color='black', zorder=10, linewidth = 0.8)



    ax032 = ax03.twinx()
    q_sample = np.linspace(0, 2, 501)
    ax032.plot(q_sample, F_q(50, 0.2, q_sample),  linestyle = '-', color = linecolor,label=r'$F_3$, $A=10$, $B=0.2$', linewidth = 2.0)
    bin_edges3 = np.arange(0, 0.801, 0.016)
    ax03.hist(Gamma_hump, bins=bin_edges3, density=True, alpha=1.0, color=color1, edgecolor=edgecolor1, label=r'2D-hump')
    counts, bin_edges = np.histogram(Gamma_asj, bins=bin_edges3, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax03.bar(bin_centers, -counts, width=np.diff(bin_edges), alpha=1.0, color=color0, edgecolor=edgecolor0, label=r'ASJ', align='center')
    ax03.set_xlabel(r'$\phi_3$', fontsize = 14)
    ax03.set_ylabel(r'PDF of $\phi_3$', fontsize = 14)
    ax03.tick_params(labelsize=12)
    ax032.set_ylabel(r'$F_3$', fontsize = 14, color = linecolor)
    ax032.tick_params(labelsize=12, color = linecolor, labelcolor=linecolor)
    ax032.spines['right'].set_color(linecolor) 
    ax032.yaxis.label.set_color(linecolor)
    ax03.set_xlim(0,0.8)
    ax03.set_ylim(-4,4)
    ax03.yaxis.set_major_locator(MultipleLocator(2.0))  
    def abs_formatter(x, pos):
        return f"{abs(x):g}"
    ax03.yaxis.set_major_formatter(FuncFormatter(abs_formatter))
    ax032.set_ylim(0,1.005)
    ax032.yaxis.set_major_locator(MultipleLocator(0.25))  
    handles1, labels1 = ax03.get_legend_handles_labels()
    handles2, labels2 = ax032.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax03.legend(handles, labels, loc='lower right', ncol=1, fontsize=10, frameon=False)  
    ax03.axhline(y=0, color='black', zorder=10, linewidth = 0.8)



    ax01.text(-0.2, 1.0, r'(a)', fontsize=16, transform=ax01.transAxes)
    ax02.text(-0.2, 1.0, r'(b)', fontsize=16, transform=ax02.transAxes)
    ax03.text(-0.2, 1.0, r'(c)', fontsize=16, transform=ax03.transAxes)


    plt.tight_layout()
    plt.savefig('F_q.png',dpi=200)
