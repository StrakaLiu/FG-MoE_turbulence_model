def getLatestTime():
    import os
    # get the latest time
    t = os.listdir('./postProcessing/sample_left/')
    for i in range(len(t)):
        t[i] = int(t[i])
    t_latest = np.array(t).max()
    return t_latest


def getInterpolate2D(xyCFD, xyDNS, uDNS):
    from scipy import interpolate
    interp = interpolate.LinearNDInterpolator(xyDNS, uDNS)
    uCFD = interp(xyCFD[:,0], xyCFD[:,1])
    return uCFD


def getUMean():
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.tri as tri
    # load baseline 
    t_latest = 1
    postPath = '../../inputs/baseline/postProcessing/sample_left/'
    filePath = os.path.join(postPath, f'{t_latest}', 'U_left.raw')
    U_prof_base = np.loadtxt(filePath, skiprows = 2)


    # load trained 
    t_latest = getLatestTime()
    postPath = './postProcessing/sample_left/'
    filePath = os.path.join(postPath, f'{t_latest}', 'U_left.raw')
    U_prof_cfd = np.loadtxt(filePath, skiprows = 2)
    filePath1 = os.path.join(postPath, f'{t_latest}', 'V_left.raw')
    cellV_cfd = np.loadtxt(filePath1 , skiprows = 2)[:,3]*1e5
    filePath2 = os.path.join(postPath, f'{t_latest}', 'c1__left.raw')
    c1_cfd = np.loadtxt(filePath2 , skiprows = 2)[:,3]/1.8 - 1
    filePath3 = os.path.join(postPath, f'{t_latest}', 'c2__left.raw')
    c2_cfd = np.loadtxt(filePath3 , skiprows = 2)[:,3]/0.555555 - 1
    filePath4 = os.path.join(postPath, f'{t_latest}', 'g3__left.raw')
    g3_cfd = np.loadtxt(filePath4 , skiprows = 2)[:,3]

    # load DNS data
    filePath = os.path.join(
        '../../inputs/data/dns/', 'DNS_Re=40000.dat')
    U_prof_dns = np.loadtxt(filePath, skiprows = 20)

    scale_ux = 1.0
    scale_uy = 50.0
    scale_uz = 50.0
    U_ref = 1.0

    A_reguMax = 0.05
    step_to_Max = 5
    A_regu = min(A_reguMax/step_to_Max*(t_latest-1), A_reguMax)

    
    Ux_obs_array = U_prof_cfd[:, 3] * scale_ux / U_ref
    Uy_obs_array = U_prof_cfd[:, 4] * scale_uy / U_ref
    Uz_obs_array = U_prof_cfd[:, 5] * scale_uz / U_ref

    Ux_base_array = U_prof_base[:, 3] * scale_ux / U_ref
    Uy_base_array = U_prof_base[:, 4] * scale_uy / U_ref
    Uz_base_array = U_prof_base[:, 5] * scale_uz / U_ref

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
    
    U_obs = np.concatenate([Ux_obs_array*cellV_cfd, Uy_obs_array*cellV_cfd, Uz_obs_array*cellV_cfd,
                            c1_cfd*cellV_cfd*A_regu, c2_cfd*cellV_cfd*A_regu, g3_cfd*cellV_cfd*A_regu])
    U_dns = np.concatenate([Ux_dns_array*cellV_cfd, Uy_dns_array*cellV_cfd, Uz_dns_array*cellV_cfd,
                            cellV_cfd*0.0, cellV_cfd*0.0, cellV_cfd*0.0])

    np.savetxt(postPath + f'/{t_latest}/Array-obs.dat', U_obs)
    np.savetxt(postPath + f'/{t_latest}/Array-dns.dat', U_dns)
    misfit = np.sum(np.abs(U_obs - U_dns))
    np.savetxt(postPath + f'/{t_latest}/misfit.dat', np.array([misfit]))
    print('time=', t_latest, 'misfit=', misfit)
    file = open('./misfit.dat', "a")
    file.write(str(t_latest) + ', ' + str(misfit) + '\n')
    file.close()

if __name__ == '__main__':
    import numpy as np
    getUMean()

    postPath = './postProcessing/sample_left'
