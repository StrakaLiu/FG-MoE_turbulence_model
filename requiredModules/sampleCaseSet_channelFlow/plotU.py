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
    yPlus = utau * y / nu
    UPlus = U / utau
    #print(yPlus[:5], UPlus[:5])
    return(yPlus, UPlus, utau)

def getUMean():
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.tri as tri


    # load CFD
    postPath = './postProcessing/sample_in'
    y_CFD = np.loadtxt('./postProcessing/sample_in/1/U_left.raw', skiprows = 2)[:, 1]
    U_CFD = np.loadtxt('./postProcessing/sample_in/1/U_left.raw', skiprows = 2)[:, 3] 

    # load DNS data
    utauDNS = 4.14872e-02
    y_DNS = np.loadtxt('truth.dat', skiprows = 1)[:, 0]
    U_DNS = np.loadtxt('truth.dat', skiprows = 1)[:,2]*utauDNS
    



    # ------------- plot velocity contour ---------------- #
    nu = 8e-6
    fig, ax1 = plt.subplots()

    yPlus, UPlus, utau = yPlus_UPlus(y_DNS, U_DNS, nu)
    ax1.semilogx( yPlus, UPlus , marker='o', linestyle = 'none', color='black',markerfacecolor='none', markevery = 0.02, markersize=6,label=r'DNS' )


    yPlus, UPlus, utau = yPlus_UPlus(y_CFD, U_CFD, nu)
    ax1.semilogx( yPlus, UPlus , marker='none', linestyle = '-', color='red',label=r'FG-MoE' )

    ax1.legend(loc='lower right',ncol=1, fontsize=16, frameon=False)
    #plt.text(2, 25, 'RANS', fontsize=20)
    ax1.set_ylabel(r'$U^+$',fontsize=22)
    #ax2.set_ylabel(r'$\it{f_d}$',fontsize=22)
    ax1.set_xlabel(r'$y^+$',fontsize=22, labelpad=1)
    ax1.tick_params(labelsize=20)
    #ax2.tick_params(labelsize=16)
    ax1.set_xlim(1,10000)
    ax1.set_ylim(0,30)
    ax1.minorticks_on()
    ax1.tick_params(direction='in')
    ax1.tick_params(which="minor", direction='in')
    fig.tight_layout()
    plt.savefig("channel_U.png", dpi = 100)
    plt.close()

    


if __name__ == '__main__':
    import numpy as np
    getUMean()

