import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

nsample = 32
tstart = 1

minMisfit = 1e15
minMisfit_i = 0
minMisfit_glo = 1e15
minMisfit_i_glo = 0
minMisfit_i_glo_time = 0
fig, ax1 = plt.subplots()
for i in range(nsample):
    file = './results_ensemble/sample_' + str(i) + '/misfit.dat'
    t, misfit =  np.loadtxt(file, delimiter=',', skiprows = 0)[:,0], np.loadtxt(file, delimiter=',', skiprows = 0)[:,1] 
    ax1.plot( t-tstart, misfit , marker='none', linestyle = '-', label = str(i) )
    if misfit[-1] < minMisfit:
        minMisfit = misfit[-1]
        minMisfit_i = i
    if min(misfit[:]) < minMisfit_glo:
        minMisfit_i_glo_time = np.argmin(misfit[:])
        minMisfit_glo = min(misfit[:])
        minMisfit_i_glo = i
print('min misfit at final time is', minMisfit, 'at sample', minMisfit_i)
print('global min misfit is', minMisfit_glo, 'at sample', minMisfit_i_glo, 'at time', t[minMisfit_i_glo_time])

ax1.legend(ncol=4, fontsize=10, frameon=False)
ax1.set_ylabel('misfit',fontsize=22)
ax1.set_xlabel('iteration',fontsize=22)
ax1.tick_params(labelsize=20)
ax1.tick_params(direction='out')
plt.tight_layout()
plt.savefig('misfit.png',bbox_inches = 'tight')
plt.close()
