import os
import numpy as np
import scipy
from scipy import interpolate
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def calStateP(phi1,phi2,phi3):
    F1 = 1.0 / (1.0 + np.exp(-np.clip(40*(phi1 - 0.25),-1e8,20)))
    F2 = 1.0 / (1.0 + np.exp(-np.clip(1000*(phi2 - 0.01),-1e8,20)))
    F3 = 1.0 / (1.0 + np.exp(-np.clip(50*(phi3 - 0.2),-1e8,20)))
    s000 = (1-F1)*(1-F2)*(1-F3)
    s100 = (F1)*(1-F2)*(1-F3)
    s010 = (1-F1)*(F2)*(1-F3)
    s110 = (F1)*(F2)*(1-F3)
    s001 = (1-F1)*(1-F2)*(F3)
    s101 = (F1)*(1-F2)*(F3)
    s011 = (1-F1)*(F2)*(F3)
    s111 = (F1)*(F2)*(F3)
    return s000, s100, s010, s110, s001, s101, s011, s111


def calExpertP(s000, s100, s010, s110, s001, s101, s011, s111):
    e1 = s101 + s111
    e2 = s011
    e3 = s010 + s110
    e4 = s000 + s100 + s001
    return e1, e2, e3, e4

def meanP(V, s000, s100, s010, s110, s001, s101, s011, s111, e1, e2, e3, e4):
    sumV = np.sum(V)
    return  np.sum(s000*V)/sumV, np.sum(s100*V)/sumV, np.sum(s010*V)/sumV, np.sum(s110*V)/sumV, np.sum(s001*V)/sumV, np.sum(s101*V)/sumV, np.sum(s011*V)/sumV, np.sum(s111*V)/sumV, np.sum(e1*V)/sumV, np.sum(e2*V)/sumV, np.sum(e3*V)/sumV, np.sum(e4*V)/sumV

def calCase(postPath,casename, xmax, xmin, ymax, ymin, zmax, zmin, useCellV = True):
    ncell = int(np.loadtxt(postPath + '/V', skiprows = 21, max_rows = 1))
    Cx = np.loadtxt(postPath + '/Cx', skiprows = 23, max_rows = ncell)
    Cy = np.loadtxt(postPath + '/Cy', skiprows = 23, max_rows = ncell)
    Cz = np.loadtxt(postPath + '/Cz', skiprows = 23, max_rows = ncell)
    mask = ((Cx>xmin) & (Cx<xmax) & (Cy>ymin) & (Cy<ymax) & (Cz>zmin) & (Cz<zmax))
    cellV = np.loadtxt(postPath + '/V', skiprows = 23, max_rows = ncell)[mask]
    phi1 = np.loadtxt(postPath + '/Q_sep', skiprows = 23, max_rows = ncell)[mask]
    phi2 = np.loadtxt(postPath + '/Q_3D', skiprows = 23, max_rows = ncell)[mask]
    phi3 = np.loadtxt(postPath + '/Q_mix', skiprows = 23, max_rows = ncell)[mask]
    s000, s100, s010, s110, s001, s101, s011, s111 = calStateP(phi1,phi2,phi3)
    e1, e2, e3, e4 = calExpertP(s000, s100, s010, s110, s001, s101, s011, s111)
    if useCellV == False:
        cellV[:] = 1.0
    return meanP(cellV, s000, s100, s010, s110, s001, s101, s011, s111, e1, e2, e3, e4)


with open('caseDir.txt', 'r') as file:
    caseDir = file.read()

file = open(caseDir + '/modelState.txt', 'w')
file.write('case, s000, s100, s010, s110, s001, s101, s011, s111, e1, e2, e3, e4\n')
print('case, s000, s100, s010, s110, s001, s101, s011, s111, e1, e2, e3, e4')
#-----------------------channel--------------------------#
tlast = getLatestTime(caseDir + '/01_channel/postProcessing/sample_in')
postPath = caseDir+'/01_channel/' + str(tlast)
casename = 'channel'
xmax = 1e8; xmin = -1e8; ymax = 1e8; ymin = -1e8; zmax = 1e8; zmin = -1e8;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#-------------------------------------------PSJ----------------------------------#
tlast = getLatestTime(caseDir + '/03_planeJet/postProcessing/sampleDict')
postPath = caseDir+'/03_planeJet/' + str(tlast)
casename = 'PSJ'
scale = 0.5
xmax = 1e8*scale; xmin = 0*scale; ymax = 1e8*scale; ymin = -1e8*scale; zmax = 5*scale; zmin = 0*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#-------------------------------------------square duct----------------------------------#
tlast = getLatestTime(caseDir + '/2_sqrDuct_Re=40000/postProcessing/sample_left')
postPath = caseDir+'/2_sqrDuct_Re=40000/' + str(tlast)
casename = 'SqrDuct'
scale = 1
xmax = 1e8*scale; xmin = -1e8*scale; ymax = 1e8*scale; ymin = -1e8*scale; zmax = 1e8*scale; zmin = -1e8*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#-------------------------------------------rec duct----------------------------------#
tlast = getLatestTime(caseDir + '/2.1_recDuct/postProcessing/extractPlane')
postPath = caseDir+'/2.1_recDuct/' + str(tlast)
casename = 'RecDuct'
scale = 1
xmax = 1e8*scale; xmin = -1e8*scale; ymax = 1e8*scale; ymin = -1e8*scale; zmax = 1e8*scale; zmin = -1e8*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#-----------------------ZPG plate--------------------------#
tlast = getLatestTime(caseDir + '/02_ZPGPlate/postProcessing/sample_down')
postPath = caseDir+'/02_ZPGPlate/' + str(tlast)
casename = 'ZPG plate'
scale = 1.0
xmax = 1e8*scale; xmin = 0.0*scale; ymax = 0.1*scale; ymin = -1e8*scale; zmax = 1e8*scale; zmin = -1e8*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#-------------------------------------------ASJ----------------------------------#
tlast = getLatestTime(caseDir + '/3_ASJ/postProcessing/sampleDict')
postPath = caseDir+'/3_ASJ/' + str(tlast)
casename = 'ASJ'
scale = 0.0254
xmax = 1e8*scale; xmin = 0*scale; ymax = 1e8*scale; ymin = -1e8*scale; zmax = 5*scale; zmin = 0*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#-------------------------------------------ANSJ----------------------------------#
tlast = getLatestTime(caseDir + '/3.1_ANSJ/postProcessing/sampleDict')
postPath = caseDir+'/3.1_ANSJ/' + str(tlast)
casename = 'ANSJ'
scale = 0.0254
xmax = 1e8*scale; xmin = 0*scale; ymax = 1e8*scale; ymin = -1e8*scale; zmax = 5*scale; zmin = 0*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#--------------------------------------------NACA0012-------------------------------#
tlast = getLatestTime(caseDir + '/4_NACA0012_AOA15/postProcessing/sampleDict')
postPath = caseDir+'/4_NACA0012_AOA15/' + str(tlast)
casename = 'NACA0012'
scale = 1.0
xmax = 1.2*scale; xmin = -0.2*scale; ymax = 1e8*scale; ymin = -1e8*scale; zmax = 0.2*scale; zmin = -0.2*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#--------------------------------------------2D-hump-------------------------------#
tlast = getLatestTime(caseDir + '/1_nasaHump/postProcessing/sample_lines_U')
postPath = caseDir+'/1_nasaHump/' + str(tlast)
casename = '2D-hump'
scale = 0.42
xmax = 1.3*scale; xmin = 0.0*scale; ymax = 0.15*scale; ymin = -1e8*scale; zmax = 1e8*scale; zmin = -1e8*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#--------------------------------------------2D-bump-------------------------------#
tlast = getLatestTime(caseDir + '/5_bump/postProcessing/sample_lines_U9')
postPath = caseDir+'/5_bump/' + str(tlast)
casename = '2D-bump'
scale = 0.305
xmax = 1.3*scale; xmin = 0.0*scale; ymax = 0.15*scale; ymin = -1e8*scale; zmax = 1e8*scale; zmin = -1e8*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#--------------------------------------------peHills-------------------------------#
tlast = getLatestTime(caseDir + '/6_pehill/postProcessing/sampleDict')
postPath = caseDir+'/6_pehill/' + str(tlast)
casename = '2D-peHills'
scale = 1
xmax = 9*scale; xmin = 0.0*scale; ymax = 3*scale; ymin = -1e8*scale; zmax = 1e8*scale; zmin = -1e8*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")
#--------------------------------------------FAITH HILL-------------------------------#
tlast = getLatestTime(caseDir + '/7_FAITHhill/postProcessing/sample_plane_z0')
postPath = caseDir+'/7_FAITHhill/' + str(tlast)
casename = '3D-hill'
scale = 0.2286
xmax = 2*scale; xmin = -1*scale; ymax = 0.8*scale; ymin = 0.0*scale; zmax = 1e8*scale; zmin = -1e8*scale;  
s000a, s100a, s010a, s110a, s001a, s101a, s011a, s111a, e1a, e2a, e3a, e4a = calCase(postPath, casename, xmax, xmin, ymax, ymin, zmax, zmin)
print(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}")
file.write(casename + ', ' + f"{s000a:.3f}, {s100a:.3f}, {s010a:.3f}, {s110a:.3f}, {s001a:.3f}, {s101a:.3f}, {s011a:.3f}, {s111a:.3f}, {e1a:.3f}, {e2a:.3f}, {e3a:.3f}, {e4a:.3f}\n")

file.close()


with open(caseDir + '/modelState.txt', 'r') as f:
    lines = f.readlines()

cases = []
data_values = []

for i, line in enumerate(lines):
    if not line.strip():
        continue
    parts = [x.strip() for x in line.split(',')]
    if i == 0:
        headers = parts[1:]
    else:
        cases.append(parts[0])
        data_values.append([float(x) for x in parts[1:]])




headers = ['Parallel-shear, 2D, Free', 'Non-parallel-shear, 2D, Free', 'Parallel-shear, 3D, Free', 'Non-parallel-shear, 3D, Free', 
            'Parallel-shear, 2D, Near-wall', 'Non-parallel-shear, 2D, Near-wall', 'Parallel-shear, 3D, Near-wall', 'Non-parallel-shear, 3D, Near-wall', 
            'NPS', 'SEC', 'FS', 'BASE']

data_array = np.array(data_values)

data_heatmap1 = data_array[:, :8]*100  
data_heatmap2 = data_array[:, 8:]*100  


x = np.arange(len(headers[:8]) + 1) - 0.5  
y = np.arange(len(cases) + 1) - 0.5       

fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

ax1 = plt.subplot(gs[0])
mesh1 = ax1.pcolormesh(x, y, data_heatmap1, cmap='plasma', edgecolors='white', linewidth=1)
for i in range(data_heatmap1.shape[0]):
    for j in range(data_heatmap1.shape[1]):
        val = data_heatmap1[i, j]
        if val > 10:
            color = 'black' if val > 50 else 'white'
            ax1.text(j, i, f"{round(val)}%", ha='center', va='center', fontsize=10, color=color)
ax1.set_xticks(range(len(headers[:8])))
ax1.set_xticklabels([])  
ax1.set_xticklabels(headers[:8], rotation=45, ha='right', rotation_mode='anchor', fontsize = 10)
ax1_top = ax1.secondary_xaxis('top')
ax1_top.set_xticks(range(len(headers[:8])))
new_headers_top = ["[0,0,0]", "[1,0,0]", "[0,1,0]", "[1,1,0]", "[0,0,1]", "[1,0,1]", "[0,1,1]", "[1,1,1]"] 
ax1_top.set_xticklabels(new_headers_top, fontsize=10) 
ax1_top.tick_params(axis='both', which='both', length=0)
ax1.set_yticks(range(len(cases)))
ax1.set_yticklabels(cases, fontsize = 10)
ax1.tick_params(axis='both', which='both', length=0)
ax1.set_title('State probability', fontsize = 14)
ax1.invert_yaxis()
ax1.set_aspect(0.5)
ax2 = plt.subplot(gs[1])
x2 = np.arange(len(headers[8:]) + 1) - 0.5
y2 = np.arange(data_heatmap2.shape[0] + 1) - 0.5
mesh2 = ax2.pcolormesh(x2, y2, data_heatmap2, cmap='plasma', edgecolors='white', linewidth=1)
for i in range(data_heatmap2.shape[0]):
    for j in range(data_heatmap2.shape[1]):
        val = data_heatmap2[i, j]
        if val > 10:
            color = 'black' if val > 50 else 'white'
            ax2.text(j, i, f"{round(val)}%", ha='center', va='center', fontsize=10, color=color)
ax2.set_xticks(range(len(headers[8:])))
ax2.set_xticklabels(headers[8:], rotation=0, fontsize = 10)
ax2.xaxis.set_tick_params(labeltop=True)
ax2.xaxis.set_tick_params(labelbottom=False)
ax2.set_yticks([])
ax2.tick_params(axis='both', which='both', length=0)
ax2.set_title('Expert probability', fontsize = 14)
ax2.invert_yaxis()
ax2.set_aspect(0.5)



ax1.annotate('(a)', xy=(-0.1, 1.15), xycoords='axes fraction', fontsize=16, 
             xytext=(5, -5), textcoords='offset points', va='top', ha='left')
ax2.annotate('(b)', xy=(-0.1, 1.15), xycoords='axes fraction', fontsize=16,
             xytext=(5, -5), textcoords='offset points', va='top', ha='left')


cbar_ax = fig.add_axes([0.6, 0.1, 0.3, 0.02]) 
cbar = plt.colorbar(mesh2, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Probability, %', fontsize = 14)


plt.subplots_adjust(wspace=0.1)  

plt.savefig('hotplot.png', dpi=200,bbox_inches='tight')





