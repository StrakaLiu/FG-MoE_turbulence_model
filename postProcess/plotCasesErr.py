import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.ticker import AutoMinorLocator



color1 = '#5b89d8'
color2 = '#ef767a'
color3 = '#48c0aa'


SSTDict = '../refData/baselineData_SST/'
baseDict = '../refData/baselineData_EARSM05/'

with open('caseDir.txt', 'r') as file:
    caseDir = file.read()


data = pd.read_csv(baseDict+'modelErr.txt', skiprows=0)
data.columns = data.columns.str.strip()
base_err = data['err']

data = pd.read_csv(caseDir+'/modelErr.txt', skiprows=0)
data.columns = data.columns.str.strip()
cases = data['case']
moe_err = data['err']

data = pd.read_csv(SSTDict+'modelErr.txt', skiprows=0)
data.columns = data.columns.str.strip()
sst_err = data['err']



fig, ax = plt.subplots(figsize=(10, 5))

bar_width = 0.15
index = range(len(cases))

bars1 = ax.bar([i - 1.3*bar_width for i in index], sst_err*100, bar_width, label=r'$k-\omega$ SST', color=color1, edgecolor='#1e4c9d')
bars2 = ax.bar([i for i in index], base_err*100, bar_width, label='baseline', color=color2, edgecolor='#aa4144')
bars3 = ax.bar([i + 1.3*bar_width for i in index], moe_err*100, bar_width, label='MoE', color=color3, edgecolor='#039177')

ax.set_ylabel('relative error, %', fontsize = 16)
ax.set_xticks(index)
ax.set_xticklabels(cases, rotation=45, ha='right', fontsize = 16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(loc = 'upper left', fontsize = 16, frameon=True, facecolor='white', framealpha=1.0)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()


ax.text(-1.8, 41.5, '(a)', fontsize = 16)


plt.savefig('err.png',dpi=200)


