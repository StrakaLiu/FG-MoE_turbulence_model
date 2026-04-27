#!/bin/bash
#SBATCH --job-name=ASJ
#SBATCH -p xahcnormal
#SBATCH --output=log.%j
#SBATCH --error=err.%j
#SBATCH -n 1


module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.4.1-gcc-7.3.1

# source ~/wuchutian/envs/PyFoam.sh
# #source ~/wuchutian/envs/py310.sh
# source ~/LuoQY/env/py.sh
# export UCX_NET_DEVICES=mlx5_0:1

source ~/LuoQY/env/mpi.sh
source ~/LuoQY/env/py.sh
source ~/LuoQY/Downloads/PyOpenFOAM-v2212/prep_env.sh



# PysimpleFoam > log.PysimpleFoam
#decomposePar
PysimpleFoam
#PysimpleFoam
#reconstructPar -latestTime
# postProcess -func writeCellVolumes
# postProcess -func sample_left
postProcess -func sampleDict
python3 plot_Cp.py
#rm -rf processor*
