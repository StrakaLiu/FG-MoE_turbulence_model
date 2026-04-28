
# Introduction

**FG-MoE (Factorized-Gating Mixture of Experts)** is a modeling framework designed to capture complex turbulent behaviors across different flow regimes. It decomposes the turbulence model into regime-specific experts and combines them through spatially varying gating probabilities. Moreover, a **factorized gating mechanism** is introduced to reduce model complexity, improve interpretability, and enable cross-regime generalization.

This guide provides step-by-step instructions for installation, compilation, and usage.

# Requirements
| Category        | Requirement                                                                                                                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OS**          | Ubuntu 22.04 (or other compatible Linux distributions)                                                                                                                                          |
| **OpenFOAM**    | v2212 compiled and sourced (recommended: GCC 11.4.0, OpenMPI 4.1.2, CMake 3.22.1)                                                                                                               |
| **Python**      | ≥3.8 (3.10 recommended) with pip, numpy (1.24.3 recommended), scipy (1.11.1 recommended), TensorFlow >2.1 (2.13.0 recommended, required for FG-MoE turbulence model), pandas (2.3.3 recommended) |
| **System Libs** | build-essential, cmake, flex, bison, zlib1g-dev, libreadline-dev, etc. (typically present if OpenFOAM is already built)                                                                         |
| **Optional**    | ParaView (recommended for visualization)                                                                                                                                                        |

# Installation

The following steps install the **DAFI** framework, the **PythonFOAM solvers**, and the **FG-MoE turbulence models**. 
The installation process takes less than 15 minutes on a typical desktop when the requirements are satisfied.

- Create working directory
```bash
export INSTALL_LOCATION=$HOME/software         # change if needed
mkdir -p $INSTALL_LOCATION && cd $INSTALL_LOCATION
```

### 1. DAFI

- Download the DAFI code used for model learning
```bash
git clone https://github.com/XinleiZhang/ENKL.git
source $INSTALL_LOCATION/ENKL/init.sh
```

### 2. PythonFOAM

- Download the PythonFOAM code, which provides a coupled OpenFOAM-Python solver. 
```bash
git clone  https://github.com/argonne-lcf/PythonFOAM.git
```

- Edit the `prep_env.sh` file in `$INSTALL_LOCATION/PythonFOAM` and replace all placeholder paths in the file with your actual installation paths.
```bash
echo "Make sure your paths to OpenFOAM/Python headers/libraries are set appropriately"
if [ "$1" == "-debug" ]; then
	echo ""
	echo "Using OpenFOAM in Debug mode"
	source $YOUR_OPENFOAM_PATH/etc/bashrc
else
	echo "Using OpenFOAM in Optimized mode"
	source $YOUR_OPENFOAM_PATH/etc/bashrc
fi

export PYTHON_LIB_PATH=$YOUR_PYTHON_LIB_PATH
export PYTHON_BIN_PATH=$YOUR_PYTHON_BIN_PATH
export PYTHON_INCLUDE_PATH=$YOUR_PYTHON_INCLUDE_PATH
export NUMPY_INCLUDE_PATH=$YOUR_NUMPY_INCLUDE_PATH
export PYTHON_LIB_NAME=lpython3.10         # change with your python version
```

- Source the PythonFOAM
```bash
cd $INSTALL_LOCATION/PythonFOAM
./prep_env.sh
```

### 3. FG-MoE turbulence models

- Download the FG-MoE code
```bash
git clone  https://github.com/StrakaLiu/FG-MoE_turbulence_model.git
```

- Compile the CFD solvers that can couple with a neural network-based turbulence model
```bash
cd $INSTALL_LOCATION/FG-MoE_turbulence_model/requiredModules/flowSolvers/PysimpleFoam
wclean && wmake

cd $INSTALL_LOCATION/FG-MoE_turbulence_model/requiredModules/flowSolvers/PyrhoSimpleFoam
wclean && wmake
```

- Compile the neural network-based turbulence models
```bash
cd $INSTALL_LOCATION/FG-MoE_turbulence_model/requiredModules/turbModel/FG_MoE/incompressible/
wclean && wmake

cd $INSTALL_LOCATION/FG-MoE_turbulence_model/requiredModules/turbModel/FG_MoE/compressible/
wclean && wmake
```
# Datasets

The datasets generated and used in the present study are provided at [Link](https://doi.org/10.5281/zenodo.19811876).

Users can download the refData.tar file to the `$INSTALL_LOCATION/FG-MoE_turbulence_model/` directory and decompress it using:

```bash
tar -zxvf $INSTALL_LOCATION/FG-MoE_turbulence_model/refData.tar
```
This step is required to run the benchmark test cases and reproduce the results presented in the present study.

The data include:

| directory | Description |
| :--- | :--- |
| truthData | The data from experiments, DNS, or high-resolution LES of different cases. Used as ground truth values for evaluating model performance. |
| trainedExpertModelData | The data from the three training cases, each run by the corresponding trained expert model. |
| FG_MoEData | The simulation results of the test cases obtained by the trained FG-MoE turbulence model. |
| baselineData_EARSM05 | The simulation results of the test cases obtained by the baseline EARSM05 model. |
| baselineData_SST | The simulation results of the test cases obtained by the $k$-$\omega$ SST model. |


# Get started

### 1. Usage of the FG-MoE model 

The trained FG-MoE turbulence model can be directly used like any of the RAS turbulence models in OpenFOAM. The required implementations include:

1. Add the following library at the beginning of the `$CASE/system/controlDict` file:

```C++
libs
( 
  "lib_FG_MoE_incompressibleRAS"
) ;
```

2. In the `$CASE/constant/turbulenceProperties` file, set the `RAS` dictionary as:

```C++
RAS
{
    RASModel            FG_MoE; 	
    useBaselineModel    false;	//'true' if use the baseline EARSM05 model
    homogeneousZ        true; 	//'true' if the case is 2D and homogeneous in Z direction
    homogeneousY        false; 	//'true' if the case is 2D and homogeneous in Y direction
    kInf                0.1;	//freestream value of k. Set to zero for internal cases
    omegaInf            100;	//freestream value of omega. Set to zero for internal cases
}
```

3. Copy the required modules into your case directory by:

```bash
cp $INSTALL_LOCATION/FG-MoE_turbulence_model/requiredModules/* $CASE_DIR/
```


4. Run the case with the `PysimpleFoam` or the `PyrhoSimpleFoam` solvers.

An example of the case setup can be found at `$INSTALL_LOCATION/FG-MoE_turbulence_model/requiredModules/sampleCaseSet_channelFlow/` directory. 
This is a channel flow at $Re_\tau=5200$. 
It can be run and post-processed by:

```bash
PysimpleFoam
postProcess -func sample_in
python3 plotU.py
```

The simulation will finish in about 200 seconds.

### 2. Train experts for each flow scenario

The initial datasets for training the three expert models are located in the `$INSTALL_LOCATION/FG-MoE_turbulence_model/trainingExperts/` directory. To run the training process, use the `training.sh` script in each training case.

The `inputs/` directory contains two subdirectories:
- `data/`: contains the ground truth data for training (from experiments or DNS).
- `baseline/`: contains the prediction results from the baseline model, which are used as the initial guess for the training process.

During training, a `results_ensemble/` directory will be created, containing samples of the ensemble training as `sample_*/`. 
In each sample directory, the prediction results at each iteration are listed in time directories. 
The related neural-network weight parameters are listed as `nn_weights_flatten_*.dat`. 

The mean absolute error of each sample's predictions during the iteration can be plotted using the script `plot_misfit.py`. Any convergence criterion can be used to end the training process. The neural-network weight parameters (defined by the corresponding `nn_weights_flatten_*.dat` file) for the sample with the lowest prediction error can be chosen as a trained expert. 

The four expert models (three trained and one baseline) in the present study are provided as `$INSTALL_LOCATION/FG-MoE_turbulence_model/requiredModules/*.dat`.



### 3. Test the FG-MOE model in benchmark cases

Several benchmark test cases are provided to evaluate the capability of the trained FG-MoE turbulence model. These cases are located in the `$INSTALL_LOCATION/FG-MoE_turbulence_model/runTestCases/initialCases/` directory.

To run the cases, use the `$INSTALL_LOCATION/FG-MoE_turbulence_model/runTestCases/*.sh` scripts. 
The expected simulation results of the test cases obtained by the FG-MoE model can be found in the `$INSTALL_LOCATION/FG-MoE_turbulence_model/refData/FG_MoEData/` directory.

The test cases are as follows:

| Case name | Description | Source |
| :--- | :--- | :--- |
| 01_channel | channel flow at Re_tau=5200 | [Link](https://turbulence.oden.utexas.edu/channel2015/content/Data_2015_5200.html) |
| 02_ZPGPlate | zero-pressure gradient plate | [Link](https://tmbwg.github.io/turbmodels/flatplate.html) |
| 03_planeJet | 2D plane jet | [Link](https://tmbwg.github.io/turbmodels/shear.html) |
| 1_nasaHump | NASA wall-mounted hump | [Link](https://tmbwg.github.io/turbmodels/nasahump_val.html) |
| 2_sqrDuct_Re=40000 | square duct flow at Re=40000 | [Link](http://newton.dma.uniroma1.it/square_duct/) |
| 2.1_recDuct | rectangular duct at Re_tau=360, aspect ratio=3 | [Link](https://www.vinuesalab.com/duct/) |
| 3_ASJ | Axisymmetric Subsonic Jet | [Link](https://tmbwg.github.io/turbmodels/jetsubsonic_val.html) |
| 3.1_ANSJ | Axisymmetric near sonic Jet | [Link](https://tmbwg.github.io/turbmodels/jetnearsonic_val.html) |
| 4* | NACA0012 at different AOA | [Link](https://tmbwg.github.io/turbmodels/naca0012_val.html) |
| 5_bump | 2D bump | [Link](https://tmbwg.github.io/turbmodels/Other_LES_Data/family_of_bumps.html) |
| 6_pehill | periodic hill | [Link](https://turbmodels.larc.nasa.gov/Other_DNS_Data/parameterized_periodic_hills.html) |
| 7_FAITHhill | FAITH 3D hill | [Link](https://tmbwg.github.io/turbmodels/Other_exp_Data/FAITH_hill_exp.html) |
| 8_CRMHL | High lift common research model | [Link](https://aiaa-hlpw.org/HLPW5/) |

**Note:** Prior downloading of the datasets is required to run cases `7_FAITHhill` and `8_CRMHL`. See the **Datasets** section for details.


# Reproductivity

The results presented in the paper can be reproduced using the Python scripts located in the `$INSTALL_LOCATION/FG-MoE_turbulence_model/postProcess/` directory.
Prior downloading of the datasets is required. See the **Datasets** section for details.

The descriptions of the files are as follows:

| File name | Description |
| :--- | :--- |
| caseDir.txt | Defines the directory of the test cases for post-processing. |
| plotExpertResults.py | Plots the detailed results of the trained expert models in the corresponding training cases (Fig. 2). |
| calculateCasesErr.py | Calculates the relative error for each test case and writes the results to a file named `modelErr.txt` in the test case directory. |
| plotCasesErr.py | Plots the relative errors and compares them with the baseline and the $k$-$\omega$ SST models (Fig. 3a). |
| plotCasesResults.py | Plots the detailed results for representative cases (Fig. 3b-f). |
| plotCasesStates.py | Calculates and plots the global probabilities of the states and experts for each test case (Fig. 4). |
| plot_CRMHL_Cp.py | Plots the pressure coefficient results of the HL-CRM case (Fig. 5). |
| plotExpertData.py | Plots the PDFs of the discriminant input features $\phi_m$ and the resulting discriminant functions $F_m$ (Fig. 6, left column). |
| plotExpertGates.py | Plots the contour plots of $F_m$ in the training cases (Fig. 6, middle and right columns). |

# Contact

Hao-Chen Liu, liu.h.c@imech.ac.cn

Xin-Lei Zhang, zhangxinlei@imech.ac.cn


