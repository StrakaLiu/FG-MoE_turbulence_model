#!/bin/bash

cp -r ./initialCases/01_channel/ .
cp ../requiredModules/* 01_channel/
cd 01_channel
PysimpleFoam
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sample_in
cd ..

cp -r ./initialCases/02_ZPGPlate/ .
cp ../requiredModules/* 02_ZPGPlate/
cd 02_ZPGPlate
decomposePar
mpirun  -n 8 PysimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sample_down
cd ..

cp -r ./initialCases/03_planeJet/ .
cp ../requiredModules/* 03_planeJet/
cd 03_planeJet 
decomposePar
mpirun -n 8 PyrhoSimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sampleDict
cd ..

cp -r ./initialCases/1_nasaHump/ .
cp ../requiredModules/* 1_nasaHump/
cd 1_nasaHump
PysimpleFoam
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sample_down
postProcess -func sample_lines_U
cd ..

cp -r ./initialCases/2_sqrDuct_Re=40000/ .
cp ../requiredModules/* 2_sqrDuct_Re=40000/
cd 2_sqrDuct_Re=40000
PysimpleFoam
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sample_left
cd ..

cp -r ./initialCases/2.1_recDuct/ .
cp ../requiredModules/* 2.1_recDuct/
cd 2.1_recDuct
PysimpleFoam
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func extractPlane
cd ..

cp -r ./initialCases/3_ASJ/ .
cp ../requiredModules/* 3_ASJ/
cd 3_ASJ
decomposePar
mpirun -n 8 PyrhoSimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sampleDict
cd ..

cp -r ./initialCases/3.1_ANSJ/ .
cp ../requiredModules/* 3.1_ANSJ/
cd 3.1_ANSJ
decomposePar
mpirun -n 8 PyrhoSimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sampleDict
cd ..

cp -r ./initialCases/4.2_NACA0012_AOA0/ .
cp ../requiredModules/* 4.2_NACA0012_AOA0/
cd 4.2_NACA0012_AOA0
PysimpleFoam
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sampleDict
cd ..

cp -r ./initialCases/4.1_NACA0012_AOA10/ .
cp ../requiredModules/* 4.1_NACA0012_AOA10/
cd 4.1_NACA0012_AOA10
PysimpleFoam
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sampleDict
cd ..

cp -r ./initialCases/4_NACA0012_AOA15/ .
cp ../requiredModules/* 4_NACA0012_AOA15/
cd 4_NACA0012_AOA15 
PysimpleFoam
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sampleDict
cd ..

cp -r ./initialCases/5_bump/ .
cp ../requiredModules/* 5_bump/
cd 5_bump
decomposePar
mpirun -n 8 PysimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func sample_down
postProcess -func sample_lines_U9
postProcess -func sample_left_U
cd ..

cp -r ./initialCases/6_pehill/ .
cp ../requiredModules/* 6_pehill/
cd 6_pehill
decomposePar
mpirun -n 4 PysimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func writeCellCentres
postProcess -func writeCellVolumes
postProcess -func 'sampleDict' -latestTime
postProcess -func 'sample_down' -latestTime
postProcess -func 'sample_left_U' -latestTime
cd ..


