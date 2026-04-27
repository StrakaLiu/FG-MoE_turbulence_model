#!/bin/bash

cp -r ../refData/FG_MoEData/7_FAITHhill .
cd 7_FAITHhill
cp ../requiredModules/* .
rm -rf 10000 postProcessing
decomposePar
mpirun  -n 128 PysimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func sample_plane_z0
postProcess -func writeCellCentres
postProcess -func writeCellVolumes

