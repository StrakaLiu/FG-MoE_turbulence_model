#!/bin/bash

cp -r ../refData/FG_MoEData/8_CRMHL .
cd 8_CRMHL
cp ../requiredModules/* .
rm -rf 8000 postProcessing
decomposePar
mpirun  -n 1024 PysimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func sample_wall
