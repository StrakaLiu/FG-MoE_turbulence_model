#!/bin/bash

decomposePar
mpirun -n 4 PyrhoSimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func sampleDict
python3 changeTimeDir.py
rm -rf ./postProcessing/sampleDict
postProcess -func sampleDict
python3 get-CFD-obs.py
