#!/bin/bash

decomposePar
mpirun -n 2 PysimpleFoam -parallel
reconstructPar -latestTime
rm -rf processor*
postProcess -func sample_down
python3 changeTimeDir.py
rm -rf ./postProcessing/sample_down
postProcess -func sample_down
postProcess -func sample_lines_U
postProcess -func sample_lines_C
python3 get-CFD-obs.py
