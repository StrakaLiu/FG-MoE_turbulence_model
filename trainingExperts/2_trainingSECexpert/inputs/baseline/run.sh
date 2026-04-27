#!/bin/bash

PysimpleFoam
postProcess -func writeCellVolumes
postProcess -func sample_left
python3 get-CFD-obs.py
