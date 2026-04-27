import numpy as np
import os
import shutil

def getSecondMaxTime(dir):
    # get the latest time
    t = os.listdir(dir)
    for i in range(len(t)):
        t[i] = int(t[i])
    t.sort(reverse=True)
    t_2ndMax = t[1]
    #print('latest time', t_latest)
    return t_2ndMax

t_2ndMax = getSecondMaxTime('./postProcessing/sample_down')
os.rename('./5000', './'+str(t_2ndMax+1))


