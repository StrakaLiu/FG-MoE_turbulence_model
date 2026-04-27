import numpy as np
import os
import shutil

def getSecondMaxTime(dir):
    t = os.listdir(dir)
    for i in range(len(t)):
        t[i] = int(t[i])
    t.sort(reverse=True)
    t_2ndMax = t[1]
    return t_2ndMax


def getDivergeTime(dir):
    num = 0
    t = os.listdir(dir)
    tDiverge = []
    for i in range(len(t)):
        t[i] = float(t[i])
        if t[i] > 0.0 and t[i] < 1.0:
            tDiverge = np.append(tDiverge, t[i])
    divergeTime = np.max(tDiverge)
    #print('latest time', t_latest)
    return divergeTime


def count_folders(directory):
    contents = os.listdir(directory)
    folders = [item for item in contents if os.path.isdir(os.path.join(directory, item))]
    return len(folders)

num = count_folders('./postProcessing/sampleDict')
if os.path.exists('./10000'):
    os.rename('./10000', './'+str(num-1))
else:
    divergeTime = getDivergeTime('./postProcessing/sampleDict')
    os.rename('./' + str(divergeTime), './'+str((num-1)*0.0001))

