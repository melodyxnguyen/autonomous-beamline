import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import pyFAI, fabio
import fnmatch
import pandas as pd
import numpy as np
import sys

# XDart imports 
sys.path.append('C:\\Users\\Public\\Documents\\repos\\xdart')
from xdart.modules.pySSRL_bServer.bServer_funcs import *
from xdart.utils import get_from_pdi, get_motor_val, query, query_yes_no
from xdart.utils import read_image_file, smooth_img, get_fit, fit_images_2D
from xdart.modules.pySSRL_bServer.bServer_funcs import specCommand, wait_until_SPECfinished, get_console_output


def run_sample_scan(start, stop, steps): 
    """Function to run sample scan

    Arguments:
        start {float} -- minimum 2th value to scan. Scan is performed from start to stop
        stop  {float} -- maximum 2th value for scan
        steps {int} -- number of steps
    """
    command = f'ascan  tth {start} {stop} {steps} 1'
    print(f'Running sample scan [{command}]')
    try:
        specCommand(command, queue=True)
    except Exception as e:
        print(e)
        print(f"Command '{command}' not sent")
        sys.exit()
        
    # Wait till Scan is finished to continue
    print('Waiting for scan to finish..')
    wait_until_SPECfinished(polling_time=5)
    time.sleep(5)
    print('Done', '\n')



ai = pyFAI.load("LaB6_2DetSetup.poni")

cur_raw = input('What is the raw file? ')

im = open(cur_raw, 'rb')
arr = np.fromstring(im.read(), dtype='int32')
im.close()
arr.shape = (195, 487)
plt.figure(1)
plt.clf()
plt.imshow(arr)
plt.clim(np.min(arr),np.max(arr)/5)
plt.colorbar()
plt.pause(0.01)
plt.draw()

if not os.path.exists(cur_raw.replace(".raw",".tif")):
    im = Image.fromarray(arr)
    im.save(cur_raw.replace(".raw",".tif"))
    
cur_tif = cur_raw.replace(".raw",".tif")
cur_xy = cur_tif.replace('.tif','.xy')

if not os.path.exists(cur_xy):
    img = fabio.open(cur_tif)
    img_array = img.data
    res = ai.integrate1d(img_array,
                        250,
                        unit="2th_deg",
                        filename=cur_xy)

    
    #Delete excess pyFAI created lines so they can be refined with TOPAS
    Script = pd.read_csv(cur_xy,skiprows=23,header=None,delim_whitespace=True)
    Script.columns = ['radians','I']
    Script.to_csv(cur_xy,index=False,float_format='%.6f',sep='\t')
    print(cur_xy)

xyOpen = np.genfromtxt(cur_xy, dtype=float, delimiter='\t')
xyDeg = xyOpen[:,0]
xyOb = xyOpen[:,1]

SmNum = 3

xyObs = [xyOb[0]]  ##Smoothing
for row in range(1, SmNum - 1):
    xyObs.append(xyOb[row])

for row in range(SmNum, len(xyOb) - SmNum):
    xyObs.append((xyOb[row - 3] + xyOb[row - 2] + xyOb[row - 1] + xyOb[row] + xyOb[row + 1] + xyOb[row + 2] + xyOb[row + 3])/7)

for row in range(SmNum + 1, 0, -1):
    xyObs.append(xyOb[len(xyObs) - SmNum])

peakwidth = 0.3

PeakInts = []
PeakPos = []
ScanLows = []
ScanHighs = []
ScanInts = []
TNB = []
TNBI = []

ScanHigh = 0
for row in range(0, len(xyObs)-1):
    if row > 3 and row < len(xyObs) - 3\
    and xyObs[row - 4] < xyObs[row - 3] < xyObs[row - 2] < xyObs[row - 1] < xyObs[row] > xyObs[row + 1]\
    and xyObs[row + 4] < xyObs[row + 3] < xyObs[row + 2] < xyObs[row + 1]:
        PeakInt = xyObs[row]
        PeakInts.append(PeakInt)
        PeakP = xyDeg[row]
        PeakPos.append(PeakP)
        
        if xyDeg[row + 1] - peakwidth  > ScanHigh:
            ScanLow = xyDeg[row + 1] - peakwidth 
        else:
            ScanLow = ScanHigh
        ScanHigh = xyDeg[row + 1] + peakwidth
        ScanLows.append(ScanLow)
        ScanHighs.append(ScanHigh)
        ScanInts.append(xyObs[row])
        ScanInts.append(xyObs[row])
        for val in range(0, len(xyObs)-1):
            if ScanLow < xyDeg[val] < ScanHigh:
                TNB.append(xyDeg[val])
                TNBI.append(xyOb[val])
            else:
                TNBI.append(0)
                TNB.append(xyDeg[val])

pltNewDat = plt.figure(3)
plt.clf()
plt.plot(xyDeg,xyOb, color='blue')
plt.plot(TNB,TNBI, color='green')
plt.ylabel("Intensity")
plt.xlabel("2\u03B8 (degrees)")

plt.pause(0.01)
plt.draw()

for scans in range(0, len(ScanHighs)):
    print(scans)
    print(round(ScanHighs[scans],3))
    print(round(ScanLows[scans],3))
    print((peakwidth*2)/0.005)
    run_sample_scan(round(ScanLows[scans],3), round(ScanHighs[scans],3), int((peakwidth*2) / 0.005))