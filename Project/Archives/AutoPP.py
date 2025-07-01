import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import pyFAI, fabio
import fnmatch
import pandas as pd
import numpy as np
import time

file = input('What is the raw file? ')

t0 = time.time()

ai = pyFAI.load("LaB6_2.poni")

if file.endswith(".raw") == False:
    tfile = file + ".raw"
    file = tfile

#print "Reading RAW file here..."

im = open(file, 'rb')
arr = np.fromstring(im.read(), dtype='int32')
im.close()
arr.shape = (195, 487)
plt.figure(1)
plt.imshow(arr)
plt.clim(np.min(arr),np.max(arr)/5)
plt.colorbar()
plt.pause(0.01)
plt.draw()

if not os.path.exists(file.replace(".raw",".tif")):
    im = Image.fromarray(arr)
    im.save(file.replace(".raw",".tif"))
    
cur_tif = file.replace(".raw",".tif")
cur_xy = cur_tif.replace('.tif','.xy')
if not os.path.exists(cur_xy):
    img = fabio.open(cur_tif)
    img_array = img.data
    res = ai.integrate1d(img_array,
                        250, #This can be shrunk down if many patterns are being taken, keep value consistent with calibration
                        #mask = mask_array, #Line can be commented out if no mask is made
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

#xyObs = xyOb ##No Smoothing

xyObs = [xyOb[0]]  ##Smoothing
for row in range(1, SmNum - 1):
    xyObs.append(xyOb[row])

for row in range(SmNum, len(xyOb) - SmNum):
    xyObs.append((xyOb[row - 3] + xyOb[row - 2] + xyOb[row - 1] + xyOb[row] + xyOb[row + 1] + xyOb[row + 2] + xyOb[row + 3])/7)

for row in range(SmNum + 1, 0, -1):
    xyObs.append(xyOb[len(xyObs) - SmNum])

#Get first and second "derivatives"
xyD1 = []
xyD2 = []
PeakInts = []
PeakPos = []
ScanDeg = []
ScanInts = []
TNB = []
TNBI = []

for row in range(0, len(xyObs)-1):
    xyDiff = xyObs[row + 1] - xyObs[row - 1]
    xyD1.append(xyDiff)
xyD1.append(0)

for row in range(0, len(xyObs)-1):
    xyDiff = xyD1[row + 1] - xyD1[row - 1]
    xyD2.append(xyDiff)
xyD2.append(0)
ScanHigh = 0
for row in range(0, len(xyObs)-1):
    if row > 3 and row < len(xyObs) - 3\
    and xyObs[row - 4] < xyObs[row - 3] < xyObs[row - 2] < xyObs[row - 1] < xyObs[row] > xyObs[row + 1]\
    and xyObs[row + 4] < xyObs[row + 3] < xyObs[row + 2] < xyObs[row + 1]:
        PeakInt = xyObs[row]
        PeakInts.append(PeakInt)
        PeakP = xyDeg[row]
        PeakPos.append(PeakP)
        
        if xyDeg[row] - 0.375 > ScanHigh:
            ScanLow = xyDeg[row] - 0.375
        else:
            ScanLow = ScanHigh
        ScanHigh = xyDeg[row] + 0.5
        ScanDeg.append(ScanLow)
        ScanDeg.append(ScanHigh)
        ScanInts.append(xyObs[row])
        ScanInts.append(xyObs[row])
        for val in range(0, len(xyObs)-1):
            if ScanLow < xyDeg[val] < ScanHigh:
                TNB.append(xyDeg[val])
                TNBI.append(xyObs[val])
            else:
                TNBI.append(0)
                TNB.append(xyDeg[val])

print(time.time() - t0)

pltAll = plt.figure(2)
plt.plot(xyDeg,xyObs, color='orange')
#plt.plot(xyDeg,xyD2, color='red')
#plt.plot(xyDeg,xyD1, color='blue')
#plt.plot(ScanDeg,ScanInts, color='green')
plt.plot(PeakPos,PeakInts, "x", color='blue')
plt.ylabel("Intensity")
plt.xlabel("2\u03B8 (degrees)")    

pltNewDat = plt.figure(3)
plt.plot(xyDeg,xyObs, color='blue')
plt.plot(TNB,TNBI, color='green')
plt.ylabel("Intensity")
plt.xlabel("2\u03B8 (degrees)")

#pltNewDat = plt.figure(4)
#plt.plot(xyDeg,xyOb, color='blue')
#plt.plot(xyDeg,xyObs, color='orange')
#plt.ylabel("Intensity")
#plt.xlabel("2\u03B8 (degrees)")
#plt.legend(['Obs','Smoothed'])
#
#pltNewDat = plt.figure(4)
#plt.plot(xyDeg,xyOb, color='blue')
#plt.ylabel("Intensity")
#plt.xlabel("2\u03B8 (degrees)")

plt.show()