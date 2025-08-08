###Early peak picking and 2theta scan code
###By Monty Cosby and Kevin Stone of SSRL
###To be used at beamline 2-1 in a 2 detector config
###April 2022

#Import all the python libraries needed
import time
import os
import shutil
import pandas as pd
import numpy as genfromtxt
import numpy as np
import matplotlib.pyplot as plt

t0 = time.time()

#Ask for xy file
#Current code will be nested in xy file
xyfile = input('What is the xy file? ')

#Open xy file and turn into an array
xyOpen = np.genfromtxt(xyfile, dtype=float, delimiter='\t')

#Split into radians and observed
xyDeg = xyOpen[:,0]
xyOb = xyOpen[:,1]

SmNum = 3

xyObs = [xyOb[0]]
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

#for row in range(0, len(xyObs)-1):
#    xyDiff = xyD1[row + 1] - xyD1[row - 1]
#    xyD2.append(xyDiff)
#xyD2.append(0)

ScanHigh = 0
for row in range(0, len(xyObs)-1):
    if row > 3 and row < len(xyObs) - 3\
    and xyObs[row - 5] < xyObs[row - 4] < xyObs[row - 3] < xyObs[row - 2] < xyObs[row - 1] < xyObs[row] > xyObs[row + 1]\
    and xyObs[row + 5] < xyObs[row + 4] < xyObs[row + 3] < xyObs[row + 2] < xyObs[row + 1]:
        PeakInt = xyObs[row]
        PeakInts.append(PeakInt)
        PeakP = xyDeg[row]
        PeakPos.append(PeakP)
        
        if xyDeg[row] - 0.375 > ScanHigh:
            ScanLow = xyDeg[row] - 0.375
        else:
            ScanLow = ScanHigh
        ScanHigh = xyDeg[row] + 0.375
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

pltAll = plt.figure(1)
plt.plot(xyDeg,xyObs, color='orange')
#plt.plot(xyDeg,xyD2, color='red')
#plt.plot(xyDeg,xyD1, color='blue')
#plt.plot(ScanDeg,ScanInts, color='green')
plt.plot(PeakPos,PeakInts, "x", color='blue')
plt.ylabel("Intensity")
plt.xlabel("2\u03B8 (degrees)")    

pltNewDat = plt.figure(2)
plt.plot(xyDeg,xyObs, color='blue')
plt.plot(TNB,TNBI, color='green')
plt.ylabel("Intensity")
plt.xlabel("2\u03B8 (degrees)")

pltNewDat = plt.figure(3)
plt.plot(xyDeg,xyOb, color='blue')
plt.plot(xyDeg,xyObs, color='orange')
plt.ylabel("Intensity")
plt.xlabel("2\u03B8 (degrees)")
plt.legend(['Obs','Smoothed'])

pltNewDat = plt.figure(4)
plt.plot(xyDeg,xyOb, color='blue')
plt.ylabel("Intensity")
plt.xlabel("2\u03B8 (degrees)")

plt.show()