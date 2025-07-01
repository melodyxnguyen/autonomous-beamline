### General code outline ###

#This is for step zero of the code
#Intentions are to take an incoming pattern and identify the peak positions from a poor scan
#This code will then tell a second detector to move to those positions +/- 2theta

## Steps used to make the code work
#1 - Accept xy pattern, converting it into an x,y
#2 - Take dirty derivative of the pattern to get rise vs run
#This will tell us the general area of all peaks that are ~0 from + to -
#3 - Adjust magnitudes of which peaks are caught by test
#3a - Possibly use early data as a background identifier
#4 - 

#Import all the python libraries needed
import time
import os
import shutil
import pandas as pd
import numpy as genfromtxt
import numpy as np
import matplotlib.pyplot as plt

#Ask for xy file
#Current code will be nested in xy file
xyfile = input('What is the xy file? ')

#Open xy file and turn into an array
xyOpen = np.genfromtxt(xyfile, dtype=float, delimiter='\t')

#Split into radians and observed
xyDeg = xyOpen[:,0]
xyObs = xyOpen[:,1]

#Get first and second "derivatives"
xyD1 = [0]
xyD2 = [0]
PeakInts = []
PeakPos = []
ScanDeg = []
ScanInts = []

for row in range(0, len(xyObs)-1):
    xyDiff = xyObs[row + 1] - xyObs[row]
    xyD1.append(xyDiff)

for row in range(0, len(xyObs)-1):
    xyDiff = xyD1[row + 1] - xyD1[row]
    xyD2.append(xyDiff)

ScanHigh = 0
for row in range(0, len(xyObs)-1):
    if row > 3 and row < len(xyD1) - 3 and xyD1[row - 1] > xyD1[row] > xyD1[row + 1]\
    and xyObs[row - 2] < xyObs[row - 1] < xyObs[row] > xyObs[row + 1] > xyObs[row + 2]\
    and 0 > xyD2[row - 1] and xyD2[row + 1] < 0 and xyD2[row] < 0:
        PeakInt = xyObs[row]
        PeakInts.append(PeakInt)
        PeakP = xyDeg[row]
        PeakPos.append(PeakP)
        
        if xyDeg[row] - 0.5 > ScanHigh:
            ScanLow = xyDeg[row] - 0.5
        else:
            ScanLow = ScanHigh
        ScanHigh = xyDeg[row] + 0.5
        ScanDeg.append(ScanLow)
        ScanDeg.append(ScanHigh)
        ScanInts.append(xyObs[row])
        ScanInts.append(xyObs[row])


pltAll = plt.figure(2)
plt.plot(xyDeg,xyObs, color='orange')
plt.plot(xyDeg,xyD1, color='blue')
plt.plot(xyDeg,xyD2, color='red')
plt.plot(ScanDeg,ScanInts, color='green')
plt.plot(PeakPos,PeakInts, "x")
plt.legend(['Obs','D1','D2'])    

plt.show()