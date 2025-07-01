import numpy as genfromtxt
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences

#Ask for xy file
#Current code will be nested in xy file
xyfile = input('What is the xy file? ')

#Open xy file and turn into an array
xyOpen = np.genfromtxt(xyfile, dtype=float, delimiter='\t')

#Split into radians and observed
xyDeg = xyOpen[:,0]
xyObs = xyOpen[:,1]

#Get first and second "derivatives"
Peaks = []

peaks, _ = find_peaks(xyObs, prominence=1, width = 4)
plt.plot(xyObs)
plt.plot(peaks, xyObs[peaks], "x")
plt.show()