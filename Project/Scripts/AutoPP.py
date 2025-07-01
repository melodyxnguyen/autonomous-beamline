'''
(Fast, low-res area detector)
Quickly scans full angular range.
Smooths raw data, peak detection, outputs .csv of useful scan regions.
No beamline dependencies :) Does NOT run scans, only processes.
Melody's updated AutoPP.py - June 2025 
'''

import numpy as np
from PIL import Image  # save image as .tif
import os
from matplotlib import pyplot as plt
import pyFAI, fabio  # beamline integration + image reader
import fnmatch
import pandas as pd
import time


# === USER INPUT ===
file = input('What is the raw file? ')
t0 = time.time()

# load .poni calibration file for geometry
ai = pyFAI.load("LaB6_2.poni")  

# Append .raw if not included
if not file.endswith(".raw"):
    file = file + ".raw"

# === READ .RAW FILE ===
with open(file, 'rb') as im:
    arr = np.frombuffer(im.read(), dtype='int32')  # read binary
arr.shape = (195, 487)  # reshape to detector dimensions


# === Optional: PLOT DETECTOR IMAGE ===
# Comment out later for automation
plt.figure(1)
plt.imshow(arr)
plt.clim(np.min(arr), np.max(arr) / 5)
plt.colorbar()
plt.pause(0.01)
plt.draw()

# === INTEGRATE TO 1D (.xy) ===
cur_xy = file.replace('.raw', '.xy')

if not os.path.exists(cur_xy):
    res = ai.integrate1d(arr, 250, unit="2th_deg", filename=cur_xy)
    df = pd.read_csv(cur_xy, skiprows=23, header=None, delim_whitespace=True)
    df.columns = ['radians', 'I']
    df.to_csv(cur_xy, index=False, float_format='%.6f', sep='\t')
    print(cur_xy)

# === READ AND SPLIT DATA ===
xy = np.genfromtxt(cur_xy, dtype=float, delimiter='\t')
xy = xy[~np.isnan(xy).any(axis=1)]  # Remove rows with NaNs
xyDeg, xyOb = xy[:, 0], xy[:, 1]

# === UPDATES: FEATURE EXTRACTION ===
mean_I = np.mean(xyOb)
std_I = np.std(xyOb)
max_I = np.max(xyOb)
snr = mean_I / std_I if std_I != 0 else 0  # signal-to-noise-ratio

# Automating SmNum and Intensity Threshold
if snr < 3:
    SmNum = 7  # strong (good for noisy patterns)
    intensity_threshold = mean_I + std_I
elif snr < 10:
    SmNum = 5
    intensity_threshold = mean_I + 0.5 * std_I
else:
    SmNum = 3  # light smoothing
    intensity_threshold = mean_I + 0.2 * std_I

# === SMOOTHING ===
def smooth(signal, window_size):
    smoothed = [signal[0]]
    for i in range(1, window_size - 1):
        smoothed.append(signal[i])
    for i in range(window_size, len(signal) - window_size):
        smoothed.append(np.mean(signal[i - window_size:i + window_size + 1]))
    for i in range(window_size + 1, 0, -1):
        smoothed.append(signal[-window_size])
    return smoothed

xyObs = smooth(xyOb, SmNum)


# === PEAK COUNT ESTIMATION ===
rough_peak_count = 0
for i in range(1, len(xyObs) - 1):
    if xyObs[i - 1] < xyObs[i] > xyObs[i + 1]:
        rough_peak_count += 1

# Debugging
print(f"Mean intensity: {mean_I:.2f}")
print(f"Standard deviation: {std_I:.2f}")
print(f"Max intensity: {max_I:.2f}")
print(f"SNR: {snr:.2f}")
print(f"Approximate peak count: {rough_peak_count}")
print(f"Chosen smoothing window (SmNum): {SmNum}")
print(f"Chosen intensity threshold: {intensity_threshold:.2f}")

# === Peak detection function ===
def detect_peaks(angles_deg, intensities_smoothed, threshold):
    """
    Detect peaks in the smoothed intensity array.
    angles_deg: list of 2θ angles (x-axis)
    intensities_smoothed: list of intensity values (y-axis)
    threshold: minimum height to consider a peak

    Returns: list of (angle, intensity) tuples
    """
    peaks = []
    for i in range(4, len(intensities_smoothed) - 4):
        if (
            intensities_smoothed[i - 4] < intensities_smoothed[i - 3] < intensities_smoothed[i - 2] < intensities_smoothed[i - 1] < intensities_smoothed[i] >
            intensities_smoothed[i + 1] > intensities_smoothed[i + 2] > intensities_smoothed[i + 3] > intensities_smoothed[i + 4]
            and intensities_smoothed[i] > threshold
        ):
            peaks.append((angles_deg[i], intensities_smoothed[i]))
    return peaks


# === PEAK DETECTION + PICKING (Original logic) ===
# Exports *_peaks.csv of peak postions 
xyD1, xyD2 = [], []
PeakInts, PeakPos = [], []
ScanDeg, ScanInts, TNB, TNBI = [], [], [], []

# Derivatives (used for optional debugging/plotting)
for i in range(0, len(xyObs)-1):
    xyD1.append(xyObs[i + 1] - xyObs[i - 1])
xyD1.append(0)
for i in range(0, len(xyObs)-1):
    xyD2.append(xyD1[i + 1] - xyD1[i - 1])
xyD2.append(0)

ScanHigh = 0
for i in range(0, len(xyObs)-1):
    if i > 3 and i < len(xyObs) - 4 and \
       xyObs[i - 4] < xyObs[i - 3] < xyObs[i - 2] < xyObs[i - 1] < xyObs[i] > xyObs[i + 1] and \
       xyObs[i + 4] < xyObs[i + 3] < xyObs[i + 2] < xyObs[i + 1]:

        PeakInt = xyObs[i]
        PeakPos.append(xyDeg[i])
        PeakInts.append(PeakInt)

        # Setting max to avoid overlaps
        ScanLow = max(ScanHigh, xyDeg[i] - 0.375) # start
        ScanHigh = xyDeg[i] + 0.5 # end 

        # Only high-intensity peaks regions
        ScanDeg += [ScanLow, ScanHigh]
        ScanInts += [PeakInt, PeakInt]

        for val in range(len(xyObs)):
            if ScanLow < xyDeg[val] < ScanHigh:
                TNB.append(xyDeg[val])
                TNBI.append(xyObs[val])
            else:
                TNB.append(xyDeg[val])
                TNBI.append(0)

# === Save peak summary ===
summary_file = file.replace(".raw", "_peaks.csv")
pd.DataFrame({"angle_2theta_deg": PeakPos, "intensity": PeakInts}).to_csv(summary_file, index=False)
print(f"Saved peak summary to {summary_file}")
print(f"Detected {len(PeakPos)} peaks.")

print(time.time() - t0)

# === PLOT RESULTS ===
plt.figure(2)
plt.plot(xyDeg, xyObs, color='orange')
plt.plot(PeakPos, PeakInts, "x", color='blue') 
plt.ylabel("Intensity")
plt.xlabel("2θ (degrees)")

plt.figure(3)
plt.plot(xyDeg, xyObs, color='blue')
plt.plot(TNB, TNBI, color='green')
plt.ylabel("Intensity")
plt.xlabel("2θ (degrees)")

plt.show()