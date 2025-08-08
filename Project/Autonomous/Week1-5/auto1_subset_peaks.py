# OPTION 1 (Local): Subset Peak Scanner

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pyFAI
import fabio
import time

# Paths
raw_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth/images")
xy_output_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth")
ai = pyFAI.load("/Users/Melody/Documents/SLAC/Project/Autonomous/AutomatedScanTest/LaB6_2DetSetup.poni")
scan_df = pd.read_csv("~/Documents/SLAC/Project/Data/Ground_truth/TiO2_heating_pilatus_only_scan1.csv")
scan_df.columns = scan_df.columns.str.strip() # clean commas
print(scan_df)

# Helper Functions
def readRAW(path):
    with open(path, 'rb') as f:
        arr = np.frombuffer(f.read(), dtype='int32')
    arr.shape = (195, 487)
    return arr

def smooth(x, y):
    mean_I = np.mean(y)
    std_I = np.std(y)
    snr = mean_I / std_I if std_I != 0 else 0
    SmNum = 2
    threshold = (mean_I + std_I) / 2 if snr < 3 else (mean_I + 0.5 * std_I) / 2 if snr < 10 else (mean_I + 0.2 * std_I) / 2
    if len(y) < 2 * SmNum + 1:
        return list(y), threshold
    smoothed = [y[0]]
    for i in range(1, SmNum - 1): smoothed.append(y[i])
    for i in range(SmNum, len(y) - SmNum):
        smoothed.append(np.mean(y[i - SmNum:i + SmNum + 1]))
    for i in range(SmNum + 1, 0, -1): smoothed.append(y[-SmNum])
    return smoothed, threshold

def detect_peaks(x, y, threshold):
    peaks = []
    for i in range(4, len(y) - 4):
        if (y[i - 4] < y[i - 3] < y[i - 2] < y[i - 1] < y[i] >
            y[i + 1] > y[i + 2] > y[i + 3] > y[i + 4] and y[i] > threshold):
            peaks.append((x[i], y[i]))
    return peaks

# Main Loop (Local)
def main():
    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith(".raw")])

    for scan_number, raw_file in enumerate(raw_files):
        print(f"\nProcessing {raw_file}...")

        raw_path = os.path.join(raw_folder, raw_file)
        xy_name = raw_file.replace(".raw", ".xy")
        xy_path = os.path.join(xy_output_folder, xy_name)

        # Integrate .raw → .xy
        arr = readRAW(raw_path)
        if not os.path.exists(xy_path):
            ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_path)
            df = pd.read_csv(xy_path, skiprows=23, header=None, delim_whitespace=True)
            df.columns = ['2theta_deg', 'I']
            df.to_csv(xy_path, index=False, sep='\t', float_format="%.6f")

        # Analyze .xy
        xy = np.genfromtxt(xy_path, delimiter="\t", skip_header=1)
        xyDeg, xyOb = xy[20:, 0], xy[20:, 1]
        mask = xyDeg >= 14 # Ignore noise & false peaks
        xyDeg = xyDeg[mask]
        xyOb = xyOb[mask]
        xySmoothed, threshold = smooth(xyDeg, xyOb)
        peaks = detect_peaks(xyDeg, xySmoothed, threshold)

        # Filter & sort strongest peaks
        peaks = [p for p in peaks if p[0] >= 11]
        strongest = sorted(peaks, key=lambda p: p[1], reverse=True)[:3]
        strongest = sorted(strongest, key=lambda p: p[0])

        # Simulate scanning each strongest peak
        scan_windows = []
        scan_high = 0
        for angle, intensity in strongest:
            start = max(scan_high, angle - 0.5)
            stop = angle + 0.5
            if stop - start >= 0.1:
                scan_windows.append((start, stop))
                scan_high = stop

        # Grab temperature for this scan
        try:
            ctemp = scan_df["CTEMP"].iloc[scan_number]
        except IndexError:
            ctemp = None  # fallback if CSV has fewer rows than scans
        if ctemp is not None:
            plt.title(f"Scan {scan_number + 1} at {ctemp:.1f}˚C")
        else:
            plt.title(f"Scan {scan_number + 1}")

        # Plot
        plt.figure()
        plt.plot(xyDeg, xySmoothed, label="Smoothed")
        for idx, (a, _) in enumerate(strongest):
            label = "Strongest Peaks" if idx == 0 else None
            plt.axvline(a, color='red', linestyle='--', label=label)
        if ctemp is not None:
            plt.title(f"Subset Peaks (Scan {scan_number + 1} at {ctemp:.1f}˚C)")
        else:
            plt.title(f"Subset Peaks (Scan {scan_number + 1})")
        plt.xlabel("2θ (°)")
        plt.ylabel("Intensity")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.show(block=False) # open plot without blocking
        plt.pause(0.5)
        plt.close() 
        
if __name__ == "__main__":
    main()