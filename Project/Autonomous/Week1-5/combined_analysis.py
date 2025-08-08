# Unified Peak Qualification and Classification
# Combines: peak subset, shift tracking, change regions, fitting & phase labeling
# July 2025

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pyFAI
import fabio

# Settings
raw_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth/images")
xy_output_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth")
poni_file = os.path.expanduser("~/Documents/SLAC/Project/Autonomous/AutomatedScanTest/LaB6_2DetSetup.poni")
scan_csv = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth/TiO2_heating_pilatus_only_scan1.csv")
output_csv = "combined_peaks_classified.csv"

shift_threshold = 0.02
angle_tolerance = 0.3
TOP_N = 3

# Functions 
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def smooth(y):
    mean_I = np.mean(y)
    std_I = np.std(y)
    snr = mean_I / std_I if std_I != 0 else 0
    SmNum = 2
    threshold = (mean_I + (std_I if snr < 3 else 0.5 * std_I if snr < 10 else 0.2 * std_I)) / 2
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
        if y[i - 4] < y[i - 3] < y[i - 2] < y[i - 1] < y[i] > y[i + 1] > y[i + 2] > y[i + 3] > y[i + 4] and y[i] > threshold:
            peaks.append((x[i], y[i]))
    return peaks

def write_header(path):
    with open(path, "w") as f:
        f.write("scan,ctemp,angle,intensity,shift,fitted_center,fitted_sigma,is_shifted,is_changing\n")

def append_row(path, data):
    with open(path, "a") as f:
        f.write(",".join(str(x) for x in data) + "\n")

# Main
def main():
    ai = pyFAI.load(poni_file)
    scan_df = pd.read_csv(scan_csv)
    scan_df.columns = scan_df.columns.str.strip()

    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith(".raw")])
    previous_peaks = []
    previous_integral = None

    write_header(output_csv)

    for scan_number, raw_file in enumerate(raw_files):
        print(f"\n[Scan {scan_number}] {raw_file}")
        raw_path = os.path.join(raw_folder, raw_file)
        xy_name = raw_file.replace(".raw", ".xy")
        xy_path = os.path.join(xy_output_folder, xy_name)

        # Integrate if not already done
        if not os.path.exists(xy_path):
            with open(raw_path, 'rb') as f:
                arr = np.frombuffer(f.read(), dtype='int32')
            arr.shape = (195, 487)
            ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_path)
            df = pd.read_csv(xy_path, skiprows=23, header=None, delim_whitespace=True)
            df.columns = ['2theta_deg', 'I']
            df.to_csv(xy_path, sep='\t', index=False, float_format='%.6f')

        # Load data
        try:
            xy = np.genfromtxt(xy_path, delimiter="\t", skip_header=1)
            xyDeg, xyOb = xy[20:, 0], xy[20:, 1]
            mask = xyDeg >= 14
            xyDeg = xyDeg[mask]
            xyOb = xyOb[mask]
        except Exception as e:
            print(f"  ✗ Failed to read {xy_path}: {e}")
            continue

        smoothed, threshold = smooth(xyOb)
        peaks = detect_peaks(xyDeg, smoothed, threshold)
        peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:TOP_N]

        # Track shift
        shifted = []
        if previous_peaks:
            for curr_angle, curr_int in peaks:
                closest = min(previous_peaks, key=lambda p: abs(p[0] - curr_angle), default=None)
                if closest and abs(curr_angle - closest[0]) > shift_threshold:
                    shifted.append((curr_angle, curr_int, curr_angle - closest[0]))

        # Region of change
        changing_regions = []
        current_integral = np.array(smoothed)
        if previous_integral is not None:
            min_len = min(len(current_integral), len(previous_integral))
            diff = np.abs(current_integral[:min_len] - previous_integral[:min_len])
            diff_threshold = np.mean(diff) + np.std(diff)
            for peak in peaks:
                idx = np.argmin(np.abs(xyDeg - peak[0]))
                if diff[idx] > diff_threshold:
                    changing_regions.append(peak[0])

        previous_peaks = peaks.copy()
        previous_integral = current_integral.copy()

        # Fit and write output
        try:
            ctemp = scan_df["CTEMP"].iloc[scan_number]
        except:
            ctemp = -1

        for angle, intensity in peaks:
            is_shifted = any(abs(angle - s[0]) < 0.01 for s in shifted)
            is_changing = any(abs(angle - c) < 0.01 for c in changing_regions)
            x_fit = xyDeg[(xyDeg >= angle - 0.2) & (xyDeg <= angle + 0.2)]
            y_fit = np.array(smoothed)[(xyDeg >= angle - 0.2) & (xyDeg <= angle + 0.2)]

            if len(x_fit) < 5:
                continue
            try:
                popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[intensity, angle, 0.05])
                fitted_center, fitted_sigma = popt[1], popt[2]
                shift_val = next((s[2] for s in shifted if abs(angle - s[0]) < 0.01), 0.0)
                append_row(output_csv, [scan_number, ctemp, angle, intensity, shift_val, fitted_center, fitted_sigma, is_shifted, is_changing])
                print(f"  ✓ Angle: {angle:.2f} σ={fitted_sigma:.3f} shift={shift_val:.3f}")
            except:
                print(f"  ✗ Fit failed at angle {angle:.2f}")

if __name__ == "__main__":
    main()
