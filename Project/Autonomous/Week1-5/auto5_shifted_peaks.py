# OPTION 5 (local): Scan Shifted Peaks (Distance x)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import pyFAI
import fabio
import time


def main():
    # Setup paths
    raw_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth/images")
    xy_output_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth")
    ai = pyFAI.load("/Users/Melody/Documents/SLAC/Project/Autonomous/AutomatedScanTest/LaB6_2DetSetup.poni")
    scan_df = pd.read_csv("~/Documents/SLAC/Project/Data/Ground_truth/TiO2_heating_pilatus_only_scan1.csv")
    scan_df.columns = scan_df.columns.str.strip()  # clean commas
    print(scan_df)

    # Convert RAW files to .xy if not already done 
    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith(".raw")])
    for raw_file in raw_files:
        raw_path = os.path.join(raw_folder, raw_file)
        xy_name = raw_file.replace(".raw", ".xy")
        xy_path = os.path.join(xy_output_folder, xy_name)

        if not os.path.exists(xy_path):
            try:
                with open(raw_path, 'rb') as im:
                    arr = np.frombuffer(im.read(), dtype='int32') # headers
                arr.shape = (195, 487)
                ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_path)
                df = pd.read_csv(xy_path, skiprows=23, header=None, delim_whitespace=True)
                df.columns = ['2theta_deg', 'I']
                df.to_csv(xy_path, sep='\t', index=False, float_format='%.6f')
                print(f"Saved: {xy_path}")
            except Exception as e:
                print(f"Failed to process {raw_file}: {e}")

    # Settings 
    files = sorted([f for f in os.listdir(xy_output_folder) if f.endswith(".xy")])
    shift_threshold = 0.02 # in degrees (2θ)
    angle_tolerance = 0.3

    # Fit peak: independent, peak amplitude, mean (center), standard deviation
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Smooth noise
    def smooth(xyDeg, xyOb):
        mean_I = np.mean(xyOb)
        std_I = np.std(xyOb)
        snr = mean_I / std_I if std_I != 0 else 0
        SmNum = 2
        threshold = (mean_I + (std_I if snr < 3 else 0.5 * std_I if snr < 10 else 0.2 * std_I)) / 2

        if len(xyOb) < 2 * SmNum + 1:
            return list(xyOb), threshold

        smoothed = [xyOb[0]]
        for i in range(1, SmNum - 1): smoothed.append(xyOb[i])
        for i in range(SmNum, len(xyOb) - SmNum):
            smoothed.append(np.mean(xyOb[i - SmNum:i + SmNum + 1]))
        for i in range(SmNum + 1, 0, -1): smoothed.append(xyOb[-SmNum])
        return smoothed, threshold

    # Set threshold to detect peaks based on intensity
    def detect_peaks(xyDeg, xyOb, threshold):
        peaks = []
        for i in range(4, len(xyOb) - 4):
            if (
                xyOb[i - 4] < xyOb[i - 3] < xyOb[i - 2] < xyOb[i - 1] < xyOb[i] >
                xyOb[i + 1] > xyOb[i + 2] > xyOb[i + 3] > xyOb[i + 4] and
                xyOb[i] > threshold
            ):
                peaks.append((xyDeg[i], xyOb[i]))
        peaks = [p for p in peaks if p[0] >= 14]
        return peaks

    # Save fitted peaks CSV
    output_csv = "fitted_peaks_shifted.csv"
    with open(output_csv, "w") as f:
        f.write("scan,ctemp,angle,intensity,shift,fitted_center,fitted_sigma,is_shifted\n")

    # Main loop
    previous_peaks = []
    for scan_number, filename in enumerate(files):
        shifted_peaks = []

        print(f"\nProcessing scan {scan_number + 1}: {filename}...")
        path = os.path.join(xy_output_folder, filename)
        try:
            xy = np.genfromtxt(path, delimiter="\t", skip_header=1)
            xyDeg, xyOb = xy[20:, 0], xy[20:, 1]
            mask = xyDeg >= 14  # Ignore noise & false peaks
            xyDeg = xyDeg[mask]
            xyOb = xyOb[mask]
        except Exception as e:
            print(f"Error reading {filename}:", e)
            continue

        xySmoothed, threshold = smooth(xyDeg, xyOb)
        current_peaks = detect_peaks(xyDeg, xySmoothed, threshold)
        N = 3 # Only keep top N strongest peaks
        current_peaks = sorted(current_peaks, key=lambda p: p[1], reverse=True)[:N]

        # Comparison of shifted peaks
        if previous_peaks:
            for curr_angle, curr_int in current_peaks:
                # Find closest previous peak
                closest = None
                min_diff = angle_tolerance
                for prev_angle, _ in previous_peaks:
                    diff = abs(curr_angle - prev_angle)
                    if diff < min_diff:
                        closest = prev_angle
                        min_diff = diff
                if closest is not None:
                    shift = curr_angle - closest
                    if abs(shift) > shift_threshold:
                        shifted_peaks.append((curr_angle, curr_int, shift))
                        print(f"Peak shift: {closest:.2f} → {curr_angle:.2f}  (∆ = {shift:.3f})")

        previous_peaks = current_peaks.copy()

        try:
            ctemp = scan_df["CTEMP"].iloc[scan_number]
        except IndexError:
            ctemp = None
        if ctemp is not None:
            plt.title(f"Scan {scan_number + 1} at {ctemp:.1f}˚C")
        else:
            plt.title(f"Scan {scan_number + 1}")

        # Fit all detected peaks, marking whether they shifted
        for angle, intensity in current_peaks:
            shift = 0.0
            is_shifted = False

            # Check if it's shifted
            for s_angle, s_intensity, s_shift in shifted_peaks:
                if abs(angle - s_angle) < 0.01:
                    shift = s_shift
                    is_shifted = True
                    break

            mask = (xyDeg >= angle - 0.2) & (xyDeg <= angle + 0.2)
            x_fit = xyDeg[mask]
            y_fit = np.array(xySmoothed)[mask]

            if len(x_fit) < 5:
                continue

            try:
                popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[intensity, angle, 0.05])
                fitted_center, fitted_sigma = popt[1], popt[2]
                print(f"  → Fitted center: {fitted_center:.3f}, sigma: {fitted_sigma:.3f}, shifted: {is_shifted}")
                with open(output_csv, "a") as f:
                    f.write(f"{scan_number},{ctemp},{angle},{intensity},{shift},{fitted_center},{fitted_sigma},{is_shifted}\n")
            except RuntimeError:
                print("  → Fit failed")

        # === Update Plot ===
        plt.figure()
        plt.plot(xyDeg, xySmoothed, label="Smoothed Intensity")

        # Separate peak types for plotting
        shifted_angles = []
        shifted_ints = []
        normal_angles = []
        normal_ints = []

        for angle, intensity in current_peaks:
            if any(abs(angle - s[0]) < 0.01 for s in shifted_peaks):
                shifted_angles.append(angle)
                shifted_ints.append(intensity)
            else:
                normal_angles.append(angle)
                normal_ints.append(intensity)

        # Plot all peaks with distinction
        plt.scatter(normal_angles, normal_ints, color='gray', label="Non-Shifted Peaks", s=20)
        plt.scatter(shifted_angles, shifted_ints, color='red', marker='*', s=60, label="Shifted Peaks")
        plt.axhline(threshold, color='green', linestyle='--', alpha=0.3, label="Threshold")

        if ctemp is not None:
            plt.title(f"Shifted Peaks (Scan {scan_number + 1} at {ctemp:.1f}˚C)")
        else:
            plt.title(f"Shifted Peaks (Scan {scan_number + 1})")

        plt.xlabel("2θ (°)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()


if __name__ == "__main__":
    main()