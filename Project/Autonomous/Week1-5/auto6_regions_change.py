# OPTION 6 (Local): Regions of Greatest Change (Integrals)
'''
Take difference of integration from previous
Find regions of greatest change
Scan those regions
Repeat
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyFAI
import fabio

def main():
    # Setup paths
    raw_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth/images")
    xy_output_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth")
    ai = pyFAI.load("/Users/Melody/Documents/SLAC/Project/Autonomous/AutomatedScanTest/LaB6_2DetSetup.poni")
    scan_df = pd.read_csv("~/Documents/SLAC/Project/Data/Ground_truth/TiO2_heating_pilatus_only_scan1.csv")
    scan_df.columns = scan_df.columns.str.strip() # clean commas
    print(scan_df)

    # Integrate each .raw image into .xy if needed
    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith(".raw")])
    for raw_file in raw_files:
        raw_path = os.path.join(raw_folder, raw_file)
        xy_name = raw_file.replace(".raw", ".xy")
        xy_path = os.path.join(xy_output_folder, xy_name)

        if os.path.exists(xy_path):
            continue  # Skip if already processed

        try:
            with open(raw_path, 'rb') as im:
                arr = np.frombuffer(im.read(), dtype='int32')
            arr.shape = (195, 487)
            ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_path)
            df = pd.read_csv(xy_path, skiprows=23, header=None, delim_whitespace=True)
            df.columns = ['2theta_deg', 'I']
            df.to_csv(xy_path, sep='\t', index=False, float_format='%.6f')
            print(f"Saved: {xy_path}")
        except Exception as e:
            print(f"Failed to process {raw_file}: {e}")

    # Begin scan logic
    files = sorted([f for f in os.listdir(xy_output_folder) if f.endswith(".xy")])
    data_folder = xy_output_folder
    previous_integration = None

    print(f"Found {len(files)} .xy files to process in {data_folder}")

    def smooth(xyDeg, xyOb):
        mean_I = np.mean(xyOb)
        std_I = np.std(xyOb)
        snr = mean_I / std_I if std_I != 0 else 0
        SmNum = 2
        threshold = (mean_I + (std_I if snr < 3 else 0.5 * std_I if snr < 10 else 0.2 * std_I)) / 2
        if len(xyOb) < 2 * SmNum + 1:
            return list(xyOb), threshold
        smoothed = []
        for i in range(SmNum, len(xyOb) - SmNum):
            smoothed.append(np.mean(xyOb[i - SmNum:i + SmNum + 1]))
        return smoothed, threshold

    for scan_number, filename in enumerate(files):
        print(f"\nProcessing {filename}...")
        path = os.path.join(data_folder, filename)

        try:
            xy = np.genfromtxt(path, delimiter="\t", skip_header=1)
            xyDeg, xyOb = xy[20:, 0], xy[20:, 1]
            mask = xyDeg >= 14 # Ignore noise & false peaks
            xyDeg = xyDeg[mask]
            xyOb = xyOb[mask]
        except Exception as e:
            print(f"Error reading {filename}:", e)
            continue

        xySmoothed, threshold = smooth(xyDeg, xyOb)
        integration = np.array(xySmoothed)

        # Compare with previous 
        if previous_integration is not None:
            min_len = min(len(integration), len(previous_integration))
            current = integration[:min_len]
            previous = previous_integration[:min_len]
            diff = np.abs(current - previous)

            diff_peaks = []
            diff_threshold = np.mean(diff) + np.std(diff)
            for i in range(4, len(diff) - 4):
                if (
                    diff[i - 4] < diff[i - 3] < diff[i - 2] < diff[i - 1] < diff[i] >
                    diff[i + 1] > diff[i + 2] > diff[i + 3] > diff[i + 4]
                    and diff[i] > diff_threshold
                ):
                    diff_peaks.append(xyDeg[i])
            # filter noise
            diff_peaks = [a for a in diff_peaks if a >= 14]

            if diff_peaks:
                scan_windows = []
                scan_high = 0
                for angle in sorted(set(diff_peaks)):
                    start = max(scan_high, angle - 0.5)
                    stop = angle + 0.5
                    if stop - start >= 0.1:
                        scan_windows.append((start, stop))
                        scan_high = stop
                print(f"{len(scan_windows)} windows with strong change")
        
        try: # Grab temperature for this scan
            ctemp = scan_df["CTEMP"].iloc[scan_number]
        except IndexError:
            ctemp = None  # fallback if CSV has fewer rows than scans

        if ctemp is not None:
            plt.title(f"Scan {scan_number + 1} at {ctemp:.1f}˚C")
        else:
            plt.title(f"Scan {scan_number + 1}")

        # Graph scans
        plt.figure()
        plt.plot(xyDeg[2:-2], xySmoothed, label='Smoothed')
        if previous_integration is not None and diff_peaks:
            angles = diff_peaks
            # nearest neighbors 
            intensities = [xyOb[np.argmin(np.abs(xyDeg - a))] for a in angles]
            plt.scatter(angles, intensities, c='red', s=60, marker='*', label='Change')
        if ctemp is not None:
            plt.title(f"Change Regions (Scan {scan_number + 1} at {ctemp:.1f}˚C)") 
        else:
            plt.title(f"Change Regions (Scan {scan_number + 1})")
        plt.xlabel("2θ (°)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()

        plt.show(block=False) # open plot without blocking
        plt.pause(0.5)
        plt.close() 

        # Update for next scan
        previous_integration = integration.copy()

if __name__ == "__main__":
    main()