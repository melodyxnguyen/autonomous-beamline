# Smart Peak Qualifier
# Combining logic to filter scans 

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyFAI
import fabio
import time

"""
Return a list of peaks that meet any combination of:
- Significant intensity change (Option 2)
- Growing or shrinking trend (Option 3 and 4)
"""

def main():
    # Setup paths
    raw_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth/images")
    xy_output_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth")
    scan_df = pd.read_csv("~/Documents/SLAC/Project/Data/Ground_truth/TiO2_heating_pilatus_only_scan1.csv")
    scan_df.columns = scan_df.columns.str.strip() # clean commas
    print(scan_df)

    # Load calibration file
    ai = pyFAI.load("/Users/Melody/Documents/SLAC/Project/Autonomous/AutomatedScanTest/LaB6_2DetSetup.poni")

    # List of raw files
    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith(".raw")])

    # Integrate each image
    for raw_file in raw_files:
        raw_path = os.path.join(raw_folder, raw_file)
        xy_name = raw_file.replace(".raw", ".xy")
        xy_path = os.path.join(xy_output_folder, xy_name)

        try:
            # Read raw image
            with open(raw_path, 'rb') as im:
                arr = np.frombuffer(im.read(), dtype='int32')
            arr.shape = (195, 487)

            # Integrate to 1D
            res = ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_path)

            # Optionally, convert to tab-delimited file
            df = pd.read_csv(xy_path, skiprows=23, header=None, delim_whitespace=True)
            df.columns = ['2theta_deg', 'I']
            df.to_csv(xy_path, sep='\t', index=False, float_format='%.6f')

            print(f"Saved: {xy_path}")
            
        except Exception as e:
            print(f"Failed to process {raw_file}: {e}")

    # Refresh list of integrated .xy files
    files = sorted([f for f in os.listdir(xy_output_folder) if f.endswith(".xy")])
    data_folder = xy_output_folder  

    print(f"Found {len(files)} .xy files to process in {data_folder}")

    # Thresholds (for what counts as a change) 
    intensity_change_threshold = 0.15   # 15% change is interesting
    angle_tolerance = 0.3   # Peaks must be close to compare
    growth_threshold = 0.1    # +10% increase = growing
    shrink_threshold = -0.1     # -10% decrease = shrinking

    # === Helper functions ===
    def smooth(xyDeg, xyOb):
        if len(xyDeg) == 0 or len(xyOb) == 0:
            print("Warning: Empty input data for smoothing")
            return [], 0
        if len(xyDeg) != len(xyOb):
            print("Error: Angle and intensity arrays must have same length")
            return [], 0
        
        mean_I = np.mean(xyOb)
        std_I = np.std(xyOb)
        snr = mean_I / std_I if std_I != 0 else 0
        SmNum = 2
        threshold = (mean_I + (std_I if snr < 3 else 0.5 * std_I if snr < 10 else 0.2 * std_I)) / 2
        if len(xyOb) < 2 * SmNum + 1:
            return list(xyOb), threshold
        smoothed = [xyOb[0]]
        for i in range(1, SmNum - 1):
            smoothed.append(xyOb[i])
        for i in range(SmNum, len(xyOb) - SmNum):
            smoothed.append(np.mean(xyOb[i - SmNum:i + SmNum + 1]))
        for i in range(SmNum + 1, 0, -1):
            smoothed.append(xyOb[-SmNum])
        return smoothed, threshold

    def detect_peaks(xyDeg, xyOb, threshold):
        if len(xyDeg) == 0 or len(xyOb) == 0 or len(xyOb) < 9:
            return []
        if len(xyDeg) != len(xyOb):
            return []
        peaks = []
        for i in range(4, len(xyOb) - 4): 
            if (
                xyOb[i - 4] < xyOb[i - 3] < xyOb[i - 2] < xyOb[i - 1] < xyOb[i] > 
                xyOb[i + 1] > xyOb[i + 2] > xyOb[i + 3] > xyOb[i + 4] and 
                xyOb[i] > threshold
            ):
                peaks.append((xyDeg[i], xyOb[i]))
        return peaks

    def qualify_peaks(current_peaks, previous_peaks):
        qualified = []
        if not previous_peaks: # Check if there's old data to compare
            return qualified
        
        # Loop over new peaks (where & how tall)
        for angle, intensity in current_peaks:
            closest = None
            min_diff = 0.3
            # find closest old peak to compare, based on angle
            for prev_angle in previous_peaks:
                if abs(angle - prev_angle) < min_diff:
                    closest = prev_angle
                    min_diff = abs(angle - prev_angle)
            if closest:
                prev_intensity = previous_peaks[closest]
                if prev_intensity == 0:
                    continue
                # How much did this peak grow or shrink compared to last?
                change = (intensity - prev_intensity) / prev_intensity
                
                if ( # Filter peaks using combined logic
                    abs(change) > intensity_change_threshold 
                    or change > growth_threshold 
                    or change < shrink_threshold 
                    or abs(angle - closest) > 0.1 # shift
                ):
                    qualified.append((angle, intensity, change))
        return qualified


    # Main Loop
    previous_peaks = {}
    for scan_number, filename in enumerate(files):
        print(f"Processing {filename}...")
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
        smoothed, threshold = smooth(xyDeg, xyOb)
        peaks = detect_peaks(xyDeg, smoothed, threshold)
        qualified_peaks = qualify_peaks(peaks, previous_peaks)
        previous_peaks = {a: i for a, i in peaks}

        try: # Grab temperature for this scan
            ctemp = scan_df["CTEMP"].iloc[scan_number]
        except IndexError:
            ctemp = None  # fallback if CSV has fewer rows than scans

        if scan_number < len(scan_df):
            ctemp = scan_df["CTEMP"].iloc[scan_number]
        else:
            ctemp = None

        plt.figure()
        plt.plot(xyDeg, smoothed, label="Smoothed Data")
        plt.axhline(threshold, color='green', linestyle='--', alpha=0.3, label='Threshold')
        if qualified_peaks:
            angles, intensities, _ = zip(*qualified_peaks)
            plt.scatter(angles, intensities, color='red', label="Interesting Peaks")
        
        plt.title(f"Autonomous (Scan {scan_number + 1} at {ctemp:.1f}˚C)")
        plt.xlabel("2θ (degrees)")
        plt.ylabel("Intensity")
        plt.legend()

        plt.show(block=False) # open plot without blocking
        plt.pause(0.5)
        plt.close() 

    plt.show()

    # Collect labeled peak data for ML training
    with open("real_peak_data.csv", "a") as f:
        for angle, intensity, change in qualified_peaks:
            label = 1  # interesting
            f.write(f"{angle},{intensity},{change},{label}\n")
        for angle, intensity in peaks:
            if not any(abs(angle - qa) < 0.1 for qa, _, _ in qualified_peaks):
                label = 0  # not interesting
                f.write(f"{angle},{intensity},0,{label}\n")

if __name__ == "__main__":
    main()