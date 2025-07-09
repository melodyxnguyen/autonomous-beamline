# Smart Peak Qualifier
# Combining logic to filter scans 

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Return a list of peaks that meet any combination of:
- Significant intensity change (Option 2)
- Growing or shrinking trend (Option 3 and 4)
"""

# Finding data
data_folder = os.path.expanduser("~/Documents/SLAC/Project/Data/Ground_truth")
files = sorted([f for f in os.listdir(data_folder) if f.endswith(".xy")])

# Thresholds (for what counts as a change) 
intensity_change_threshold = 0.15   # 15% change is interesting
angle_tolerance = 0.3   # Peaks must be close to compare
growth_threshold = 0.1    # +10% increase = growing
shrink_threshold = -0.1     # -10% decrease = shrinking

# === Helper functions ===
def smooth(xyDeg, xyOb):
    if len(xyDeg) == 0 or len(xyOb) == 0:
        return [], 0
    if len(xyDeg) != len(xyOb):
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


# === Main Loop ===
previous_peaks = {}
for scan_number, filename in enumerate(files):
    print(f"Processing {filename}...")
    path = os.path.join(data_folder, filename)
    try:
        xy = np.genfromtxt(path, delimiter="\t", skip_header=1)
        xyDeg, xyOb = xy[20:, 0], xy[20:, 1]
    except Exception as e:
        print(f"Error reading {filename}:", e)
        continue
    smoothed, threshold = smooth(xyDeg, xyOb)
    peaks = detect_peaks(xyDeg, smoothed, threshold)
    qualified_peaks = qualify_peaks(peaks, previous_peaks)
    previous_peaks = {a: i for a, i in peaks}

    plt.figure()
    plt.plot(xyDeg, smoothed, label="Smoothed Data")
    if qualified_peaks:
        angles, intensities, _ = zip(*qualified_peaks)
        plt.scatter(angles, intensities, color='red', label="Interesting Peaks")
    plt.title(f"Scan {scan_number + 1}")
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.pause(0.1)

plt.show()
