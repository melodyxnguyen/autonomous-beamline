# Smart Peak Qualifier
# Combining logic to filter scans 

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from auto2_intensity_change import detect_peaks, smooth
from scipy.spatial.distance import cdist

"""
Return a list of peaks that meet any combination of:
- Significant intensity change (Option 2)
- Growing or shrinking trend (Option 3 and 4)
"""

# 1. Finding the data
data_folder = os.path.expanduser("~/Documents/Project/Data/Ground_truth")
files = sorted([f for f in os.listdir(data_folder) if f.endswith(".xy")])

# 2. Settings for what counts as a change
angle_tolerance = 0.3   # Peaks must be close to compare
growth_threshold = 0.1    # More than 10% increase = growing
shrink_threshold = -0.1     # More than 10% decrease = shrinking
intensity_change_threshold = 0.15   # 15% change is interesting

previous_peaks = {}

# Helper function
def qualify_peaks(current_peaks, previous_peaks):
    qualified = []
    if not previous_peaks: # Check if there's old data to compare
        return qualified
    
    # Loop over all new peaks (where it happens, how tall)
    for angle, intensity in current_peaks:
        closest = None
        min_diff = 0.3
        for prev_angle in previous_peaks:
            if abs(angle - prev_angle) < min_diff:
                closest = prev_angle
                min_diff = abs(angle - prev_angle)

        # find closest old peak to compare, based on angle
        if closest:
            prev_intensity = previous_peaks[closest]
            if prev_intensity == 0:
                continue
            # How much did this peak grow or shrink compared to last?
            change = (intensity - prev_intensity) / prev_intensity

            # Combine all logic
            if (
                abs(change) > intensity_change_threshold
                or change > growth_threshold
                or change < shrink_threshold
                or abs(angle - closest) > 0.1
            ):
                qualified.append((angle, intensity, change))
    return qualified

# 3. Main loop
for filename in files:
    print(f"Looking at {filename}...")
    
    # Read the file
    path = os.path.join(data_folder, filename)
    try:
        data = np.genfromtxt(path, delimiter="\t", skip_header=1)
        if len(data) <30:
            continue # not enough data to be useful
    
        # Keep only useful parts of data
        angles = data[20:, 0]       # 20 values
        intensity = data[20:, 1]    # brightness at each angle 

    except Exception as e:
        print("Can't read this file:", e)
        continue

    # 4. Smooth data and find peaks
    smoothed, threshold = smooth(angles, intensity)
    peaks = detect_peaks(angles, smoothed, threshold) # List of (angle, intensity)

    # 5. Compare to last peaks and find interesting ones
    previous_peaks_dict = {a: i for a, i in previous_peaks}
    interesting_peaks = qualify_peaks(peaks, previous_peaks_dict)
    previous_peaks = {a: i for a, i in peaks}

    # 6. Graph
    plt.figure()
    plt.plot(angles, smoothed, label="Smoothed")

    if interesting_peaks:
        special_angles = [a for a, i, _ in interesting_peaks]
        special_intensity = [i for a, i, _ in interesting_peaks]
        plt.scatter(special_angles, special_intensity, color="red", label="Interesting Peaks")

    plt.title(f"Scan: {filename}")
    plt.xlabel("Angle (2θ)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)

plt.show()