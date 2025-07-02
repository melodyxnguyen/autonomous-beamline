# OPTION 3: Scan growing peaks

'''
find the peaks
compare to previous peaks
scan any peaks that are growing
repeat
'''

import numpy as np
from PIL import Image  # save image as .tif
import os
import sys
from matplotlib import pyplot as plt
import pyFAI, fabio  # beamline integration + image reader
import fnmatch
import pandas as pd
import time
from scipy.spatial.distance import cdist

# XDart imports
sys.path.append('C:\\Users\\Public\\Documents\\repos\\xdart')
from xdart.modules.pySSRL_bServer.bServer_funcs import *
from xdart.utils import get_from_pdi, get_motor_val, query, query_yes_no
from xdart.utils import read_image_file, smooth_img, get_fit, fit_images_2D
from xdart.modules.pySSRL_bServer.bServer_funcs import specCommand, wait_until_SPECfinished, get_console_output

# === Beamline Command Functions ===
def sendSPECcmd(command):
    ''' Send a command to SPEC and wait for it to finish '''
    try:
        specCommand(command, queue=True)
    except Exception as e:
        print(e)
        print(f"Command '{command}' not sent")
        sys.exit()
    print('Waiting for scan to finish...')
    wait_until_SPECfinished(polling_time=1)
    time.sleep(1)
    print('Done.\n')
    

def create_SPEC_file(path, name):
    """Create new SPEC scan file on remote system."""
    print(f'Creating new SPEC file for sample: {name} at {path}')
    sendSPECcmd(f'newfile {path}/{name}')


def set_PD_savepath(img_path):
    """Set Pilatus Detector save path on SPEC side."""
    print(f'Setting PD save path to {img_path}')
    sendSPECcmd(f'pd savepath {img_path}')


def run_sample_scan(start, stop, steps):
    """Run a 1D scan over 2theta with given step count."""
    command = f'ascan tth {start} {stop} {steps} 0.5'
    print(f'Running sample scan: {command}')
    sendSPECcmd(command)
    
def readRAW(file):
    # Append .raw if not included
    if not file.endswith(".raw"):
        file = file + ".raw"

    # === READ .RAW FILE ===
    with open(file, 'rb') as im:
        arr = np.frombuffer(im.read(), dtype='int32')  # read binary
    arr.shape = (195, 487)  # reshape to detector dimensions
    return arr
    

def smooth(xyDeg, xyOb):
    if len(xyDeg) == 0 or len(xyOb) == 0:
        print("Warning: Empty input data for smoothing")
        return [], 0
    if len(xyDeg) != len(xyOb):
        print("Error: Angle and intensity arrays must have same length")
        return [], 0

    mean_I = np.mean(xyOb)
    std_I = np.std(xyOb)
    snr = mean_I / std_I if std_I != 0 else 0  # signal-to-noise-ratio
 
    # Simplified threshold since same SmNum value
    SmNum = 2
    threshold = (mean_I + (std_I if snr < 3 else 0.5 * std_I if snr < 10 else 0.2 * std_I)) / 2

    # Handle edge cases for small arrays
    if len(xyOb) < 2 * SmNum + 1: # total pts = SmNum(left) + 1(center) + SmNum(right)
        print(f"Warning: Array too small for smoothing (length {len(xyOb)}, need {2*SmNum+1})")
        return list(xyOb), threshold
    
    smoothed = []

    # Handle edges and middle section consistently
    for i in range(len(xyOb)):
        start_idx = max(0, i - SmNum)
        end_idx = min(len(xyOb), i + SmNum + 1)
        smoothed.append(np.mean(xyOb[start_idx:end_idx]))
        
    return smoothed, threshold


def detect_peaks(xyDeg, xyObs, threshold):
    peaks = []
    # Check we have enough points for peak detection
    if len(xyObs) < 9:  # Need at least 9 points for 4-point buffer on each side
        return peaks
    
    for i in range(4, len(xyObs) - 4):
        # Check if current point is a local maximum with increasing/decreasing pattern
        if (xyObs[i - 4] < xyObs[i - 3] < xyObs[i - 2] < xyObs[i - 1] < xyObs[i] and
            xyObs[i] > xyObs[i + 1] > xyObs[i + 2] > xyObs[i + 3] > xyObs[i + 4] and
            xyObs[i] > threshold):
            peaks.append((xyDeg[i], xyObs[i]))
    return peaks


# TRYING DISTANCE MEASUREMENTS
def match_peaks_optimized(current_peaks, previous_peaks, angle_tolerance=0.3):
    """Optimized peak matching using scipy's distance functions."""
    if not previous_peaks:
        return []
    
    current_angles = np.array([angle for angle, _ in current_peaks])
    previous_angles = np.array(list(previous_peaks.keys()))
    
    # Calculate distance matrix
    distances = cdist(current_angles.reshape(-1, 1), previous_angles.reshape(-1, 1))
    
    matched_peaks = []
    for i, (angle, intensity) in enumerate(current_peaks):
        min_dist_idx = np.argmin(distances[i])
        min_dist = distances[i, min_dist_idx]
        
        if min_dist < angle_tolerance:
            closest_angle = previous_angles[min_dist_idx]
            matched_peaks.append((angle, intensity, closest_angle, previous_peaks[closest_angle]))
    
    return matched_peaks

# === Configure beamline paths ===
remote_path = "~/data/July2025/SelfDriving_algo3_test"
remote_wpath = "X:/bl2-1/July2025/SelfDriving_algo3_test"
remote_scan_path = f"{remote_path}/scans"
remote_img_path = f"{remote_path}/images"
remote_img_wpath = f"{remote_wpath}/images"
remote_xye_wpath = f"{remote_wpath}/xye"
spec_filename = f"SelfDriving_algo3_test_1"

# === Setup beamline environment ===
create_SPEC_file(remote_scan_path, spec_filename)
set_PD_savepath(remote_img_path)

# load .poni calibration file for geometry
ai = pyFAI.load("X:/bl2-1/July2025/Si_fixed_detector.poni")

sendSPECcmd("umv tth 35")

# === MAIN LOOP: Scan if peaks are growing ===
growth_threshold = 0.11 # adjustable %
angle_tolerance = 0.3
max_scans = 20
previous_peaks = {}

for scan_num in range(1, max_scans + 1):
    print(f"\n=== Scan {scan_num} ===")

    # Run the quick scan
    sendSPECcmd("loopscan 1 5 0")

    raw_file = f"{remote_img_wpath}/b_stone_{spec_filename}_scan1_0000.raw"
    xy_file = f"{remote_xye_wpath}/b_stone_{spec_filename}_scan1_0000.xy"
    
    # Read and process the raw data
    arr = readRAW(raw_file)
    if arr is None:
        print("Failed to read RAW file, skipping this scan")
        continue

    # Generate XY file if it doesn't exist
    if not os.path.exists(xy_file):
        res = ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_file)
        df = pd.read_csv(xy_file, skiprows=23, header=None, delim_whitespace=True)
        df.columns = ['2theta_deg', 'I']
        df.to_csv(xy_file, index=False, float_format='%.6f', sep='\t')
        print(f"Created: {xy_file}")

    # Load and process XY data
    try:
        xy = np.genfromtxt(xy_file, dtype=float, delimiter='\t', skip_header=1)
        #xy = xy[~np.isnan(xy).any(axis=1)]  # Remove rows with NaNs
        
        if len(xy) < 30:  # Need reasonable amount of data
            print("Insufficient data points. Skipping.")
            continue
        
        # Skip first 20 points if they're noisy
        start_idx = min(20, len(xy) // 4)  # Skip first 20 or 25% of data, whichever is smaller
        xyDeg, xyOb = xy[start_idx:, 0], xy[start_idx:, 1]
        
    except Exception as e:
        print(f"Error loading XY data: {e}")
        continue
    
    # Smooth data and detect peaks
    xyObs, threshold = smooth(xyDeg, xyOb)
    current_peaks = detect_peaks(xyDeg, xyObs, threshold)
    print(f"Found {len(current_peaks)} peaks")

    # Find growing peaks
    growing_peaks = []
    for angle, intensity in current_peaks:
        closest = None
        min_diff = angle_tolerance

        # Find closest previous peak
        for prev_angle in previous_peaks:
            if abs(angle - prev_angle) < min_diff:
                closest = prev_angle
                min_diff = abs(angle - prev_angle)
        
    # Find growing peaks
    growing_peaks = []
    if previous_peaks:
        matched_peaks = match_peaks_optimized(current_peaks, previous_peaks, angle_tolerance)
        
        for angle, intensity, prev_angle, prev_intensity in matched_peaks:
            if prev_intensity > 0:  # Avoid division by zero
                growth_rate = (intensity - prev_intensity) / prev_intensity
                if growth_rate > growth_threshold:
                    growing_peaks.append((angle, intensity, growth_rate))
                    print(f"Growing peak at {angle:.2f} degrees | Rate: {growth_rate:.1%}")

    # Update previous peaks for next iteration
    previous_peaks = {angle: intensity for angle, intensity in current_peaks}

    # Run detailed scans on growing peaks
    for angle, intensity in growing_peaks:
        start = round(angle - 0.375, 3)
        stop = round(angle + 0.5, 3)
        steps = int((stop - start) / 0.002)

        if stop - start > 0.1 and steps > 0: # Only scan if range is reasonable
            print(f"Running detailed scan on peak at {angle:.2f} (growth: {growth_rate:.1%})")
            run_sample_scan(start, stop, steps)

    # graph to show growth
    plt.plot(xyDeg, xyObs, 'r-', label='Smoothed')
    if growing_peaks:
        g_angles, g_intensities = zip(*growing_peaks)
        plt.scatter(g_angles, g_intensities, c='orange', s=50, marker='*', label='Growing')

    plt.title(f"Scan {scan_num + 1}")
    plt.xlabel("2θ (°)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

