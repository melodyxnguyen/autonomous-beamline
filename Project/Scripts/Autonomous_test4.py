# OPTION 3: Scan shrinking peaks

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
    mean_I = np.mean(xyOb)
    std_I = np.std(xyOb)
    snr = mean_I / std_I if std_I != 0 else 0  # signal-to-noise-ratio

    # Simplified threshold since same SmNum value
    # SmNum = 2
    # threshold = mean_I + (std_I if snr < 3 else 0.5 * std_I if snr < 10 else 0.2 * std_I)
        
    # Automating SmNum and Intensity Threshold
    if snr < 3:
        SmNum = 2  # strong (good for noisy patterns)
        threshold = (mean_I + std_I)/2
    elif snr < 10:
        SmNum = 2
        threshold = (mean_I + 0.5 * std_I)/2
    else:
        SmNum = 2  # light smoothing
        threshold = (mean_I + 0.2 * std_I)/2

    # Handle edge cases for small arrays
    if len(xyOb) < 2 * SmNum + 1: # total pts = SmNum(left) + 1(center) + SmNum(right)
        print(f"Warning: Array too small for smoothing (length {len(xyOb)}, need {2*SmNum+1})")
        return list(xyOb), threshold
    
    smoothed = [xyOb[0]]

    # Handle beginning of array
    for i in range(1, SmNum - 1):
        smoothed.append(xyOb[i])

    # smooth middle section
    for i in range(SmNum, len(xyOb) - SmNum):
        smoothed.append(np.mean(xyOb[i - SmNum:i + SmNum + 1]))

    # end of array
    for i in range(SmNum + 1, 0, -1):
        smoothed.append(xyOb[-SmNum])
        
    # print(smoothed)
    return smoothed, threshold

# === Peak detection function ===
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

def get_latest_scan_files(base_path, spec_filename):
    """Find the most recent scan files based on scan number."""
    # We could update based on actual file naming convention
    # For now, assuming scan numbering increments
    scan_dirs = [d for d in os.listdir(base_path) if d.startswith('scan')]
    if not scan_dirs:
        return None, None
    
    latest_scan = max(scan_dirs, key=lambda x: int(x.replace('scan', '')))
    raw_file = os.path.join(base_path, latest_scan, f"b_stone_{spec_filename}_0000.raw")
    xy_file = raw_file.replace('.raw', '.xy').replace('images', 'xye')
    
    return raw_file, xy_file

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
growth_threshold = 0.15 # adjustable %
angle_tolerance = 0.3
max_scans = 20
previous_peaks = {}

for scan_num in range(max_scans):
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

    xy = np.genfromtxt(xy_file, dtype=float, delimiter='\t')
    xy = xy[~np.isnan(xy).any(axis=1)]  # Remove rows with NaNs
    xyDeg, xyOb = xy[:, 0], xy[:, 1]
    
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
        
        # Check for growth
        if closest is not None:
            prev_intensity = previous_peaks[closest]
            if prev_intensity > 0:  # Avoid division by zero
                growth_rate = (intensity - prev_intensity) / prev_intensity
                if growth_rate > growth_threshold:
                    growing_peaks.append((angle, intensity, growth_rate))
                    print(f"Growing peak at {angle:.2f}°: {growth_rate:.1%} growth")

    # Update previous peaks for next iteration
    previous_peaks = {angle: intensity for angle, intensity in current_peaks}

    # Run detailed scans on growing peaks
    for angle, intensity in growing_peaks:
        start = round(angle - 0.375, 3)
        stop = round(angle + 0.5, 3)
        steps = int((stop - start) / 0.002)

        if stop - start > 0.1: # Only scan if range is reasonable
            print(f"Running detailed scan on peak at {angle:.2f}° (growth: {growth_rate:.1%})")
            run_sample_scan(start, stop, steps)

plt.plot(xyDeg, xyOb, 'b-', alpha=0.6)
plt.plot(xyDeg, xyObs, 'r-')
plt.axhline(y=threshold, color='g', linestyle='--', alpha=0.4)

if current_peaks:
    angles, intensities = zip(*current_peaks)
    plt.scatter(angles, intensities, color='red', s=40)

if growing_peaks:
    g_angles = [p[0] for p in growing_peaks]
    g_intensities = [p[1] for p in growing_peaks]
    plt.scatter(g_angles, g_intensities, color='orange', s=80, marker='*')

plt.title(f"Scan {scan_num + 1}")
plt.xlabel("2θ (°)")
plt.ylabel("Intensity")
plt.grid(alpha=0.2)
plt.show()
