# OPTION 2: Scan peaks that change by a percentage difference

'''
1. Store previous peaks 
    list to hold (angle, intenisty) from last scan
2. Compare current vs. previous peaks 
    for each peak in current scan
        find closest match in prev list
        calc % intensity change
        if change > threshold (15%):
            mark for scanning
3. Trigger scan on changed peaks
    define scan window
    run sendSPECcmd()
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
    if not file.endswith(".raw"):
        file += ".raw"
    if not os.path.exists(file):
        raise FileNotFoundError(f"RAW file not found: {file}")
    with open(file, 'rb') as im:
        arr = np.frombuffer(im.read(), dtype='int32')
    arr.shape = (195, 487)
    return arr


# Smoothing
def smooth(xyDeg, xyOb):
    mean_I = np.mean(xyOb)
    std_I = np.std(xyOb)
    snr = mean_I / std_I if std_I != 0 else 0  # signal-to-noise-ratio
    
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
    if len(xyOb) < 2 * SmNum + 1: 
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
    
# === Peak detection ===
def detect_peaks(xyDeg, xyOb, threshold):
    
    """
    Detect peaks in the smoothed intensity array.

    Args:
        xyDeg: list of 2theta angles (x-axis)
        xyOb: list of intensity values (y-axis) - should be smoothed
        threshold: minimum height to consider a peak

    Returns: list of (angle, intensity) tuples
    """
    peaks = []
    for i in range(4, len(xyOb) - 4): 
        if (
            # left slope rising
            xyOb[i - 4] < xyOb[i - 3] < xyOb[i - 2] < xyOb[i - 1] < xyOb[i] > 
            # right slop falling
            xyOb[i + 1] > xyOb[i + 2] > xyOb[i + 3] > xyOb[i + 4]
            # filters out small bumps
            and xyOb[i] > threshold
        ):
            peaks.append((xyDeg[i], xyOb[i]))
    return peaks


# === Configure beamline paths ===
remote_path = "~/data/July2025/SelfDriving_algo2_test"
remote_wpath = "X:/bl2-1/July2025/SelfDriving_algo2_test"
remote_scan_path = f"{remote_path}/scans"
remote_img_path = f"{remote_path}/images"
remote_img_wpath = f"{remote_wpath}/images"
remote_xye_wpath = f"{remote_wpath}/xye"
spec_filename = f"SelfDriving_algo2_test_1"

# beamline prep
create_SPEC_file(remote_scan_path, spec_filename)
set_PD_savepath(remote_img_path)

# load .poni calibration file for geometry
ai = pyFAI.load("X:/bl2-1/July2025/Si_fixed_detector.poni")
sendSPECcmd("umv tth 35") # rotate to correct angle

# parameters
max_scans = 10
intensity_change_threshold = 0.15 # THIS IS THE SET 15% (adjustable)
angle_tolerance = 0.4  # degrees

previous_peaks = {} # angle: intensity

for scan_num in range(max_scans):
    print(f"\n=== Scan {scan_num + 1} ===")
    sendSPECcmd("loopscan 1 5 0")

    # Raw image & intergrate to 1D (.xy) ===
    raw_file = f"{remote_img_wpath}/b_stone_{spec_filename}_scan1_0000.raw"
    xy_file = f"{remote_xye_wpath}/b_stone_{spec_filename}_scan1_0000.xy"
    
    # check if raw file exists
    if not os.path.exists(raw_file):
        print(f"Warning: Raw file not found: {raw_file}")
        continue
    arr = readRAW(raw_file)

    if not os.path.exists(xy_file):
        res = ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_file)
        df = pd.read_csv(xy_file, skiprows=23, header=None, delim_whitespace=True)
        df.columns = ['2theta_deg', 'I']
        df.to_csv(xy_file, index=False, float_format='%.6f', sep='\t')

    # Read & smooth data
    xy = np.genfromtxt(xy_file, dtype=float, delimiter='\t')
    # xy = xy[~np.isnan(xy).any(axis=1)]  # Remove rows with NaNs
    xyDeg, xyOb = xy[20:, 0], xy[20:, 1]
    xyObs, threshold = smooth(xyDeg, xyOb)
    current_peaks = detect_peaks(xyDeg, xyObs, threshold)

    print(f"Found {len(current_peaks)} peaks")

    # Compare with previous peaks (skip first scan since no previous data)
    changed_peaks = []
    if scan_num > 0 and previous_peaks:
        for angle, intensity in current_peaks:
            closest_match = None    
            min_diff = angle_tolerance 
            for prev_angle in previous_peaks:
                diff = abs(angle - prev_angle)
                if diff < min_diff: 
                    closest_match = prev_angle
                    min_diff = diff

            if closest_match:
                prev_intensity = previous_peaks[closest_match]
                if prev_intensity > 0: 
                    frac_change = abs(intensity - prev_intensity) / prev_intensity
                    # found transformation!
                    if frac_change > intensity_change_threshold: 
                        changed_peaks.append((angle, intensity))
    
    # === Run scan on changed peaks ===
    if changed_peaks:
        print(f"Running detailed scans on {len(changed_peaks)} on changed peaks")
        for angle, intensity in changed_peaks:
            start = round(angle - 0.375, 3)
            stop = round(angle + 0.5, 3)
            steps = int((stop - start) / 0.002)
            if steps > 0: # check for postive step count
                run_sample_scan(start, stop, steps)
    else:
        print("No significant peak changes found")

    # Update peak history
    previous_peaks = {angle: intensity for angle, intensity in current_peaks}


    # optional plot for sanity check
    plt.figure()
    plt.plot(xyDeg, xyObs)
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Intensity")
    plt.title(f"Scan {scan_num + 1}")
    plt.pause(0.1)

plt.show()
