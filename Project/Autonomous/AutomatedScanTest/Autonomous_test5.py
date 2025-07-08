# OPTION 5: Scan Shifted Peaks (Distance x)
'''
find the peaks
compare to previous peaks
scan any peaks that have shifted or sharped
repeat
'''

import numpy as np
from PIL import Image
import os
import sys
from matplotlib import pyplot as plt
import pyFAI, fabio
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
    print(f'Creating new SPEC file for sample: {name} at {path}')
    sendSPECcmd(f'newfile {path}/{name}')

def set_PD_savepath(img_path):
    print(f'Setting PD save path to {img_path}')
    sendSPECcmd(f'pd savepath {img_path}')

def run_sample_scan(start, stop, steps):
    command = f'ascan tth {start} {stop} {steps} 0.5'
    print(f'Running sample scan: {command}')
    sendSPECcmd(command)

def readRAW(file):
    if not file.endswith(".raw"):
        file = file + ".raw"
    if not os.path.exists(file):
        raise FileNotFoundError(f"RAW file not found: {file}")
    try:
        with open(file, 'rb') as im:
            arr = np.frombuffer(im.read(), dtype='int32')
        arr.shape = (195, 487)
        return arr
    except Exception as e:
        print(f"Error reading RAW file {file}: {e}")
        raise

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
    if snr < 3:
        SmNum = 2
        threshold = (mean_I + std_I) / 2
    elif snr < 10:
        SmNum = 2
        threshold = (mean_I + 0.5 * std_I) / 2
    else:
        SmNum = 2
        threshold = (mean_I + 0.2 * std_I) / 2

    if len(xyOb) < 2 * SmNum + 1:
        print(f"Warning: Array too small for smoothing (length {len(xyOb)}, need {2*SmNum+1})")
        return list(xyOb), threshold

    smoothed = [xyOb[0]]
    for i in range(1, SmNum - 1): # Handle beginning of array
        smoothed.append(xyOb[i])
    for i in range(SmNum, len(xyOb) - SmNum): # Handle middle section with proper averaging
        smoothed.append(np.mean(xyOb[i - SmNum:i + SmNum + 1]))
    for i in range(SmNum + 1, 0, -1): # Handle end of array
        smoothed.append(xyOb[-SmNum])
    return smoothed, threshold

def detect_peaks(xyDeg, xyOb, threshold):
    if len(xyDeg) == 0 or len(xyOb) == 0:
        print("Warning: Empty input data for peak detection")
        return []
    if len(xyDeg) != len(xyOb):
        print("Error: Angle and intensity arrays must have same length")
        return []
    if len(xyOb) < 9:
        print(f"Warning: Array too small for peak detection (length {len(xyOb)}, need >=9)")
        return []

    peaks = []
    for i in range(4, len(xyOb) - 4):
        if (xyOb[i - 4] < xyOb[i - 3] < xyOb[i - 2] < xyOb[i - 1] < xyOb[i] >
            xyOb[i + 1] > xyOb[i + 2] > xyOb[i + 3] > xyOb[i + 4] and xyOb[i] > threshold):
            peaks.append((xyDeg[i], xyOb[i]))
    return peaks

# === Beamline setup ===
remote_path = "~/data/July2025/SelfDriving_algo6_test"
remote_wpath = "X:/bl2-1/July2025/SelfDriving_algo6_test"
remote_scan_path = f"{remote_path}/scans"
remote_img_path = f"{remote_path}/images"
remote_img_wpath = f"{remote_wpath}/images"
remote_xye_wpath = f"{remote_wpath}/xye"
spec_filename = f"SelfDriving_algo6_test_1"

for directory in [remote_wpath, remote_xye_wpath]:
    if not os.path.exists(directory):
        print(f"Warning: Directory does not exist: {directory}")
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Could not create directory {directory}: {e}")

create_SPEC_file(remote_scan_path, spec_filename)
set_PD_savepath(remote_img_path)
sendSPECcmd("csettemp 820")

ai = pyFAI.load("X:/bl2-1/July2025/Si_fixed_detector.poni")

# Initialize variables before loop
previous_peaks = []  # Store previous peak positions
scan_number = 0 
shift_threshold = 0.11  # degrees — adjustable
angle_tolerance = 0.3   # degrees — for matching peaks between scans

# Switch to (while True:) to run "forever", ctrl c to "kill"
while scan_number < 10: 
    scan_number += 1

    # Perform standard scan
    sendSPECcmd("umv tth 35")
    sendSPECcmd("loopscan 1 5 0")

    # File paths for this scan
    raw_file = f"{remote_img_wpath}/b_stone_{spec_filename}_scan{scan_number}_0000.raw"
    xy_file = f"{remote_xye_wpath}/b_stone_{spec_filename}_scan{scan_number}_0000.xy"

    arr = readRAW(raw_file)

    if not os.path.exists(xy_file):
        res = ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_file)
        df = pd.read_csv(xy_file, skiprows=23, header=None, delim_whitespace=True)
        df.columns = ['2theta_deg', 'I']
        df.to_csv(xy_file, index=False, float_format='%.6f', sep='\t')

    # Read xy data
    xy = np.genfromtxt(xy_file, dtype=float, delimiter='\t')
    xyDeg, xyOb = xy[20:, 0], xy[20:, 1]

    # Smooth data and detect peaks
    xySmoothed, threshold = smooth(xyDeg, xyOb)
    current_peaks = detect_peaks(xyDeg, xySmoothed, threshold)
    print(f"Detected {len(current_peaks)} peaks in scan {scan_number}")

    # Compare with previous peaks if this isn't the first scan
    shifted_peaks = []
    shift_threshold = 0.11  # degrees — adjustable
    angle_tolerance = 0.3
    
    if scan_number > 1 and previous_peaks:
        print(f"Comparing with {len(previous_peaks)} peaks from previous scan...")

        for current_angle, current_intensity in current_peaks:
            # Find closest previous peak
            closest_prev_angle = None
            min_diff = angle_tolerance

            for prev_angle, _ in previous_peaks:
                diff = abs(current_angle - prev_angle)
                if diff < min_diff:
                    closest_prev_angle = prev_angle
                    min_diff = diff
        
                # Check if peak has shifted significantly
                if closest_prev_angle is not None:
                    shift = current_angle - closest_prev_angle
                    if abs(shift) > shift_threshold:
                        shifted_peaks.append((current_angle, current_intensity, shift))
                        print(f"Peak shifted: {closest_prev_angle:.3f} degrees to {current_angle:.3f} degrees (shift: {shift:+.3f} degrees)")

        # Scan shifted peaks
        if shifted_peaks:
            print(f"\nFound {len(shifted_peaks)} shifted peaks. Running detailed scans...")
            for angle, intensity, shift in shifted_peaks:
                start = round(angle - 0.375, 3)
                stop = round(angle + 0.5, 3)
                steps = int((stop - start) / 0.002)

                if stop - start > 0.1:  # Only scan if range is reasonable
                    print(f"Scanning shifted peak at {angle:.3f}° (range: {start:.3f} to {stop:.3f}°, {steps} steps)")
                    run_sample_scan(start, stop, steps)
                else:
                    print(f"Skipping peak at {angle:.3f}° - scan range too small")
        else:
            print("No significantly shifted peaks detected.")

        # Update previous peaks for next iteration
        previous_peaks = current_peaks.copy()

plt.figure()
plt.plot()
plt.title(f"Scan {scan_number}")
plt.xlabel("2θ (°)")
plt.ylabel("Intensity")
plt.legend()
plt.show()