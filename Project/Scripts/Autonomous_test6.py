# OPTION 6: Regions of Greatest Change
'''
take difference of integration from previous
find regions of greatest change
scan those regions
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

previous_integration = None # Intialize before loop
scan_number = 0 

# Switch to (while True:) to run "forever", ctrl c to "kill"
while scan_number < 10: 
    sendSPECcmd("umv tth 35")
    sendSPECcmd("loopscan 1 5 0")
    scan_number += 1

    raw_file = f"{remote_img_wpath}/b_stone_{spec_filename}_scan{scan_number}_0000.raw"
    xy_file = f"{remote_xye_wpath}/b_stone_{spec_filename}_scan{scan_number}_0000.xy"

    arr = readRAW(raw_file)

    if not os.path.exists(xy_file):
        res = ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_file)
        df = pd.read_csv(xy_file, skiprows=23, header=None, delim_whitespace=True)
        df.columns = ['2theta_deg', 'I']
        df.to_csv(xy_file, index=False, float_format='%.6f', sep='\t')

    # read xy data
    xy = np.genfromtxt(xy_file, dtype=float, delimiter='\t')
    xyDeg, xyOb = xy[20:, 0], xy[20:, 1]

    xySmoothed, threshold = smooth(xyDeg, xyOb)
    peaks = detect_peaks(xyDeg, xySmoothed, threshold)
    print(f"Detected {len(peaks)} peaks.")

    # === Compare integration with previous ===
    integration = xySmoothed  # this scan's smoothed data

    if previous_integration is not None:
        # Check arrays have same length for comparison
        min_len = min(len(integration), len(previous_integration))
        if min_len > 0:
            current_data = np.array(integration[:min_len])
            previous_data = np.array(previous_integration[:min_len])
            diff = np.abs(current_data - previous_data)

        diff_peaks = []
        if len(diff) >= 9: # Need at least 9 points for 4-point window
            diff_threshold = np.mean(diff) + np.std(diff)
            for i in range(4, len(diff) - 4):
                if (
                    diff[i - 4] < diff[i - 3] < diff[i - 2] < diff[i - 1] < diff[i] >
                    diff[i + 1] > diff[i + 2] > diff[i + 3] > diff[i + 4]
                    and diff[i] > diff_threshold
                ):
                    if i < len(xyDeg):
                        diff_peaks.append(xyDeg[i])

        # Deduplicate nearby peaks & make scan windows
        if diff_peaks:
            diff_peaks = sorted(set(diff_peaks))
            scan_windows = []
            scan_high = 0
            for angle in diff_peaks:
                start = max(scan_high, angle - 0.5)
                stop = angle + 0.5
                if stop - start >= 0.1:
                    scan_windows.append((start, stop))
                    scan_high = stop

            print(f"{len(scan_windows)} windows with strong change")

            # === Scan those regions
            sendSPECcmd("pd nosave; pd disable")
            for start, stop in scan_windows:
                steps = int((stop - start) / 0.005)
                scan_command = f"ascan tth {start:.3f} {stop:.3f} {steps} 0.5"
                run_sample_scan(start, stop, steps)
            sendSPECcmd("pd save; pd enable")

    # update for next round
    previous_integration = integration.copy() if isinstance(integration, list) else integration

    # optional plot
    plt.plot(xyDeg, xySmoothed, label='Smoothed')
    plt.title(f"Scan {scan_number}: Option 6 - Change Regions")
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Intensity")
    plt.grid(alpha=0.2)
    plt.pause(0.1)
    plt.show()
