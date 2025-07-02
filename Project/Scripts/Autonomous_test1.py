# OPTION 1: Subset Peak Scanner'''
'''
find the peaks
scan some subset of the peaks
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
    if not os.path.exists(file): # error fix
        raise FileNotFoundError(f"RAW file not found: {file}")
    try:
        with open(file, 'rb') as im:
            arr = np.frombuffer(im.read(), dtype='int32')  # read binary
        arr.shape = (195, 487)  # reshape to detector dimensions
        return arr
    except Exception as e:
        print(f"Error reading RAW file {file}: {e}")
        raise
    

# smoothing intensity data & deciding threshold
def smooth(xyDeg, xyOb):
    if len(xyDeg) == 0 or len(xyOb) == 0:  # checks 
        print("Warning: Empty input data for smoothing")
        return [], 0
    
    if len(xyDeg) != len(xyOb):
        print("Error: Angle and intensity arrays must have same length")
        return [], 0

    mean_I = np.mean(xyOb)
    std_I = np.std(xyOb)
    snr = mean_I / std_I if std_I != 0 else 0  # signal-to-noise-ratio
    
    # Automating SmNum and Intensity Threshold
    if snr < 3:
        SmNum = 2  # strong (good for noisy patterns)
        threshold = (mean_I + std_I) / 2
    elif snr < 10:
        SmNum = 2
        threshold = (mean_I + 0.5 * std_I)
    else:
        SmNum = 2  # light smoothing
        threshold = (mean_I + 0.2 * std_I) / 7
        
    # Handle edge cases for small arrays
    if len(xyOb) < 2 * SmNum + 1: # total pts = SmNum(left) + 1(center) + SmNum(right)
        print(f"Warning: Array too small for smoothing (length {len(xyOb)}, need {2*SmNum+1})")
        return list(xyOb), threshold
    
    smoothed = []

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
    # error fix: add input validation
    if len(xyDeg) == 0 or len(xyOb) == 0:
        print("Warning: Empty input data for peak detection")
        return []
    
    if len(xyDeg) != len(xyOb):
        print("Error: Angle and intensity arrays must have same length")
        return []
    
    if len(xyOb) < 9:  # Need at least 9 points for the detection algorithm
        print(f"Warning: Array too small for peak detection (length {len(xyOb)}, need ≥9)")
        return []

    peaks = []
    # skipping first & last 4 intensity values to avoid out-of-bounds
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
remote_path = "~/data/Jun2025/SelfDriving_debug"
remote_wpath = "X:/bl2-1/Jun2025/SelfDriving_debug"
remote_scan_path = f"{remote_path}/scans"
remote_img_path = f"{remote_path}/images"
remote_img_wpath = f"{remote_wpath}/images"
remote_xye_wpath = f"{remote_wpath}/xye"
spec_filename = f"SelfDriving_debug_12"

# directory existence checks
for directory in [remote_wpath, remote_xye_wpath]:
    if not os.path.exists(directory):
        print(f"Warning: Directory does not exist: {directory}")
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Could not create directory {directory}: {e}")

# === Setup beamline environment ===
create_SPEC_file(remote_scan_path, spec_filename)
set_PD_savepath(remote_img_path)

# load .poni calibration file for geometry
ai = pyFAI.load("X:/bl2-1/Jun2025/Si_fixed_detector.poni")

# take initial starting image (fast scan)
sendSPECcmd("umv tth 35")
sendSPECcmd("loopscan 1 5 0")

# === Integrate raw image to 1D (.xy) ===
raw_file = f"{remote_img_wpath}/b_stone_{spec_filename}_scan1_0000.raw"
xy_file = f"{remote_xye_wpath}/b_stone_{spec_filename}_scan1_0000.xy"

arr = readRAW(raw_file)

if not os.path.exists(xy_file):
    res = ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_file)
    df = pd.read_csv(xy_file, skiprows=23, header=None, delim_whitespace=True)
    df.columns = ['2theta_deg', 'I']
    df.to_csv(xy_file, index=False, float_format='%.6f', sep='\t')
    # print(xy_file)

# === Read .xy data ===
xy = np.genfromtxt(xy_file, dtype=float, delimiter='\t')
xy = xy[~np.isnan(xy).any(axis=1)]  # Remove rows with NaNs
xyDeg, xyOb = xy[:, 0], xy[:, 1]
# print(xyDeg)


# === Smoothing and Peak Detection ===
xySmoothed, threshold = smooth(xyDeg, xyOb)
peaks = detect_peaks(xyDeg, xySmoothed, threshold)
print(f"Detected {len(peaks)} peaks.")

# Select subset of peaks 
subset_size = 3 # adjustable
# Filter out peaks with postions below 11 degrees
peaks = sorted(peaks, key=lambda p: p[0], reverse=False) 
for i in range(0, len(peaks)):
    if peaks[i][0] >= 11:
        peaks = peaks[i:]
        break
strongest_peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:subset_size]
strongest_peaks = sorted(strongest_peaks, key=lambda p: p[0], reverse=False)

# === Generate scan windows ===
scan_windows = [] # about 0.875° wide
scan_high = 0
for angle, intensity in strongest_peaks:
    start = max(scan_high, angle - 0.375) # buffer space
    stop = angle + 0.5 
    if stop - start < 0.1:
        continue
    scan_windows.append((start, stop))
    scan_high = stop

# === Autonomous Scan Loop ===
print("\nStarting autonomous scan of strongest peaks...")
for start, stop in scan_windows:
    steps = int((stop - start) / 0.002) # small steps for high-res
    scan_command = f"ascan tth {start:.3f} {stop:.3f} {steps} 0.5"
    sendSPECcmd(scan_command)

# optional plot
plt.figure()
plt.plot(xyDeg, xySmoothed)
for peak in strongest_peaks:
    plt.axvline(peak[0], color='red', linestyle='--')
plt.title('Strongest Peaks')
plt.pause(0.1)
plt.show()