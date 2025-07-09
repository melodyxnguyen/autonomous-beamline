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
    wait_until_SPECfinished(polling_time=5)
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
    

# === SMOOTHING ===
def smooth(xyDeg, xyOb):
    # === UPDATES: FEATURE EXTRACTION ===
    mean_I = np.mean(xyOb)
    std_I = np.std(xyOb)
    max_I = np.max(xyOb)
    snr = mean_I / std_I if std_I != 0 else 0  # signal-to-noise-ratio
    
    # Automating SmNum and Intensity Threshold
    if snr < 3:
        SmNum = 2  # strong (good for noisy patterns)
        intensity_threshold = mean_I + std_I
    elif snr < 10:
        SmNum = 2
        intensity_threshold = mean_I + 0.5 * std_I
    else:
        SmNum = 2  # light smoothing
        intensity_threshold = mean_I + 0.2 * std_I
        
    smoothed = [xyOb[0]]
    for i in range(1, SmNum - 1):
        smoothed.append(xyOb[i])
    for i in range(SmNum, len(xyOb) - SmNum):
        smoothed.append(np.mean(xyOb[i - SmNum:i + SmNum + 1]))
    for i in range(SmNum + 1, 0, -1):
        smoothed.append(xyOb[-SmNum])
        
    print(smoothed)
    return smoothed, intensity_threshold

# === Peak detection function ===
def detect_peaks_old(xyDeg, xyObs, threshold):
    PeakInts, PeakPos = [], []
    
    ScanHigh = 0
    for i in range(0, len(xyObs)-1):
        if i > 3 and i < len(xyObs) - 4 and \
           xyObs[i - 4] < xyObs[i - 3] < xyObs[i - 2] < xyObs[i - 1] < xyObs[i] > xyObs[i + 1] and \
           xyObs[i + 4] < xyObs[i + 3] < xyObs[i + 2] < xyObs[i + 1]:
    
            PeakInt = xyObs[i]
            PeakPos.append(xyDeg[i])
            PeakInts.append(PeakInt)
            
    return PeakPos, PeakInt
    
# === Peak detection function ===
def detect_peaks(angles_deg, intensities_smoothed, threshold):
    """
    Detect peaks in the smoothed intensity array.
    angles_deg: list of 2theta angles (x-axis)
    intensities_smoothed: list of intensity values (y-axis)
    threshold: minimum height to consider a peak

    Returns: list of (angle, intensity) tuples
    """
    peaks = []
    for i in range(4, len(intensities_smoothed) - 4):
        if (
            intensities_smoothed[i - 4] < intensities_smoothed[i - 3] < intensities_smoothed[i - 2] < intensities_smoothed[i - 1] < intensities_smoothed[i] >
            intensities_smoothed[i + 1] > intensities_smoothed[i + 2] > intensities_smoothed[i + 3] > intensities_smoothed[i + 4]
            and intensities_smoothed[i] > threshold
        ):
            peaks.append((angles_deg[i], intensities_smoothed[i]))
    return peaks

# === Generate scan windows ===
scan_windows = []
scan_high = 0

# === Configure beamline paths ===
remote_path = "~/data/Jun2025/SelfDriving_debug"
remote_wpath = "X:/bl2-1/Jun2025/SelfDriving_debug"
remote_scan_path = f"{remote_path}/scans"
remote_xye_wpath = f"{remote_wpath}/xye"
remote_img_path = f"{remote_path}/images"
remote_img_wpath = f"{remote_wpath}/images"
spec_filename = f"SelfDriving_debug_12"

# === Setup beamline environment ===
create_SPEC_file(remote_scan_path, spec_filename)
set_PD_savepath(remote_img_path)

# load .poni calibration file for geometry
ai = pyFAI.load("X:/bl2-1/Jun2025/Si_fixed_detector.poni")

# take initial starting image
sendSPECcmd("umv tth 35")
sendSPECcmd("loopscan 1 5 0")
# === INTEGRATE TO 1D (.xy) ===
file = remote_img_wpath + "/b_stone_" + spec_filename + "_scan1_0000.raw"
arr = readRAW(file)
cur_xy = remote_xye_wpath + "/b_stone_" + spec_filename + "_scan1_0000.xy"

if not os.path.exists(cur_xy):
    res = ai.integrate1d(arr, 500, unit="2th_deg", filename=cur_xy)
    df = pd.read_csv(cur_xy, skiprows=23, header=None, delim_whitespace=True)
    df.columns = ['radians', 'I']
    df.to_csv(cur_xy, index=False, float_format='%.6f', sep='\t')
    print(cur_xy)
# === READ AND SPLIT DATA ===
xy = np.genfromtxt(cur_xy, dtype=float, delimiter='\t')
xy = xy[~np.isnan(xy).any(axis=1)]  # Remove rows with NaNs
xyDeg = xy[:, 0]
xyOb = xy[:, 1]

print(xyDeg)
    
xyObs, threshold = smooth(xyDeg, xyOb)
peaks = detect_peaks(xyDeg, xyObs, threshold)

print(peaks)
plt.figure()
plt.plot(xyDeg, xyOb)

plt.figure()
plt.plot(xyDeg, xyObs)

plt.show()

'''
# option 1:
find the peaks
scan some subset of the peaks
repeat

# option 2:
find the peaks
compare to previous peaks
scan any peaks that changed by some fraction
repeat

# option 3:
find the peaks
compare to previous peaks
scan any peaks that are growing
repeat

# option 4:
find the peaks
compare to previous peaks
scan any peaks that are shrinking
repeat

# option 5:
find the peaks
compare to previous peaks
scan any peaks that have shifted
repeat

# option 6
take difference of integration from previous
find regions of greatest change
scan those regions
repeat
'''