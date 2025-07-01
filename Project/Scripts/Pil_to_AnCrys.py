'''
Beamline Executor
(Slow, high-res detector)
Auto scanning: Zooms in and collect high-quality data at peak locations only.
Loads peaks.csv from AutoPP.py -> converts peak postions into angle windows 
Melody's updated Pil_to_AnCrys.py - June 2025
'''

import os
import time
import pandas as pd
import sys

# === XDart Beamline Imports ===
sys.path.append('C:\\Users\\Public\\Documents\\repos\\xdart')  # Update this path if needed

# Control + utility imports
from xdart.modules.pySSRL_bServer.bServer_funcs import (specCommand, wait_until_SPECfinished, get_console_output)
from xdart.utils import (get_from_pdi, get_motor_val, query, query_yes_no, read_image_file, smooth_img, get_fit, fit_images_2D)


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
    """Run a 1D scan over 2θ with given step count."""
    command = f'ascan tth {start} {stop} {steps} 0.5'
    print(f'Running sample scan: {command}')
    sendSPECcmd(command)


# === Load Peak Data and Trigger Scans ===
peak_csv = input("Enter path to *_peaks.csv from AutoPP.py: ")
if not os.path.exists(peak_csv):
    print("File not found.")
    sys.exit()

df = pd.read_csv(peak_csv)
peak_positions = df["angle_2theta_deg"].tolist()
sample_name = os.path.splitext(os.path.basename(peak_csv))[0].replace("_peaks", "")


# === Configure beamline paths ===
remote_path = "~/data/AutoXRDTest"
remote_scan_path = f"{remote_path}/scans"
remote_img_path = f"{remote_path}/images"

# === Setup beamline environment ===
create_SPEC_file(remote_scan_path, sample_name)
set_PD_savepath(remote_img_path)

# === Generate scan windows ===
scan_windows = []
scan_high = 0

for peak in peak_positions:
    scan_low = max(scan_high, peak - 0.375)
    scan_high = peak + 0.5
    scan_windows.append((round(scan_low, 3), round(scan_high, 3)))


# === RUN SCANS FOR EACH WINDOW ===
for i, (start, stop) in enumerate(scan_windows):
    steps = int((stop - start) / 0.002)
    print(f"⟶ Running scan {i+1}: {start}° to {stop}° | {steps} steps")
    run_sample_scan(start, stop, steps)
