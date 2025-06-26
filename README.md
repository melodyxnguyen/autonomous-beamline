# Multi-modal autonomous beamline experimentation
SSRL Materials Sciences Division

SLAC 2025 Internship

The goal of this proposal is to create an infrastructure for autonomous beamline experiments utilizing multi-modal approaches at SSRL. Utilizing a 2-detector setup and clever Python scripting to automate high-quality data acquisition.  Automated analysis of the fast, low-resolution data was used to autonomously collect slower, high-resolution data only in the information-rich regions around diffraction peaks. 

<img width="347" alt="image" src="https://github.com/user-attachments/assets/d489b5ee-d93b-4dbd-b0ad-e390755211a0" />

For the powder diffraction experiment, the goal is for the computer to determine where and when to collect high-quality diffraction data autonomously, rather than scanning everything manually.

I will use two detectors:
- A fast but low-res area detector → scans broadly and quickly
- A slow but high-res crystal detector → zooms in on useful parts only

Areas of improvement:
1. Peak picking - Add an intensity threshold, or better logic
2. Smoothing -  Make SmNum user-settable or more robust
3. Output - Save a summary of peak positions to .csv
4. Scan range logic - Add comments, clean up if-else flow
5. Plotting - Label plots, use plt.title() etc.

