# Peak Qualification and Classification
# SULI Project: Aug 2025 (Melody)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import pyFAI
import seaborn as sn

# Settings & Paths
SCRIPT_DIR = os.path.dirname(__file__)
RAW_FOLDER = os.path.join(SCRIPT_DIR, "images")
XY_OUTPUT_FOLDER = SCRIPT_DIR
PONI_FILE = os.path.join(SCRIPT_DIR, "Si_fixed_detector.poni")
SCAN_CSV = os.path.join(SCRIPT_DIR, "TiO2_heating_pilatus_only_scan1.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "final_classified_peaks.csv")

SHIFT_THRESHOLD = 0.02 # Degree
TOP_N = 4 # Top number of peaks by intensity to keep

# Plot graph with status of peaks
STATUS_COLORS = {"growing": "green", "shrinking": "red", "stable": "blue"}
STATUS_HANDLES = [
    Line2D([0], [0], color=STATUS_COLORS["growing"], lw=2, linestyle="--", label="growth"),
    Line2D([0], [0], color=STATUS_COLORS["shrinking"], lw=2, linestyle="--", label="shrink"),
    Line2D([0], [0], color=STATUS_COLORS["stable"], lw=2, linestyle="--", label="stable"),
]


# Reference peaks for phase scoring (2θ, relative intensity) 
# Standard powder X-ray diffraction reference patterns for titanium dioxide (TiO₂) in anatase and rutile— Cu Kα defaults
# TUNE centers/tol for geometry  — Cu Kα defaults

ANATASE_REF = [(25.30, 100), (37.80, 32), (48.05, 22), (54.00, 18), (55.10, 18)]
RUTILE_REF  = [(27.40, 100), (36.10, 45), (41.30, 32), (54.30, 30), (56.60, 26)]
PEAK_TOL = 0.35  # deg tolerance for a reference match (tune)

# FUNCTIONS
# Gaussian (normal) distribution for peak fitting
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x-x0) ** 2 / (2 * sigma ** 2))

# Smoothing
def smooth(y):
    mean_I, std_I = np.mean(y), np.std(y)
    snr = mean_I / std_I if std_I !=0 else 0
    SmNum = 2
    threshold = (mean_I + (std_I if snr < 3 else 0.5 * std_I if snr < 10 else 0.2 * std_I)) / 2
    if len(y) < 2 * SmNum + 1: 
        return list(y), threshold
    smoothed = [y[0]]
    for i in range(1, SmNum - 1): 
        smoothed.append(y[i])
    for i in range(SmNum,len(y) - SmNum):
        smoothed.append(np.mean(y[i - SmNum:i + SmNum + 1]))
    for i in range(SmNum + 1, 0, -1): 
        smoothed.append(y[-SmNum])
    return smoothed, threshold

# Peak finding: angles (2θ position), intensity, minimum to consider 
def detect_peaks(x, y, threshold):
    peaks = []
    for i in range(4, len(y) - 4):
        # looking for local maxima above threshold
        if (y[i-4] < y[i-3] < y[i-2] < y[i-1] < y[i] > # rise before peak
            y[i+1] > y[i+2] > y[i+3] > y[i+4]) and y[i] > threshold: # fall after
            peaks.append((x[i], y[i]))
    return peaks

def classify_growth(current_peaks, previous_peaks):
    # Return dictionary: angle 'growing' 'shrinking' 'stable'
    growth_labels = {}
    for angle, curr_int in current_peaks:
        # Find closest peak in previous scan
        closest = min(previous_peaks, key=lambda p: abs(p[0] - angle), default=None)
        if closest:
            prev_intensity = closest[1]
            if curr_int > prev_intensity * 1.05: # 5% increase
                growth_labels[angle] = "growing"
            elif curr_int < prev_intensity * 0.95: # 5% decrease
                growth_labels[angle] = "shrinking"
            else:
                growth_labels[angle] = "stable"
        else:
            growth_labels[angle] = "stable" 
    return growth_labels

def write_header(path):
    with open(path,"w") as f:
        f.write("scan,ctemp,angle,intensity,shift,fitted_center,fitted_sigma,is_shifted,is_changing,status\n")

def append_row(path,row):
    with open(path,"a") as f:
        f.write(",".join(str(x) for x in row) + "\n")

# Multi-peak phase scoring
# Matching algorithm, compare to reference crystal peak phases (anatase or rutile)
def score_phase(peaks, refs):
    score = 0.0
    for ref_angle, ref_relI in refs:
        a, I = min(peaks, key=lambda p: abs(p[0] - ref_angle), default=(None, None))
        if a is None:
            continue
        delta = abs(a - ref_angle)
        if delta <= PEAK_TOL:
            closeness = max(0.0, 1.0 - delta / PEAK_TOL)
            score += closeness * ref_relI * I
    return score

def classify_phase_multi(peaks):
    a_score = score_phase(peaks, ANATASE_REF)
    r_score = score_phase(peaks, RUTILE_REF)
    if a_score == 0 and r_score == 0:
        return "Mixed"
    margin = 1.15  # need 15% advantage to call a pure phase
    if a_score > r_score * margin:
        return "Anatase"
    if r_score > a_score * margin:
        return "Rutile"
    return "Mixed"

# === Main ===
def main():
    ai = pyFAI.load(PONI_FILE)
    scan_df = pd.read_csv(SCAN_CSV)
    scan_df.columns = scan_df.columns.str.strip()

    raw_files = sorted([f for f in os.listdir(RAW_FOLDER) if f.endswith(".raw")])
    previous_peaks=[]
    previous_integral=None
    write_header(OUTPUT_CSV)

    for scan_number, raw_file in enumerate(raw_files):
        raw_path = os.path.join(RAW_FOLDER, raw_file)
        xy_name = raw_file.replace(".raw", ".xy")
        xy_path = os.path.join(XY_OUTPUT_FOLDER, xy_name)

        # Integrate raw 1D -> xy 2D if needed
        if not os.path.exists(xy_path): 
            with open(raw_path, 'rb') as f:
                arr = np.frombuffer(f.read(), dtype='int32')
            arr.shape = (195,487)
            ai.integrate1d(arr, 500, unit="2th_deg", filename=xy_path)
            df_xy = pd.read_csv(xy_path, skiprows=23, header=None, sep=r"\s+", engine="python")
            df_xy.columns = ["2theta_deg", "I"]
            df_xy.to_csv(xy_path, sep='\t', index=False, float_format='%.6f')
            print(xy_path)

        # Load xy 1D pattern with pandas
        df_xy = pd.read_csv(xy_path, sep="\t")  
        xyDeg = df_xy["2theta_deg"].to_numpy()
        xyOb  = df_xy["I"].to_numpy()

        # drop low-angle startup and apply mask
        xyDeg = xyDeg[20:]
        xyOb  = xyOb[20:]
        mask = xyDeg >= 14
        xyDeg = xyDeg[mask]
        xyOb  = xyOb[mask]

        if xyDeg.size == 0 or xyOb.size == 0:
            print(f"Skipping {raw_file}: empty pattern after masking.")
            if os.path.exists(xy_path):
                os.remove(xy_path)
            continue

        # Delete xy file after processing
        if os.path.exists(xy_path):
            os.remove(xy_path)

        smoothed,threshold = smooth(xyOb)
        peaks = detect_peaks(xyDeg, smoothed, threshold)
        peaks = sorted(peaks, key=lambda p:p[1], reverse=True)[:TOP_N]

        # Track shift vs. previous scan
        shifted = []
        if previous_peaks:
            for curr_angle, curr_int in peaks:
                closest = min(previous_peaks, key=lambda p:abs(p[0]-curr_angle), default=None)
                if closest and abs(curr_angle - closest[0]) > SHIFT_THRESHOLD:
                    shifted.append((curr_angle, curr_int, curr_angle - closest[0]))

        # Detect global change vs. previous scan (simple integral difference)
        changing = []
        current_integral = np.array(smoothed)
        if previous_integral is not None:
            min_len = min(len(current_integral), len(previous_integral))
            diff = np.abs(current_integral[:min_len] - previous_integral[:min_len])
            diff_thr = np.mean(diff) + np.std(diff)
            for peak in peaks:
                idx = np.argmin(np.abs(xyDeg - peak[0]))
                if diff[idx] > diff_thr: changing.append(peak[0])
        
        # Status labels (growth/shrink/stable) vs previous scan
        growth_labels = classify_growth(peaks, previous_peaks)
        previous_peaks = peaks.copy()
        previous_integral = current_integral.copy()

        # Temperature for title/CSV
        try: 
            ctemp = scan_df["CTEMP"].iloc[scan_number]
        except Exception: 
            ctemp = None

        # Visualization
        plt.figure()
        plt.plot(xyDeg,smoothed, label="Smoothed")
        for angle, _int in peaks:
            state = growth_labels.get(angle, "stable")
            plt.axvline(angle, color=STATUS_COLORS[state], linestyle="--")
        plt.title(f"Scan {scan_number+1}"+(f" at {ctemp:.1f}°C" if ctemp is not None else ""))
        plt.xlabel("2θ (deg)")
        plt.ylabel("Intensity")
        plt.grid(alpha=0.3)
        plt.legend(handles=STATUS_HANDLES, loc="upper right", title="Peak status")
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        
        # Classify phase for this scan
        phase_call = classify_phase_multi(peaks)

        # Save each peak's data
        for angle, intensity in peaks:
            is_shifted = any(abs(angle-s[0]) < 0.01 for s in shifted)
            is_changing = any(abs(angle-c) < 0.01 for c in changing)
            
            # Local Gaussian fit for better center/sigma
            mask = (xyDeg >= angle - 0.2) & (xyDeg <= angle + 0.2)
            x_fit,y_fit = xyDeg[mask], np.array(smoothed)[mask]
            if len(x_fit) < 5: 
                continue
            try:
                popt,_ = curve_fit(gaussian, x_fit, y_fit, p0 = [intensity, angle, 0.05])
                fitted_center, fitted_sigma = popt[1], popt[2]
            except:
                fitted_center, fitted_sigma = angle, 0.0 
            shift_val = next((s[2] for s in shifted if abs(angle - s[0]) < 0.01), 0.0)
            status = growth_labels.get(angle, "stable")
            append_row(OUTPUT_CSV,
                       [scan_number, ctemp,angle, intensity, shift_val, fitted_center, 
                       fitted_sigma, is_shifted, is_changing, status])
    post_analysis()

# Post analysis and Visualizations
def post_analysis():
    df = pd.read_csv(OUTPUT_CSV)
    df.columns = df.columns.str.strip()

    plt.figure(figsize=(10, 6))
    for s, g in df.groupby("status"):
        plt.scatter(g["angle"], g["intensity"], label=s, color=STATUS_COLORS.get(s, "gray"), alpha=0.6)
    plt.title("Peaks by Status")
    plt.xlabel("2θ (deg)")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Train decision tree to predict phase from features
    features = df[["angle", "intensity", "shift", "fitted_sigma"]]
    labels = df["status"]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    clf_phase = DecisionTreeClassifier(max_depth = 4, random_state = 42)
    clf_phase.fit(X_train, y_train)

    plt.figure(figsize = (12, 6))
    plot_tree(clf_phase, feature_names=features.columns, class_names=clf_phase.classes_, filled = True)
    plt.title("Decision Tree: Peak Type")
    plt.tight_layout()
    plt.show()

    print("\nPhase Classification Report:\n", classification_report(y_test, clf_phase.predict(X_test)))

if __name__ == "__main__":
    main()  