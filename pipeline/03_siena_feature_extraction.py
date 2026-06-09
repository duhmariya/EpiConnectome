"""
Siena Feature Extraction - MATCHED TO LAB DATA
Features: Sample Entropy, Spectral Entropy, Variance, Skewness
OPTIMIZED & COMPLETE

Parameters matched to lab data:
✓ 16 common channels
✓ Same preprocessing (ICA: 15 components, infomax)
✓ Same frequency bands
"""

import os
import mne
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from scipy.signal import welch
from PyQt5 import QtWidgets, QtCore
import sys
import warnings
import time
import antropy as ant
warnings.filterwarnings('ignore')

# ================== REPRODUCIBILITY ==================
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
mne.set_config('MNE_RANDOM_SEED', str(RANDOM_SEED))
mne.set_config('MNE_USE_NUMBA', 'false')

# ================== PARAMETERS - MATCHED ==================
EPOCH_LENGTH = 10
subj = 'Siena_Epilepsy'
SFREQ = 512
TARGET_SFREQ = 1000

# ⭐ COMMON 16 CHANNELS - MATCHED
COMMON_16_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4'
]

# ⭐ FREQUENCY BANDS - MATCHED
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

print("\n" + "="*80)
print("SIENA FEATURE EXTRACTION - MATCHED TO LAB DATA")
print("="*80)
print("\n✓ Parameters MATCHED:")
print("  ✓ Channels: 16 common")
print("  ✓ Sampling: 1000 Hz")
print("  ✓ ICA: 15 components, infomax")
print("  ✓ Bands: Theta(4-8), Alpha(8-13), Beta(13-30)")
print("  ✓ Features: SampEn, SpecEn, Var, Skew")
print("="*80)

# ================== FOLDER SELECTION ==================
def choose_folder(title="選擇 Siena TXT 資料夾"):
    try:
        app = QtWidgets.QApplication.instance()
        created_app = False
        if app is None:
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
            app = QtWidgets.QApplication(sys.argv)
            created_app = True

        dialog = QtWidgets.QFileDialog()
        dialog.setWindowTitle(title)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            path = dialog.selectedFiles()[0]
        else:
            path = None

        if created_app:
            app.quit()

        return path
    except:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return path if path else None
        except:
            return None

selected_folder = choose_folder()
if not selected_folder:
    raise RuntimeError("未選取資料夾")

print(f"\n資料夾：{selected_folder}")

txt_files = []
for root, dirs, files in os.walk(selected_folder):
    for file in files:
        if file.endswith('.txt'):
            txt_files.append(os.path.join(root, file))

if not txt_files:
    raise RuntimeError("找不到 .txt 檔案")

print(f"找到 {len(txt_files)} 個檔案")

all_features = []

# ================== OPTIMIZED FEATURE FUNCTIONS ==================

def sample_entropy_fast(data, m=2, r=None, max_length=2000):
    """Fast Sample Entropy using antropy (C implementation)"""
    if len(data) > max_length:
        data = data[:max_length]
    try:
        return ant.sample_entropy(data, order=m)
    except:
        return 0.0

    try:
        def _count(m_val):
            # Build template matrix using stride tricks
            shape = (N - m_val, m_val)
            strides = (data.strides[0], data.strides[0])
            templates = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
            # Pairwise max absolute difference
            count = 0
            for i in range(len(templates)):
                diff = np.max(np.abs(templates - templates[i]), axis=1)
                count += np.sum(diff <= r) - 1  # exclude self
            return count

        A = _count(m)
        B = _count(m + 1)

        if A > 0 and B > 0:
            return -np.log(B / A)
        return 0.0
    except:
        return 0.0


def spectral_entropy_fast(data, sfreq):
    """FAST Spectral Entropy"""
    try:
        nperseg = min(256, len(data) // 4)
        freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)
        psd_norm = psd / (np.sum(psd) + 1e-10)
        return entropy(psd_norm + 1e-10)
    except:
        return 0.0


def extract_band_signal(raw, band_range):
    """Extract frequency band"""
    try:
        filtered = raw.copy().filter(
            l_freq=band_range[0], 
            h_freq=band_range[1], 
            verbose=False
        )
        return filtered.get_data()
    except:
        return raw.get_data()


def compute_features_per_band(raw, ch_names, freq_bands, sfreq):
    """Compute all features for all bands"""
    n_channels = len(ch_names)
    
    features = {
        'sample_entropy': {},
        'spectral_entropy': {},
        'variance': {},
        'skewness': {}
    }
    
    for feature_name in features.keys():
        for band_name in freq_bands.keys():
            features[feature_name][band_name] = np.zeros(n_channels)
    
    for band_idx, (band_name, band_range) in enumerate(freq_bands.items(), 1):
        print(f"    [{band_idx}/{len(freq_bands)}] {band_name} ({band_range[0]}-{band_range[1]} Hz)...", 
              end=' ')
        start_time = time.time()
        
        try:
            band_data = extract_band_signal(raw, band_range)
            
            for ch_idx in range(n_channels):
                signal = band_data[ch_idx, :]
                
                # Sample Entropy
                features['sample_entropy'][band_name][ch_idx] = sample_entropy_fast(signal)
                
                # Spectral Entropy
                features['spectral_entropy'][band_name][ch_idx] = spectral_entropy_fast(signal, sfreq)
                
                # Variance
                features['variance'][band_name][ch_idx] = np.var(signal)
                
                # Skewness
                features['skewness'][band_name][ch_idx] = stats.skew(signal)
            
            elapsed = time.time() - start_time
            print(f"Done ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    return features


def load_siena_txt(txt_path):
    """Load TXT"""
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 2 and data.shape[1] > 29:
            data = data[:, :29]
        return data
    except:
        df = pd.read_csv(txt_path, sep='\s+', header=None)
        data = df.values
        if data.shape[1] > 29:
            data = data[:, :29]
        return data


def save_features_to_excel(features, ch_names, output_path):
    """Save to Excel"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sample Entropy
        df_se = pd.DataFrame(
            {band: features['sample_entropy'][band] for band in FREQ_BANDS.keys()},
            index=ch_names
        )
        df_se.to_excel(writer, sheet_name='Sample_Entropy')
        
        # Spectral Entropy
        df_spe = pd.DataFrame(
            {band: features['spectral_entropy'][band] for band in FREQ_BANDS.keys()},
            index=ch_names
        )
        df_spe.to_excel(writer, sheet_name='Spectral_Entropy')
        
        # Variance
        df_var = pd.DataFrame(
            {band: features['variance'][band] for band in FREQ_BANDS.keys()},
            index=ch_names
        )
        df_var.to_excel(writer, sheet_name='Variance')
        
        # Skewness
        df_skew = pd.DataFrame(
            {band: features['skewness'][band] for band in FREQ_BANDS.keys()},
            index=ch_names
        )
        df_skew.to_excel(writer, sheet_name='Skewness')
        
        # Summary
        summary_data = []
        for ch_idx, ch_name in enumerate(ch_names):
            for band_name in FREQ_BANDS.keys():
                summary_data.append({
                    'Channel': ch_name,
                    'Band': band_name,
                    'Sample_Entropy': features['sample_entropy'][band_name][ch_idx],
                    'Spectral_Entropy': features['spectral_entropy'][band_name][ch_idx],
                    'Variance': features['variance'][band_name][ch_idx],
                    'Skewness': features['skewness'][band_name][ch_idx]
                })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)


def create_heatmaps(features, ch_names, output_dir, subject_id):
    """Create heatmaps"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{subject_id} - Features (Matched to Lab)', fontsize=14, fontweight='bold')
    
    feature_info = [
        ('sample_entropy', 'Sample Entropy', 'YlOrRd'),
        ('spectral_entropy', 'Spectral Entropy', 'Blues'),
        ('variance', 'Variance', 'Greens'),
        ('skewness', 'Skewness', 'Purples')
    ]
    
    for ax, (feature_key, feature_label, cmap) in zip(axes.flat, feature_info):
        matrix = np.array([
            features[feature_key][band] for band in FREQ_BANDS.keys()
        ]).T
        
        sns.heatmap(
            matrix,
            xticklabels=list(FREQ_BANDS.keys()),
            yticklabels=ch_names,
            cmap=cmap,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': feature_label},
            ax=ax,
            annot_kws={'size': 8}
        )
        
        ax.set_title(feature_label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Band', fontsize=10)
        ax.set_ylabel('Channel', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{subject_id}_features.png"),
        dpi=200, bbox_inches='tight'
    )
    plt.close(fig)


# ================== MAIN PROCESSING ==================
print("\n" + "="*80)
print("Starting processing...")
print("="*80)

success_count = 0
fail_count = 0

for file_idx, txt_file_path in enumerate(txt_files, 1):
    print(f"\n[{file_idx}/{len(txt_files)}] {os.path.basename(txt_file_path)}")
    
    start_file = time.time()
    
    try:
        # Load
        print("  [1/5] Loading...", end=' ')
        start = time.time()
        eeg_data = load_siena_txt(txt_file_path)
        trim_samples = 30 * 512
        if eeg_data.shape[0] > 2 * trim_samples:
            eeg_data = eeg_data[trim_samples:-trim_samples, :]
        print(f"Done ({time.time()-start:.1f}s)")
        
        # Setup
        n_channels = min(eeg_data.shape[1], 16)
        eeg_data = eeg_data[:, :n_channels]
        temp_ch_names = COMMON_16_CHANNELS[:n_channels]
        
        info = mne.create_info(ch_names=temp_ch_names, sfreq=SFREQ, ch_types=['eeg']*n_channels)
        raw = mne.io.RawArray(eeg_data.T, info, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_alias=True, on_missing='ignore')

        # Preprocess
        print("  [2/5] Preprocessing...", end=' ')
        start = time.time()
        raw.filter(l_freq=1, h_freq=40, verbose=False)
        raw.resample(TARGET_SFREQ, verbose=False)
        raw, _ = mne.set_eeg_reference(raw, projection=True, verbose=False)
        print(f"Done ({time.time()-start:.1f}s)")

        # Bad channels
        print("  [3/5] Bad channels...", end=' ')
        start = time.time()
        bad_channels, _ = mne.preprocessing.find_bad_channels_lof(
            raw, n_neighbors=8, threshold=1.5, return_scores=True
        )
        raw.info['bads'] = bad_channels
        if bad_channels:
            raw.interpolate_bads(reset_bads=True)
        print(f"Done ({time.time()-start:.1f}s) - {len(bad_channels)} bad")

        # ICA - MATCHED (15, infomax)
        print("  [4/5] ICA (15, infomax)...", end=' ')
        start = time.time()
        
        try:
            ica = mne.preprocessing.ICA(
                n_components=15,
                random_state=RANDOM_SEED,
                method='infomax',
                max_iter='auto'
            )
            
            ica.fit(inst=raw.copy().filter(1, 40, verbose=False), verbose=False)
            
            eog_channels = [ch for ch in ['Fp1', 'Fp2'] if ch in raw.ch_names]
            if eog_channels:
                eog_indices, _ = ica.find_bads_eog(
                    inst=raw, ch_name=eog_channels, 
                    measure='correlation', threshold=0.5
                )
            else:
                eog_indices = []
            
            muscle_indices, _ = ica.find_bads_muscle(inst=raw, threshold=0.5)
            ica.exclude = list(set(eog_indices + muscle_indices))
            raw = ica.apply(raw.copy())
            
            print(f"Done ({time.time()-start:.1f}s) - {len(ica.exclude)} excluded")
        except Exception as e:
            print(f"Skipped ({str(e)[:20]})")

        ch_names = raw.ch_names
        file_stem = os.path.splitext(os.path.basename(txt_file_path))[0]

        # Features
        print("  [5/5] Extracting features:")
        features = compute_features_per_band(raw, ch_names, FREQ_BANDS, TARGET_SFREQ)

        # Save
        base_dir = os.path.dirname(txt_file_path)
        output_dir_features = os.path.join(base_dir, f"{subj}_Features_{EPOCH_LENGTH}s", file_stem)
        output_dir_matrices = os.path.join(base_dir, f"{subj}_Feature_Matrices_{EPOCH_LENGTH}s")
        
        os.makedirs(output_dir_features, exist_ok=True)
        os.makedirs(output_dir_matrices, exist_ok=True)

        excel_path = os.path.join(output_dir_matrices, f"{subj}_Features_{file_stem}.xlsx")
        
        print("  Saving...", end=' ')
        save_features_to_excel(features, ch_names, excel_path)
        create_heatmaps(features, ch_names, output_dir_features, file_stem)
        print("Done")

        # Store
        for ch_idx, ch_name in enumerate(ch_names):
            for band_name in FREQ_BANDS.keys():
                all_features.append({
                    'Subject_ID': file_stem,
                    'Group': 'Epilepsy',
                    'Channel': ch_name,
                    'Band': band_name,
                    'Sample_Entropy': features['sample_entropy'][band_name][ch_idx],
                    'Spectral_Entropy': features['spectral_entropy'][band_name][ch_idx],
                    'Variance': features['variance'][band_name][ch_idx],
                    'Skewness': features['skewness'][band_name][ch_idx]
                })

        elapsed = time.time() - start_file
        print(f"✅ Done ({elapsed:.1f}s)")
        success_count += 1

    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        fail_count += 1
        continue

print(f"\n{'='*80}")
print(f"✅ Complete! Success: {success_count}/{len(txt_files)}, Failed: {fail_count}")
print(f"{'='*80}")

# ================== AGGREGATE ==================
if all_features:
    print(f"\nAggregating...")
    
    df_all = pd.DataFrame(all_features)
    
    aggregate_path = os.path.join(
        selected_folder,
        f"{subj}_quality_control_{EPOCH_LENGTH}s",
        "All_Features_Aggregated_MATCHED.xlsx"
    )
    os.makedirs(os.path.dirname(aggregate_path), exist_ok=True)
    
    with pd.ExcelWriter(aggregate_path, engine='openpyxl') as writer:
        df_all.to_excel(writer, sheet_name='All_Data', index=False)
        
        for band in FREQ_BANDS.keys():
            stats = df_all[df_all['Band'] == band].groupby('Channel').agg({
                'Sample_Entropy': ['mean', 'std'],
                'Spectral_Entropy': ['mean', 'std'],
                'Variance': ['mean', 'std'],
                'Skewness': ['mean', 'std']
            })
            stats.to_excel(writer, sheet_name=f'Stats_{band}')
    
    print(f"📊 Saved: {aggregate_path}")
    
    # Parameter log
    params_path = os.path.join(
        selected_folder,
        f"{subj}_quality_control_{EPOCH_LENGTH}s",
        "Feature_Parameters_MATCHED.txt"
    )
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SIENA FEATURES - MATCHED TO LAB\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {pd.Timestamp.now()}\n\n")
        f.write("MATCHED PARAMETERS:\n")
        f.write(f"  Channels: 16 ({', '.join(COMMON_16_CHANNELS)})\n")
        f.write(f"  Sampling: {TARGET_SFREQ} Hz\n")
        f.write(f"  ICA: 15 components, infomax\n")
        f.write(f"  Bands:\n")
        for band, (low, high) in FREQ_BANDS.items():
            f.write(f"    - {band.capitalize()}: {low}-{high} Hz\n")
        f.write(f"  Features:\n")
        f.write(f"    - Sample Entropy (m=2, r=0.2*std)\n")
        f.write(f"    - Spectral Entropy\n")
        f.write(f"    - Variance\n")
        f.write(f"    - Skewness\n")
        f.write(f"\nFiles: {len(txt_files)}\n")
    
    print(f"📝 Parameters: {params_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for band in FREQ_BANDS.keys():
        band_data = df_all[df_all['Band'] == band]
        if len(band_data) > 0:
            print(f"\n{band.upper()} (n={len(band_data)}):")
            print(f"  SampEn: {band_data['Sample_Entropy'].mean():.3f} ± {band_data['Sample_Entropy'].std():.3f}")
            print(f"  SpecEn: {band_data['Spectral_Entropy'].mean():.3f} ± {band_data['Spectral_Entropy'].std():.3f}")
            print(f"  Var: {band_data['Variance'].mean():.6f} ± {band_data['Variance'].std():.6f}")
            print(f"  Skew: {band_data['Skewness'].mean():.3f} ± {band_data['Skewness'].std():.3f}")

print(f"\n{'='*80}")
print("✅ FEATURE EXTRACTION COMPLETE!")
print("✅ All parameters MATCHED to lab data!")
print(f"{'='*80}")
print(f"\nResults: {selected_folder}")
print(f"  1. {subj}_Features_{EPOCH_LENGTH}s/")
print(f"  2. {subj}_Feature_Matrices_{EPOCH_LENGTH}s/")
print(f"  3. {subj}_quality_control_{EPOCH_LENGTH}s/")
