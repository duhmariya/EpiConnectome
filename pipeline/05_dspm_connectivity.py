"""
EpiConnectome — Source-Level Connectivity Pipeline (Phase 2)
dSPM source localization + HCPMMP1 parcellation + wPLI connectivity

Pipeline:
    TXT files → Preprocessing → dSPM → 18 ROIs → wPLI → Excel + Plots

ROI Network Definitions (9 bilateral networks = 18 ROIs):
    ROI1: Prefrontal (ACC + IFC + DLPFC)
    ROI2: Motor (Somatosensory + Premotor + Cingulate)
    ROI3: Visual (V1 + MT+ + Dorsal stream)
    ROI4: Orbital Frontal
    ROI5: Temporal (Lateral + Ventral Visual)
    ROI6: Superior Parietal
    ROI7: Inferior Parietal + PCC + TPOJ
    ROI8: Auditory + Insula + Operculum
    ROI9: Medial Temporal Lobe (Hippocampus + PHG + Entorhinal) ← key for epilepsy

Usage:
    python 05_dspm_connectivity.py

Requirements:
    pip install mne mne-connectivity nilearn
    MNE fsaverage dataset (downloaded automatically on first run)
"""

import os
import glob
import random
import warnings
warnings.filterwarnings('ignore')

import mne
import mne_connectivity
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mne.set_log_level('WARNING')

# ── Reproducibility ───────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
mne.set_config('MNE_RANDOM_SEED', str(RANDOM_SEED))
mne.set_config('MNE_USE_NUMBA', 'false')

# ══════════════════════════════════════════════════════════════
#  CHANGE THESE PATHS FOR YOUR MACHINE
# ══════════════════════════════════════════════════════════════
TXT_FOLDER  = r"C:\Users\mariy\Desktop\Siena\siena-scalp-eeg-database-1.0.0\siena_txt_format"
OUTPUT_DIR  = r"C:\Users\mariy\Desktop\Siena\dSPM_Results"
# ══════════════════════════════════════════════════════════════

# ── Pipeline parameters ───────────────────────────────────────
SFREQ        = 512     # Original Siena sampling rate
TARGET_SFREQ = 1000    # Resampled rate
EPOCH_LENGTH = 10      # Epoch duration (seconds)
LAMBDA2      = 1.0 / 9.0  # Regularization for dSPM inverse

FREQ_BANDS = {
    'theta': (6,  8),
    'alpha': (8,  12),
    'beta':  (12, 30),
}

# Standard 16 channels
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3',  'C3',  'Cz', 'C4', 'T4',
    'T5',  'P3',  'Pz', 'P4'
]

# ── ROI definitions (HCPMMP1_combined parcellation) ──────────
ROI_DEFINITIONS = {
    'ROI1': {
        'name': 'Prefrontal',
        'areas': ['anterior cingulate and medial prefrontal cortex',
                  'inferior frontal cortex',
                  'dorsolateral prefrontal cortex'],
    },
    'ROI2': {
        'name': 'Motor',
        'areas': ['somatosensory and motor cortex',
                  'paracentral lobular and mid cingulate cortex',
                  'premotor cortex'],
    },
    'ROI3': {
        'name': 'Visual',
        'areas': ['dorsal stream visual cortex',
                  'early visual cortex',
                  'mt+ complex and neighboring visual areas',
                  'primary visual cortex (v1)'],
    },
    'ROI4': {
        'name': 'Orbital Frontal',
        'areas': ['orbital and polar frontal cortex'],
    },
    'ROI5': {
        'name': 'Temporal',
        'areas': ['lateral temporal cortex',
                  'medial temporal cortex',
                  'ventral stream visual cortex'],
    },
    'ROI6': {
        'name': 'Superior Parietal',
        'areas': ['superior parietal cortex'],
    },
    'ROI7': {
        'name': 'Inferior Parietal',
        'areas': ['inferior parietal cortex',
                  'posterior cingulate cortex',
                  'temporo-parieto-occipital junction'],
    },
    'ROI8': {
        'name': 'Auditory-Insula',
        'areas': ['auditory association cortex',
                  'early auditory cortex',
                  'insular and frontal opercular cortex',
                  'posterior opercular cortex'],
    },
    'ROI9': {
        'name': 'Medial Temporal (Hippocampal)',
        'areas': ['posterior cingulate cortex'],
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# STEP 1 — BUILD FORWARD MODEL (done once, reused for all files)
# ════════════════════════════════════════════════════════════════

def build_forward_model(sample_info, subjects_dir):
    """
    Build the dSPM forward model using fsaverage template.
    This is computed ONCE and reused for all seizure files.

    Returns: src, bem, fwd
    """
    print("\n── Building forward model (done once) ──────────────────")

    # Source space: oct5 gives ~20,000 sources (good balance of speed vs resolution)
    print("   Setting up source space (oct5)...")
    src = mne.setup_source_space(
        subject='fsaverage',
        spacing='oct5',
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False
    )

    # BEM model (3-layer: scalp, skull, brain)
    print("   Building BEM model...")
    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(
        subject='fsaverage',
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir,
        verbose=False
    )
    bem = mne.make_bem_solution(model, verbose=False)

    # Forward solution
    print("   Computing forward solution...")
    fwd = mne.make_forward_solution(
        sample_info,
        trans='fsaverage',
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        verbose=False
    )

    print(f"   ✅ Forward model ready — {fwd['nsource']} sources")
    return src, bem, fwd


# ════════════════════════════════════════════════════════════════
# STEP 2 — BUILD ROI LABELS FROM HCPMMP1
# ════════════════════════════════════════════════════════════════

def build_roi_labels(subjects_dir):
    """
    Load HCPMMP1_combined parcellation and combine into 18 ROIs
    (9 bilateral networks × 2 hemispheres).

    Returns: dict of {roi_key: mne.Label}
    """
    print("\n── Loading HCPMMP1 parcellation ────────────────────────")

    labels = mne.read_labels_from_annot(
        subject='fsaverage',
        parc='HCPMMP1_combined',
        subjects_dir=subjects_dir,
        verbose=False
    )
    print(f"   Loaded {len(labels)} HCPMMP1 labels")

    roi_labels = {}

    for roi_key, roi_def in ROI_DEFINITIONS.items():
        for hemi in ['lh', 'rh']:
            hemi_key = f"{roi_key}_{hemi.upper()}"
            combined = None

            for area_name in roi_def['areas']:
                matching = [
                    l for l in labels
                    if area_name.lower() in l.name.lower()
                    and l.hemi == hemi
                ]
                for m in matching:
                    combined = m if combined is None else combined + m

            if combined is not None and len(combined.vertices) > 0:
                roi_labels[hemi_key] = combined
                print(f"   {hemi_key}: {len(combined.vertices)} vertices")
            else:
                print(f"   ⚠️  {hemi_key}: no matching labels found")

    print(f"   ✅ {len(roi_labels)}/18 ROIs built")
    return roi_labels


# ════════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING
# ════════════════════════════════════════════════════════════════

def preprocess(eeg_data):
    """Full preprocessing: filter → resample → ref → bad channels → ICA."""
    trim = 30 * SFREQ
    if eeg_data.shape[0] > 2 * trim:
        eeg_data = eeg_data[trim:-trim, :]
    eeg_data = eeg_data[:, :16]

    info = mne.create_info(
        ch_names=CHANNEL_NAMES, sfreq=SFREQ, ch_types=['eeg'] * 16)
    raw = mne.io.RawArray(eeg_data.T, info, verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_alias=True, on_missing='ignore')

    raw.filter(l_freq=4, h_freq=40, verbose=False)
    raw.resample(TARGET_SFREQ, verbose=False)
    raw, _ = mne.set_eeg_reference(raw, projection=True, verbose=False)

    bad_channels, _ = mne.preprocessing.find_bad_channels_lof(
        raw, n_neighbors=8, threshold=1.5, return_scores=True)
    raw.info['bads'] = bad_channels
    if bad_channels:
        raw.interpolate_bads(reset_bads=True)

    ica = mne.preprocessing.ICA(
        n_components=15, random_state=RANDOM_SEED,
        method='infomax', max_iter='auto')
    ica.fit(inst=raw.copy().filter(4, 40, verbose=False), verbose=False)

    eog_ch = [ch for ch in ['Fp1', 'Fp2'] if ch in raw.ch_names]
    eog_idx = []
    if eog_ch:
        eog_idx, _ = ica.find_bads_eog(
            inst=raw, ch_name=eog_ch,
            measure='correlation', threshold=0.5)
    muscle_idx, _ = ica.find_bads_muscle(inst=raw, threshold=0.5)
    ica.exclude = list(set(eog_idx + muscle_idx))
    raw = ica.apply(raw.copy())

    return raw, bad_channels, ica.exclude


def make_epochs(raw):
    """Cut continuous raw into non-overlapping 10s epochs."""
    interval = TARGET_SFREQ * EPOCH_LENGTH
    events = np.array([[i, 0, 1] for i in range(0, raw.n_times, interval)])
    return mne.Epochs(
        raw, events, event_id=1, tmin=0,
        tmax=EPOCH_LENGTH - 1/TARGET_SFREQ,
        baseline=None, preload=True, verbose=False)


# ════════════════════════════════════════════════════════════════
# STEP 4 — dSPM SOURCE LOCALIZATION
# ════════════════════════════════════════════════════════════════

def run_dspm(epochs, fwd):
    """
    Compute dSPM inverse solution for each epoch.
    Returns list of SourceEstimate objects.
    """
    # Compute noise covariance from epochs
    noise_cov = mne.compute_covariance(
        epochs, method='empirical', verbose=False)

    # Build inverse operator
    inverse_op = make_inverse_operator(
        epochs.info, fwd, noise_cov,
        loose=0.2, depth=0.8, verbose=False)

    # Apply inverse to all epochs
    stcs = apply_inverse_epochs(
        epochs, inverse_op,
        lambda2=LAMBDA2,
        method='dSPM',
        pick_ori=None,
        return_generator=False,
        verbose=False)

    return stcs


# ════════════════════════════════════════════════════════════════
# STEP 5 — EXTRACT ROI TIME COURSES
# ════════════════════════════════════════════════════════════════

def extract_roi_timecourses(stcs, roi_labels, src):
    """
    Extract mean time course for each ROI from each epoch.
    Returns array of shape (n_epochs, n_rois, n_times).
    """
    roi_keys  = list(roi_labels.keys())
    roi_label_list = [roi_labels[k] for k in roi_keys]

    roi_tc = []
    for stc in stcs:
        epoch_tc = []
        for label in roi_label_list:
            ts = mne.extract_label_time_course(
                stc, [label], src, mode='mean', verbose=False)
            epoch_tc.append(ts.squeeze())
        roi_tc.append(np.array(epoch_tc))

    return np.array(roi_tc), roi_keys  # (n_epochs, n_rois, n_times)


# ════════════════════════════════════════════════════════════════
# STEP 6 — wPLI CONNECTIVITY ON ROI TIME COURSES
# ════════════════════════════════════════════════════════════════

def compute_roi_wpli(roi_tc, roi_keys, sfreq):
    """
    Compute wPLI connectivity between ROIs for each frequency band.
    Returns dict of {band: n_rois × n_rois matrix}.
    """
    n_epochs, n_rois, n_times = roi_tc.shape

    # Create MNE EpochsArray from ROI time courses
    info = mne.create_info(
        ch_names=roi_keys, sfreq=sfreq, ch_types=['eeg'] * n_rois)
    epochs_roi = mne.EpochsArray(roi_tc, info, verbose=False)

    fmin = [b[0] for b in FREQ_BANDS.values()]
    fmax = [b[1] for b in FREQ_BANDS.values()]

    con = mne_connectivity.spectral_connectivity_epochs(
        epochs_roi,
        method='wpli',
        mode='multitaper',
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        sfreq=sfreq,
        verbose=False
    )

    raw_matrix = con.get_data(output='dense')

    conn_dict = {}
    for idx, band in enumerate(FREQ_BANDS.keys()):
        m = raw_matrix[:, :, idx]
        m = m + m.T
        np.fill_diagonal(m, 0)
        conn_dict[band] = m

    return conn_dict


# ════════════════════════════════════════════════════════════════
# STEP 7 — SAVE OUTPUTS
# ════════════════════════════════════════════════════════════════

def save_outputs(conn_dict, roi_keys, file_stem, out_dir):
    """Save connectivity matrices to Excel and circle plots to PNG."""
    os.makedirs(out_dir, exist_ok=True)

    # Short ROI names for plots
    short_names = [k.replace('ROI', 'R').replace('_LH', '-L').replace('_RH', '-R')
                   for k in roi_keys]

    # Excel
    excel_path = os.path.join(out_dir, f'{file_stem}_source_connectivity.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for band, matrix in conn_dict.items():
            df = pd.DataFrame(matrix, index=roi_keys, columns=roi_keys)
            df.to_excel(writer, sheet_name=band.capitalize())

        # ROI key reference sheet
        roi_ref = []
        for key, roi_def in ROI_DEFINITIONS.items():
            for hemi in ['LH', 'RH']:
                full_key = f"{key}_{hemi}"
                if full_key in roi_keys:
                    roi_ref.append({
                        'ROI': full_key,
                        'Network': roi_def['name'],
                        'Hemisphere': 'Left' if hemi == 'LH' else 'Right',
                        'Areas': ', '.join(roi_def['areas'])
                    })
        pd.DataFrame(roi_ref).to_excel(writer, sheet_name='ROI_Reference', index=False)

    # Circle plots
    n_rois = len(roi_keys)
    idx_row, idx_col = np.triu_indices(n_rois, k=1)
    indices = (idx_row, idx_col)

    for band, con_matrix in conn_dict.items():
        upper_tri = con_matrix[idx_row, idx_col]
        try:
            fig, _ = plot_connectivity_circle(
                upper_tri,
                short_names,
                indices=indices,
                vmin=0.0,
                vmax=max(con_matrix.max(), 0.01),
                title=f'{file_stem} — {band.capitalize()} (Source Level)',
                colormap='RdYlBu_r',
                colorbar=True,
                show=False
            )
            fig.savefig(
                os.path.join(out_dir, f'source_circle_{band}.png'),
                dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"   ⚠️  Circle plot failed for {band}: {e}")

    return excel_path


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EpiConnectome — dSPM Source-Level Connectivity Pipeline")
    print("=" * 60)

    # Find TXT files
    txt_files = sorted([
        f for f in glob.glob(os.path.join(TXT_FOLDER, '*.txt'))
        if os.path.basename(f).startswith('PN')
    ])
    print(f"\nFound {len(txt_files)} seizure files")

    # ── Download fsaverage if needed ──────────────────────────
    print("\n── Setting up MNE fsaverage template ───────────────────")
    data_path   = mne.datasets.sample.data_path()
    subjects_dir = data_path / "subjects"
    mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
    print(f"   subjects_dir: {subjects_dir}")

    # ── Build ROI labels (once) ───────────────────────────────
    roi_labels = build_roi_labels(subjects_dir)
    if len(roi_labels) == 0:
        raise RuntimeError("No ROI labels built — check HCPMMP1 parcellation")

    # ── Build forward model using first file (once) ───────────
    print("\n── Loading first file to build forward model ───────────")
    first_data = np.loadtxt(txt_files[0])
    if first_data.shape[0] > 30 * SFREQ * 2:
        first_data = first_data[30*SFREQ:-30*SFREQ, :]
    first_data = first_data[:, :16]

    info_tmp = mne.create_info(
        ch_names=CHANNEL_NAMES, sfreq=SFREQ, ch_types=['eeg'] * 16)
    raw_tmp = mne.io.RawArray(first_data.T, info_tmp, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_tmp.set_montage(montage, match_alias=True, on_missing='ignore')
    raw_tmp.resample(TARGET_SFREQ, verbose=False)
    raw_tmp, _ = mne.set_eeg_reference(raw_tmp, projection=True, verbose=False)

    src, bem, fwd = build_forward_model(raw_tmp.info, subjects_dir)

    # ── Process all files ─────────────────────────────────────
    all_results = []
    success, failed = 0, 0

    for file_idx, txt_path in enumerate(txt_files, 1):
        file_stem = os.path.splitext(os.path.basename(txt_path))[0]
        print(f"\n{'='*60}")
        print(f"[{file_idx}/{len(txt_files)}] {file_stem}")
        print(f"{'='*60}")

        out_dir = os.path.join(OUTPUT_DIR, file_stem)

        # Skip if already processed
        excel_check = os.path.join(out_dir, f'{file_stem}_source_connectivity.xlsx')
        if os.path.exists(excel_check):
            print(f"   ⏭️  Already processed, skipping")
            skipped_data = pd.read_excel(excel_check, sheet_name='Theta', index_col=0)
            for band in FREQ_BANDS.keys():
                m = pd.read_excel(excel_check, sheet_name=band.capitalize(), index_col=0).values
                all_results.append({
                    'Subject': file_stem,
                    'Band': band,
                    'Mean_wPLI': round(np.mean(m[m != 0]), 4),
                    'Max_wPLI': round(m.max(), 4),
                    'N_ROIs': m.shape[0],
                })
            success += 1
            continue

        try:
            # Load
            print("   [1/5] Loading...", end=' ')
            try:
                eeg_data = np.loadtxt(txt_path)
            except:
                eeg_data = pd.read_csv(txt_path, sep=r'\s+', header=None).values
            print(f"shape {eeg_data.shape}")

            # Preprocess
            print("   [2/5] Preprocessing...", end=' ')
            raw, bad_ch, excluded_ica = preprocess(eeg_data)
            print(f"bad={bad_ch if bad_ch else 'none'}, ICA={len(excluded_ica)}")

            # Epoch
            print("   [3/5] Epoching...", end=' ')
            epochs = make_epochs(raw)
            print(f"{len(epochs)} epochs")

            if len(epochs) < 2:
                print("   ⚠️  Too few epochs, skipping")
                failed += 1
                continue

            # dSPM
            print("   [4/5] Running dSPM source localization...")
            stcs = run_dspm(epochs, fwd)

            # ROI time courses
            print("   [5/5] Extracting ROI time courses + wPLI...", end=' ')
            roi_tc, roi_keys = extract_roi_timecourses(stcs, roi_labels, src)
            print(f"shape {roi_tc.shape}")

            # wPLI on ROI time courses
            conn_dict = compute_roi_wpli(roi_tc, roi_keys, TARGET_SFREQ)

            for band, m in conn_dict.items():
                print(f"      {band}: mean wPLI = {np.mean(m[m!=0]):.3f}")

            # Save
            excel_path = save_outputs(conn_dict, roi_keys, file_stem, out_dir)
            print(f"   ✅ Saved → {out_dir}")

            # Store summary
            for band, m in conn_dict.items():
                all_results.append({
                    'Subject':   file_stem,
                    'Band':      band,
                    'Mean_wPLI': round(np.mean(m[m != 0]), 4),
                    'Max_wPLI':  round(m.max(), 4),
                    'N_ROIs':    m.shape[0],
                    'N_Epochs':  len(epochs),
                    'Bad_Ch':    len(bad_ch),
                    'ICA_Out':   len(excluded_ica),
                })
            success += 1

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

    # ── Save summary atlas ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"COMPLETE — Success: {success}, Failed: {failed}")
    print(f"{'='*60}")

    if all_results:
        df = pd.DataFrame(all_results)
        atlas_path = os.path.join(OUTPUT_DIR, 'SOURCE_connectivity_atlas.xlsx')
        df.to_excel(atlas_path, index=False)
        print(f"\n📊 Source atlas saved → {atlas_path}")
        print("\n── Mean wPLI per band (source level) ──")
        print(df.groupby('Band')[['Mean_wPLI', 'Max_wPLI']].mean().round(3).to_string())


if __name__ == "__main__":
    main()