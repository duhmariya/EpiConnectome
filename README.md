# EpiConnectome 🧠⚡

**Source-Level Seizure Severity Classification using Graph Attention Networks**

> Brainhack School 2026 · National Central University, Taiwan  
> **Mariya Nissar** · First Year PhD, Electrical Engineering · NCU Taiwan

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![MNE](https://img.shields.io/badge/MNE-1.6+-green.svg)](https://mne.tools)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Brainhack](https://img.shields.io/badge/Brainhack-School%202026-purple.svg)](https://brainhackmtl.github.io/school)

---

## 🎯 What This Project Does

EpiConnectome is an end-to-end, fully reproducible pipeline that:

1. Takes **raw clinical EEG** files (EDF format) from epilepsy patients
2. Applies **dSPM source localization** to recover 18 cortical regions of interest (ROIs)
3. Computes **wPLI connectivity matrices** across theta, alpha, and beta bands
4. Trains a **GATv2 Graph Attention Network** to classify seizure severity
5. Generates **interpretable attention maps** showing which brain regions drive severity

**NOT:** "We discovered new epilepsy biomarkers"  
**YES:** "We built a shareable, well-documented pipeline anyone can run on public data"

---

## 📊 Key Results (v4)

| Band | K-Fold Acc | K-Fold F1 | K-Fold AUC | LOSO AUC | Permutation p |
|------|-----------|-----------|-----------|---------|--------------|
| **Alpha** | **97.5%** | **97.1%** | **99.2%** | **1.00** (8/12 subj) | **0.000** |
| **Beta** | 92.1% | 91.4% | 98.8% | 1.00 (7/12 subj) | 0.000 |
| **Theta** | 92.1% | 92.1% | 97.2% | 1.00 (5/12 subj) | 0.000 |

- All bands: **p=0.000** (1000 permutation tests) — observed wPLI difference exceeds all 1000 null values
- LOSO AUC=1.00 in all evaluable subjects — rank-ordering of severity **generalizes across patients**
- GATv2 attention maps reveal **frequency-specific cortical propagation patterns**

---

## 🗂️ Repository Structure

```
EEG_Project/
├── README.md
├── requirements.txt
├── environment.yml                        # Conda environment specification
├── LICENSE
├── EpiConnectome_Tutorial.ipynb           # Step-by-step pipeline walkthrough
│
├── pipeline/
│   ├── 01_siena_to_txt_converter.py   # EDF → TXT extraction with annotation timestamps
│   ├── 02_channelbasedcode_Siena.py   # Preprocessing · ICA · Channel-level wPLI · circle plots
│   ├── 03_siena_feature_extraction.py # EEG feature extraction (SampEn, SpecEn, Var, Skew)
│   ├── 04_siencehisto.py              # Connectivity matrix visualizations / histograms
│   ├── 05_dspm_connectivity.py        # dSPM source localization · 18 ROIs · Source wPLI atlas
│   └── 06_gatv2_classification.py     # GATv2 classifier · K-Fold · LOSO · SVM baseline
│
├── results/
│   ├── GATv2_Results_v4/              # All GATv2 v4 output figures and metrics
│   │   ├── SUMMARY_FIGURE.png
│   │   ├── brain_topo_alpha/beta/theta.png
│   │   ├── confusion_*.png
│   │   ├── roc_*.png
│   │   ├── permutation_*.png
│   │   ├── heatmap_*.png
│   │   ├── gatv2_vs_svm.png
│   │   ├── GATv2_metrics.xlsx
│   │   └── subject_severity_report.xlsx
│   ├── connectivity_plots/            # Channel-level wPLI circle plots (47 seizures)
│   ├── dSPM_Results/                  # Source localization outputs
│   ├── Siena_Epilepsy_Channel_analysis_10s/
│   ├── Siena_Epilepsy_Feature_Matrices_10s/
│   ├── Siena_Epilepsy_Features_10s/
│   ├── Siena_Epilepsy_matrices_10s/
│   └── Siena_Epilepsy_quality_control_10s/
│
└── data/
    └── README.md                      # Instructions to download Siena dataset
```

---

## 🔬 Dataset

**Siena Scalp EEG Database** — publicly available on PhysioNet

| Property | Value |
|----------|-------|
| Source | PhysioNet (free, no login) |
| Citation | Detti, P. (2020) · doi:[10.13026/5d4a-j060](https://doi.org/10.13026/5d4a-j060) |
| Patients | 14 epilepsy subjects |
| Seizures | 47 total |
| Channels | 29 scalp electrodes (10-20 system) |
| Sample rate | 512 Hz |
| Format | EDF + clinical annotations |
| Size | ~13 GB |

> **The data is NOT included in this repository.** Download instructions:
> ```bash
> wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/
> ```
> Or visit: https://physionet.org/content/siena-scalp-eeg/1.0.0/

---

## 🧠 Pipeline Overview

```
Raw EDF Files  →  01: EDF→TXT Conversion  →  02: Preprocessing + Channel wPLI
                                                        ↓
                                          03: EEG Feature Extraction
                                          04: Connectivity Visualizations
                                                        ↓
                                          05: dSPM Source Localization → 18 ROIs
                                                        ↓
                                          06: GATv2 Classification → Severity + Attention Maps
```

### Script Descriptions

| Script | What it does |
|--------|-------------|
| `01_siena_to_txt_converter.py` | Reads EDF files, extracts ictal segments (±30s) using clinical annotation timestamps |
| `02_channelbasedcode_Siena.py` | Bandpass filter (4–40 Hz), bad channel detection (LOF), ICA, channel-level wPLI, circle plots, bar charts |
| `03_siena_feature_extraction.py` | Sample Entropy, Spectral Entropy, Variance, Skewness per channel per band |
| `04_siencehisto.py` | Histogram and matrix visualizations of connectivity results |
| `05_dspm_connectivity.py` | dSPM source localization on fsaverage, HCPMMP1 parcellation, 18×18 source wPLI atlas |
| `06_gatv2_classification.py` | GATv2 graph classifier, 5-fold CV, LOSO, permutation tests, SVM baseline, attention maps |

### Why wPLI?
Standard coherence is inflated by **volume conduction** — the same cortical source appearing in multiple scalp electrodes. wPLI discards zero-lag interactions, retaining only true brain-to-brain connectivity.

### Why dSPM?
Scalp electrodes mix signals from many cortical areas. dSPM inverts the EEG forward model to recover source-level signals, giving us **18 anatomically meaningful ROIs** instead of 29 scalp positions.

### 9 Bilateral Cortical Networks (18 ROIs total)

| # | Network | Key for epilepsy? |
|---|---------|-------------------|
| R1 | Prefrontal (ACC + DLPFC) | |
| R2 | Motor cortex | |
| R3 | Visual cortex (V1 + MT+) | |
| R4 | Orbital frontal | |
| R5 | Temporal (lateral + ventral) | |
| R6 | Superior parietal | |
| R7 | Inferior parietal + PCC | |
| R8 | Auditory + Insula | |
| R9 | **Medial Temporal (Hippocampal)** | ⭐ Primary epilepsy hub |

---

## 🤖 GATv2 Model

```
Input: 18-node graph (nodes = ROIs, edges = wPLI weights)
Node features: [mean_wPLI, max_wPLI, std_wPLI, ROI_identity (×18), hemispheric_asymmetry]
               → 22-dimensional feature vector per node

GATv2 Layer 1: 16 hidden × 2 heads, ELU activation
               + Attention entropy regularization (λ=0.01)
GATv2 Layer 2: 8 hidden × 2 heads
               + MC Dropout (p=0.5, 50 passes at inference)
Global Mean Pool → 16-dim graph embedding
Classifier: Linear(16→32) → ReLU → Dropout → Linear(32→2)

Output: High / Low severity label + attention weight per ROI
```

**Key design decisions:**
- **Attention regularization** (entropy penalty): prevents attention collapse — forces the model to attend to multiple ROIs
- **MC Dropout** (50 passes): uncertainty estimation on attention weights
- **Balanced classes** (50th percentile split): ~20 High / ~20 Low per band — prevents trivially zero LOSO F1
- **Dual validation**: 5-fold CV (within-subject) + LOSO (cross-subject generalization)
- **SVM baseline**: RBF-SVM on flat wPLI features for honest comparison

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/duhmariya/EEG_Project.git
cd EEG_Project

# Option A: Conda environment (recommended)
conda env create -f environment.yml
conda activate epiconnectome

# Option B: Virtual environment
python -m venv epienv
source epienv/bin/activate  # Windows: epienv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** For GPU support, install PyTorch with CUDA before running `pip install torch-geometric`. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 🚀 Usage

### Step 1 — Download the data
Download the Siena Scalp EEG Database from PhysioNet:
```bash
wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/
```

### Step 2 — Update paths
At the top of each script, update the `INPUT_DIR` and `OUTPUT_DIR` variables to point to your local data.

### Step 3 — Run the full pipeline in order

```bash
# Step 1: Extract seizure segments from EDF files
python pipeline/01_siena_to_txt_converter.py

# Step 2: Preprocess and compute channel-level wPLI + circle plots
python pipeline/02_channelbasedcode_Siena.py

# Step 3: Extract EEG features (entropy, variance, skewness)
python pipeline/03_siena_feature_extraction.py

# Step 4: Visualize connectivity matrices
python pipeline/04_siencehisto.py

# Step 5: dSPM source localization + source-level wPLI atlas
python pipeline/05_dspm_connectivity.py

# Step 6: GATv2 classification + all results figures
python pipeline/06_gatv2_classification.py
```

> Scripts 01–04 can run independently on channel-level data.  
> Script 05 requires MNE's fsaverage template (auto-downloaded on first run).  
> Script 06 requires the output atlas from Script 05.  
> Expected runtime for Script 06: ~20–40 minutes on CPU, ~5–10 minutes on GPU.

### 📓 Tutorial Notebook
For a guided walkthrough of the full pipeline with explanations and demo outputs, open:
```bash
jupyter notebook EpiConnectome_Tutorial.ipynb
```

---

## 📈 Reproducibility

- Fixed random seed: `RANDOM_SEED = 42` throughout
- All hyperparameters are defined at the top of each script — no hidden configuration files
- Results in `results/` were generated with the exact code in this repository
- Data source is public and freely available — no institutional access required

---

## 🏗️ What's in the Current Version

| Fix | Description |
|-----|-------------|
| **FIX 1** | LOSO AUC now computed only over evaluable folds (both classes present); `n_valid/n_total` reported explicitly |
| **FIX 2** | Permutation test now runs for all bands including `combined` |
| **FIX 3** | Label threshold changed from 75th → **50th percentile** (median split) for balanced High/Low classes |
| **FIX 4** | LOSO ROC curves only plot folds with valid AUC — no misleading NaN lines |
| **FIX 5** | **Balanced accuracy** added to all metric tables (more honest under class imbalance) |
| **FIX 6** | **SVM baseline** added for GATv2 comparison |

---

## 📂 Outputs Generated by Script 06

| File | Description |
|------|-------------|
| `SUMMARY_FIGURE.png` | Classification performance + permutation test significance |
| `confusion_{band}.png` | K-Fold confusion matrix per band |
| `confusion_{band}_loso.png` | LOSO confusion matrix per band |
| `roc_{band}.png` | K-Fold ROC curves per band |
| `roc_{band}_loso.png` | LOSO ROC curves (valid folds only) |
| `attention_{band}_v4.png` | Bar chart of ROI attention weights |
| `brain_topo_{band}.png` | Brain topography of attention weights |
| `heatmap_{band}.png` | wPLI connectivity Low vs High vs Difference |
| `permutation_{band}.png` | Permutation test histogram |
| `gatv2_vs_svm.png` | GATv2 vs SVM comparison |
| `subject_severity_heatmap.png` | Patient-level wPLI severity overview |
| `GATv2_metrics.xlsx` | All metrics (K-Fold + LOSO + SVM baseline + Summary) |
| `subject_severity_report.xlsx` | Per-seizure severity labels and thresholds |

---

## 🔭 Future Work

- [ ] Clinical ground truth labels (seizure duration, post-ictal duration) as severity proxy
- [ ] Pre-ictal vs ictal connectivity comparison
- [ ] Larger dataset validation (TUH EEG Corpus)
- [ ] Patient-adaptive severity thresholds
- [ ] Graph embedding visualization (t-SNE/UMAP) for subject vs severity separation

---

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{nissar2026epiconnectome,
  author       = {Nissar, Mariya},
  title        = {EpiConnectome: Source-Level Seizure Severity Classification 
                  using Graph Attention Networks},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/duhmariya/EEG_Project}},
  note         = {Brainhack School 2026}
}
```

**Dataset citation:**
> Detti, P. (2020). Siena Scalp EEG Database. PhysioNet. https://doi.org/10.13026/5d4a-j060

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Brainhack School 2026** organizers and TAs
- **PhysioNet** for open data access
- **MNE-Python** and **PyTorch Geometric** communities

---

*EpiConnectome · Brainhack School 2026 · github.com/duhmariya/EEG_Project*
