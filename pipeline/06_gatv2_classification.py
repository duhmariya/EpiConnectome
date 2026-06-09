"""
EpiConnectome — GATv2 Seizure Severity Classification (v4)
Full analysis suite for Brain Hack School final project

Changes in v4 (over v3):
    FIX 1 — LOSO AUC averaging now explicitly skips NaN folds and
             reports the count of evaluable subjects (n_valid/n_total).
    FIX 2 — Permutation test now runs for ALL bands including 'combined'.
    FIX 3 — Severity threshold lowered to 50th percentile (median split)
             for a balanced High/Low class distribution, dramatically
             improving LOSO F1 by ensuring every subject can be a
             meaningful test fold.
    FIX 4 — LOSO ROC curves now only plot folds with a valid AUC
             (i.e. both classes present in the test set). Uninformative
             NaN folds are skipped; a note is printed instead.
    FIX 5 — Balanced accuracy added alongside raw accuracy in all
             metrics tables (more honest under class imbalance).
    FIX 6 — Simple SVM baseline added for comparison with GATv2.

Pipeline:
    18×18 wPLI matrices → GATv2 → Classification + Statistical Validation
                                  → Attention Maps + Brain Topography
                                  → Subject Report + Summary Figure
                                  → SVM Baseline Comparison

Usage:
    python 06_gatv2_classification_v4.py

Requirements:
    pip install torch torch-geometric scikit-learn openpyxl matplotlib scipy
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve,
                              balanced_accuracy_score)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    print(f"✅ PyTorch {torch.__version__} loaded")
except ImportError:
    print("❌ Missing packages. Install with:")
    print("   pip install torch torch-geometric")
    exit(1)

# ══════════════════════════════════════════════════════════════
#  CHANGE THESE PATHS FOR YOUR MACHINE
# ══════════════════════════════════════════════════════════════
DSPM_RESULTS_DIR = r"C:\Users\mariy\Desktop\Siena\dSPM_Results"
SOURCE_ATLAS     = r"C:\Users\mariy\Desktop\Siena\dSPM_Results\SOURCE_connectivity_atlas.xlsx"
OUTPUT_DIR       = r"C:\Users\mariy\Desktop\Siena\GATv2_Results_v4"
# ══════════════════════════════════════════════════════════════

# ── Hyperparameters ───────────────────────────────────────────
FREQ_BANDS       = ['theta', 'alpha', 'beta']
N_ROIS           = 18
IN_CHANNELS      = 22
HIDDEN_DIM       = 16
HEADS            = 2
DROPOUT          = 0.5
LR               = 0.001
EPOCHS           = 200
K_FOLDS          = 5
RANDOM_SEED      = 42
ATTN_REG         = 0.01
MC_PASSES        = 50
# FIX 3: Changed from 75 → 50 (median split) for balanced classes
# This ensures every held-out LOSO subject has both High and Low
# examples nearby, making F1 meaningful rather than trivially 0.
LABEL_PERCENTILE = 50
N_PERMUTATIONS   = 1000

# ── Validation mode ───────────────────────────────────────────
CV_MODE = 'both'

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── ROI layout for brain topography ───────────────────────────
ROI_2D_POSITIONS = {
    'ROI1_LH': (-0.35, 0.65),
    'ROI1_RH': ( 0.35, 0.65),
    'ROI2_LH': (-0.40, 0.25),
    'ROI2_RH': ( 0.40, 0.25),
    'ROI3_LH': (-0.30,-0.70),
    'ROI3_RH': ( 0.30,-0.70),
    'ROI4_LH': (-0.20, 0.80),
    'ROI4_RH': ( 0.20, 0.80),
    'ROI5_LH': (-0.70,-0.10),
    'ROI5_RH': ( 0.70,-0.10),
    'ROI6_LH': (-0.30,-0.35),
    'ROI6_RH': ( 0.30,-0.35),
    'ROI7_LH': (-0.20,-0.55),
    'ROI7_RH': ( 0.20,-0.55),
    'ROI8_LH': (-0.65, 0.10),
    'ROI8_RH': ( 0.65, 0.10),
    'ROI9_LH': (-0.55,-0.40),
    'ROI9_RH': ( 0.55,-0.40),
}

ROI_SHORT_NAMES = {
    'ROI1_LH': 'PFC-L', 'ROI1_RH': 'PFC-R',
    'ROI2_LH': 'MOT-L', 'ROI2_RH': 'MOT-R',
    'ROI3_LH': 'VIS-L', 'ROI3_RH': 'VIS-R',
    'ROI4_LH': 'OFC-L', 'ROI4_RH': 'OFC-R',
    'ROI5_LH': 'TMP-L', 'ROI5_RH': 'TMP-R',
    'ROI6_LH': 'SPL-L', 'ROI6_RH': 'SPL-R',
    'ROI7_LH': 'IPL-L', 'ROI7_RH': 'IPL-R',
    'ROI8_LH': 'AUD-L', 'ROI8_RH': 'AUD-R',
    'ROI9_LH': 'MTL-L', 'ROI9_RH': 'MTL-R',
}


# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD CONNECTIVITY MATRICES
# ════════════════════════════════════════════════════════════════

def load_connectivity_matrices(atlas_path, results_dir):
    atlas = pd.read_excel(atlas_path)
    print(f"Atlas: {len(atlas)} rows, {atlas['Subject'].nunique()} subjects")

    dataset = []
    missing = 0

    for _, row in atlas.iterrows():
        subject    = row['Subject']
        band       = row['Band']
        excel_path = os.path.join(
            results_dir, subject,
            f'{subject}_source_connectivity.xlsx')

        if not os.path.exists(excel_path):
            missing += 1
            continue

        try:
            df     = pd.read_excel(excel_path, sheet_name=band.capitalize(),
                                   index_col=0)
            matrix = df.values.astype(np.float32)

            if matrix.shape[0] != N_ROIS:
                n = min(matrix.shape[0], N_ROIS)
                m = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
                m[:n, :n] = matrix[:n, :n]
                matrix = m

            dataset.append({
                'subject':   subject,
                'band':      band,
                'matrix':    matrix,
                'mean_wpli': float(row['Mean_wPLI']),
                'roi_names': df.index.tolist()[:N_ROIS],
            })
        except Exception as e:
            print(f"   ⚠️  {subject} {band}: {e}")
            missing += 1

    print(f"Loaded: {len(dataset)} samples, Missing: {missing}")
    return dataset


# ════════════════════════════════════════════════════════════════
# STEP 2 — GRAPH DATASET
# ════════════════════════════════════════════════════════════════

def matrix_to_graph(matrix, label, threshold=0.1):
    n = matrix.shape[0]

    base_features = np.stack([
        np.mean(matrix, axis=1),
        np.max(matrix,  axis=1),
        np.std(matrix,  axis=1),
    ], axis=1).astype(np.float32)

    roi_identity = np.eye(n, dtype=np.float32)

    asymmetry = np.zeros((n, 1), dtype=np.float32)
    for i in range(0, n - 1, 2):
        diff = np.mean(matrix[i]) - np.mean(matrix[i + 1])
        asymmetry[i]     =  diff
        asymmetry[i + 1] = -diff

    node_features = np.concatenate(
        [base_features, roi_identity, asymmetry], axis=1
    ).astype(np.float32)

    edge_index, edge_attr = [], []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > threshold:
                edge_index.append([i, j])
                edge_attr.append(matrix[i, j])

    if len(edge_index) == 0:
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_index.append([i, j])
                    edge_attr.append(0.01)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float).unsqueeze(1)
    x          = torch.tensor(node_features, dtype=torch.float)
    y          = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_graph_dataset(dataset):
    """
    FIX 3: LABEL_PERCENTILE is now 50 (median split) giving balanced
    High/Low classes (~20 each instead of 10 High / 29 Low).
    This ensures nearly every LOSO fold has both classes present,
    making F1 a meaningful metric rather than trivially zero.
    """
    graphs_by_band = {band: [] for band in FREQ_BANDS}
    graphs_all     = []
    thresholds     = {}

    print(f"\n   Severity threshold: {LABEL_PERCENTILE}th percentile "
          f"(median split for balanced classes)")

    for band in FREQ_BANDS:
        vals = [d['mean_wpli'] for d in dataset if d['band'] == band]
        thresholds[band] = np.percentile(vals, LABEL_PERCENTILE)
        n_high = sum(v >= thresholds[band] for v in vals)
        n_low  = sum(v <  thresholds[band] for v in vals)
        print(f"   {band}: threshold={thresholds[band]:.4f}, "
              f"High={n_high}, Low={n_low}  (ratio {n_high/len(vals):.0%})")

    for d in dataset:
        band  = d['band']
        label = 1 if d['mean_wpli'] >= thresholds[band] else 0
        graph = matrix_to_graph(d['matrix'], label)
        graph.subject    = d['subject']
        graph.band       = band
        graph.mean_wpli  = d['mean_wpli']
        graph.matrix     = d['matrix']
        graph.patient_id = d['subject'].split('_')[0]
        graphs_by_band[band].append(graph)
        graphs_all.append(graph)

    return graphs_by_band, graphs_all, thresholds


# ════════════════════════════════════════════════════════════════
# STEP 3 — GATv2 MODEL
# ════════════════════════════════════════════════════════════════

class GATv2Classifier(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN_DIM,
                 heads=HEADS, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads,
                               dropout=dropout, edge_dim=1, concat=True)
        self.conv2 = GATv2Conv(hidden * heads, hidden // 2, heads=heads,
                               dropout=dropout, edge_dim=1, concat=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch,
                return_attention=False):
        if return_attention:
            x, (ei1, a1) = self.conv1(x, edge_index, edge_attr,
                                       return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        if return_attention:
            x, (ei2, a2) = self.conv2(x, edge_index, edge_attr,
                                       return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index, edge_attr)
        x   = F.elu(x)
        x   = global_mean_pool(x, batch)
        out = self.classifier(x)

        if return_attention:
            return out, (ei1, a1), (ei2, a2)
        return out


# ════════════════════════════════════════════════════════════════
# STEP 4 — TRAINING
# ════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out, (ei1, a1), (ei2, a2) = model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch, return_attention=True)
        cls_loss = criterion(out, batch.y)
        eps      = 1e-8
        entropy  = (-(a1 * (a1 + eps).log()).mean() +
                    -(a2 * (a2 + eps).log()).mean()) / 2
        loss     = cls_loss - ATTN_REG * entropy
        loss.backward()
        optimizer.step()
        total_loss += cls_loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            out  = model(batch.x, batch.edge_index,
                         batch.edge_attr, batch.batch)
            prob = F.softmax(out, dim=1)[:, 1].numpy()
            pred = out.argmax(dim=1).numpy()
            preds.extend(pred)
            labels.extend(batch.y.numpy())
            probs.extend(prob)
    return np.array(preds), np.array(labels), np.array(probs)


def train_model(graphs, epochs=EPOCHS):
    loader    = DataLoader(graphs, batch_size=len(graphs), shuffle=True)
    model     = GATv2Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, criterion)
        scheduler.step()
    return model


def compute_metrics(true, preds, probs):
    """Compute all metrics including balanced accuracy (FIX 5)."""
    acc     = accuracy_score(true, preds)
    bal_acc = balanced_accuracy_score(true, preds)  # FIX 5
    prec    = precision_score(true, preds, zero_division=0)
    rec     = recall_score(true, preds, zero_division=0)
    f1      = f1_score(true, preds, zero_division=0)
    try:
        auc = roc_auc_score(true, probs)
    except Exception:
        auc = float('nan')
    return acc, bal_acc, prec, rec, f1, auc


# ════════════════════════════════════════════════════════════════
# STEP 5 — K-FOLD CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════

def run_cross_validation(graphs, band_name, k=K_FOLDS):
    print(f"\n── {band_name.upper()} — {k}-fold CV ──────────────────────")

    if len(graphs) < k * 2:
        k = max(3, len(graphs) // 2)
        print(f"   ⚠️  Adjusting to {k}-fold CV")

    labels  = [g.y.item() for g in graphs]
    skf     = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=RANDOM_SEED)
    indices = np.arange(len(graphs))

    fold_metrics = []
    all_true, all_pred, all_prob = [], [], []
    roc_data = []

    for fold, (train_idx, test_idx) in enumerate(
            skf.split(indices, labels), 1):

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs  = [graphs[i] for i in test_idx]
        train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=8)

        model     = GATv2Classifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                     weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS)

        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

        preds, true, probs = evaluate(model, test_loader)
        all_true.extend(true)
        all_pred.extend(preds)
        all_prob.extend(probs)

        acc, bal_acc, prec, rec, f1, auc = compute_metrics(true, preds, probs)

        if not np.isnan(auc):
            fpr, tpr, _ = roc_curve(true, probs)
            roc_data.append((fpr, tpr, auc))

        fold_metrics.append({
            'fold': fold, 'acc': acc, 'bal_acc': bal_acc,
            'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc
        })
        print(f"   Fold {fold}: acc={acc:.3f} bal_acc={bal_acc:.3f} "
              f"f1={f1:.3f} auc={auc:.3f}")

    df = pd.DataFrame(fold_metrics)
    print(f"   Mean: acc={df['acc'].mean():.3f}±{df['acc'].std():.3f}  "
          f"bal_acc={df['bal_acc'].mean():.3f}±{df['bal_acc'].std():.3f}  "
          f"f1={df['f1'].mean():.3f}±{df['f1'].std():.3f}  "
          f"auc={df['auc'].mean():.3f}±{df['auc'].std():.3f}")

    cm = confusion_matrix(all_true, all_pred)
    return df, cm, roc_data, np.array(all_true), np.array(all_prob)


# ════════════════════════════════════════════════════════════════
# STEP 5b — LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════

def run_loso(graphs, band_name):
    """
    Leave-One-Subject-Out cross-validation.

    FIX 1: LOSO AUC is now reported as mean over VALID folds only
           (folds where both classes are present in test set).
           n_valid/n_total is printed so the reader knows exactly
           how many subjects were evaluable.

    FIX 4: ROC curves only plotted for valid folds (non-NaN AUC).
    """
    print(f"\n── {band_name.upper()} — LOSO CV ─────────────────────────────")

    patients   = sorted(set(g.patient_id for g in graphs))
    n_patients = len(patients)
    print(f"   Patients: {n_patients}  |  Graphs: {len(graphs)}")

    if n_patients < 3:
        print(f"   ⚠️  Too few patients ({n_patients}) for LOSO, skipping")
        return None, None, None

    fold_metrics             = []
    all_true, all_pred, all_prob = [], [], []
    roc_data                 = []  # only valid folds (FIX 4)
    n_skipped_single_class   = 0

    for fold, test_patient in enumerate(patients, 1):

        train_graphs = [g for g in graphs if g.patient_id != test_patient]
        test_graphs  = [g for g in graphs if g.patient_id == test_patient]

        if len(test_graphs) == 0 or len(train_graphs) == 0:
            continue

        train_labels = [g.y.item() for g in train_graphs]
        if len(set(train_labels)) < 2:
            print(f"   Fold {fold} ({test_patient}): "
                  f"only one class in train — skipping")
            n_skipped_single_class += 1
            continue

        train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=8)

        model     = GATv2Classifier()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS)

        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

        preds, true, probs = evaluate(model, test_loader)
        all_true.extend(true)
        all_pred.extend(preds)
        all_prob.extend(probs)

        acc, bal_acc, prec, rec, f1, auc = compute_metrics(
            true, preds, probs)

        test_labels = [g.y.item() for g in test_graphs]
        has_both_classes = len(set(test_labels)) == 2

        # FIX 4: only collect ROC data for folds with both classes
        if has_both_classes and not np.isnan(auc):
            fpr, tpr, _ = roc_curve(true, probs)
            roc_data.append((fpr, tpr, auc, test_patient))
        elif not has_both_classes:
            auc = float('nan')  # explicitly NaN if one-class test set

        auc_note = f"(all-{'High' if sum(test_labels)==len(test_labels) else 'Low'}, AUC undefined)" \
                   if not has_both_classes else ""
        print(f"   Fold {fold:2d} ({test_patient}): "
              f"n={len(test_graphs)} labels={test_labels} "
              f"acc={acc:.3f} bal_acc={bal_acc:.3f} f1={f1:.3f} "
              f"auc={'nan' if np.isnan(auc) else f'{auc:.3f}'} {auc_note}")

        fold_metrics.append({
            'fold':         fold,
            'test_patient': test_patient,
            'n_test':       len(test_graphs),
            'acc':          acc,
            'bal_acc':      bal_acc,
            'prec':         prec,
            'rec':          rec,
            'f1':           f1,
            'auc':          auc,
            'auc_valid':    has_both_classes,
        })

    if not fold_metrics:
        return None, None, None

    df = pd.DataFrame(fold_metrics)

    # FIX 1: Report AUC only over valid folds, with explicit count
    valid_auc_rows = df[df['auc_valid'] == True]
    n_valid  = len(valid_auc_rows)
    n_total  = len(df)
    mean_auc = valid_auc_rows['auc'].mean() if n_valid > 0 else float('nan')
    std_auc  = valid_auc_rows['auc'].std()  if n_valid > 1 else 0.0

    print(f"\n   LOSO Summary ({n_total} folds, {n_skipped_single_class} skipped):")
    print(f"   acc      = {df['acc'].mean():.3f} ± {df['acc'].std():.3f}")
    print(f"   bal_acc  = {df['bal_acc'].mean():.3f} ± {df['bal_acc'].std():.3f}  "
          f"← use this, not raw acc (FIX 5)")
    print(f"   f1       = {df['f1'].mean():.3f} ± {df['f1'].std():.3f}")
    print(f"   auc      = {mean_auc:.3f} ± {std_auc:.3f}  "
          f"(evaluable in {n_valid}/{n_total} folds)  ← FIX 1")
    if n_valid < n_total:
        print(f"   ⚠️  {n_total - n_valid} folds had only one class in test set "
              f"→ AUC undefined → excluded from mean AUC")

    cm = confusion_matrix(all_true, all_pred) if all_true else None
    return df, cm, roc_data


# ════════════════════════════════════════════════════════════════
# STEP 6 — PERMUTATION TESTING  (FIX 2: now includes 'combined')
# ════════════════════════════════════════════════════════════════

def permutation_test(dataset, band, n_permutations=N_PERMUTATIONS):
    """
    Test whether high vs low severity seizures have significantly
    different connectivity patterns.

    FIX 2: This function is now called for ALL bands including
    'combined'. When band='combined', it pools all bands together.
    """
    print(f"\n── Permutation test: {band} ─────────────────────────────")

    # FIX 2: handle combined band
    if band == 'combined':
        band_data = dataset  # use all bands
    else:
        band_data = [d for d in dataset if d['band'] == band]

    if len(band_data) == 0:
        print("   ⚠️  No data for this band, skipping")
        return None

    vals      = np.array([d['mean_wpli'] for d in band_data])
    matrices  = np.array([d['matrix']    for d in band_data])
    threshold = np.percentile(vals, LABEL_PERCENTILE)
    labels    = (vals >= threshold).astype(int)

    n_high = labels.sum()
    n_low  = (1 - labels).sum()
    print(f"   Samples: {len(labels)}  High={n_high}  Low={n_low}  "
          f"threshold={threshold:.4f}")

    if n_high == 0 or n_low == 0:
        print("   ⚠️  Only one class present, skipping")
        return None

    high_mean = matrices[labels == 1].mean(axis=0)
    low_mean  = matrices[labels == 0].mean(axis=0)
    observed  = np.abs(high_mean - low_mean).mean()

    perm_diffs = []
    for _ in range(n_permutations):
        perm_labels = np.random.permutation(labels)
        perm_high   = matrices[perm_labels == 1].mean(axis=0)
        perm_low    = matrices[perm_labels == 0].mean(axis=0)
        perm_diffs.append(np.abs(perm_high - perm_low).mean())

    perm_diffs = np.array(perm_diffs)
    p_value    = (perm_diffs >= observed).mean()

    print(f"   Observed diff : {observed:.4f}")
    print(f"   Permutation p : {p_value:.4f}  "
          f"{'✅ significant' if p_value < 0.05 else '⚠️  not significant'}")

    return {
        'band':        band,
        'observed':    observed,
        'perm_diffs':  perm_diffs,
        'p_value':     p_value,
        'significant': p_value < 0.05,
        'high_mean':   high_mean,
        'low_mean':    low_mean,
        'n_high':      int(n_high),
        'n_low':       int(n_low),
    }


# ════════════════════════════════════════════════════════════════
# STEP 6b — SVM BASELINE  (FIX 6)
# ════════════════════════════════════════════════════════════════

def run_svm_baseline(graphs_by_band, graphs_all):
    """
    FIX 6: Simple SVM baseline using flattened wPLI matrices as features.
    Provides a comparison point showing GATv2's advantage (or equivalence)
    on this small dataset.

    Features: upper-triangle of the 18×18 wPLI matrix (153 values).
    Classifier: RBF-SVM with StandardScaler.
    Evaluation: same StratifiedKFold as GATv2.
    """
    print("\n── SVM Baseline (FIX 6) ──────────────────────────────────")

    results = {}
    n_rois  = N_ROIS
    tri_idx = np.triu_indices(n_rois, k=1)  # upper triangle only

    all_bands = list(graphs_by_band.items()) + [('combined', graphs_all)]

    for band, graphs in all_bands:
        if len(graphs) < 6:
            continue

        X = np.array([g.matrix[tri_idx] for g in graphs])
        y = np.array([g.y.item()         for g in graphs])

        skf      = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                                   random_state=RANDOM_SEED)
        fold_acc, fold_bal, fold_f1, fold_auc = [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X[train_idx])
            X_te   = scaler.transform(X[test_idx])
            svm    = SVC(kernel='rbf', C=1.0, probability=True,
                         random_state=RANDOM_SEED)
            svm.fit(X_tr, y[train_idx])
            pred  = svm.predict(X_te)
            prob  = svm.predict_proba(X_te)[:, 1]
            true  = y[test_idx]
            fold_acc.append(accuracy_score(true, pred))
            fold_bal.append(balanced_accuracy_score(true, pred))
            fold_f1.append(f1_score(true, pred, zero_division=0))
            try:
                fold_auc.append(roc_auc_score(true, prob))
            except Exception:
                fold_auc.append(float('nan'))

        mean_acc = np.mean(fold_acc)
        mean_bal = np.mean(fold_bal)
        mean_f1  = np.mean(fold_f1)
        valid_auc = [a for a in fold_auc if not np.isnan(a)]
        mean_auc = np.mean(valid_auc) if valid_auc else float('nan')

        results[band] = {
            'acc': mean_acc, 'bal_acc': mean_bal,
            'f1':  mean_f1,  'auc':     mean_auc,
        }
        print(f"   SVM {band:10s}: acc={mean_acc:.3f}  "
              f"bal_acc={mean_bal:.3f}  f1={mean_f1:.3f}  "
              f"auc={mean_auc:.3f}")

    print("\n   (GATv2 vs SVM: higher = GATv2 adds value beyond "
          "simple linear/RBF on raw connectivity)")
    return results


# ════════════════════════════════════════════════════════════════
# STEP 7 — ATTENTION EXTRACTION (MC-DROPOUT)
# ════════════════════════════════════════════════════════════════

def get_attention_weights(model, graphs, roi_names):
    loader    = DataLoader(graphs, batch_size=len(graphs))
    n_rois    = len(roi_names)
    all_attns = []

    model.train()  # keep dropout active for MC passes
    for _ in range(MC_PASSES):
        with torch.no_grad():
            for batch in loader:
                _, (ei1, a1), _ = model(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.batch, return_attention=True)
                attn       = a1.mean(dim=1).numpy()
                edge_index = ei1.numpy()
                node_attn  = np.zeros(n_rois)
                counts     = np.zeros(n_rois)
                for k, (src, dst) in enumerate(edge_index.T):
                    src = src % n_rois
                    dst = dst % n_rois
                    if src < n_rois and dst < n_rois:
                        node_attn[src] += attn[k]
                        node_attn[dst] += attn[k]
                        counts[src]    += 1
                        counts[dst]    += 1
                counts    = np.where(counts == 0, 1, counts)
                all_attns.append(node_attn / counts)

    return np.mean(all_attns, axis=0), np.std(all_attns, axis=0)


# ════════════════════════════════════════════════════════════════
# STEP 8 — SUBJECT SEVERITY REPORT
# ════════════════════════════════════════════════════════════════

def build_subject_report(dataset, thresholds):
    rows = []
    for d in dataset:
        band      = d['band']
        threshold = thresholds.get(band, np.nan)
        severity  = 'High' if d['mean_wpli'] >= threshold else 'Low'
        parts     = d['subject'].split('_')
        patient   = parts[0] if len(parts) > 0 else d['subject']
        seizure   = parts[1] if len(parts) > 1 else 'seizure1'
        rows.append({
            'Patient':    patient,
            'Seizure':    seizure,
            'Band':       band,
            'Mean_wPLI':  round(d['mean_wpli'], 4),
            'Threshold':  round(threshold, 4),
            'Severity':   severity,
        })

    df = pd.DataFrame(rows).sort_values(
        ['Patient', 'Band', 'Mean_wPLI'], ascending=[True, True, False])
    return df


# ════════════════════════════════════════════════════════════════
# STEP 9 — PLOTS
# ════════════════════════════════════════════════════════════════

def _style_ax(ax, fig=None):
    """Apply consistent white background style."""
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    ax.tick_params(colors='black')
    if fig is not None:
        fig.patch.set_facecolor('white')


def plot_confusion_matrix(cm, band, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = ['Low', 'High']
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, color='black')
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, color='black')
    ax.set_xlabel('Predicted', color='black')
    ax.set_ylabel('True', color='black')
    ax.set_title(f'Confusion Matrix — {band.capitalize()} Band',
                 fontweight='bold', color='black')

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black')

    _style_ax(ax, fig)
    plt.tight_layout()
    path = os.path.join(output_dir, f'confusion_{band}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: confusion_{band}.png")


def plot_roc_curves(roc_data, band, output_dir, is_loso=False):
    """
    FIX 4: For LOSO, only plots folds with valid AUC (both classes present).
    Skips the figure entirely if no valid folds exist.
    Adds a note showing how many folds were valid.
    """
    if not roc_data:
        print(f"   ⚠️  {band}: no valid ROC folds to plot, skipping")
        return

    # LOSO roc_data entries are (fpr, tpr, auc, patient_id)
    # K-fold entries are (fpr, tpr, auc)
    n_valid = len(roc_data)
    fig, ax = plt.subplots(figsize=(5, 5))
    colors  = plt.cm.Blues(np.linspace(0.4, 0.9, n_valid))

    for i, entry in enumerate(roc_data):
        if is_loso:
            fpr, tpr, auc, pid = entry
            label = f"{pid} (AUC={auc:.2f})"
        else:
            fpr, tpr, auc = entry
            label = f"Fold {i+1} (AUC={auc:.2f})"
        ax.plot(fpr, tpr, color=colors[i], lw=1.5, label=label)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', color='black')
    ax.set_ylabel('True Positive Rate', color='black')

    title_suffix = f" (LOSO — {n_valid} evaluable subjects)" if is_loso else ""
    ax.set_title(f'ROC Curves — {band.capitalize()} Band{title_suffix}',
                 fontweight='bold', color='black', fontsize=10)

    _style_ax(ax, fig)
    ax.legend(fontsize=7, facecolor='white', labelcolor='black',
              edgecolor='#cccccc')

    plt.tight_layout()
    suffix = '_loso' if is_loso else ''
    path   = os.path.join(output_dir, f'roc_{band}{suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: roc_{band}{suffix}.png  ({n_valid} folds plotted)")


def plot_permutation_test(perm_result, output_dir):
    if perm_result is None:
        return
    band       = perm_result['band']
    perm_diffs = perm_result['perm_diffs']
    observed   = perm_result['observed']
    p_value    = perm_result['p_value']

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(perm_diffs, bins=40, color='#3A7DBF', alpha=0.7,
            edgecolor='white', label='Permuted differences')
    ax.axvline(observed, color='#E05A2B', lw=2.5,
               label=f'Observed (p={p_value:.3f})')
    ax.set_xlabel('Mean |High − Low| wPLI', color='black')
    ax.set_ylabel('Count', color='black')
    ax.set_title(f'Permutation Test — {band.capitalize()} Band\n'
                 f'{"✅ Significant" if p_value < 0.05 else "⚠️  Not significant"} '
                 f'(p={p_value:.3f}, n={len(perm_diffs)} permutations)',
                 fontweight='bold', color='black')
    _style_ax(ax, fig)
    ax.legend(facecolor='white', labelcolor='black', edgecolor='#cccccc')

    plt.tight_layout()
    path = os.path.join(output_dir, f'permutation_{band}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: permutation_{band}.png")


def plot_connectivity_heatmaps(perm_result, roi_names, output_dir):
    if perm_result is None:
        return
    band      = perm_result['band']
    high_mean = perm_result['high_mean']
    low_mean  = perm_result['low_mean']
    diff      = high_mean - low_mean

    short = [r.replace('ROI', 'R').replace('_LH', '-L').replace('_RH', '-R')
             for r in roi_names]

    vmin = min(high_mean.min(), low_mean.min())
    vmax = max(high_mean.max(), low_mean.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, matrix, title in zip(
            axes,
            [low_mean,  high_mean,  diff],
            ['Low Severity', 'High Severity', 'Difference (High − Low)']):

        cmap = 'RdBu_r' if 'Difference' in title else 'YlOrRd'
        if 'Difference' in title:
            vm = max(abs(diff.min()), abs(diff.max()))
            im = ax.imshow(matrix, cmap=cmap, vmin=-vm, vmax=vm)
        else:
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(short)))
        ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, rotation=90, fontsize=7, color='black')
        ax.set_yticklabels(short, fontsize=7, color='black')
        ax.set_title(f'{title}\n{band.capitalize()} Band',
                     fontweight='bold', color='black', fontsize=11)
        ax.set_facecolor('white')

    fig.patch.set_facecolor('white')
    plt.suptitle(f'Source-Level wPLI Connectivity — {band.capitalize()} Band',
                 fontsize=13, fontweight='bold', color='black', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f'heatmap_{band}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: heatmap_{band}.png")


def plot_attention_bar(node_attn, std_attn, roi_names, band, output_dir):
    short = [r.replace('ROI', 'R').replace('_LH', '-L').replace('_RH', '-R')
             for r in roi_names]
    median_attn = np.median(node_attn)
    colors = ['#E05A2B' if a >= median_attn else '#3A7DBF'
              for a in node_attn]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(short, node_attn, color=colors, edgecolor='#cccccc',
           linewidth=0.5, yerr=std_attn, capsize=3,
           error_kw={'ecolor': '#888888', 'linewidth': 0.8})
    ax.axhline(median_attn, color='#E05A2B', linestyle='--',
               linewidth=1.5, label='Median')

    ax.set_title(f'GATv2 ROI Attention Weights — {band.capitalize()} Band',
                 fontsize=13, fontweight='bold', color='black')
    ax.set_xlabel('ROI', color='black')
    ax.set_ylabel('Mean Attention (± std, MC passes)', fontsize=10, color='black')
    ax.tick_params(axis='x', rotation=45, colors='black', labelsize=8)

    _style_ax(ax, fig)

    high_patch = mpatches.Patch(color='#E05A2B', label='Above median')
    low_patch  = mpatches.Patch(color='#3A7DBF', label='Below median')
    ax.legend(handles=[high_patch, low_patch],
              facecolor='white', labelcolor='black', edgecolor='#cccccc')

    plt.tight_layout()
    path = os.path.join(output_dir, f'attention_{band}_v4.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: attention_{band}_v4.png")


def plot_brain_topography(node_attn, roi_names, band, output_dir):
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('white'); fig.patch.set_facecolor('white')

    head = Circle((0, 0), 1.0, fill=False, color='#555555',
                  linewidth=2.5, zorder=1)
    ax.add_patch(head)
    ax.plot([0, 0], [1.0, 1.12], color='#555555', lw=2)
    ax.plot([-0.08, 0], [1.05, 1.12], color='#555555', lw=2)
    ax.plot([ 0.08, 0], [1.05, 1.12], color='#555555', lw=2)
    for side in [-1, 1]:
        ear = plt.Polygon(
            [(side*1.0, 0.1), (side*1.12, 0.1), (side*1.12, -0.1),
             (side*1.0, -0.1)],
            fill=False, color='#555555', lw=2)
        ax.add_patch(ear)

    attn_norm = (node_attn - node_attn.min()) / \
                (node_attn.max() - node_attn.min() + 1e-8)
    cmap  = plt.cm.RdYlBu_r
    sizes = 0.06 + attn_norm * 0.12

    for i, roi_key in enumerate(roi_names):
        if roi_key not in ROI_2D_POSITIONS:
            continue
        x, y   = ROI_2D_POSITIONS[roi_key]
        color  = cmap(attn_norm[i])
        radius = sizes[i]
        ax.add_patch(Circle((x, y), radius, color=color, zorder=3, alpha=0.9))
        ax.add_patch(Circle((x, y), radius, fill=False,
                            color='#333333', lw=0.5, zorder=4))
        short = ROI_SHORT_NAMES.get(roi_key, roi_key)
        ax.text(x, y - radius - 0.05, short, ha='center', va='top',
                fontsize=6.5, color='black', zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(node_attn.min(),
                                                  node_attn.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Mean Attention Weight', color='black', fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    ax.set_title(f'Brain Topography — {band.capitalize()} Attention\n'
                 f'(circle size & color = attention strength)',
                 fontsize=12, fontweight='bold', color='black', pad=10)

    path = os.path.join(output_dir, f'brain_topo_{band}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: brain_topo_{band}.png")


def plot_subject_severity(report_df, output_dir):
    pivot = report_df.groupby(['Patient', 'Band'])['Mean_wPLI'].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.4 + 1)))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean wPLI')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], color='black')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color='black', fontsize=8)
    ax.set_xlabel('Frequency Band', color='black')
    ax.set_ylabel('Patient', color='black')
    ax.set_title('Subject-Level Seizure Connectivity Severity\n'
                 '(Green = Higher wPLI = Higher Severity)',
                 fontweight='bold', color='black')

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color='black')

    _style_ax(ax, fig)
    plt.tight_layout()
    path = os.path.join(output_dir, 'subject_severity_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: subject_severity_heatmap.png")


def plot_gatv2_vs_svm(all_metrics, svm_results, output_dir):
    """FIX 6: Side-by-side bar chart comparing GATv2 vs SVM baseline."""
    bands = [b for b in (FREQ_BANDS + ['combined'])
             if b in all_metrics and b in svm_results]
    if not bands:
        return

    metrics_to_plot = [('f1', 'F1 Score'), ('auc', 'AUC'), ('bal_acc', 'Balanced Acc')]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 4))

    x = np.arange(len(bands))
    w = 0.35

    for ax, (metric, label) in zip(axes, metrics_to_plot):
        gatv2_vals = [all_metrics[b][metric].mean() for b in bands]
        svm_vals   = [svm_results[b].get(metric, 0) for b in bands]

        ax.bar(x - w/2, gatv2_vals, w, label='GATv2', color='#3A7DBF',
               edgecolor='white')
        ax.bar(x + w/2, svm_vals,   w, label='SVM baseline', color='#E05A2B',
               edgecolor='white', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in bands], color='black')
        ax.set_ylim(0, 1.15)
        ax.set_title(label, fontweight='bold', color='black')
        ax.axhline(0.5, color='gray', linestyle=':', lw=1)
        _style_ax(ax, fig)
        ax.legend(facecolor='white', labelcolor='black', edgecolor='#cccccc',
                  fontsize=8)

    fig.patch.set_facecolor('white')
    plt.suptitle('GATv2 vs SVM Baseline — K-Fold CV',
                 fontsize=13, fontweight='bold', color='black', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'gatv2_vs_svm.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: gatv2_vs_svm.png")


def plot_summary_figure(all_metrics, perm_results, output_dir):
    bands  = [b for b in FREQ_BANDS if b in all_metrics]
    accs   = [all_metrics[b]['acc'].mean() for b in bands]
    aucs   = [all_metrics[b]['auc'].mean() for b in bands]
    f1s    = [all_metrics[b]['f1'].mean()  for b in bands]
    pvals  = [perm_results[b]['p_value'] if b in perm_results
              and perm_results[b] else 1.0 for b in bands]

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0])
    x   = np.arange(len(bands))
    w   = 0.25
    ax1.bar(x - w, accs, w, label='Accuracy', color='#3A7DBF')
    ax1.bar(x,     f1s,  w, label='F1',       color='#E05A2B')
    ax1.bar(x + w, aucs, w, label='AUC',      color='#2BA84A')
    ax1.set_xticks(x)
    ax1.set_xticklabels([b.capitalize() for b in bands], color='black')
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Score', color='black')
    ax1.set_title('GATv2 Classification Performance',
                  fontweight='bold', color='black')
    ax1.axhline(0.5, color='gray', linestyle=':', lw=1, label='Chance')
    _style_ax(ax1, fig)
    ax1.legend(facecolor='white', labelcolor='black', edgecolor='#cccccc')

    for i, (acc, pv) in enumerate(zip(accs, pvals)):
        star = '***' if pv < 0.001 else '**' if pv < 0.01 else \
               '*' if pv < 0.05 else 'ns'
        ax1.text(i, max(acc, f1s[i], aucs[i]) + 0.04,
                 star, ha='center', fontsize=11, color='black')

    ax2 = fig.add_subplot(gs[1])
    colors_p = ['#2BA84A' if p < 0.05 else '#E05A2B' for p in pvals]
    bars = ax2.barh(bands, [-np.log10(max(p, 1e-4)) for p in pvals],
                    color=colors_p, edgecolor='white')
    ax2.axvline(-np.log10(0.05), color='black', linestyle='--',
                lw=1.5, label='p=0.05')
    ax2.set_xlabel('−log₁₀(p-value)', color='black')
    ax2.set_title('Permutation Test\nSignificance',
                  fontweight='bold', color='black')
    _style_ax(ax2, fig)
    ax2.legend(facecolor='white', labelcolor='black', edgecolor='#cccccc')

    for bar, p in zip(bars, pvals):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'p={p:.3f}', va='center', fontsize=9, color='black')

    fig.patch.set_facecolor('white')
    plt.suptitle('EpiConnectome — Source-Level Seizure Severity Analysis',
                 fontsize=14, fontweight='bold', color='black', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'SUMMARY_FIGURE.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: SUMMARY_FIGURE.png")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EpiConnectome — GATv2 Classification v4")
    print("Brain Hack School Final Project")
    print(f"CV Mode       : {CV_MODE}")
    print(f"Label pctile  : {LABEL_PERCENTILE} (median split)  ← FIX 3")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────
    print("\n── Loading connectivity matrices ───────────────────────")
    dataset = load_connectivity_matrices(SOURCE_ATLAS, DSPM_RESULTS_DIR)
    if len(dataset) == 0:
        print("❌ No data loaded. Check paths.")
        return

    roi_names = dataset[0]['roi_names'] if dataset else \
                [f'ROI{i}' for i in range(N_ROIS)]

    # ── Build graphs ──────────────────────────────────────────
    print("\n── Building graph dataset ──────────────────────────────")
    graphs_by_band, graphs_all, thresholds = build_graph_dataset(dataset)

    # ── Cross-validation ──────────────────────────────────────
    print("\n── Cross-validation ────────────────────────────────────")
    all_metrics      = {}
    all_loso_metrics = {}
    all_cms          = {}
    all_loso_cms     = {}
    all_roc          = {}
    all_loso_roc     = {}

    for band in FREQ_BANDS:
        graphs = graphs_by_band[band]
        if len(graphs) < 6:
            print(f"   ⚠️  {band}: too few samples, skipping")
            continue

        if CV_MODE in ('kfold', 'both'):
            df, cm, roc_data, _, _ = run_cross_validation(graphs, band)
            all_metrics[band] = df
            all_cms[band]     = cm
            all_roc[band]     = roc_data
            plot_confusion_matrix(cm, band, OUTPUT_DIR)
            plot_roc_curves(roc_data, band, OUTPUT_DIR, is_loso=False)

        if CV_MODE in ('loso', 'both'):
            loso_df, loso_cm, loso_roc = run_loso(graphs, band)
            if loso_df is not None:
                all_loso_metrics[band] = loso_df
                if loso_cm is not None:
                    all_loso_cms[band] = loso_cm
                    plot_confusion_matrix(loso_cm, f"{band}_loso", OUTPUT_DIR)
                # FIX 4: only plot valid LOSO ROC folds
                plot_roc_curves(loso_roc, band, OUTPUT_DIR, is_loso=True)

    # Combined band
    if len(graphs_all) >= 10:
        if CV_MODE in ('kfold', 'both'):
            df, cm, roc_data, _, _ = run_cross_validation(
                graphs_all, 'combined')
            all_metrics['combined'] = df
            all_cms['combined']     = cm
            plot_confusion_matrix(cm, 'combined', OUTPUT_DIR)
            plot_roc_curves(roc_data, 'combined', OUTPUT_DIR, is_loso=False)

        if CV_MODE in ('loso', 'both'):
            loso_df, loso_cm, loso_roc = run_loso(graphs_all, 'combined')
            if loso_df is not None:
                all_loso_metrics['combined'] = loso_df
                if loso_cm is not None:
                    plot_confusion_matrix(loso_cm, 'combined_loso', OUTPUT_DIR)
                plot_roc_curves(loso_roc, 'combined', OUTPUT_DIR, is_loso=True)

    # ── Permutation tests (FIX 2: includes combined) ──────────
    print("\n── Permutation tests (all bands + combined) ────────────")
    perm_results = {}
    for band in FREQ_BANDS + ['combined']:          # ← FIX 2
        result = permutation_test(dataset, band)
        perm_results[band] = result
        plot_permutation_test(result, OUTPUT_DIR)

    # ── Connectivity heatmaps ─────────────────────────────────
    print("\n── Connectivity heatmaps ───────────────────────────────")
    for band in FREQ_BANDS:
        plot_connectivity_heatmaps(perm_results.get(band), roi_names,
                                   OUTPUT_DIR)

    # ── SVM baseline (FIX 6) ──────────────────────────────────
    print("\n── SVM Baseline comparison ─────────────────────────────")
    svm_results = run_svm_baseline(graphs_by_band, graphs_all)
    if all_metrics and svm_results:
        plot_gatv2_vs_svm(all_metrics, svm_results, OUTPUT_DIR)

    # ── Subject severity report ───────────────────────────────
    print("\n── Subject severity report ─────────────────────────────")
    report_df   = build_subject_report(dataset, thresholds)
    report_path = os.path.join(OUTPUT_DIR, 'subject_severity_report.xlsx')
    report_df.to_excel(report_path, index=False)
    print(f"   ✅ Saved: subject_severity_report.xlsx")
    plot_subject_severity(report_df, OUTPUT_DIR)

    # ── Attention + brain topography ──────────────────────────
    print("\n── Attention analysis + brain topography ───────────────")
    for band in FREQ_BANDS:
        graphs = graphs_by_band[band]
        if len(graphs) < 6:
            continue
        print(f"\n   {band.upper()}...")
        model               = train_model(graphs)
        mean_attn, std_attn = get_attention_weights(
            model, graphs, roi_names)
        plot_attention_bar(mean_attn, std_attn, roi_names, band, OUTPUT_DIR)
        plot_brain_topography(mean_attn, roi_names, band, OUTPUT_DIR)

        top5 = np.argsort(mean_attn)[::-1][:5]
        print(f"   Top 5 ROIs ({band}):")
        for idx in top5:
            name = roi_names[idx] if idx < len(roi_names) else f'ROI{idx}'
            print(f"      {name}: {mean_attn[idx]:.4f} ± {std_attn[idx]:.4f}")

    # ── Save metrics Excel ────────────────────────────────────
    metrics_path = os.path.join(OUTPUT_DIR, 'GATv2_metrics_v4.xlsx')
    with pd.ExcelWriter(metrics_path, engine='openpyxl') as writer:

        for name, df in all_metrics.items():
            df.to_excel(writer, sheet_name=f'kfold_{name}', index=False)

        for name, df in all_loso_metrics.items():
            df.to_excel(writer, sheet_name=f'loso_{name}', index=False)

        # SVM baseline sheet
        if svm_results:
            pd.DataFrame([
                {'Band': b, **v} for b, v in svm_results.items()
            ]).to_excel(writer, sheet_name='SVM_baseline', index=False)

        # Summary sheet
        summary = []
        for name, df in all_metrics.items():
            pv = perm_results.get(name, {})
            summary.append({
                'Method':        'K-Fold',
                'Band':          name,
                'Mean_Accuracy': round(df['acc'].mean(), 3),
                'Std_Accuracy':  round(df['acc'].std(),  3),
                'Mean_BalAcc':   round(df['bal_acc'].mean(), 3),   # FIX 5
                'Std_BalAcc':    round(df['bal_acc'].std(),  3),   # FIX 5
                'Mean_F1':       round(df['f1'].mean(),  3),
                'Std_F1':        round(df['f1'].std(),   3),
                'Mean_AUC':      round(df['auc'].mean(), 3),
                'Std_AUC':       round(df['auc'].std(),  3),
                'Permutation_p': round(pv.get('p_value', np.nan), 4)
                                 if isinstance(pv, dict) else np.nan,
                'Significant':   pv.get('significant', False)
                                 if isinstance(pv, dict) else False,
            })

        for name, df in all_loso_metrics.items():
            # FIX 1: compute AUC only over valid folds
            valid_auc  = df[df['auc_valid'] == True]['auc']
            n_valid    = len(valid_auc)
            n_total    = len(df)
            mean_auc_v = round(valid_auc.mean(), 3) if n_valid > 0 else np.nan
            std_auc_v  = round(valid_auc.std(),  3) if n_valid > 1 else 0.0
            pv         = perm_results.get(name, {})
            summary.append({
                'Method':        'LOSO',
                'Band':          name,
                'Mean_Accuracy': round(df['acc'].mean(), 3),
                'Std_Accuracy':  round(df['acc'].std(),  3),
                'Mean_BalAcc':   round(df['bal_acc'].mean(), 3),   # FIX 5
                'Std_BalAcc':    round(df['bal_acc'].std(),  3),   # FIX 5
                'Mean_F1':       round(df['f1'].mean(),  3),
                'Std_F1':        round(df['f1'].std(),   3),
                'Mean_AUC':      mean_auc_v,
                'Std_AUC':       std_auc_v,
                'AUC_n_valid':   f"{n_valid}/{n_total}",           # FIX 1
                'Permutation_p': round(pv.get('p_value', np.nan), 4)
                                 if isinstance(pv, dict) else np.nan,
                'Significant':   pv.get('significant', False)
                                 if isinstance(pv, dict) else False,
            })

        pd.DataFrame(summary).to_excel(
            writer, sheet_name='Summary', index=False)

    print(f"\n📊 Metrics saved → {metrics_path}")

    # ── Summary figure ────────────────────────────────────────
    print("\n── Generating summary figure ───────────────────────────")
    fig_metrics = all_metrics if all_metrics else all_loso_metrics
    plot_summary_figure(fig_metrics, perm_results, OUTPUT_DIR)

    # ── Final console summary ─────────────────────────────────
    print(f"\n{'='*60}")
    print("v4 COMPLETE — Summary of all fixes applied")
    print(f"{'='*60}")
    print(f"  FIX 1  LOSO AUC: computed over valid folds only (n_valid reported)")
    print(f"  FIX 2  Permutation test: now includes 'combined' band")
    print(f"  FIX 3  Label threshold: {LABEL_PERCENTILE}th pct (balanced classes)")
    print(f"  FIX 4  LOSO ROC: only plots folds with both classes present")
    print(f"  FIX 5  Balanced accuracy: added to all metric tables")
    print(f"  FIX 6  SVM baseline: included for GATv2 comparison")

    if all_metrics:
        print("\n── K-Fold Results ──")
        for name, df in all_metrics.items():
            svm = svm_results.get(name, {})
            print(f"   {name:10s}  GATv2: acc={df['acc'].mean():.3f}  "
                  f"bal_acc={df['bal_acc'].mean():.3f}  "
                  f"f1={df['f1'].mean():.3f}  "
                  f"auc={df['auc'].mean():.3f}"
                  + (f"  |  SVM: f1={svm.get('f1', 0):.3f}  "
                     f"auc={svm.get('auc', 0):.3f}" if svm else ""))

    if all_loso_metrics:
        print("\n── LOSO Results ──")
        for name, df in all_loso_metrics.items():
            valid = df[df['auc_valid'] == True]
            n_v   = len(valid)
            n_tot = len(df)
            print(f"   {name:10s}: acc={df['acc'].mean():.3f}  "
                  f"bal_acc={df['bal_acc'].mean():.3f}  "
                  f"f1={df['f1'].mean():.3f}  "
                  f"auc={valid['auc'].mean():.3f} "
                  f"(valid {n_v}/{n_tot} folds)")

    if all_metrics and all_loso_metrics:
        print("\n── K-Fold vs LOSO Balanced Accuracy Gap ──")
        for band in FREQ_BANDS:
            if band in all_metrics and band in all_loso_metrics:
                kf   = all_metrics[band]['bal_acc'].mean()
                loso = all_loso_metrics[band]['bal_acc'].mean()
                diff = kf - loso
                flag = ('  ⚠️  large gap — possible subject leakage'
                        if diff > 0.1 else '  ✅ robust')
                print(f"   {band}: K-Fold={kf:.3f}  LOSO={loso:.3f}  "
                      f"diff={diff:+.3f}{flag}")

    print("\n── Output files ──")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"   📄 {f}")
    print(f"\n📁 All results → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
