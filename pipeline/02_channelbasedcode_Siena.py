# %%
"""
Siena Scalp EEG Database - Channel Connectivity Analysis
Dataset: https://physionet.org/content/siena-scalp-eeg/1.0.0/
Using pre-converted TXT files
COMPLETE VERSION - Ready to run
"""

import os
import mne
import random
import mne_connectivity
from mne_connectivity.viz import plot_connectivity_circle
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import networkx as nx
from PyQt5 import QtWidgets, QtCore
import sys

# ================== REPRODUCIBILITY SETTINGS ==================
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

mne.set_config('MNE_RANDOM_SEED', str(RANDOM_SEED))
mne.set_config('MNE_USE_NUMBA', 'false')

# %%
# ================== PARAMETERS ==================
EPOCH_LENGTH = 10
subj = 'Siena_Epilepsy'  # Siena dataset contains ONLY epilepsy patients
SFREQ = 512  # Siena original sampling rate
METHOD = 'wpli'

# Thresholds
CHANNEL_CONNECTIVITY_THRESHOLD = 0.5  # For network metrics
VISUALIZATION_THRESHOLD = 0.0  # For visualizations

# Standard 16 channels (matching reference format)
STANDARD_16_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4'
]

# ================== FOLDER SELECTION ==================
def choose_folder(title="選擇包含 Siena TXT 檔案的資料夾"):
    try:
        app = QtWidgets.QApplication.instance()
        created_app = False
        if app is None:
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
            app = QtWidgets.QApplication(sys.argv)
            created_app = True

        dialog = QtWidgets.QFileDialog()
        dialog.setWindowTitle(title)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            path = dialog.selectedFiles()[0]
        else:
            path = None

        if created_app:
            app.quit()

        return path

    except Exception:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            try:
                root.call('tk', 'scaling', 1.5)
            except Exception:
                pass
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return path if path else None
        except Exception:
            return None

# ================== FIND FILES ==================
selected_folder = choose_folder()
if not selected_folder:
    raise RuntimeError("未選取任何資料夾，程式已停止。")

print(f"你選擇的資料夾是：{selected_folder}")

txt_files = []
for root, dirs, files in os.walk(selected_folder):
    for file in files:
        if file.endswith('.txt'):
            txt_files.append(os.path.join(root, file))

if not txt_files:
    raise RuntimeError(f"在資料夾 {selected_folder} 中沒有找到任何 .txt 檔案！")

print(f"找到 {len(txt_files)} 個 txt 檔案：")
for i, file in enumerate(txt_files, 1):
    print(f"  {i}. {os.path.basename(file)}")

# ================== STORAGE ==================
all_bad_channels = []
all_channel_metrics = []

# ================== HELPER FUNCTIONS ==================
def load_siena_txt(txt_path):
    """Load Siena TXT file"""
    print(f"Loading: {os.path.basename(txt_path)}")
    
    try:
        data = np.loadtxt(txt_path)
        print(f"  Loaded shape: {data.shape}")
        
        if data.ndim == 2 and data.shape[1] > 29:
            data = data[:, :29]
        elif data.ndim == 1:
            raise ValueError("Data is 1D, expected 2D array")
            
        return data
        
    except Exception as e:
        print(f"  Trying pandas...")
        try:
            df = pd.read_csv(txt_path, sep='\s+', header=None)
            data = df.values
            if data.shape[1] > 29:
                data = data[:, :29]
            print(f"  Loaded shape: {data.shape}")
            return data
        except Exception as e2:
            raise RuntimeError(f"Could not load {txt_path}: {str(e)} | {str(e2)}")


def compute_channel_connectivity(epochs, method, freq_bands_dict):
    """Compute connectivity for multiple frequency bands"""
    fmin = [band[0] for band in freq_bands_dict.values()]
    fmax = [band[1] for band in freq_bands_dict.values()]
    
    con = mne_connectivity.spectral_connectivity_epochs(
        epochs, 
        method=method, 
        mode='multitaper',
        fmin=fmin, 
        fmax=fmax, 
        faverage=True,
        sfreq=epochs.info['sfreq'],
        verbose=False
    )
    
    con_matrix = con.get_data(output='dense')
    
    conn_dict = {}
    for idx, band_name in enumerate(freq_bands_dict.keys()):
        matrix = con_matrix[:, :, idx]
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 0)
        conn_dict[band_name] = matrix
    
    return conn_dict


def compute_network_metrics(con_matrix, ch_names, threshold):
    """Compute graph theory metrics"""
    G = nx.Graph()
    n_nodes = con_matrix.shape[0]
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if con_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=con_matrix[i, j])
    
    metrics = {}
    
    if G.number_of_edges() > 0:
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        metrics['global_efficiency'] = nx.global_efficiency(G)
        metrics['avg_clustering'] = nx.average_clustering(G, weight='weight')
        
        node_strength = np.sum(con_matrix, axis=0) + np.sum(con_matrix, axis=1)
        metrics['node_strength'] = node_strength
        metrics['avg_node_strength'] = np.mean(node_strength)
        
        degree = np.sum(con_matrix > threshold, axis=0) + np.sum(con_matrix > threshold, axis=1)
        metrics['degree'] = degree
        metrics['avg_degree'] = np.mean(degree)
        
        try:
            communities = nx.community.greedy_modularity_communities(G)
            metrics['modularity'] = nx.community.modularity(G, communities)
            metrics['n_communities'] = len(communities)
        except:
            metrics['modularity'] = np.nan
            metrics['n_communities'] = np.nan
    else:
        metrics = {
            'n_edges': 0,
            'density': 0,
            'global_efficiency': 0,
            'avg_clustering': 0,
            'node_strength': np.zeros(n_nodes),
            'avg_node_strength': 0,
            'degree': np.zeros(n_nodes),
            'avg_degree': 0,
            'modularity': np.nan,
            'n_communities': np.nan
        }
    
    return metrics


def compute_channel_mean_connectivity(con_matrix, ch_names):
    """Compute mean connectivity for each channel (excluding diagonal)"""
    n_channels = con_matrix.shape[0]
    mean_connectivity = np.zeros(n_channels)
    
    for i in range(n_channels):
        connections = np.concatenate([con_matrix[i, :i], con_matrix[i, i+1:]])
        mean_connectivity[i] = np.mean(connections)
    
    return mean_connectivity


def save_channel_connectivity_excel(channel_conn_dict, ch_names, output_path, metadata):
    """Save connectivity matrices to Excel"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for band_name, con_matrix in channel_conn_dict.items():
            df = pd.DataFrame(con_matrix, index=ch_names, columns=ch_names)
            df.to_excel(writer, sheet_name=f'{band_name.capitalize()}')
        
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)


def create_channel_visualizations(channel_conn_dict, ch_names, output_dir, subject_id):
    """Create connectivity visualizations"""
    
    for band_name, con_matrix in channel_conn_dict.items():
        # 1. Circle Plot
        fig, _ = plot_connectivity_circle(
            con_matrix, ch_names,
            title=f'{subject_id} - {band_name.upper()}',
            colormap='RdYlBu_r', facecolor='white', textcolor='black',
            vmin=0.0, vmax=1.0, n_lines=None, fontsize_names=10, show=False
        )
        fig.savefig(os.path.join(output_dir, f"{subject_id}_{band_name}_circle.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(con_matrix, xticklabels=ch_names, yticklabels=ch_names,
                   cmap='RdYlBu_r', vmin=0, vmax=1, square=True,
                   cbar_kws={'label': f'{METHOD.upper()}'}, ax=ax)
        ax.set_title(f'{subject_id} - {band_name.upper()}', fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{subject_id}_{band_name}_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 3. Network Graph
        G = nx.Graph()
        for i in range(len(ch_names)):
            for j in range(i+1, len(ch_names)):
                if con_matrix[i, j] > 0:
                    G.add_edge(ch_names[i], ch_names[j], weight=con_matrix[i, j])
        
        if G.number_of_edges() > 0:
            fig, ax = plt.subplots(figsize=(12, 12))
            
            node_strength = np.sum(con_matrix, axis=0) + np.sum(con_matrix, axis=1)
            normalize_strength = (node_strength - np.min(node_strength)) / (np.max(node_strength) - np.min(node_strength) + 1e-10)
            node_sizes = [s * 800 + 100 for s in normalize_strength]
            
            pos = nx.spring_layout(G, k=2, iterations=50, seed=RANDOM_SEED)
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=normalize_strength, 
                                  cmap='Reds', alpha=0.9, ax=ax)
            
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            normalized_widths = [w/max_weight * 2.5 for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=normalized_widths, alpha=0.3,
                                  edge_color=weights, edge_cmap=plt.cm.Blues,
                                  edge_vmin=0, edge_vmax=1, ax=ax)
            
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
            
            ax.set_title(f'{subject_id} - {band_name.upper()} Network', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{subject_id}_{band_name}_network.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)


def create_channel_bar_plots(channel_conn_dict, ch_names, output_dir, subject_id):
    """Create bar plots with threshold line"""
    
    for band_name, con_matrix in channel_conn_dict.items():
        mean_conn = compute_channel_mean_connectivity(con_matrix, ch_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Reds(mean_conn / (np.max(mean_conn) + 1e-10))
        bars = ax.bar(ch_names, mean_conn, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.axhline(y=CHANNEL_CONNECTIVITY_THRESHOLD, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold = {CHANNEL_CONNECTIVITY_THRESHOLD}')
        
        ax.set_ylabel('Mean Connectivity', fontsize=12, fontweight='bold')
        ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
        ax.set_title(f'{subject_id} - {band_name.upper()} Channel Mean Connectivity', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 0.9])
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, mean_conn)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{subject_id}_{band_name}_channel_bar.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)


# ================== MAIN PROCESSING LOOP ==================
for file_idx, txt_file_path in enumerate(txt_files, 1):
    print(f"\n{'='*80}")
    print(f"正在處理第 {file_idx}/{len(txt_files)} 個檔案：{os.path.basename(txt_file_path)}")
    print(f"{'='*80}\n")
    
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    try:
        # Load data
        eeg_data = load_siena_txt(txt_file_path)
        
        # Trim edges (skip if file too short)
        trim_samples = 30 * 512
        min_samples  = 90 * 512  # need at least 90s to trim 30s each side
        if eeg_data.shape[0] > min_samples:
            eeg_data = eeg_data[trim_samples:-trim_samples, :]
            print(f"  Trimmed to: {eeg_data.shape}")
        else:
            print(f"  ⚠️  Short file ({eeg_data.shape[0]/512:.1f}s) — skipping trim")
        
        # Use first 16 channels
        n_channels = min(eeg_data.shape[1], 16)
        eeg_data = eeg_data[:, :n_channels]
        
        temp_ch_names = STANDARD_16_CHANNELS[:n_channels]
        ch_type = ['eeg'] * n_channels
        info = mne.create_info(ch_names=temp_ch_names, sfreq=SFREQ, ch_types=ch_type)

        raw = mne.io.RawArray(eeg_data.T, info, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_alias=True, on_missing='ignore')

        # Preprocessing
        print("  Filtering 4-40 Hz...")
        raw.filter(l_freq=4, h_freq=40, verbose=False)
        
        print("  Resampling to 1000 Hz...")
        raw.resample(1000, verbose=False)
        
        print("  Setting reference...")
        raw, _ = mne.set_eeg_reference(raw, projection=True, verbose=False)

        # Bad channels
        bad_channels, scores = mne.preprocessing.find_bad_channels_lof(
            raw, n_neighbors=8, threshold=1.5, return_scores=True
        )
        print(f"  壞通道: {bad_channels if bad_channels else 'None'}")
        
        file_stem = os.path.splitext(os.path.basename(txt_file_path))[0]
        all_bad_channels.append({
            '檔案名稱': file_stem,
            '壞通道': ', '.join(bad_channels) if bad_channels else '無',
            '壞通道數量': len(bad_channels)
        })
        
        raw.info['bads'] = bad_channels
        if bad_channels:
            print(f"  Interpolating {len(bad_channels)} bad channels...")
            raw.interpolate_bads(reset_bads=True)

        # ICA
        filter_eeg = raw.copy().filter(l_freq=4, h_freq=40, verbose=False)
        
        ica = mne.preprocessing.ICA(
            n_components=15, random_state=RANDOM_SEED,
            method='infomax', max_iter='auto'
        )
        
        print("  Running ICA...")
        ica.fit(inst=raw.copy().filter(4, 40, verbose=False), verbose=False)
        
        eog_channels = [ch for ch in ['Fp1', 'Fp2'] if ch in raw.ch_names]
        if eog_channels:
            eog_indices, _ = ica.find_bads_eog(
                inst=filter_eeg, ch_name=eog_channels, 
                measure='correlation', threshold=0.5
            )
        else:
            eog_indices = []
            
        muscle_indices, _ = ica.find_bads_muscle(inst=filter_eeg, threshold=0.5)

        ica.exclude = list(set(eog_indices + muscle_indices))
        print(f"  Excluded {len(ica.exclude)} ICA components")
        
        filter_ica = ica.apply(filter_eeg.copy())
        raw = filter_ica

        ch_names = raw.ch_names
        print(f"  Final channels ({len(ch_names)}): {ch_names}")

        # Create epochs
        times = raw.n_times
        interval = 1000 * EPOCH_LENGTH
        events = np.array([[i, 0, 1] for i in range(0, times, interval)])

        epoch_eeg = mne.Epochs(
            raw, events, event_id=1, tmin=0, 
            tmax=EPOCH_LENGTH - 1/1000, baseline=None, 
            preload=True, verbose=False
        )
        print(f"  Created {len(epoch_eeg)} epochs")

        # Compute connectivity
        print("\n  Computing connectivity...")
        
        freq_bands_dict = {
            'theta': (6, 8),
            'alpha': (8, 12),
            'beta': (12, 30)
        }
        
        channel_conn_dict = compute_channel_connectivity(epoch_eeg, METHOD, freq_bands_dict)
        
        # Network metrics
        channel_metrics_dict = {}
        for band_name, con_matrix in channel_conn_dict.items():
            metrics = compute_network_metrics(con_matrix, ch_names, CHANNEL_CONNECTIVITY_THRESHOLD)
            channel_metrics_dict[band_name] = metrics
            
            print(f"\n    {band_name.upper()}:")
            print(f"      Global Efficiency: {metrics['global_efficiency']:.4f}")
            print(f"      Clustering: {metrics['avg_clustering']:.4f}")
            print(f"      Density: {metrics['density']:.4f}")
            
            all_channel_metrics.append({
                'Subject_ID': file_stem,
                'Group': 'Epilepsy',  # Siena dataset = Epilepsy patients
                'Frequency_Band': band_name,
                'N_Epochs': len(epoch_eeg),
                'N_Channels': len(ch_names),
                'N_Bad_Channels': len(bad_channels),
                'N_Excluded_ICA': len(ica.exclude),
                'N_Edges': metrics['n_edges'],
                'Density': metrics['density'],
                'Global_Efficiency': metrics['global_efficiency'],
                'Avg_Clustering': metrics['avg_clustering'],
                'Avg_Node_Strength': metrics['avg_node_strength'],
                'Avg_Degree': metrics['avg_degree'],
                'Modularity': metrics['modularity'],
                'N_Communities': metrics['n_communities']
            })

        # Save results
        base_dir = os.path.dirname(txt_file_path)
        
        output_dir_channel = os.path.join(base_dir, f"{subj}_Channel_analysis_{EPOCH_LENGTH}s", file_stem)
        output_dir_matrix = os.path.join(base_dir, f"{subj}_matrices_{EPOCH_LENGTH}s")
        
        os.makedirs(output_dir_channel, exist_ok=True)
        os.makedirs(output_dir_matrix, exist_ok=True)

        # Excel
        excel_path = os.path.join(output_dir_matrix, f"{subj}_Channel_{EPOCH_LENGTH}s_{file_stem}.xlsx")
        
        metadata = {
            'Subject_ID': file_stem,
            'Group': 'Epilepsy',  # Siena = Epilepsy patients
            'Dataset': 'Siena Scalp EEG Database',
            'N_Epochs': len(epoch_eeg),
            'N_Channels': len(ch_names),
            'Channels': ', '.join(ch_names),
            'Bad_Channels': ', '.join(bad_channels) if bad_channels else 'None',
            'Excluded_ICA': len(ica.exclude),
            'Method': METHOD,
            'Threshold': CHANNEL_CONNECTIVITY_THRESHOLD
        }
        
        save_channel_connectivity_excel(channel_conn_dict, ch_names, excel_path, metadata)
        print(f"\n  📊 Saved: {os.path.basename(excel_path)}")

        # Visualizations
        print("  Creating figures...")
        create_channel_visualizations(channel_conn_dict, ch_names, output_dir_channel, file_stem)
        create_channel_bar_plots(channel_conn_dict, ch_names, output_dir_channel, file_stem)
        print(f"  📁 Saved: {output_dir_channel}")

        # Mean connectivity
        mean_file = os.path.join(output_dir_matrix, f"Channel_平均值_{METHOD}.txt")
        with open(mean_file, 'a', encoding='utf-8') as f:
            for band_name, con_matrix in channel_conn_dict.items():
                mean_conn = np.mean(con_matrix[con_matrix != 0])
                f.write(f"{file_stem} - {band_name.upper()}: {mean_conn:.3f}\n")

        print(f"✅ Completed: {file_stem}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n🎉 所有 {len(txt_files)} 個檔案處理完成！")

# ================== SAVE SUMMARIES ==================
if all_bad_channels:
    bad_channels_df = pd.DataFrame(all_bad_channels)
    bad_path = os.path.join(selected_folder, f"{subj}_quality_control_{EPOCH_LENGTH}s", "所有檔案_壞通道統計.xlsx")
    os.makedirs(os.path.dirname(bad_path), exist_ok=True)
    bad_channels_df.to_excel(bad_path, index=False, engine='openpyxl')
    print(f"\n📊 Saved: {bad_path}")

if all_channel_metrics:
    metrics_df = pd.DataFrame(all_channel_metrics)
    metrics_path = os.path.join(selected_folder, f"{subj}_quality_control_{EPOCH_LENGTH}s", "所有檔案_Channel網路指標統計.xlsx")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df.to_excel(metrics_path, index=False, engine='openpyxl')
    print(f"📊 Saved: {metrics_path}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    for band in ['theta', 'alpha', 'beta']:
        band_data = metrics_df[metrics_df['Frequency_Band'] == band]
        if len(band_data) > 0:
            print(f"\n{band.upper()} (n={len(band_data)}):")
            print(f"  Global Efficiency: {band_data['Global_Efficiency'].mean():.4f} ± {band_data['Global_Efficiency'].std():.4f}")
            print(f"  Clustering: {band_data['Avg_Clustering'].mean():.4f} ± {band_data['Avg_Clustering'].std():.4f}")
            print(f"  Modularity: {band_data['Modularity'].mean():.4f} ± {band_data['Modularity'].std():.4f}")
            print(f"  Density: {band_data['Density'].mean():.4f} ± {band_data['Density'].std():.4f}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"Results: {selected_folder}")
print(f"  1. Figures: {subj}_Channel_analysis_{EPOCH_LENGTH}s/")
print(f"  2. Matrices: {subj}_matrices_{EPOCH_LENGTH}s/")
print(f"  3. QC: {subj}_quality_control_{EPOCH_LENGTH}s/")