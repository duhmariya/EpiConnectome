import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import mne_connectivity
from mne_connectivity.viz import plot_connectivity_circle
from nilearn import plotting
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns

# ================== CONFIGURATION ==================
DATA_DIR = r"C:\Users\mariy\Desktop\mne_plot_con_circle\mne_plot_con_circle\siena-scalp-eeg-database-1.0.0\siena_txt_format"
EPOCH_LENGTH = 10    # seconds
SFREQ = 1000         # Hz
METHOD = 'wpli'      # Connectivity method
subj = 'Siena'       # Dataset name

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
mne.set_config('MNE_RANDOM_SEED', str(RANDOM_SEED))

# ================== FIND TXT FILES ==================
txt_files = []
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith('.txt') and 'seizure' in file.lower():
            txt_files.append(os.path.join(root, file))

txt_files.sort()
print(f"找到 {len(txt_files)} 個 txt 檔案")

if not txt_files:
    raise RuntimeError("沒有找到任何 .txt 檔案！")

# Collect bad channels info
all_bad_channels = []

# ================== PROCESS EACH FILE ==================
for file_idx, decoded_file_path in enumerate(txt_files, 1):
    print(f"\n{'='*80}")
    print(f"正在處理第 {file_idx}/{len(txt_files)} 個檔案：{decoded_file_path}")
    print(f"{'='*80}\n")
    
    np.random.seed(RANDOM_SEED + file_idx)
    
    try:
        # ================== LOAD DATA ==================
        eeg_data = np.loadtxt(decoded_file_path)  # (timepoints, channels)
        n_samples, n_channels = eeg_data.shape
        
        print(f'SHAPE OF EEG DATA: {eeg_data.shape}')
        print(f'Duration: {n_samples/SFREQ:.1f} seconds')
        
        # ================== CHANNEL SETUP ==================
        ch_names_full = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        ch_names = ch_names_full[:n_channels]
        ch_type = ['eeg'] * n_channels
        
        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_type)
        raw = mne.io.RawArray(eeg_data.T, info)
        
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_alias=False)
        
        # ================== PREPROCESSING ==================
        print("Applying bandpass filter (4-40 Hz)...")
        raw.filter(l_freq=4, h_freq=40)
        print("Setting average reference...")
        raw, _ = mne.set_eeg_reference(raw, projection=True)
        
        # Bad channel detection
        bad_channels, scores = mne.preprocessing.find_bad_channels_lof(raw, n_neighbors=8, threshold=1.5, return_scores=True)
        print(f"標記的壞通道: {bad_channels}")
        
        file_stem = os.path.splitext(os.path.basename(decoded_file_path))[0]
        all_bad_channels.append({
            '檔案名稱': file_stem,
            '壞通道': ', '.join(bad_channels) if bad_channels else '無',
            '壞通道數量': len(bad_channels)
        })
        
        raw.info['bads'] = bad_channels
        if bad_channels:
            print(f"Interpolating {len(bad_channels)} bad channel(s)...")
            raw.interpolate_bads(reset_bads=True)
        
        # ================== ICA ==================
        filter_eeg = raw.copy().filter(l_freq=4, h_freq=40, verbose=False)
        
        ica = mne.preprocessing.ICA(n_components=min(8, n_channels-1), random_state=97, method='infomax')
        ica.fit(inst=raw.copy().filter(4, 40))
        
        eog_indices, eog_scores = ica.find_bads_eog(inst=filter_eeg, ch_name=[ch_names[0], ch_names[1]], measure='correlation', threshold=0.5)
        muscle_noise_indices, muscle_noise_scores = ica.find_bads_muscle(inst=filter_eeg, threshold=0.5)
        ica.exclude = list(set(eog_indices + muscle_noise_indices))
        
        filter_ica = ica.apply(filter_eeg.copy())
        raw = filter_ica
        
        # ================== CREATE EPOCHS ==================
        times = raw.n_times
        event_id = 1
        interval = 1000 * EPOCH_LENGTH
        events = np.array([[i, 0, event_id] for i in range(0, times, interval)])
        
        epoch_eeg = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=EPOCH_LENGTH - 1/SFREQ, baseline=None)
        print(f"Epochs shape: {events.shape}")
        
        # ================== SOURCE LOCALIZATION ==================
        data_path = mne.datasets.sample.data_path()
        subjects_dir = data_path / "subjects"
        
        if file_idx == 1:
            fsaverage_path = mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
            print(f"fsaverage 已下載並儲存在: {fsaverage_path}")
        
        src = mne.setup_source_space(subject='fsaverage', spacing='oct4', subjects_dir=subjects_dir)
        
        conductivity = (0.3, 0.006, 0.3)
        model = mne.make_bem_model(subject='fsaverage', ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        
        fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem, meg=False, eeg=True, mindist=5.0)
        noise_cov = mne.compute_covariance(epoch_eeg, method='empirical')
        inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
        
        stcs = apply_inverse_epochs(epoch_eeg, inverse_operator, lambda2=1./9., method='dSPM', pick_ori=None, return_generator=False)
        
        # ================== ROI DEFINITION ==================
        labels = mne.read_labels_from_annot(subject='fsaverage', parc='HCPMMP1_combined', subjects_dir=subjects_dir)
        
        roi1_names = ['anterior cingulate and medial prefrontal cortex', 'inferior frontal cortex', 'dorsolateral prefrontal cortex']
        roi2_names = ['somatosensory and motor cortex','paracentral lobular and mid cingulate cortex','premotor cortex']
        roi3_names = ['dorsal stream visual cortex', 'early visual cortex', 'mt+ complex and neighboring visual areas', 'primary visual cortex (v1)']
        roi4_names = ['orbital and polar frontal cortex']
        roi5_names = ['lateral temporal cortex','medial temporal cortex','ventral stream visual cortex']
        roi6_names = ['superior parietal cortex']
        roi7_names = ['inferior parietal cortex','posterior cingulate cortex','temporo-parieto-occipital junction']
        roi8_names = ['auditory association cortex','early auditory cortex','insular and frontal opercular cortex','posterior opercular cortex']
        roi9_names = ['???']
        
        roi1_labels = [l for l in labels if any(name in l.name.lower() for name in roi1_names)]
        roi2_labels = [l for l in labels if any(name in l.name.lower() for name in roi2_names)]
        roi3_labels = [l for l in labels if any(name in l.name.lower() for name in roi3_names)]
        roi4_labels = [l for l in labels if any(name in l.name.lower() for name in roi4_names)]
        roi5_labels = [l for l in labels if any(name in l.name.lower() for name in roi5_names)]
        roi6_labels = [l for l in labels if any(name in l.name.lower() for name in roi6_names)]
        roi7_labels = [l for l in labels if any(name in l.name.lower() for name in roi7_names)]
        roi8_labels = [l for l in labels if any(name in l.name.lower() for name in roi8_names)]
        roi9_labels = [l for l in labels if any(name in l.name.lower() for name in roi9_names)]
        
        roi_labels = {}
        for i, roi_group in enumerate([roi1_labels, roi2_labels, roi3_labels, roi4_labels, roi5_labels,
                                       roi6_labels, roi7_labels, roi8_labels, roi9_labels], start=1):
            for hemi in ['lh', 'rh']:
                label_name = f'ROI{i}_{hemi.upper()}'
                roi_label = None
                for label in roi_group:
                    if label.hemi == hemi:
                        if roi_label is None:
                            roi_label = label
                        else:
                            roi_label += label
                if roi_label is not None and len(roi_label.vertices) > 0:
                    roi_labels[label_name] = roi_label
        
        valid_roi_labels = list(roi_labels.values())
        
        custom_roi_names = {
            'ROI1_LH': 'ROI1_LH ACC + IFC + DPC (LH)',
            'ROI1_RH': 'ROI1_RH ACC + IFC + DPC (RH)',
            'ROI2_LH': 'ROI2_LH Motor + Premotor + Cingulate (LH)',
            'ROI2_RH': 'ROI2_RH Motor + Premotor + Cingulate (RH)',
            'ROI3_LH': 'ROI3_LH Visual V1 + MT+ etc. (LH)',
            'ROI3_RH': 'ROI3_RH Visual V1 + MT+ etc. (RH)',
            'ROI4_LH': 'ROI4_LH Orbital Frontal (LH)',
            'ROI4_RH': 'ROI4_RH Orbital Frontal (RH)',
            'ROI5_LH': 'ROI5_LH Temporal + Ventral Visual (LH)',
            'ROI5_RH': 'ROI5_RH Temporal + Ventral Visual (RH)',
            'ROI6_LH': 'ROI6_LH Superior Parietal (LH)',
            'ROI6_RH': 'ROI6_RH Superior Parietal (RH)',
            'ROI7_LH': 'ROI7_LH Inferior Parietal + PCC + TPOJ (LH)',
            'ROI7_RH': 'ROI7_RH Inferior Parietal + PCC + TPOJ (RH)',
            'ROI8_LH': 'ROI8_LH Auditory + Insula + Operculum (LH)',
            'ROI8_RH': 'ROI8_RH Auditory + Insula + Operculum (RH)',
            'ROI9_LH': 'ROI9_LH ??? (LH)',
            'ROI9_RH': 'ROI9_RH ??? (RH)',
        }
        valid_roi_names = [custom_roi_names[key] for key in roi_labels.keys()]
        
        # ================== EXTRACT ROI TIME COURSES ==================
        roi_tc = []
        for stc in stcs:
            epoch_tc = []
            for label in valid_roi_labels:
                ts = mne.extract_label_time_course(stc, [label], src, mode='mean')
                epoch_tc.append(ts.squeeze())
            roi_tc.append(np.array(epoch_tc))
        
        roi_tc = np.array(roi_tc)
        print(f"✅ roi_tc.shape = {roi_tc.shape}")
        
        # ================== ROI COORDINATES ==================
        roi_coords_18 = []
        roi_names_18 = []
        
        roi_groups_lr = {
            'ROI1_LH': ['anterior cingulate and medial prefrontal cortex-lh', 'inferior frontal cortex-lh', 'dorsolateral prefrontal cortex-lh'],
            'ROI1_RH': ['anterior cingulate and medial prefrontal cortex-rh', 'inferior frontal cortex-rh', 'dorsolateral prefrontal cortex-rh'],
            'ROI2_LH': ['somatosensory and motor cortex-lh', 'paracentral lobular and mid cingulate cortex-lh', 'premotor cortex-lh'],
            'ROI2_RH': ['somatosensory and motor cortex-rh', 'paracentral lobular and mid cingulate cortex-rh', 'premotor cortex-rh'],
            'ROI3_LH': ['dorsal stream visual cortex-lh', 'early visual cortex-lh', 'mt+ complex and neighboring visual areas-lh', 'primary visual cortex (v1)-lh'],
            'ROI3_RH': ['dorsal stream visual cortex-rh', 'early visual cortex-rh', 'mt+ complex and neighboring visual areas-rh', 'primary visual cortex (v1)-rh'],
            'ROI4_LH': ['orbital and polar frontal cortex-lh'],
            'ROI4_RH': ['orbital and polar frontal cortex-rh'],
            'ROI5_LH': ['lateral temporal cortex-lh', 'medial temporal cortex-lh', 'ventral stream visual cortex-lh'],
            'ROI5_RH': ['lateral temporal cortex-rh', 'medial temporal cortex-rh', 'ventral stream visual cortex-rh'],
            'ROI6_LH': ['superior parietal cortex-lh'],
            'ROI6_RH': ['superior parietal cortex-rh'],
            'ROI7_LH': ['inferior parietal cortex-lh', 'posterior cingulate cortex-lh', 'temporo-parieto-occipital junction-lh'],
            'ROI7_RH': ['inferior parietal cortex-rh', 'posterior cingulate cortex-rh', 'temporo-parieto-occipital junction-rh'],
            'ROI8_LH': ['auditory association cortex-lh', 'early auditory cortex-lh', 'insular and frontal opercular cortex-lh', 'posterior opercular cortex-lh'],
            'ROI8_RH': ['auditory association cortex-rh', 'early auditory cortex-rh', 'insular and frontal opercular cortex-rh', 'posterior opercular cortex-rh'],
            'ROI9_LH': ['???-lh'],
            'ROI9_RH': ['???-rh'],
        }
        
        for roi_name, label_keys in roi_groups_lr.items():
            coords = []
            for key in label_keys:
                for label in labels:
                    if key.lower() in label.name.lower():
                        hemi = 0 if label.hemi == 'lh' else 1
                        center = label.center_of_mass(subject='fsaverage', subjects_dir=subjects_dir)
                        mni_coord = mne.vertex_to_mni([center], hemis=hemi, subject='fsaverage', subjects_dir=subjects_dir)
                        coords.append(mni_coord[0])
                        break
            
            if coords:
                avg_coord = np.mean(coords, axis=0)
                roi_coords_18.append(avg_coord)
                roi_names_18.append(roi_name)
            else:
                roi_coords_18.append(np.array([np.nan, np.nan, np.nan]))
                roi_names_18.append(roi_name)
        
        roi_coords_18 = np.array(roi_coords_18)
        print(f"ROI座標形狀: {roi_coords_18.shape}")
        
        # ================== COMPUTE CONNECTIVITY ==================
        roi_tc_list = [epoch for epoch in roi_tc]
        
        con = mne_connectivity.spectral_connectivity_epochs(
            roi_tc_list, method=METHOD, mode='multitaper',
            fmin=(6, 8, 12), fmax=(8, 12, 30), faverage=True, sfreq=1000
        )
        
        con_matrix = con.get_data(output='dense')
        
        con_theta = con_matrix[:, :, 0]
        con_alpha = con_matrix[:, :, 1]
        con_beta = con_matrix[:, :, 2]
        
        # Symmetrize
        con_theta = con_theta + con_theta.T
        con_alpha = con_alpha + con_alpha.T
        con_beta = con_beta + con_beta.T
        
        # ================== VISUALIZATION ==================
        if len(valid_roi_labels) > 1:
            base_dir = os.path.dirname(decoded_file_path)
            output_dir = os.path.join(base_dir, f"{subj}_con_figures_{EPOCH_LENGTH}s", file_stem)
            output_dir_matrix = os.path.join(base_dir, f"{subj}_matrix_{EPOCH_LENGTH}s")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir_matrix, exist_ok=True)
            
            # ================== SAVE EXCEL ==================
            excel_path1 = os.path.join(output_dir, f"{subj}_{EPOCH_LENGTH}s_{file_stem}.xlsx")
            excel_path2 = os.path.join(output_dir_matrix, f"{subj}_{EPOCH_LENGTH}s_{file_stem}.xlsx")
            
            def save_connectivity_excel(path):
                with pd.ExcelWriter(path, engine='openpyxl') as writer:
                    df_theta = pd.DataFrame(con_theta, index=valid_roi_names, columns=valid_roi_names)
                    df_theta.to_excel(writer, sheet_name='Theta (6-8 Hz)')
                    
                    df_alpha = pd.DataFrame(con_alpha, index=valid_roi_names, columns=valid_roi_names)
                    df_alpha.to_excel(writer, sheet_name='Alpha (8-12 Hz)')
                    
                    df_beta = pd.DataFrame(con_beta, index=valid_roi_names, columns=valid_roi_names)
                    df_beta.to_excel(writer, sheet_name='Beta (12-30 Hz)')
            
            save_connectivity_excel(excel_path1)
            save_connectivity_excel(excel_path2)
            print(f"📊 已儲存連結性矩陣至: {excel_path1}")
            
            # ================== CONNECTOME PLOTS ==================
            if len(roi_coords_18) == len(valid_roi_labels):
                connectome_matrices = [con_theta, con_alpha, con_beta]
                freq_names = ['Theta', 'Alpha', 'Beta']
                
                for matrix, freq_name in zip(connectome_matrices, freq_names):
                    node_strength = np.sum(matrix, axis=0) + np.sum(matrix, axis=1)
                    scaler = MinMaxScaler(feature_range=(10, 300))
                    node_size = scaler.fit_transform(node_strength.reshape(-1, 1)).flatten()
                    normalize_strength = (node_strength - np.min(node_strength)) / (np.max(node_strength) - np.min(node_strength))
                    
                    # 3D Connectome
                    fig = plt.figure(figsize=(20, 13))
                    display = plotting.plot_connectome(
                        adjacency_matrix=matrix,
                        node_coords=roi_coords_18,
                        node_color='red',
                        node_size=node_size,
                        edge_threshold=0.3,  # ← Changed from 0.5 to 0.3
                        title=f'{freq_name} Band Connectivity',
                        display_mode='lzr',
                        black_bg=False,
                        figure=fig,
                        edge_kwargs={'linewidth': 2.5},
                        edge_cmap='rainbow',
                        edge_vmin=0,
                        edge_vmax=1
                    )
                    display.title(f'{freq_name} Band Connectivity', size=35)
                    
                    connectome_filename = f"connectome_1000_{freq_name.lower()}.png"
                    plt.savefig(os.path.join(output_dir, connectome_filename), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    # 2D Top View
                    x = roi_coords_18[:, 0]
                    y = roi_coords_18[:, 1]
                    
                    fig2 = plt.figure(figsize=(10, 10))
                    ax2 = fig2.add_subplot(111)
                    
                    for i in range(len(roi_coords_18)):
                        for j in range(i + 1, len(roi_coords_18)):
                            if matrix[i, j] > 0:
                                ax2.plot([x[i], x[j]], [y[i], y[j]], color='blue', linewidth=matrix[i, j]*3)
                    
                    ax2.scatter(x, y, s=node_size, c=normalize_strength, cmap='Reds')
                    ax2.axis('off')
                    
                    plt.title(f'2D Network Graph - {freq_name}', fontsize=25, loc='center')
                    plt.tight_layout()
                    view_connectome_filename = f"view_connectome_1000_{freq_name.lower()}.png"
                    plt.savefig(os.path.join(output_dir, view_connectome_filename), dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                
                # ================== GROUPED BAR CHART (CONNECTIONS >= 0.3) ==================
                # Create grouped bar chart showing inter-hemispheric connections >= 0.3
                # Extract ROI numbers (ROI1-ROI9) for left and right hemispheres
                roi_numbers = []
                roi_hemis = []
                for name in valid_roi_names:
                    # Extract ROI number (e.g., "ROI1_LH" -> 1)
                    roi_num = int(name.split('_')[0].replace('ROI', ''))
                    hemi = name.split('_')[1]
                    roi_numbers.append(roi_num)
                    roi_hemis.append(hemi)
                
                # Find all left-right pairs (inter-hemispheric connections)
                lr_connections = []
                for i, (num_i, hemi_i) in enumerate(zip(roi_numbers, roi_hemis)):
                    for j, (num_j, hemi_j) in enumerate(zip(roi_numbers, roi_hemis)):
                        if hemi_i == 'LH' and hemi_j == 'RH':
                            conn_strength = {
                                'pair': f"L{num_i}-R{num_j}",
                                'left_idx': i,
                                'right_idx': j,
                                'theta': connectome_matrices[0][i, j],
                                'alpha': connectome_matrices[1][i, j],
                                'beta': connectome_matrices[2][i, j]
                            }
                            # Only include if ANY frequency band >= 0.3
                            if (conn_strength['theta'] >= 0.3 or 
                                conn_strength['alpha'] >= 0.3 or 
                                conn_strength['beta'] >= 0.3):
                                lr_connections.append(conn_strength)
                
                # Sort by average strength (highest first)
                for conn in lr_connections:
                    conn['avg'] = (conn['theta'] + conn['alpha'] + conn['beta']) / 3
                lr_connections.sort(key=lambda x: x['avg'], reverse=True)
                
                # Use all connections >= 0.3 (no limit)
                top_connections = lr_connections
                
                # Create grouped bar chart
                if len(top_connections) == 0:
                    print(f"  ⚠️  No connections >= 0.3 found for {file_stem}")
                else:
                    print(f"  📊 Found {len(top_connections)} connections >= 0.3")
                    
                    fig5, (ax_alpha, ax_beta, ax_theta) = plt.subplots(1, 3, figsize=(max(18, len(top_connections)*1.5), 5))
                
                x_labels = [conn['pair'] for conn in top_connections]
                x_pos = np.arange(len(x_labels))
                width = 0.6
                
                # Alpha
                alpha_vals = [conn['alpha'] for conn in top_connections]
                bars_alpha = ax_alpha.bar(x_pos, alpha_vals, width, color='turquoise', edgecolor='black')
                ax_alpha.set_ylabel('Connection Strength', fontsize=12)
                ax_alpha.set_title('Alpha(8-12 Hz) Connection Strength', fontsize=12, fontweight='bold')
                ax_alpha.set_xticks(x_pos)
                ax_alpha.set_xticklabels(x_labels, rotation=45, ha='right')
                ax_alpha.set_ylim(0, 0.9)
                ax_alpha.grid(True, alpha=0.3, axis='y')
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars_alpha, alpha_vals)):
                    ax_alpha.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Beta
                beta_vals = [conn['beta'] for conn in top_connections]
                bars_beta = ax_beta.bar(x_pos, beta_vals, width, color='lightblue', edgecolor='black')
                ax_beta.set_ylabel('Connection Strength', fontsize=12)
                ax_beta.set_title('Beta(12-30 Hz) Connection Strength', fontsize=12, fontweight='bold')
                ax_beta.set_xticks(x_pos)
                ax_beta.set_xticklabels(x_labels, rotation=45, ha='right')
                ax_beta.set_ylim(0, 0.9)
                ax_beta.grid(True, alpha=0.3, axis='y')
                for i, (bar, val) in enumerate(zip(bars_beta, beta_vals)):
                    ax_beta.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Theta
                theta_vals = [conn['theta'] for conn in top_connections]
                bars_theta = ax_theta.bar(x_pos, theta_vals, width, color='lightcoral', edgecolor='black')
                ax_theta.set_ylabel('Connection Strength', fontsize=12)
                ax_theta.set_title('Theta(6-8 Hz) Connection Strength', fontsize=12, fontweight='bold')
                ax_theta.set_xticks(x_pos)
                ax_theta.set_xticklabels(x_labels, rotation=45, ha='right')
                ax_theta.set_ylim(0, 0.9)
                ax_theta.grid(True, alpha=0.3, axis='y')
                for i, (bar, val) in enumerate(zip(bars_theta, theta_vals)):
                    ax_theta.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Add overall title
                fig5.suptitle(f'{file_stem} - Top Inter-Hemispheric Connections\nEpoch=10s, Threshold=0.3, ICA method=infomax',
                             fontsize=14, fontweight='bold', y=1.02)
                
                plt.tight_layout()
                grouped_bar_filename = f"grouped_bar_interhemispheric.png"
                plt.savefig(os.path.join(output_dir, grouped_bar_filename), dpi=300, bbox_inches='tight')
                plt.close(fig5)
            
            # ================== CIRCLE PLOTS ==================
            freq_names_circle = ['Theta (5-8 Hz)', 'Alpha (8-12 Hz)', 'Beta (12-30 Hz)']
            
            for freq_idx, freq_name in enumerate(freq_names_circle):
                fig, _ = plot_connectivity_circle(
                    con_matrix[:, :, freq_idx],
                    valid_roi_names,
                    title=f'Brain Connectivity - {freq_name}',
                    colormap='rainbow',
                    facecolor='white',
                    textcolor='black',
                    vmin=0.0,
                    vmax=1,
                    n_lines=80,
                    fontsize_names=8,
                    show=False
                )
                
                filename = f"circle_{freq_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}.png"
                fig.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close(fig)
                
                # Calculate mean
                mean_conn = np.mean(con_matrix[:, :, freq_idx][con_matrix[:, :, freq_idx] != 0])
                mean_output_dir = os.path.join(base_dir, f"{subj}_con_figures_{EPOCH_LENGTH}s")
                os.makedirs(mean_output_dir, exist_ok=True)
                
                mean_file_path = os.path.join(mean_output_dir, f"平均值_1000_brain_{METHOD}.txt")
                with open(mean_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"{file_stem} - {freq_name} 平均連結值: {mean_conn:.3f}\n")
            
            print(f"📁 已儲存圖片至資料夾: {output_dir}")
        
        print(f"✅ 完成處理檔案: {os.path.basename(decoded_file_path)}")
        
    except Exception as e:
        print(f"❌ 處理檔案 {decoded_file_path} 時發生錯誤：")
        print(f"   錯誤訊息: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n🎉 所有 {len(txt_files)} 個檔案處理完成！")

# ================== SAVE BAD CHANNELS INFO ==================
if all_bad_channels:
    bad_channels_df = pd.DataFrame(all_bad_channels)
    bad_channels_excel_path = os.path.join(DATA_DIR, f"{subj}_con_figures_{EPOCH_LENGTH}s", "所有檔案_壞通道統計.xlsx")
    os.makedirs(os.path.dirname(bad_channels_excel_path), exist_ok=True)
    bad_channels_df.to_excel(bad_channels_excel_path, index=False, engine='openpyxl')
    print(f"\n📊 已儲存所有檔案的壞通道統計至: {bad_channels_excel_path}")