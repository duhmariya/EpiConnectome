import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
import sys

def choose_folder(title="Select folder containing Excel files"):
    """Open a folder dialog to select the folder"""
    try:
        app = QtWidgets.QApplication.instance()
        created_app = False
        if app is None:
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
            app = QtWidgets.QApplication(sys.argv)
            created_app = True

        folder = QtWidgets.QFileDialog.getExistingDirectory(None, title)
        
        if created_app:
            app.quit()

        return folder

    except Exception:
        # fallback to Tkinter
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        return folder if folder else None

def read_excel_matrix(file_path, sheet_name=None):
    """Read the Excel sheet as a connectivity matrix"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
        return df
    except Exception as e:
        print(f"❌ Failed to read {sheet_name if sheet_name else 'default sheet'} in {file_path}: {e}")
        return None

def plot_connectivity(df, output_path, threshold=0.3):
    """Plot connectivity strength for all channels with threshold line"""
    channels = df.columns.tolist()
    strengths = df.values.flatten()  # flatten the full matrix

    # optional: only upper triangle without diagonal
    upper_tri_indices = np.triu_indices_from(df, k=1)
    strengths = df.values[upper_tri_indices]
    pairs = [f"{channels[i]}-{channels[j]}" for i,j in zip(*upper_tri_indices)]

    plt.figure(figsize=(20,6))
    plt.bar(pairs, strengths, color='#4ECDC4', alpha=0.7, edgecolor='black')
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    plt.ylabel('Connectivity Strength', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=10)
    plt.title(os.path.basename(output_path).replace('.png',''), fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {output_path}")

def main():
    folder = choose_folder("Select folder containing Excel files")
    if not folder:
        print("❌ No folder selected. Exiting.")
        return

    # Recursively find Excel files
    excel_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(('.xlsx', '.xls')):
                excel_files.append(os.path.join(root, f))

    if not excel_files:
        print(f"❌ No Excel files found in {folder}")
        return

    print(f"\n📂 Found {len(excel_files)} Excel files")

    output_dir = os.path.join(folder, "connectivity_plots")
    os.makedirs(output_dir, exist_ok=True)

    for file_path in excel_files:
        # Read all sheets if they exist
        sheet_names = ['Theta', 'Alpha', 'Beta']
        for sheet in sheet_names:
            df = read_excel_matrix(file_path, sheet_name=sheet)
            if df is not None:
                file_stem = os.path.splitext(os.path.basename(file_path))[0]
                out_file = os.path.join(output_dir, f"{file_stem}_{sheet}.png")
                plot_connectivity(df, out_file, threshold=0.3)

    print(f"\n✅ All plots saved in {output_dir}")

if __name__ == "__main__":
    main()
