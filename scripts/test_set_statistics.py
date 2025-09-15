import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from source.data_management.path_repo import PathRepo

filename = os.path.join(PathRepo().get_hdf5_path(), 'metrics_all.h5')
df = pd.read_hdf(filename, key="metrics")
df5 = df[df["model"] == "model_5"]
df7 = df[df["model"] == "model_7"]

psnrs_5 = df5["PSNR (dB)"].to_numpy()
pesqs_5 = df5["PESQ"].to_numpy()
stois_5 = df5["STOI"].to_numpy()
f0_rmses_5 = df5["F0 RMSE (Hz)"].to_numpy()

psnrs_7 = df7["PSNR (dB)"].to_numpy()
pesqs_7 = df7["PESQ"].to_numpy()
stois_7 = df7["STOI"].to_numpy()
f0_rmses_7 = df7["F0 RMSE (Hz)"].to_numpy()

metrics_5 = {
    "PSNR (dB)": psnrs_5,
    "PESQ": pesqs_5,
    "STOI": stois_5,
    "F0 RMSE (Hz)": f0_rmses_5,
}

metrics_7 = {
    "PSNR (dB)": psnrs_7,
    "PESQ": pesqs_7,
    "STOI": stois_7,
    "F0 RMSE (Hz)": f0_rmses_7,
}

base_path = PathRepo().get_base_path()
figspath = os.path.join(base_path, 'outputs', 'statistics')

# Estadísticas:

print('===== Modelo 5 =====')

print('--- PSNR ---')
print(f'Media = {np.mean(psnrs_5):.2f} dB')
print(f'Desviación estándar = {np.std(psnrs_5):.2f} dB')
print(f'Valor mínimo = {np.min(psnrs_5):.2f} dB')
print(f'Valor máximo = {np.max(psnrs_5):.2f} dB')

print('--- PESQ ---')
print(f'Media = {np.mean(pesqs_5):.2f}')
print(f'Desviación estándar = {np.std(pesqs_5):.2f}')
print(f'Valor mínimo = {np.min(pesqs_5):.2f}')
print(f'Valor máximo = {np.max(pesqs_5):.2f}')

print('--- STOI ---')
print(f'Media = {np.mean(stois_5):.2f}')
print(f'Desviación estándar = {np.std(stois_5):.2f}')
print(f'Valor mínimo = {np.min(stois_5):.2f}')
print(f'Valor máximo = {np.max(stois_5):.2f}')

print('--- F0-RMSE ---')
print(f'Media = {np.mean(f0_rmses_5):.2f} Hz')
print(f'Desviación estándar = {np.std(f0_rmses_5):.2f} Hz')
print(f'Valor mínimo = {np.min(f0_rmses_5):.2f} Hz')
print(f'Valor máximo = {np.max(f0_rmses_5):.2f} Hz')

print('===== Modelo 7 =====')

print('--- PSNR ---')
print(f'Media = {np.mean(psnrs_7):.2f} dB')
print(f'Desviación estándar = {np.std(psnrs_7):.2f} dB')
print(f'Valor mínimo = {np.min(psnrs_7):.2f} dB')
print(f'Valor máximo = {np.max(psnrs_7):.2f} dB')

print('--- PESQ ---')
print(f'Media = {np.mean(pesqs_7):.2f}')
print(f'Desviación estándar = {np.std(pesqs_7):.2f}')
print(f'Valor mínimo = {np.min(pesqs_7):.2f}')
print(f'Valor máximo = {np.max(pesqs_7):.2f}')

print('--- STOI ---')
print(f'Media = {np.mean(stois_7):.2f}')
print(f'Desviación estándar = {np.std(stois_7):.2f}')
print(f'Valor mínimo = {np.min(stois_7):.2f}')
print(f'Valor máximo = {np.max(stois_7):.2f}')

print('--- F0-RMSE ---')
print(f'Media = {np.mean(f0_rmses_7):.2f} Hz')
print(f'Desviación estándar = {np.std(f0_rmses_7):.2f} Hz')
print(f'Valor mínimo = {np.min(f0_rmses_7):.2f} Hz')
print(f'Valor máximo = {np.max(f0_rmses_7):.2f} Hz')

# ========== Modelo 5: Histogramas ==========
for name, values in metrics_5.items():
    plt.figure(figsize=(6,4))
    sns.histplot(values, bins=30, kde=True, color="steelblue", stat='probability')
    plt.title(f"Histograma de {name}")
    plt.xlabel(name)
    plt.ylabel("Probabilidad")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    figname = 'model_5_histogram_' + str(name) + '.pdf'
    plt.savefig(os.path.join(figspath, figname))

# ========== Modelo 5: Boxplots ==========
plt.figure(figsize=(8,5))
sns.boxplot(data=[psnrs_5, pesqs_5, stois_5, f0_rmses_5])
plt.xticks([0,1,2,3], list(metrics_5.keys()))
plt.title("Distribución de métricas (Boxplot)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
figname = 'model_5_boxplots.pdf'
plt.savefig(os.path.join(figspath, figname))

# (Opcional) ========== Modelo 5: Violin plots ==========
plt.figure(figsize=(8,5))
sns.violinplot(data=[psnrs_5, pesqs_5, stois_5, f0_rmses_5], inner="box")
plt.xticks([0,1,2,3], list(metrics_5.keys()))
plt.title("Distribución de métricas (Violin plot)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
figname = 'model_5_violinplots.pdf'
plt.savefig(os.path.join(figspath, figname))

# ========== Modelo 7: Histogramas ==========
for name, values in metrics_7.items():
    plt.figure(figsize=(6,4))
    sns.histplot(values, bins=30, kde=True, color="steelblue", stat='probability')
    plt.title(f"Histograma de {name}")
    plt.xlabel(name)
    plt.ylabel("Probabilidad")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    figname = 'model_7_histogram_' + str(name) + '.pdf'
    plt.savefig(os.path.join(figspath, figname))

# ========== Modelo 7: Boxplots ==========
plt.figure(figsize=(8,5))
sns.boxplot(data=[psnrs_7, pesqs_7, stois_7, f0_rmses_7])
plt.xticks([0,1,2,3], list(metrics_7.keys()))
plt.title("Distribución de métricas (Boxplot)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
figname = 'model_7_boxplots.pdf'
plt.savefig(os.path.join(figspath, figname))

# (Opcional) ========== Modelo 7: Violin plots ==========
plt.figure(figsize=(8,5))
sns.violinplot(data=[psnrs_7, pesqs_7, stois_7, f0_rmses_7], inner="box")
plt.xticks([0,1,2,3], list(metrics_7.keys()))
plt.title("Distribución de métricas (Violin plot)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
figname = 'model_7_violinplots.pdf'
plt.savefig(os.path.join(figspath, figname))
