import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Wczytanie koteczka ---
cat_path = "koteczek.jpg"
cat = cv2.imread(cat_path, cv2.IMREAD_GRAYSCALE)

if cat is None:
    raise FileNotFoundError("❌ Nie znaleziono pliku koteczek.jpg")

# --- FFT + przesunięcie widma ---
fft = np.fft.fft2(cat)
fft_shift = np.fft.fftshift(fft)

# --- Magnituda widma (log) ---
fft_mag = np.log(1 + np.abs(fft_shift))

# --- Przygotowanie danych 3D ---
X = np.arange(0, fft_mag.shape[1])
Y = np.arange(0, fft_mag.shape[0])
X, Y = np.meshgrid(X, Y)
Z = fft_mag

# --- Wizualizacja 3D ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("Widmo FFT koteczek.jpg – wizualizacja 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Amplituda (log)")

plt.show()
