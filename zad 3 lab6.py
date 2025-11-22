import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from mpl_toolkits.mplot3d import Axes3D

# --- Wczytanie koteczka ---
cat_path = "koteczek.jpg"
cat = cv2.imread(cat_path, cv2.IMREAD_GRAYSCALE)

if cat is None:
    raise FileNotFoundError("❌ Nie znaleziono pliku koteczek.jpg")

# --- DCT 2D ---
dct_img = dct(dct(cat.T, norm='ortho').T, norm='ortho')

# --- magnituda logarytmiczna ---
dct_mag = np.log(1 + np.abs(dct_img))

# --- dane do wykresu ---
X = np.arange(0, dct_mag.shape[1])
Y = np.arange(0, dct_mag.shape[0])
X, Y = np.meshgrid(X, Y)

Z_dct = dct_mag

# --- wizualizacja 3D ---
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z_dct, cmap='inferno')
ax.set_title("Widmo DCT koteczek.jpg – wizualizacja 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Amplituda (log)")

plt.show()
