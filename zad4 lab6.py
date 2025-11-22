import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# --- Funkcje DCT 2D i IDCT 2D ---
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# --- Wczytanie obrazu koteczka ---
cat_path = "koteczek.jpg"
cat = cv2.imread(cat_path, cv2.IMREAD_GRAYSCALE)

if cat is None:
    raise FileNotFoundError("❌ Nie znaleziono pliku koteczek.jpg")

# --- DCT pełnego obrazu ---
dct_img = dct2(cat.astype(np.float32))

plt.imshow(np.log(np.abs(dct_img) + 1), cmap='gray')
plt.title("Widmo DCT (log) – koteczek.jpg")
plt.axis('off')
plt.show()

# --- wersja 8×8 ---
cat_small = cv2.resize(cat, (8, 8), interpolation=cv2.INTER_AREA)

# --- DCT 8×8 ---
dct_matrix = dct2(cat_small.astype(np.float32))

print("\nMacierz współczynników DCT koteczek.jpg (8×8 po 2D-DCT):\n")
print(np.round(dct_matrix, 2))
