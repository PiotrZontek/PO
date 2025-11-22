import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# --- Funkcje DCT 2D i IDCT 2D ---
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# --- Wczytanie koteczka ---
cat_path = "koteczek.jpg"
cat = cv2.imread(cat_path, cv2.IMREAD_GRAYSCALE)

if cat is None:
    raise FileNotFoundError("❌ Nie znaleziono pliku koteczek.jpg")

# --- DCT pełnego obrazu ---
dct_img = dct2(cat.astype(np.float32))

# --- Rekonstrukcja obrazu z DCT ---
img_dct_rec = idct2(dct_img)

plt.imshow(img_dct_rec, cmap='gray')
plt.title("Rekonstrukcja obrazu koteczek.jpg z DCT")
plt.axis('off')
plt.show()
