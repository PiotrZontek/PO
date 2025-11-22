import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Wczytanie obrazu koteczka ---
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("❌ Nie znaleziono pliku koteczek.jpg")

img = np.float32(img)

# --- DCT ---
dct_img = cv2.dct(img)

# --- IDCT ---
img_idct = cv2.idct(dct_img)

# --- Wizualizacja ---
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("koteczek.jpg – oryginał")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(np.log(np.abs(dct_img) + 1), cmap='gray')
plt.title("Widmo DCT (log)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_idct, cmap='gray')
plt.title("Rekonstrukcja z IDCT")
plt.axis("off")

plt.show()
