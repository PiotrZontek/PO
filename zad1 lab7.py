import cv2
import numpy as np
import matplotlib.pyplot as plt

# === WCZYTANIE OBRAZU koteczek.jpg ===
img_path = "koteczek.jpg"   # <-- obraz z uploadu
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

if img is None:
    raise FileNotFoundError("Nie udało się wczytać koteczek.jpg")

# --- Podgląd ---
plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.title("Obraz oryginalny – koteczek")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# 1. FFT + przesunięcie widma
# ---------------------------------------------------------
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

# ---------------------------------------------------------
# 2. Tworzenie maski LPF (Low-Pass Filter)
# ---------------------------------------------------------
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

r = 40  # promień filtru dolnoprzepustowego
mask_lpf = np.zeros((rows, cols), np.uint8)

y, x = np.ogrid[:rows, :cols]
mask_lpf[(x - ccol)**2 + (y - crow)**2 <= r*r] = 1

# ---------------------------------------------------------
# 3. Filtracja w dziedzinie częstotliwości
# ---------------------------------------------------------
lpf_shift = fft_shift * mask_lpf
lpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(lpf_shift)))

# ---------------------------------------------------------
# 4. Wynik
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(mask_lpf, cmap="gray")
plt.title("Maska LPF")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(lpf_back, cmap="gray")
plt.title("Obraz po LPF – koteczek")
plt.axis("off")

plt.show()
