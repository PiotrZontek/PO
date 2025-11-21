import numpy as np
import matplotlib.pyplot as plt
import cv2

# === WCZYTANIE OBRAZU KOTECZEK ===
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# --- FFT ---
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

# --- MASKA LPF (z poprzedniego kroku) ---
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

r = 40                     # promień LPF
y, x = np.ogrid[:rows, :cols]
mask_lpf = ((x - ccol)**2 + (y - crow)**2 <= r*r).astype(np.uint8)

# === MASKA HPF ===
mask_hpf = 1 - mask_lpf

# --- Filtracja HPF ---
hpf_shift = fft_shift * mask_hpf
hpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(hpf_shift)))

# --- Wykres ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(mask_hpf, cmap="gray")
plt.title("Maska HPF")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(hpf_back, cmap="gray")
plt.title("Obraz po HPF")
plt.axis("off")

plt.show()
