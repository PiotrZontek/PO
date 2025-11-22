import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Wczytanie obrazu ---
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("❌ Nie znaleziono pliku koteczek.jpg")

img = img.astype(np.float32)

# --- Transformacja FFT ---
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)
img_fft_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shift)))

# --- Transformacja DCT ---
dct = cv2.dct(img)
img_dct_back = cv2.idct(dct)

# --- Funkcja PSNR ---
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 999.0, 0
    return 20 * np.log10(255.0 / np.sqrt(mse)), mse

psnr_fft, mse_fft = psnr(img, img_fft_back)
psnr_dct, mse_dct = psnr(img, img_dct_back)

# --- Obliczenie różnic ---
diff_fft = img - img_fft_back
diff_dct = img - img_dct_back

abs_fft = np.abs(diff_fft)
abs_dct = np.abs(diff_dct)

# --- WIZUALIZACJA ---
plt.figure(figsize=(14, 10))

# FFT error
plt.subplot(2,3,1)
plt.imshow(abs_fft, cmap='inferno')
plt.title(f"Błąd FFT (abs)\nPSNR={psnr_fft:.2f} dB, MSE={mse_fft:.2e}")
plt.axis("off")

plt.subplot(2,3,4)
plt.hist(abs_fft.flatten(), bins=50, color='black')
plt.title("Histogram błędu FFT")

# DCT error
plt.subplot(2,3,2)
plt.imshow(abs_dct, cmap='inferno')
plt.title(f"Błąd DCT (abs)\nPSNR={psnr_dct:.2f} dB, MSE={mse_dct:.2e}")
plt.axis("off")

plt.subplot(2,3,5)
plt.hist(abs_dct.flatten(), bins=50, color='black')
plt.title("Histogram błędu DCT")

# Porównanie rekonstrukcji
plt.subplot(2,3,3)
plt.imshow(np.abs(img_fft_back - img_dct_back), cmap='inferno')
plt.title("Różnica między FFT a DCT")
plt.axis("off")

plt.subplot(2,3,6)
plt.hist(np.abs(img_fft_back - img_dct_back).flatten(), bins=50, color='blue')
plt.title("Histogram różnic FFT vs DCT")

plt.tight_layout()
plt.show()
