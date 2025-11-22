import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Funkcja PSNR
# ---------------------------------------------------
def compute_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return np.inf
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# ---------------------------------------------------
# Wczytanie obrazu — koteczka
# ---------------------------------------------------
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("❌ Nie znaleziono pliku koteczek.jpg")

img = img.astype(np.float32)

# ---------------------------------------------------
# 1. FFT
# ---------------------------------------------------
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)
fft_mag = np.log(np.abs(fft_shift) + 1)

# Rekonstrukcja FFT
img_fft_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shift)))

# ---------------------------------------------------
# 2. DCT
# ---------------------------------------------------
dct = cv2.dct(img)
dct_mag = np.log(np.abs(dct) + 1)

# Rekonstrukcja DCT
img_dct_back = cv2.idct(dct)

# ---------------------------------------------------
# 3. Obliczenie PSNR
# ---------------------------------------------------
psnr_fft = compute_psnr(img, img_fft_back)
psnr_dct = compute_psnr(img, img_dct_back)

print("=== PSNR rekonstrukcji (koteczek.jpg) ===")
print(f"PSNR FFT: {psnr_fft:.4f} dB")
print(f"PSNR DCT: {psnr_dct:.4f} dB")

# ---------------------------------------------------
# 4. Wizualizacja
# ---------------------------------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("koteczek.jpg — oryginał")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(fft_mag, cmap='gray')
plt.title("Widmo FFT (log)")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(dct_mag, cmap='gray')
plt.title("Widmo DCT (log)")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(img_fft_back, cmap='gray')
plt.title(f"Rekonstrukcja FFT\nPSNR = {psnr_fft:.2f} dB")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(img_dct_back, cmap='gray')
plt.title(f"Rekonstrukcja DCT\nPSNR = {psnr_dct:.2f} dB")
plt.axis("off")

plt.tight_layout()
plt.show()
