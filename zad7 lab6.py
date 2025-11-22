import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# ===================================================
# 1. Generowanie syntetycznej ilustracji rozproszenia widma
# ===================================================
fft_grid = np.random.rand(10,10)  # widmo FFT – rozproszone wartości
dct_grid = np.zeros((10,10))
dct_grid[:3,:3] = np.random.rand(3,3)  # widmo DCT – energia skupiona w narożniku

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(fft_grid, cmap='gray')
plt.title("FFT – rozproszone widmo")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(dct_grid, cmap='gray')
plt.title("DCT – skondensowane widmo")
plt.axis('off')
plt.show()

# ===================================================
# 2. Wczytanie obrazu koteczka
# ===================================================
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("❌ Nie znaleziono koteczek.jpg")

img = img.astype(np.float32)

# ===================================================
# 3. FFT obrazu
# ===================================================
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)
magnitude_fft = np.log(np.abs(fft_shift) + 1)

# ===================================================
# 4. DCT obrazu
# ===================================================
dct_2d = dct(dct(img.T, norm='ortho').T, norm='ortho')
dct_mag = np.log(np.abs(dct_2d) + 1)

# ===================================================
# 5. Wizualizacja widm amplitudowych
# ===================================================
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(magnitude_fft, cmap='gray')
plt.title("FFT — widmo amplitudowe (koteczek)")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(dct_mag, cmap='gray')
plt.title("DCT — widmo amplitudowe (koteczek)")
plt.axis('off')

plt.tight_layout()
plt.show()

# ===================================================
# 6. Funkcja energii
# ===================================================
def energy(matrix):
    return np.sum(matrix.astype(np.float64)**2)

# ===================================================
# 7. Koncentracja energii FFT vs DCT
# ===================================================

# FFT energia całkowita
fft_energy_total = energy(np.abs(fft_shift))

# FFT energia niskich częstotliwości (centrum 40×40)
fft_energy_low = energy(
    np.abs(fft_shift)[
        int(img.shape[0]//2 - 20) : int(img.shape[0]//2 + 20),
        int(img.shape[1]//2 - 20) : int(img.shape[1]//2 + 20)
    ]
)

# DCT energia całkowita
dct_energy_total = energy(np.abs(dct_2d))

# DCT energia niskich częstotliwości (lewy górny róg 40×40)
dct_energy_low = energy(np.abs(dct_2d)[:40, :40])

# ===================================================
# 8. Wyniki porównania
# ===================================================
print("=== KONCENTRACJA ENERGII (koteczek.jpg) ===")
print(f"FFT: energia niskich częstotliwości = {fft_energy_low / fft_energy_total :.4f}")
print(f"DCT: energia niskich częstotliwości = {dct_energy_low / dct_energy_total :.4f}")
