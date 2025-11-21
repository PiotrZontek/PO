import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# FUNKCJE DO TWORZENIA MASEK
# =========================================================


def make_bpf_mask(rows, cols, r1, r2):
    """Band-pass filter mask (pierścień)."""
    y, x = np.ogrid[:rows, :cols]
    crow, ccol = rows//2, cols//2
    dist = (x-ccol)**2 + (y-crow)**2
    return ((dist >= r1*r1) & (dist <= r2*r2)).astype(np.uint8)

# =========================================================
# FUNKCJA DO FILTRACJI W DZIEDZINIE CZĘSTOTLIWOŚCI
# =========================================================
def apply_frequency_filter(img, mask):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    filtered = fft_shift * mask
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return result

# =========================================================
# WCZYTANIE OBRAZU
# =========================================================
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
if img is None:
    raise FileNotFoundError("Nie znaleziono pliku koteczek.jpg")

rows, cols = img.shape

# =========================================================
# MASKI
# =========================================================
mask_bpf = make_bpf_mask(rows, cols, r1=20, r2=80)

# =========================================================
# FILTRACJE
# =========================================================

bpf_result = apply_frequency_filter(img, mask_bpf)

# =========================================================
# WIZUALIZACJA
# =========================================================
plt.figure(figsize=(14,10))

# Oryginał
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Oryginał")
plt.axis("off")



# BPF
plt.subplot(2, 2, 4)
plt.imshow(bpf_result, cmap='gray')
plt.title("BPF – pasmowoprzepustowy")
plt.axis("off")

plt.tight_layout()
plt.show()
