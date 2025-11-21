import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 1. WCZYTANIE OBRAZU
# ================================
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.title("Oryginał")
plt.axis("off")
plt.show()

# ================================
# 2. FFT + SHIFT
# ================================
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# -------------------------------
# LOW PASS FILTER (LPF)
# -------------------------------
r = 40
y, x = np.ogrid[:rows, :cols]
mask_lpf = ((x - ccol)**2 + (y - crow)**2) <= r*r

lpf_shift = fft_shift * mask_lpf
lpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(lpf_shift)))

# -------------------------------
# HIGH PASS FILTER (HPF)
# -------------------------------
mask_hpf = 1 - mask_lpf

hpf_shift = fft_shift * mask_hpf
hpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(hpf_shift)))

# -------------------------------
# BAND PASS FILTER (BPF)
# -------------------------------
r1, r2 = 20, 80
mask_bpf = np.logical_and(
    (x - ccol)**2 + (y - crow)**2 >= r1*r1,
    (x - ccol)**2 + (y - crow)**2 <= r2*r2
)

bpf_shift = fft_shift * mask_bpf
bpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(bpf_shift)))

# ================================
# 3. WIZUALIZACJA 2×2
# ================================
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Oryginał")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(lpf_back, cmap='gray')
plt.title("LPF — wygładzenie")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(hpf_back, cmap='gray')
plt.title("HPF — krawędzie")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(bpf_back, cmap='gray')
plt.title("BPF — częstotliwości średnie")
plt.axis("off")

plt.show()
