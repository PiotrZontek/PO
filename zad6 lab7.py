import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. WCZYTANIE OBRAZU
# ============================
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)



# ============================
# 2. FFT + MASKI
# ============================
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
y, x = np.ogrid[:rows, :cols]

# LPF maska
r = 40
mask_lpf = ((x - ccol)**2 + (y - crow)**2 <= r*r).astype(np.uint8)

# HPF maska
mask_hpf = 1 - mask_lpf

# BPF maska
r1, r2 = 20, 80
mask_bpf = (((x - ccol)**2 + (y - crow)**2 >= r1*r1) &
            ((x - ccol)**2 + (y - crow)**2 <= r2*r2)).astype(np.uint8)

# ============================
# 3. FILTRACJE
# ============================
# LPF
lpf_shift = fft_shift * mask_lpf
lpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(lpf_shift)))

# HPF
hpf_shift = fft_shift * mask_hpf
hpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(hpf_shift)))

# BPF
bpf_shift = fft_shift * mask_bpf
bpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(bpf_shift)))

# ============================


# ============================
# 5. MSE & PSNR
# ============================
def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    MSE = mse(a, b)
    if MSE == 0:
        return 999
    return 10 * np.log10((255 * 255) / MSE)

print("\n=== MSE ===")
print("LPF:", mse(img, lpf_back))
print("HPF:", mse(img, hpf_back))
print("BPF:", mse(img, bpf_back))

print("\n=== PSNR ===")
print("LPF:", psnr(img, lpf_back))
print("HPF:", psnr(img, hpf_back))
print("BPF:", psnr(img, bpf_back))

# ============================
# 6. HISTOGRAMY BŁĘDÓW
# ============================
errors = {
    "LPF": img - lpf_back,
    "HPF": img - hpf_back,
    "BPF": img - bpf_back,
}

plt.figure(figsize=(12,8))

for i, (name, err) in enumerate(errors.items()):
    plt.subplot(3,1,i+1)
    plt.hist(err.flatten(), bins=50, color='k')
    plt.title(f"Histogram błędu — {name}")

plt.tight_layout()
plt.show()
