import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. WCZYTANIE OBRAZU
# ---------------------------------------------------------
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)



# ---------------------------------------------------------
# 2. FFT + przesunięcie widma
# ---------------------------------------------------------
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

y, x = np.ogrid[:rows, :cols]

# ---------------------------------------------------------
# 3. LPF — filtr dolnoprzepustowy
# ---------------------------------------------------------
r = 40
mask_lpf = np.zeros((rows, cols), np.uint8)
mask_lpf[(x - ccol)**2 + (y - crow)**2 <= r*r] = 1

lpf_shift = fft_shift * mask_lpf
lpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(lpf_shift)))

# ---------------------------------------------------------
# 4. HPF — filtr górnoprzepustowy
# ---------------------------------------------------------
mask_hpf = 1 - mask_lpf

hpf_shift = fft_shift * mask_hpf
hpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(hpf_shift)))

# ---------------------------------------------------------
# 5. BPF — filtr pasmowoprzepustowy
# ---------------------------------------------------------
mask_bpf = np.zeros((rows, cols), np.uint8)
r1, r2 = 20, 80

mask_bpf[((x - ccol)**2 + (y - crow)**2 >= r1*r1) &
         ((x - ccol)**2 + (y - crow)**2 <= r2*r2)] = 1

bpf_shift = fft_shift * mask_bpf
bpf_back = np.abs(np.fft.ifft2(np.fft.ifftshift(bpf_shift)))
# ---------------------------------------------------------
# 7. MSE i PSNR
# ---------------------------------------------------------
def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    MSE = mse(a, b)
    if MSE == 0:
        return 999
    return 10 * np.log10((255 * 255) / MSE)

# MSE
mse_lpf = mse(img, lpf_back)
mse_hpf = mse(img, hpf_back)
mse_bpf = mse(img, bpf_back)

print("=== MSE ===")
print("LPF:", mse_lpf)
print("HPF:", mse_hpf)
print("BPF:", mse_bpf)

# PSNR
print("\n=== PSNR ===")
print("LPF:", psnr(img, lpf_back))
print("HPF:", psnr(img, hpf_back))
print("BPF:", psnr(img, bpf_back))
