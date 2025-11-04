import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from skimage.metrics import structural_similarity as ssim

# === FUNKCJA IoU ===
def iou_mask(mask1, mask2):
    # Zamiana na maski binarne 0/1
    m1 = (mask1 > 0).astype(np.uint8).ravel()
    m2 = (mask2 > 0).astype(np.uint8).ravel()

    return jaccard_score(m1, m2)  # IoU

# === FUNKCJA DICE ===
def dice_mask(mask1, mask2):
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)

    intersection = np.sum(m1 * m2)
    return (2 * intersection) / (np.sum(m1) + np.sum(m2) + 1e-8)

# === SSIM między maskami ===
def compute_ssim(mask1, mask2):
    m1 = (mask1 > 0).astype(np.uint8) * 255
    m2 = (mask2 > 0).astype(np.uint8) * 255
    return ssim(m1, m2)

# Wczytanie obrazu (w odcieniach szarości)
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku moj_obraz.jpg — upewnij się, że jest w tym samym folderze!")

# Zadanie 1: test z różnymi kernelami mediany
results_med = []
kernels = [1,3,5,7]  # kernel=1 => brak filtra (identity)

for k in kernels:
    if k == 1:
        img_proc = img_gray.copy()
    else:
        img_proc = cv2.medianBlur(img_gray, k)

    # segmentacje
    _, mask_otsu = cv2.threshold(img_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_adapt = cv2.adaptiveThreshold(img_proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    iou_val = iou_mask(mask_otsu, mask_adapt)
    dice_val = dice_mask(mask_otsu, mask_adapt)
    ssim_val = compute_ssim(mask_otsu, mask_adapt)

    results_med.append({'kernel':k, 'iou':iou_val, 'dice':dice_val, 'ssim_mask':ssim_val})
    print(f"kernel={k}: IoU={iou_val:.4f}, Dice={dice_val:.4f}, SSIM(mask)={ssim_val:.4f}")

df_med = pd.DataFrame(results_med)
print(df_med)
