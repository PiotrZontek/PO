import cv2
from skimage.metrics import structural_similarity as ssim

# Wczytanie obrazu (w odcieniach szarości)
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

# Obliczenie masek dla porównania (użyjemy obrazu oryginalnego)
_, mask_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mask_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

ssim_val_masks = ssim(mask_otsu, mask_adapt)
print("SSIM (Otsu vs Adaptacyjne):", ssim_val_masks)
