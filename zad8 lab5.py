import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# === FUNKCJE IoU i DICE JEŚLI NIE MASZ ===
def iou_mask(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

# === WCZYTANIE DRUGIEGO OBRAZU ===
img_path2 = "jelen.jpg"  # <-- tu wstaw swój drugi plik
img_rgb2 = cv2.imread(img_path2)
if img_rgb2 is None:
    raise FileNotFoundError(f"Nie znaleziono pliku {img_path2}")

# Wczytanie obrazu (w odcieniach szarości)
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

gauss = cv2.GaussianBlur(img_gray, (5,5), 0)
img_rgb2 = cv2.cvtColor(img_rgb2, cv2.COLOR_BGR2RGB)
_, mask_otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# pipeline na drugim obrazie
img_gray2 = cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2GRAY)
start = time.perf_counter()
gauss2 = cv2.GaussianBlur(img_gray2, (5,5), 1.0)
_, mask_otsu2 = cv2.threshold(gauss2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges2 = cv2.Canny(gauss2, 100, 200)
contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
t_elapsed = time.perf_counter() - start
print(f"Obliczenia dla obrazu 2: czas = {t_elapsed:.4f}s, konturów wykryto = {len(contours2)}")
# dopasowanie rozmiarów masek
h1, w1 = mask_otsu.shape
mask_otsu2_resized = cv2.resize(mask_otsu2, (w1, h1), interpolation=cv2.INTER_NEAREST)

# Porównanie z pierwszym obrazem IoU między maskami Otsu
iou_otsu = iou_mask(mask_otsu, mask_otsu2_resized)
print("IoU (Otsu obraz1 vs obraz2):", iou_otsu)
