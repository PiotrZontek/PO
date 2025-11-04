import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from skimage.metrics import structural_similarity as ssim

# === FUNKCJA IoU ===
def iou_mask(mask1, mask2):
    # Zamiana na maski binarne 0/1
    m1 = (mask1 > 0).astype(np.uint8).ravel()
    m2 = (mask2 > 0).astype(np.uint8).ravel()

    return jaccard_score(m1, m2)  # IoU

# Wczytanie obrazu (w odcieniach szarości)
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku moj_obraz.jpg — upewnij się, że jest w tym samym folderze!")

# przygotowanie referencyjnej maski Otsu na obrazie oryginalnym (bez filtracji)
_, ref_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def edges_to_mask(edges, dilate_iter=2):
    # rozszerz krawędzie, by mogły być porównane z maską regionów
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=dilate_iter)
    return mask

canny_params = [(50,150), (75,200), (100,200), (50,200), (30,100)]
res_canny = []
for low, high in canny_params:
    edges = cv2.Canny(img_gray, low, high)
    mask_from_edges = edges_to_mask(edges, dilate_iter=3)
    # porównaj maskę z ref_otsu
    iou_val = iou_mask(mask_from_edges, ref_otsu)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_canny.append({'low':low, 'high':high, 'contours':len(contours), 'iou':iou_val})
    print(f"Canny({low},{high}) -> contours: {len(contours)}, IoU vs Otsu: {iou_val:.4f}")

df_canny = pd.DataFrame(res_canny)
print(df_canny)
