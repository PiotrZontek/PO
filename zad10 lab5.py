import cv2
from skimage.metrics import structural_similarity as ssim
import time
import numpy as np

# ===== dodatkowe funkcje metryk =====
def iou_mask(mask1, mask2):
    m1 = (mask1 > 0).astype(bool)
    m2 = (mask2 > 0).astype(bool)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union != 0 else 0.0

def dice_mask(mask1, mask2):
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)
    intersection = np.sum(m1 * m2)
    denom = np.sum(m1) + np.sum(m2)
    return (2.0 * intersection) / denom if denom != 0 else 0.0

def safe_ssim(mask1, mask2):
    # Przygotuj maski jako uint8 (0/255) i ustaw data_range=255
    m1 = (mask1 > 0).astype(np.uint8) * 255
    m2 = (mask2 > 0).astype(np.uint8) * 255
    try:
        return ssim(m1, m2, data_range=255)
    except Exception:
        # jako awaryjna opcja: zamiana na float i obliczenie bez data_range
        return ssim(m1.astype(float), m2.astype(float))

# ===== wczytanie obrazu (przykład) =====
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

img_rgb = cv2.imread("sloneczniki.png")
if img_rgb is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

# ===== pipeline =====
def run_pipeline(img_rgb_input, params):
    res = {}
    timings = {}

    img_gray_local = cv2.cvtColor(img_rgb_input, cv2.COLOR_BGR2GRAY)

    # filtr
    t0 = time.perf_counter()
    if params.get('filter') == 'gauss':
        k = params.get('filter_k', 5)
        proc = cv2.GaussianBlur(img_gray_local, (k, k), 0)
    elif params.get('filter') == 'median':
        k = params.get('filter_k', 5)
        proc = cv2.medianBlur(img_gray_local, k)
    elif params.get('filter') == 'bilateral':
        proc = cv2.bilateralFilter(img_gray_local, 9, 75, 75)
    else:
        proc = img_gray_local.copy()
    timings['filter'] = time.perf_counter() - t0
    res['proc'] = proc

    # Otsu
    t0 = time.perf_counter()
    if params.get('otsu', True):
        _, mask_otsu_local = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask_otsu_local = None
    timings['otsu'] = time.perf_counter() - t0
    res['mask_otsu'] = mask_otsu_local

    # adaptive
    t0 = time.perf_counter()
    if params.get('adaptive', True):
        mask_adapt_local = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
    else:
        mask_adapt_local = None
    timings['adaptive'] = time.perf_counter() - t0
    res['mask_adapt'] = mask_adapt_local

    # canny
    t0 = time.perf_counter()
    if params.get('canny') is not None:
        low, high = params['canny']
        edges_local = cv2.Canny(proc, low, high)
    else:
        edges_local = None
    timings['canny'] = time.perf_counter() - t0
    res['edges'] = edges_local

    # contours
    t0 = time.perf_counter()
    mask_for_contour = None
    if mask_otsu_local is not None:
        mask_for_contour = mask_otsu_local
    elif mask_adapt_local is not None:
        mask_for_contour = mask_adapt_local
    else:
        mask_for_contour = (proc > 0).astype(np.uint8) * 255

    contours_local, _ = cv2.findContours(mask_for_contour.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    timings['contours'] = time.perf_counter() - t0
    res['contours'] = contours_local
    res['timings'] = timings

    # metryki
    if (mask_otsu_local is not None) and (mask_adapt_local is not None):
        # upewnij się, że mają ten sam rozmiar (zazwyczaj mają, bo są z tego samego proc)
        if mask_otsu_local.shape != mask_adapt_local.shape:
            mask_adapt_local = cv2.resize(mask_adapt_local, (mask_otsu_local.shape[1], mask_otsu_local.shape[0]), interpolation=cv2.INTER_NEAREST)
        res['iou'] = iou_mask(mask_otsu_local, mask_adapt_local)
        res['dice'] = dice_mask(mask_otsu_local, mask_adapt_local)
        res['ssim_mask'] = safe_ssim(mask_otsu_local, mask_adapt_local)
    else:
        res['iou'] = None
        res['dice'] = None
        res['ssim_mask'] = None

    return res

# ===== przykładowe wywołanie =====
params_example = {'filter':'median','filter_k':5,'otsu':True,'adaptive':True,'canny':(100,200)}
out_example = run_pipeline(img_rgb, params_example)
print("Przykładowe metryki:", {k:out_example.get(k) for k in ['iou','dice','ssim_mask']})
print("Czasy:", out_example['timings'])
