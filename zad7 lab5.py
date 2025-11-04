import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# ===== USTAWIENIA ŚCIEŻKI =====
img_path = Path("sloneczniki.png")
base_path = Path(".")  # katalog bieżący

img_rgb = cv2.imread("sloneczniki.png")
if img_rgb is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

img_shapes = img_rgb.copy()

# Wczytanie obrazu (w odcieniach szarości)
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

gauss = cv2.GaussianBlur(img_gray, (5,5), 0)
bilat = cv2.bilateralFilter(img_gray, 9, 75, 75)

# ===== SEGMENTACJA =====
_, mask_otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mask_adapt = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

# ===== HISTOGRAMY =====
hist_orig = cv2.calcHist([img_gray],[0],None,[256],[0,256])
hist_gauss = cv2.calcHist([gauss],[0],None,[256],[0,256])
hist_bilat = cv2.calcHist([bilat],[0],None,[256],[0,256])

# ===== WYKRYWANIE KRAWĘDZI =====
edges = cv2.Canny(img_gray, 100, 200)
img_shapes = img_rgb.copy()

# ===== ZAPIS PDF =====
out_pdf = base_path / "Lab5_results.pdf"

with PdfPages(out_pdf) as pdf:
    # strona tytułowa
    plt.figure(figsize=(8.27, 11.69)); plt.axis('off')
    plt.text(0.5, 0.6, 'Lab6 — Integracja Techniki Przetwarzania Obrazów', ha='center', fontsize=16)
    plt.text(0.5, 0.55, f'Plik źródłowy: {img_path.name if img_path.exists() else "skimage.astronaut()"}', ha='center')
    pdf.savefig(); plt.close()
    # oryginał i filtry
    fig = plt.figure(figsize=(8,6))
    plt.subplot(2,3,1); plt.imshow(img_rgb); plt.title('Oryginał'); plt.axis('off')
    plt.subplot(2,3,2); plt.imshow(img_gray, cmap='gray'); plt.title('Gray'); plt.axis('off')
    plt.subplot(2,3,3); plt.imshow(gauss, cmap='gray'); plt.title('Gauss'); plt.axis('off')
    plt.subplot(2,3,4); plt.imshow(bilat, cmap='gray'); plt.title('Bilateral'); plt.axis('off')
    plt.subplot(2,3,5); plt.imshow(mask_otsu, cmap='gray'); plt.title('Maska Otsu'); plt.axis('off')
    plt.subplot(2,3,6); plt.imshow(mask_adapt, cmap='gray'); plt.title('Maska Adapt.'); plt.axis('off')
    pdf.savefig(); plt.close()
    # krawędzie i kontury
    plt.figure(figsize=(8,6))
    edges = cv2.Canny(img_gray, 100, 200)
    plt.subplot(1,2,1); plt.imshow(edges, cmap='gray'); plt.title('Canny 100/200'); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(img_shapes); plt.title('Wykryte kształty'); plt.axis('off')
    pdf.savefig(); plt.close()
    # histogramy
    plt.figure(figsize=(8,6))
    plt.plot(hist_orig, label='orig'); plt.plot(hist_gauss, label='gauss'); plt.plot(hist_bilat, label='bilat')
    plt.legend(); plt.title('Histogramy (porównanie)'); plt.xlim([0,255])
    pdf.savefig(); plt.close()

print("Zapisano plik PDF z wynikami:", out_pdf)
