import cv2
import time

# Wczytanie obrazu (w odcieniach szarości)
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

timings = {}
t0 = time.perf_counter()
g = cv2.GaussianBlur(img_gray, (5,5), 1.0)
timings['gauss'] = time.perf_counter() - t0

t0 = time.perf_counter()
_, m_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
timings['otsu'] = time.perf_counter() - t0

t0 = time.perf_counter()
edges = cv2.Canny(g, 100, 200)
timings['canny'] = time.perf_counter() - t0

t0 = time.perf_counter()
cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
timings['contours'] = time.perf_counter() - t0

print("Czasy (s):", timings)
