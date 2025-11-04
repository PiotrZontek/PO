import cv2
import matplotlib.pyplot as plt

# Wczytanie obrazu (w odcieniach szarości)
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

# Filtracja
gauss = cv2.GaussianBlur(img_gray, (5,5), 1.0)
bilat = cv2.bilateralFilter(img_gray, 9, 75, 75)

# histogramy
hist_orig = cv2.calcHist([img_gray],[0],None,[256],[0,256]).flatten()
hist_gauss = cv2.calcHist([gauss],[0],None,[256],[0,256]).flatten()
hist_bilat = cv2.calcHist([bilat],[0],None,[256],[0,256]).flatten()

# porównania histogramów - korelacja i chi-square
corr_gauss = cv2.compareHist(hist_orig.astype('float32'), hist_gauss.astype('float32'), cv2.HISTCMP_CORREL)
corr_bilat = cv2.compareHist(hist_orig.astype('float32'), hist_bilat.astype('float32'), cv2.HISTCMP_CORREL)
chi_gauss = cv2.compareHist(hist_orig.astype('float32'), hist_gauss.astype('float32'), cv2.HISTCMP_CHISQR)
chi_bilat = cv2.compareHist(hist_orig.astype('float32'), hist_bilat.astype('float32'), cv2.HISTCMP_CHISQR)

print("Korelacja (orig vs gauss):", corr_gauss)
print("Korelacja (orig vs bilateral):", corr_bilat)
print("Chi-square (orig vs gauss):", chi_gauss)
print("Chi-square (orig vs bilateral):", chi_bilat)

# wykres
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.title('Oryginał'); plt.plot(hist_orig); plt.xlim([0,255])
plt.subplot(1,3,2); plt.title('Gauss'); plt.plot(hist_gauss); plt.xlim([0,255])
plt.subplot(1,3,3); plt.title('Bilateral'); plt.plot(hist_bilat); plt.xlim([0,255])
plt.tight_layout(); plt.show()
