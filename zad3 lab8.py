import cv2
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
# --- Wczytanie obrazu ---
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("❌ Błąd: obraz 'koteczek.jpg' nie został znaleziony!")

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.title("Obraz wejściowy – koteczek")
plt.axis("off")
plt.show()



# 2D korelacja własna obrazu koteczka
corr_2d = correlate2d(img, img, mode='same')

plt.figure(figsize=(6,6))
plt.imshow(corr_2d, cmap='hot')
plt.title("Mapa korelacji 2D – koteczek")
plt.axis("off")
plt.show()
