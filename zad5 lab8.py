import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# ------------------------------------------------
# 1. Wczytanie obrazu
# ------------------------------------------------
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("❌ Błąd: obraz 'koteczek.jpg' nie został znaleziony!")

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.title("Obraz wejściowy – koteczek")
plt.axis("off")
plt.show()

# ------------------------------------------------
# 2. Odchylenie standardowe
# ------------------------------------------------
std_value = np.std(img)

# ------------------------------------------------
# 3. Korelacja Pearsona przy przesunięciu 1 px
# ------------------------------------------------
shifted = np.roll(img, 1, axis=1)
corr_matrix = np.corrcoef(img.flatten(), shifted.flatten())
corr_value = corr_matrix[0, 1]

# ------------------------------------------------
# 4. Energia obrazu
# ------------------------------------------------
energy = np.sum(img.astype(np.float64)**2)

# ------------------------------------------------
# 5. 2D korelacja własna
# ------------------------------------------------
corr_2d = correlate2d(img, img, mode='same')

plt.figure(figsize=(6,6))
plt.imshow(corr_2d, cmap='hot')
plt.title("Mapa korelacji 2D – koteczek")
plt.axis("off")
plt.show()

# ------------------------------------------------
# 6. Filtr Gaussa – eksperyment
# ------------------------------------------------
blur = cv2.GaussianBlur(img, (11,11), 5)

std_blur = np.std(blur)
energy_blur = np.sum(blur.astype(np.float64)**2)
corr_blur = np.corrcoef(img.flatten(), blur.flatten())[0,1]

print("\n==== PODSUMOWANIE – parametry obrazu ====")
print(f"Odchylenie standardowe         : {std_value:.4f}")
print(f"Korelacja Pearsona (1 px)      : {corr_value:.4f}")
print(f"Energia                        : {energy:.2f}")

print("\n==== EKSPERYMENT: rozmycie Gaussa ====")
print(f"Odchylenie std po filtracji    : {std_blur:.4f}")
print(f"Korelacja z obrazem oryginalnym: {corr_blur:.4f}")
print(f"Energia po filtracji           : {energy_blur:.2f}")

# ------------------------------------------------
# 7. Wizualizacja rozmycia Gaussa
# ------------------------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Oryginał")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(blur, cmap='gray')
plt.title("Po filtrze Gaussa")
plt.axis("off")

plt.show()
