import cv2
import numpy as np
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

# przesunięcie o 1 piksel w prawo
shifted = np.roll(img, 1, axis=1)

# korelacja Pearsona
corr_matrix = np.corrcoef(img.flatten(), shifted.flatten())
corr_value = corr_matrix[0, 1]

print("📌 Korelacja Pearsona (przesunięcie 1 px w prawo):", corr_value)
