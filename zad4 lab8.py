import cv2
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

# --- Wczytanie obrazu ---
img = cv2.imread("koteczek.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("❌ Błąd: obraz 'koteczek.jpg' nie został znaleziony!")

# --- Wizualizacja ---
plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.title("Obraz wejściowy – koteczek")
plt.axis("off")
plt.show()

# --- Odchylenie standardowe ---
std_value = np.std(img)

# --- Korelacja Pearsona przy przesunięciu 1 px w prawo ---
shifted = np.roll(img, 1, axis=1)
corr_matrix = np.corrcoef(img.flatten(), shifted.flatten())
corr_value = corr_matrix[0, 1]

# --- Energia obrazu ---
energy = np.sum(img.astype(np.float64)**2)

print("\n==== PODSUMOWANIE ====")
print(f"Odchylenie standardowe: {std_value:.4f}")
print(f"Korelacja Pearsona (1 px): {corr_value:.4f}")
print(f"Energia: {energy:.2f}")

# --- 2D korelacja własna obrazu koteczka ---
corr_2d = correlate2d(img, img, mode='same')

plt.figure(figsize=(6,6))
plt.imshow(corr_2d, cmap='hot')
plt.title("Mapa korelacji 2D – koteczek")
plt.axis("off")
plt.show()
