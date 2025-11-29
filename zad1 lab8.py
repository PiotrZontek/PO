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

# --- Odchylenie standardowe ---
std_value = np.std(img)
print("📌 Odchylenie standardowe:", std_value)
