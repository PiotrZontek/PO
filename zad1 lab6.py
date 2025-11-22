import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 6)

# --- Wczytanie obrazu koteczka ---
cat_path = "koteczek.jpg"
cat = cv2.imread(cat_path, cv2.IMREAD_GRAYSCALE)

if cat is None:
    raise FileNotFoundError("❌ Nie znaleziono pliku koteczek.jpg")

# --- Wyświetlenie obrazu ---
plt.imshow(cat, cmap='gray')
plt.title("koteczek.jpg – obraz wejściowy")
plt.axis('off')
plt.show()

# --- Zmniejszamy koteczka do 8×8 ---
cat_small = cv2.resize(cat, (8, 8), interpolation=cv2.INTER_AREA)

print("Macierz pikseli koteczek.jpg (8×8):\n")
print(cat_small)


