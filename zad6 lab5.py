import cv2
import numpy as np
import matplotlib.pyplot as plt

img_rgb = cv2.imread("sloneczniki.png")
if img_rgb is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

# Detekcja koloru w HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

# zakresy dla czerwieni
lower1 = np.array([0, 100, 50])
upper1 = np.array([10, 255, 255])
lower2 = np.array([170, 100, 50])
upper2 = np.array([180, 255, 255])

mask1 = cv2.inRange(img_hsv, lower1, upper1)
mask2 = cv2.inRange(img_hsv, lower2, upper2)
mask_red = cv2.bitwise_or(mask1, mask2)

# oczyszczenie maski
kernel = np.ones((5,5), np.uint8)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

# Wyswietlanie
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
plt.title("Obraz RGB")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask_red, cmap="gray")
plt.title("Maska koloru czerwonego")
plt.axis("off")

plt.show()
