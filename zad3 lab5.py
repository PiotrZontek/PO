

import cv2

img_rgb = cv2.imread("sloneczniki.png")
if img_rgb is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")

img_shapes = img_rgb.copy()

# Wczytanie obrazu (w odcieniach szarości)
img_gray = cv2.imread("sloneczniki.png", cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Nie znaleziono pliku sloneczniki.png!")


# przygotowanie maski i konturów
_, mask_otsu_full = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(mask_otsu_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_shapes = img_rgb.copy()
ellipse_count = 0
rect_count = 0
other_count = 0
shapes = []

for c in contours:
    area = cv2.contourArea(c)
    if area < 200:  # pomiń drobne szumy
        continue
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        # prostokąt / kwadrat - sprawdź stosunek boków
        x,y,w,h = cv2.boundingRect(approx)
        ar = float(w)/h if h>0 else 0
        rect_count += 1
        label = "Rectangle"
        cv2.drawContours(img_shapes, [approx], -1, (0,255,0), 2)
        cv2.putText(img_shapes, f"Rect", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
        shapes.append(('rect', area, ar))
    else:
        # spróbuj dopasować elipsę, jeśli kontur ma minimum 5 punktów
        if len(c) >= 5:
            try:
                el = cv2.fitEllipse(c)
                (xc, yc), (major, minor), angle = el
                if minor == 0:
                    continue
                ratio = minor/major if major>0 else 0
                # jeśli ratio bliskie 1 → okrąg/elipsa
                ellipse_count += 1
                cv2.ellipse(img_shapes, el, (255,0,0), 2)
                cv2.putText(img_shapes, f"Elps", (int(xc), int(yc)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0),1)
                shapes.append(('ellipse', area, ratio))
            except Exception as e:
                other_count += 1
        else:
            other_count += 1

print(f"Prostokąty: {rect_count}, Elipsy: {ellipse_count}, Pozostałe: {other_count}")
print(img_shapes, "Wykryte kształty (prostokąty zielone, elipsy czerwone)")

# Wyświetlenie obrazu z wykrytymi kształtami
cv2.imshow("Wykryte ksztalty", img_shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
