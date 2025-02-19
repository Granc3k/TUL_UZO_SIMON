# made by Martin "Granc3k" Šimon, Jakub "Parrot2" Keršláger
import numpy as np
import cv2


# Fce pro calc těžiště obrazu (zeroth moment)
def zeroth_moment(back_projection):
    x, y = np.meshgrid(
        np.arange(back_projection.shape[1]), np.arange(back_projection.shape[0])
    )

    # Výpočet těžiště (vážený průměr souřadnic podle histogramu)
    x_t = np.sum(x * back_projection) / np.sum(back_projection)
    y_t = np.sum(y * back_projection) / np.sum(back_projection)

    return int(x_t), int(y_t)


# Load videa a vzoru
cap = cv2.VideoCapture("./data/cv02_hrnecek.mp4")
pattern = cv2.imread("./data/cv02_vzor_hrnecek.bmp")

# Transfer vzoru na HSV a výpočet histogramu
pattern_hsv = cv2.cvtColor(pattern, cv2.COLOR_BGR2HSV)
hue_pattern = pattern_hsv[:, :, 0]
roi_hist, _ = np.histogram(hue_pattern, bins=180, range=(0, 180))
roi_hist = roi_hist / np.max(roi_hist)  # Normalizace

# Fixní rozměry vzoru
pattern_h, pattern_w, _ = pattern.shape

is_first = True
prev_xy = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Převedení aktuálního snímku na HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_frame = frame_hsv[:, :, 0]

    # Projekce histogramu na snímek
    back_projection = roi_hist[hue_frame]

    if is_first:
        # Pokud first snímek, init těžiště
        x_t, y_t = zeroth_moment(back_projection)
        is_first = False
    else:
        # Calc posunu těžiště v rámci předchozí oblasti
        x_t, y_t = zeroth_moment(
            back_projection[prev_xy[1] : prev_xy[3], prev_xy[0] : prev_xy[2]]
        )
        x_t += prev_xy[0]
        y_t += prev_xy[1]

    # Výpočet souřadnic obdélníku kolem těžiště
    x1_t = max(0, x_t - pattern_w // 2)
    y1_t = max(0, y_t - pattern_h // 2)
    x2_t = min(frame.shape[1], x_t + pattern_w // 2)
    y2_t = min(frame.shape[0], y_t + pattern_h // 2)
    prev_xy = (x1_t, y1_t, x2_t, y2_t)

    # Print obdélníku
    cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 0), 2)
    cv2.imshow("Hrnecek Tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC - ukončí program
        break

cap.release()
cv2.destroyAllWindows()
