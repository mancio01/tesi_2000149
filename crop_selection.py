import os
import cv2



def mouse_callback(event, x, y, flags, param):
    global crop
    if len(crop) < 4 and event == cv2.EVENT_LBUTTONDOWN:
        crop.append((x, y))

crop = []
img_x = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\2023-01-09-08-01-27-32-Walle-Configuration 0\2023-01-09-08-01-24-89_first.png")
#img_x = cv2.resize(img_x,(224,224))
cv2.imshow("Immagine", img_x)
cv2.setMouseCallback('Immagine', mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(crop)

crop = []
img_y = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\2023-01-09-08-01-27-32-Walle-Configuration 0\2023-01-09-08-01-28-43_second.png")
#img_y = cv2.resize(img_y,(224,224))
cv2.imshow("Immagine", img_y)
cv2.setMouseCallback('Immagine', mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(crop)
