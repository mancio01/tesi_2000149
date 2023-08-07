import cv2
import os
import Image_similitary as sim

IMG_X_VERTICES = [(321, 1048), (1698, 1048), (1683, 0), (296, 0)]
IMG_Y_VERTICES = [(192, 1048), (1690, 1048), (1635, 0), (181, 0)]

img = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\2023-01-09-08-01-27-32-Walle-Configuration 0\2023-01-09-08-01-24-89_first.png")
img = sim.crop_image(img,IMG_X_VERTICES)
cv2.imshow("Immagine",img)
cv2.waitKey(0)