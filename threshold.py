import Image_similitary as sim
import cv2
import os
import json
import matplotlib.pyplot as plt

IMG_X_VERTICES = [(321, 1048), (1698, 1048), (1683, 0), (296, 0)]
IMG_Y_VERTICES = [(192, 1048), (1690, 1048), (1635, 0), (181, 0)]

with open("ratings.json", "r") as file:
    ratings = json.load(file)
rmse_t = []
rmse_f = []
path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"
folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
for i in folders:
    for j in os.listdir(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"+i):
        if j.endswith("first.png"):
            testo = path[:-2] + i + "\\" + j
            #img_x = sim.crop_image(cv2.imread(path[:-2] + "\\" + i + "\\" + j),IMG_X_VERTICES)
            img_x = cv2.imread(path[:-2] + "\\" + i + "\\" + j)        
        elif j.endswith("second.png"):
            #img_y = sim.crop_image(cv2.imread(path[:-2] + "\\" + i + "\\" + j),IMG_Y_VERTICES)
            img_y = cv2.imread(path[:-2] + "\\" + i + "\\" + j)
    if(ratings[i]):
        rmse_t.append(sim.imsimilarity(img_x, img_y, "RMSE"))
    elif(not ratings[i]):
        rmse_f.append(sim.imsimilarity(img_x, img_y, "RMSE")) 
rmse_t = sorted(rmse_t)
rmse_f = sorted(rmse_f) 
plt.scatter(rmse_f, range(len(rmse_f)), color = 'red', marker = 'o', label='false' )
plt.scatter(rmse_t, range(len(rmse_t)), color='blue', marker='o', label='true')
plt.show()