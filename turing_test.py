import os
import cv2
import json
from matplotlib import pyplot as plt

path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"
folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
ratings = {}
for i in folders:
    imm = cv2.imread(path + i + r"\DisplayImage.png")
    imm = cv2.resize(imm, (800, 600))
    plt.imshow(cv2.cvtColor(imm, cv2.COLOR_BGR2RGB))
    plt.show(block = False)
    pmp = input("dire se la foto e' idonea[i] o non idonea[n]")
    plt.close()
    if pmp.lower()=="i":
        val = True
    else:
        val = False        
    ratings[i] = val
with open(os.path.dirname(os.path.abspath(__file__))+r"\ratings.json", 'w') as json_file:
    json.dump(ratings, json_file)