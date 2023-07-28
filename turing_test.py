import os
import cv2
import json

path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"
folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
ratings = {}
for i in folders:
    imm = cv2.imread(path + i + r"\DisplayImage.png")
    imm = cv2.resize(imm, (800, 600))
    cv2.imshow("Immagine",imm)
    cv2.waitKey(0)
    pmp = input("dire se la foto è idonea[i] o non idonea[n]")
    if pmp.lower()=="i":
        val = True
    else:
        val = False        
    ratings[i] = val
with open(os.path.dirname(os.path.abspath(__file__))+r"\ratings.json", 'w') as json_file:
    json.dump(ratings, json_file)