import os
import cv2
import json
from matplotlib import pyplot as plt

train_path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"
test_path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering\test\\"
train_folders = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]
test_folders = [name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))]
train = {}
test = {}
for i in train_folders:
    imm = cv2.imread(train_path + i + r"\DisplayImage.png")
    imm = cv2.resize(imm, (800, 600))
    plt.imshow(cv2.cvtColor(imm, cv2.COLOR_BGR2RGB))
    plt.show(block = False)
    pmp = input("dire se la foto e' idonea[i] o non idonea[n]")
    plt.close()
    if pmp.lower()=="i":
        val = True
    else:
        val = False        
    train[i] = val
    
for i in test_folders:
    imm = cv2.imread(test_path + i + r"\DisplayImage.png")
    imm = cv2.resize(imm, (800, 600))
    plt.imshow(cv2.cvtColor(imm, cv2.COLOR_BGR2RGB))
    plt.show(block = False)
    pmp = input("dire se la foto e' idonea[i] o non idonea[n]")
    plt.close()
    if pmp.lower()=="i":
        val = True
    else:
        val = False        
    test[i] = val

ratings = {"train":train, "test":test}

with open(os.path.dirname(os.path.abspath(__file__))+r"\ratings.json", 'w') as json_file:
    json.dump(ratings, json_file, indent=4)