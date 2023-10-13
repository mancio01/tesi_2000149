import os
import cv2
import json
from matplotlib import pyplot as plt

train_path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"
test_path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering\test\\"
data_path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering_2\\"
folders = [test_path + name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))] + [train_path + name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))] + [data_path + name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
print(folders)
ratings = {}

for i in folders:
    print(i)
    imm = cv2.imread(i + r"\DisplayImage.png")
    imm = cv2.resize(imm, (1000, 800))
    plt.imshow(cv2.cvtColor(imm, cv2.COLOR_BGR2RGB))
    plt.show(block = False)
    pmp = input("dire se la foto e' idonea[i] o non idonea[n]")
    plt.close()
    if pmp.lower()=="i":
        val = True
    else:
        val = False        
    ratings[i] = val
 

with open(os.path.dirname(os.path.abspath(__file__))+r"\label.json", 'w') as json_file:
    json.dump(ratings, json_file, indent=4)