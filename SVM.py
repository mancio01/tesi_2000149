import os
import json
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import cv2

import imSimilarity as ims
import preprocessing as pp
'''
with open(os.path.dirname(os.path.abspath(__file__)) + r"\ratings.json", "r") as file:
    data = json.load(file)

Y_train = list(data["train"].values())
Y_test = list(data["test"].values())

X_train = []
X_test = []
sonda=0
for i in data["train"]:
    for j in os.listdir(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"+i):
        if j.endswith("first.png"):
            img_x = pp.preprocess(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"+ i + "\\" + j) 
        elif j.endswith("second.png"):
            img_y = pp.preprocess(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\" + i + "\\" + j)
    print(sonda)
    X_train.append(list(ims.imSimilarity(img_x,img_y).values()))
    sonda = sonda+1
    
for i in data["test"]:
    for j in os.listdir(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\test\\"+i):
        if j.endswith("first.png"):
            img_x = pp.preprocess(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\test\\"+ i + "\\" + j)   
        elif j.endswith("second.png"):
            img_y = pp.preprocess(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\test\\" + i + "\\" + j)
    print(sonda)
    X_test.append(list(ims.imSimilarity(img_x,img_y).values()))
    sonda=sonda+1
    
X = X_train + X_test
Y = Y_train + Y_test
data = { "X_train" : X_train, "Y_train" : Y_train, "X_test": X_test, "Y_test":Y_test}
with open(os.path.dirname(os.path.abspath(__file__)) + r"\dataset3d.json", 'w') as json_file:
    json.dump(data, json_file, indent=4)
'''
with open(os.path.dirname(os.path.abspath(__file__)) + r"\dataset.json", "r") as file:
    dataset = json.load(file)
    
X_train, X_test, Y_train, Y_test= dataset["X_train"], dataset["X_test"], dataset["Y_train"], dataset["Y_test"]
X = X_train + X_test
Y = Y_train + Y_test
size = 0.5
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=10)  

param = {
    'degree': [2, 3, 4, 5],
    'gamma': [0.1, 1, 10, 5],
    'coef0': [0.0, 1.0, 2.0, 3.0, 4.0],
}

clf = svm.SVC(C=0.1, kernel='poly', probability = True, verbose = True)
grid_search = GridSearchCV(clf, param, cv=5)  
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
print(best_params)
best_model = grid_search.best_estimator_
predict = best_model.predict(X_test)
sbagliati = sum(x != y for x, y in zip(Y_test, predict))
falsi_true = sum(x==False and y==True for x, y in zip(Y_test,predict))

print("Sbagliati",sbagliati)
print("Falsi positivi",falsi_true)

accuracy = accuracy_score(Y_test, predict)
print("acc:",accuracy)
print("test:",size)

cv_scores = cross_val_score(clf, X, Y, cv=5)
print(cv_scores)

'''
4d
{'C': 0.1, 'coef0': 2.0, 'degree': 5, 'gamma': 1, 'probability': True, 'shrinking': True, 'verbose': True}
Sbagliati 2
Falsi positivi 1
acc: 0.92
test: 0.25

{'C': 0.1, 'coef0': 0.0, 'degree': 3, 'gamma': 10, 'probability': True, 'shrinking': True, 'verbose': True}
Sbagliati 3
Falsi positivi 2
acc: 0.94
test: 0.5

{'C': 0.1, 'coef0': 0.0, 'degree': 5, 'gamma': 10, 'probability': True, 'shrinking': True, 'verbose': True}
Sbagliati 16
Falsi positivi 2
acc: 0.7866666666666666
test: 0.75
---------------------------------------------------
3d
{'C': 0.1, 'coef0': 1.0, 'degree': 3, 'gamma': 10, 'probability': True, 'shrinking': True, 'verbose': True}
Sbagliati 4
Falsi positivi 1
acc: 0.84
test: 0.25

{'C': 0.1, 'coef0': 0.0, 'degree': 5, 'gamma': 5, 'probability': True, 'shrinking': True, 'verbose': True}
Sbagliati 4
Falsi positivi 2
acc: 0.92
test: 0.5

{'C': 0.1, 'coef0': 0.0, 'degree': 4, 'gamma': 10, 'probability': True, 'shrinking': True, 'verbose': True}
Sbagliati 14
Falsi positivi 2
acc: 0.8133333333333334
test: 0.75
'''
#{'C': 0.1, 'coef0': ?, 'degree': ?, 'gamma': ?, 'probability': True, 'shrinking': True, 'verbose': True}
