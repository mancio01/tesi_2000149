import imSimilarity as ims
import os
import json
import cv2
from matplotlib import pyplot as plt
import sklearn.cluster as cl
from sklearn.preprocessing import StandardScaler
import numpy as np

'''
with open(os.path.dirname(os.path.abspath(__file__)) + r"\label.json", "r") as file:
    data = json.load(file)
    
sonda=0
for k, v in data.items():
    print(sonda)
    for j in os.listdir(k):
        if j.endswith("first.png"):
            img_x = k + "\\" + j 
        elif j.endswith("second.png"):
            img_y = k + "\\" + j 
    if(not v and ims.areSimilar(img_x,img_y,algorithm='clustering')):
        print("falso positivo")
        plt.imshow(cv2.imread(k + r"\\" + r"\DisplayImage.png"))
        plt.show()
    if(v and not ims.areSimilar(img_x,img_y,algorithm='clustering')):
        print("falso negativo")
        plt.imshow(cv2.imread(k + r"\\" + r"\DisplayImage.png"))
        plt.show()
    sonda=sonda+1
'''
with open(os.path.dirname(os.path.abspath(__file__)) + r"\measures.json", "r") as file:
    data = json.load(file)

with open(os.path.dirname(os.path.abspath(__file__)) + r"\label.json", "r") as file:
    names = json.load(file)
    
nt=[]
nf=[]
for k,v in names.items():
    if v:
        nt.append(k)
    else:
        nf.append(k)
n=nt+nf

X=[]
Y=[]
for i in data["true"]:
    X.append([i["lfv"],i["uiq"],i["lfa"],i["rmse"]])
    Y.append(True) 
 
for i in data["false"]:
    X.append([i["lfv"],i["uiq"],i["lfa"],i["rmse"]])
    Y.append(False)
        
scaler=StandardScaler()
X=scaler.fit_transform(X)

t = np.count_nonzero(Y)
model = cl.SpectralClustering(n_clusters=2, random_state=77,gamma=5)



ins=[]


cluster_labels = model.fit_predict(X)
trues = cluster_labels[:t]
falses = cluster_labels[t:]

for i in range(len(cluster_labels)):
    if not cluster_labels[i] and i<t:
        plt.imshow(cv2.imread(n[i]+r"\\"+"DisplayImage.png"))
        plt.show()

print(model)
print(trues)
print(falses)
 
