import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cl
from sklearn.preprocessing import StandardScaler

with open(os.path.dirname(os.path.abspath(__file__)) + r"\measures.json", "r") as file:
    data = json.load(file)

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

models=[]
models.append(cl.KMeans(n_clusters=2))
models.append(cl.MiniBatchKMeans(n_clusters=2, init='random'))
models.append(cl.AgglomerativeClustering(n_clusters=2, affinity='euclidean'))
models.append(cl.SpectralClustering(n_clusters=2, random_state=77,gamma=5))
models.append(cl.Birch(n_clusters=2,threshold=0.05, branching_factor=25))


ins=[]

for model in models:
    cluster_labels = model.fit_predict(X)
    trues = cluster_labels[:t]
    falses = cluster_labels[t:]
    if np.count_nonzero(falses==0) > np.count_nonzero(falses==1):
       score =  np.count_nonzero(trues==1) + np.count_nonzero(falses==0)
    else:
        score =  np.count_nonzero(trues==0) + np.count_nonzero(falses==1)
    print(model)
    print(trues)
    print(falses)
    print("score:", score, r'/', len(X),"  ", score*100/len(X),"%")
        
    
'''
SKMeans(n_clusters=2)
[0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0
 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0 1 1 0 0 0
 0 0 0 0 0 0 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 0 1
 0 1 1 1 1 0 1 1 1 0 1 0 0 0 0 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1
 0 1 0 0 1 0 1 0 1 1 1 1 1 1 1 0 0 0]
[1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
score: 121 / 202    59.9009900990099 %
------------------------------------------------------------------------------------
MiniBatchKMeans(n_clusters=2)
[1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1
 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0 0 0 1 0 0 1 1 1
 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 0 0 0 1 0 1 0 1 1 0 0 0 0 1 0 0 0 0 0 0 1 0
 1 0 0 0 0 1 0 0 0 1 0 1 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0
 1 0 1 1 0 1 1 1 0 0 0 0 0 0 1 1 1 1]
[0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
score: 127 / 202    62.87128712871287 %
-------------------------------------------------------------------------------------
AgglomerativeClustering(affinity='euclidean')
[0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0
 0 1 1 0 0 0 0 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0 1 0 0 0 0
 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 1 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 1 1 1 1 0 0
 0 1 1 1 1 0 0 1 1 0 0 0 0 0 0 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 0 0 1 0 1 0 1
 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0]
[1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1]
score: 129 / 202    63.86138613861386 %
-------------------------------------------------------------------------------------
SpectralClustering(gamma=5, n_clusters=2, random_state=77)
[1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1
 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1
 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 1]
[0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0]
score: 174 / 202    86.13861386138613 %
-------------------------------------------------------------------------------------
Birch(branching_factor=25, copy=False, n_clusters=2, threshold=0.0001)
[0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0
 0 1 1 0 0 0 0 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0 1 0 0 0 0
 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 1 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 1 1 1 1 0 0
 0 1 1 1 1 0 0 1 1 0 0 0 0 0 0 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 0 0 1 0 1 0 1
 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0]
[1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1]
score: 129 / 202    63.86138613861386 %
'''