import os
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

with open(os.path.dirname(os.path.abspath(__file__)) + r"\measures.json", "r") as file:
    index = json.load(file)
true = index["true"]
false = index["false"]

measures = []
names = ["rmse", "psnr", "ssim", "fsim", "sre", "sam", "uiq", "lfa", "lfv"]

for item in true + false:
    measures.append([item[i] for i in names])

measures = np.array(measures)
scaler=StandardScaler()
measures=scaler.fit_transform(measures)

pca = PCA()
pca.fit_transform(measures)
variance = pca.explained_variance_ratio_
PC = pca.components_
vPC=np.abs(PC * variance[:, np.newaxis]) #moltiplico le componenti per le rispettive varianze  
weights = np.sum(vPC, axis=0) #trovo la somma il valore assoluto di ogni colonna
s_weights = np.sort(weights)[::-1] 
rankings = [names[np.where(weights == weight)[0][0]] for weight in s_weights]
print(rankings)
print(s_weights)