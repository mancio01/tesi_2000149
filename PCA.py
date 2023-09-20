import os
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

with open(os.path.dirname(os.path.abspath(__file__)) + r"\index.json", "r") as file:
    index = json.load(file)
true = index["true"]
false = index["false"]

measures = []
dimension_names = ["rmse", "psnr", "ssim", "fsim", "sre", "sam", "uiq", "lfa","lfv"]
for item in true + false:
    measures.append([item[dim] for dim in dimension_names])
measures = np.array(measures)
scaler=StandardScaler()
measures=scaler.fit_transform(measures)

pca = PCA()
pca.fit_transform(measures)
variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(variance)

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.xlabel('Numero di Componenti Principali')
plt.ylabel('Varianza Cumulativa')
plt.title('Varianza')
plt.show()

#conto quante dimensioni mi servono per avere una varianza almeno del 95%

main_dimensions = sum(1 for i in cumulative_variance if i < 0.95)
components = pca.components_[:main_dimensions]
components_values = (components).sum(axis=0)
print(components_values)
print(dimension_names)
ind = sorted(abs(components_values))
main_dimension_names = [dimension_names[i] for i in ind]
print("Dimensioni principali:", main_dimension_names)