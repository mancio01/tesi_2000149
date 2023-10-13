import pickle
import os
import json
from sklearn import svm
from sklearn.cluster import SpectralClustering



with open(os.path.dirname(os.path.abspath(__file__)) + r"\measures.json", "r") as file:
    data = json.load(file)

X_train=[]
Y_train=[]
for i in data["true"]:
    X_train.append([i["lfv"],i["uiq"],i["lfa"],i["rmse"]])
    Y_train.append(True) 
 
for i in data["false"]:
    X_train.append([i["lfv"],i["uiq"],i["lfa"],i["rmse"]])
    Y_train.append(False)
    
    
 
clf = svm.SVC(coef0=5.0, degree=2, kernel='poly', probability=1)
clf.fit(X_train,Y_train)

with open('svm.pkl', 'wb') as file:
    pickle.dump(clf, file)

clf = svm.SVC(coef0=0.0, degree=1, kernel='rbf', probability=1)
clf.fit(X_train,Y_train)

with open('svm_o.pkl', 'wb') as file:
    pickle.dump(clf, file)
    
clf=SpectralClustering(n_clusters=2, random_state=77,gamma=5) #con questo randomstate i false sono a 0 
clf.fit(X_train)

with open('clustering.pkl', 'wb') as file:
    pickle.dump(clf, file)
    