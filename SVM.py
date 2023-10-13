import os
import json
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle



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

X,Y = shuffle(X,Y, random_state=77)

param = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1,2,3,4,5],
    'gamma': ['scale','auto'],
    'coef0': [0.0, 1.0, 5.0, 10.0, 15.0],
    'shrinking': [True,False],
    'probability': [True,False],
    'class_weight': [None,'balanced'],
    'verbose': [True,False],
}

clf = svm.SVC()
grid_search = GridSearchCV(clf, param, cv=10, scoring = 'accuracy', refit='accuracy')  
grid_search.fit(X, Y)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

#clf = svm.SVC(C=0.001, class_weight= 'balanced', degree=4, gamma='scale', kernel = 'poly',probability=True,verbose=True,coef0=1.0)



scorings = ['precision','accuracy','roc_auc','f1','recall']
results = []
for scoring in scorings:
    score = cross_val_score(best_model, X, Y, scoring=scoring, cv=10)
    results.append({scoring:score.mean()})


print()
#print(best_params)
print(results)


'''
parametri per ottimizzare roc_auc
'C': 0.001
'class_weight':'balanced'
'coef0':1.0
'degree':4
'gamma':'scale'
'kernel':'poly'
'probability':True
'shrinking':True
'verbose':True

Risultati: precision:0.79375 accuracy:0.35595238095238096 roc_auc:0.9276960784313726 f1:0.3361865939358199 recall:0.21617647058823528
--------------------------------------------------------------------------------
parametri per ottimizzare precision
'C':0.01
'class_weight':'balanced'
'coef0':0.0
'degree':1
'gamma':'scale'
'kernel':'sigmoid'
'probability':True
'shrinking':True
'verbose':True

Risultati: precision:0.978409090909091 accuracy:0.5947619047619047 roc_auc:0.9113664215686275 f1:0.6529548542134748 recall:0.5297794117647059
------------------------------------------------------------------------------------------------
parametri per ottimizzare accuracy
'C':0.01
'class_weight':None
'coef0':1.0
'degree':5
'gamma':'scale'
'kernel':'poly'
'probability':True
'shrinking':True
'verbose':True

Risultati: precision:0.934468524251806 accuracy:0.9114285714285714 roc_auc:0.8787377450980391 f1:0.9476329300048466 recall:0.9632352941176471
------------------------------------------------------------------------------------------------
parametri per ottimizzare f1
'C':1
'class_weight':None
'coef0':0.0
'degree':1
'gamma':'scale'
'kernel':'rbf'
'probability':True
'shrinking':True
'verbose':True

Risultati: precision:0.9311321809425526 accuracy:0.910952380952381 roc_auc:0.8674325980392158 f1:0.9462481175265806 recall:0.9636029411764706
------------------------------------------------------------------------------------------------
parametri per ottimizzare recall
'C':0.001
'class_weight':None
'coef0':0.0
'degree':1
'gamma':'scale'
'kernel':'linear'
'probability':True
'shrinking':True
'verbose':True

Risultati: precision:0.8219047619047618 accuracy:0.8219047619047618 roc_auc: 0.9007965686274509 f1:0.9020704915441758 recall:1.0

'''
