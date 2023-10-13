import os
import json
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV




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

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=77)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=77)

param = {
    'C':[0.001, 0.01, 0.1, 1],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1,2,3,4,5],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 20,'scale'],
    'coef0': [0.0, 1.0, 5.0],
    'class_weight': [None,'balanced'],
}

clf = svm.SVC(probability=True)
grid_search = GridSearchCV(clf, param, cv=5, scoring = 'f1', verbose=1,n_jobs=-1)  
grid_search.fit(X_val, Y_val)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

new_clf = best_model.fit(X_train,Y_train)
pred = new_clf.predict(X_test)

accuracy = accuracy_score(Y_test, pred)
precision = precision_score(Y_test, pred)
recall = recall_score(Y_test, pred)
f1 = f1_score(Y_test, pred)

print()
print(best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



'''
ottimizzazione di accuracy:
{'C': 0.1, 'class_weight': 'balanced', 'coef0': 1.0, 'degree': 4, 'gamma': 'scale', 'kernel': 'poly'}
Accuracy: 0.85
Precision: 0.9629629629629629
Recall: 0.8387096774193549
F1 Score: 0.896551724137931

ottimizzazione di f1:
{'C': 1, 'class_weight': None, 'coef0': 5.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
Accuracy: 0.875
Precision: 0.9642857142857143
Recall: 0.8709677419354839
F1 Score: 0.9152542372881356

ottimizzazione di precision:
{'C': 0.001, 'class_weight': 'balanced', 'coef0': 0.0, 'degree': 2, 'gamma': 20, 'kernel': 'poly'}
Accuracy: 0.775
Precision: 0.9583333333333334
Recall: 0.7419354838709677
F1 Score: 0.8363636363636364

ottimizzazione di recall:
{'C': 0.001, 'class_weight': None, 'coef0': 0.0, 'degree': 1, 'gamma': 0.001, 'kernel': 'linear'}
Accuracy: 0.775
Precision: 0.775
Recall: 1.0
F1 Score: 0.8732394366197184

'''