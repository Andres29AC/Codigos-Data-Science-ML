import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RFC
#Arboles de decision y bosques aleatorios
iris = sns.load_dataset("iris")
X = iris.drop('species', axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

arbol = DTC()
arbol.fit(X_train, y_train)

tree.plot_tree(arbol)
plt.show()

X_nombre = list(X.columns)
classes = ['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(arbol,feature_names = X_nombre, class_names=classes,filled = True)
fig.savefig('arbol.png')
#Creamos las predicciones
pred = arbol.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

rfc = RFC(n_estimators=20, random_state=33)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
