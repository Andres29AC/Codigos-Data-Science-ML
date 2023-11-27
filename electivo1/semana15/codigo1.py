import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
iris = sns.load_dataset("iris")
iris.head()
iris_v =iris[iris['species']!='setosa']
sns.pairplot(iris_v, hue='species', height=1.5)
plt.show()
X = iris_v.drop('species', axis=1)
Y = iris_v['species']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=30)
#0.30 en test_size es el 30% de los valores hacia la validacion
#0.30 en random_state con una aleatoriedad de 30%

# Crear el modelo de regresión logística
logModel = LogisticRegression()
logModel.fit(X_train, y_train)
predictions = logModel.predict(X_test)
print(predictions)

print(classification_report(y_test, predictions))

confusion_matrix(y_test, predictions)

#Ahora implementaremos la curva ROC
y_pred_prob = logModel.predict_proba(X_test)[:,1]

#Se debe escoger una de las 2 columnas virginica o versicolor
#porque los valores fluctuaran entre 0 y 1
#donde 0 se asigna a una columna y 1 hacia otra columna
roc_curve(y_test, y_pred_prob, pos_label='virginica')

fpr,tpr,threshold = roc_curve(y_test, y_pred_prob, pos_label='virginica')
plt.plot(fpr,tpr)
plt.show()
print(fpr)
print(tpr)
print(threshold)

#Dibujamos la curva ROC
plt.plot(fpr,tpr,color='red',label='Curva ROC')
plt.plot([0,1],[0,1],color='blue',linestyle='--',label='Curva ROC nula')
plt.xlabel('Tasa de Falsos Positivos-FPR')
plt.ylabel('Tasa de Verdaderos Positivos-TPR')
plt.title('Curva ROC')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=100)
