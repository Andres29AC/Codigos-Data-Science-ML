import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
iris = sns.load_dataset("iris")
iris.head()

X = iris.drop('species', axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#Predicciones
pred = knn.predict(X_test)
print(pred)

report = classification_report(y_test, pred)
tabla = confusion_matrix(y_test, pred)
print(report)
print(tabla)

#->Real->Negativo->Negativo(N)->a:(TN)
#->Real->Negativo->Positivo(P)->b:(FP)
#->Real->Positivo->Negativo(N)->c:(FN)
#->Real->Positivo->Positivo(P)->d:(TP)
#Precisión ('precision')
#Porcentaje predicciones positivas correctas
#d/b+d
#Sensibilidad exhaustividad ('recall')
#Porcentaje de casos positivos detectados
#d/(d+c)
#Especifidad ('specificity')
#Porcentaje de casos negativos detectados
#a/(a+b)
#Exactitud ('accuracy')
#Porcentaje de predicciones correctas
#No sirve en datasets poco equilibrados
#(a+d)/(a+b+c+d)
#Calculamos la exactitud
print((tabla[0][0]+tabla[1][1]+tabla[2][2])/len(y_test))

print(13+10+18)
print(13+10+18+3+1)
print(41/45)

#Calculamos la puntuación F1
knn.score(X_test, y_test)
#Calculamos la puntuación a nivel de entrenamiento
knn.score(X_train, y_train)

#Ahora establecemos un numero de vecinos
vecinos = np.arange(1, 25)
print(vecinos)

#Crear 2 matrices vacias en base a la cantidad de vecinos
#los valores seran muy pequeños que se asume que son nulos

train_2 = np.empty(len(vecinos))
test_2 = np.empty(len(vecinos))
print(train_2)
print(test_2)

#Generamos un bucle para registrar los datos en las matrices
for i,k in enumerate(vecinos):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_2[i] = knn.score(X_train, y_train)
    test_2[i] = knn.score(X_test, y_test)
print(train_2)
print(test_2)

#Ahora crearemos un grafico de vecinos versus test como vecinos versus train_test
plt.title('Numero de vecinos proximos KNN')
plt.plot(vecinos, test_2, label='Exactitud del Test')
plt.plot(vecinos, train_2, label='Exactitud del Train')
plt.legend()
plt.xlabel('Numero de vecinos')
plt.ylabel('Exactitud')
plt.show()
