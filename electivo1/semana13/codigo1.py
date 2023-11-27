#Objetivos:
#Analisis de datos descriptivo(desviacion estandar)
#Plantear la division de grupos de datos
#Exportacion de las partes
#Crear los modelos y la red neuronal
#Obtener la exactitud de los modelos
#Modelos de Regresion en machine learming:
#->Regresion lineal simple
#->Regresion polinomica
#->Regresion de vectores de soporte
#->Regresion de arbol de decisiones
#->Regresion forestal aleatoria
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
#cargando los datos
prod = pd.read_csv('Water_pond_tanks_2021.csv',encoding='latin-1')
#conociendo la informacion
prod.head()
print(prod.head())
columnas = prod.head().columns
print(columnas)

descripcion= prod.describe()
print(descripcion)

#Ecplorando
# sns.histplot(prod['Dissolved Oxygen (mg/L) (Min)'], bins=20, kde=True)
# plt.title('Distribucion de oxigeno disuelto (minimo)')
# plt.show()
sns.histplot(prod['Type Water Body'], bins=20, kde=True)
plt.title('Tipo de cuerpo de agua')
plt.show()
sns.histplot(prod['pH (Max)'], bins=20, kde=True)
plt.title('ph Maximo')
plt.show()
sns.histplot(prod['Conductivity (?mhos/cm) (Max)'], bins=20, kde=True)
plt.title('Conductividad Maxima')
plt.show()
sns.histplot(prod['Total Coliform (MPN/100ml) (Max)'], bins=20, kde=True)
plt.title('Total coliformes')
plt.show()

#Escogiendo las variables
#Variable objetivo es:
#-->Dissolved Oxygen (mg/L) (Max)
#Columnas que ayudaran:
#Temperature\n?C (Max)
#Temperature\n?C (Min)
#Conductivity (?mhos/cm) (Min)
#Conductivity (?mhos/cm) (Max)
#BOD (mg/L) (Min)
#BOD (mg/L) (Max)
data = {'c1': prod['Temperature\n?C (Max)'], 'c2': prod['Temperature\n?C (Min)']
        ,'c3': prod['Conductivity (?mhos/cm) (Min)']
        ,'c4': prod['Conductivity (?mhos/cm) (Max)']
        ,'c5': prod['BOD (mg/L) (Min)']
        ,'c6': prod['BOD (mg/L) (Max)']
        , 'Oxigeno disuelto MAX': prod['Dissolved Oxygen (mg/L) (Max)']}
df = pd.DataFrame(data)
# df.fillna(df.mean(), inplace=True)
# df[['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'Oxigeno disuelto MAX']] = df[['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'Oxigeno disuelto MAX']].apply(pd.to_numeric, errors='coerce')
# Manejo de NaN
df.replace('-', np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
df[['c3', 'c4', 'c5', 'c6','Oxigeno disuelto MAX']] = imputer.fit_transform(
    df[['c3', 'c4', 'c5', 'c6','Oxigeno disuelto MAX']])

# Verificacion de NaN residual
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

#Mostrando todo el dataframe
print(df)
df.info()
prod.info()
#utilizando plot para que las caracteristicas ayuden al obetivo
df.plot(kind='scatter',x='c1', y='Oxigeno disuelto MAX')
plt.show()
df.plot(kind='scatter',x='c2', y='Oxigeno disuelto MAX')
plt.show()
df.plot(kind='scatter',x='c3', y='Oxigeno disuelto MAX')
plt.show()
df.plot(kind='scatter',x='c4', y='Oxigeno disuelto MAX')
plt.show()
df.plot(kind='scatter',x='c5', y='Oxigeno disuelto MAX')
plt.show()
df.plot(kind='scatter',x='c6', y='Oxigeno disuelto MAX')
plt.show()
#Grafico de correlacion:
sns.heatmap(df.corr(), annot=True)
plt.savefig('correlation_plot.png')
plt.close()
#Desviacion estandar
desvi_std = df.std()
print(desvi_std)
#Division de grupos de datos en entrenamiento y prueba
X=df.drop('Oxigeno disuelto MAX',axis=1)
y=df['Oxigeno disuelto MAX']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Exportar las partes:
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

#Crear los modelos y la red neuronal
# model =LinearRegression()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)
model_tf = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=100, activation='relu',input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=100, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_tf.compile(optimizer='adam', loss='mean_squared_error')
history=model_tf.fit(X_train, y_train, epochs=30, batch_size=10 ,validation_split=0.2)

#Obtener la exactitud de los modelos
y_pred = model.predict(X_test)
y_pred_tf = model_tf.predict(X_test)
print('Modelo de regresion lineal')
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
print('Modelo de red neuronal')
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred_tf))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred_tf))

#Graficar la perdida de entrenamiento
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Progreso de entrenamiento del modelo')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend()
plt.savefig('Progreso.png')
# plt.close()
plt.show()
#Realizar predicciones
y_pred = model.predict(X_test)
#Calcular metricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R^2): {r2}")
#Visualizar las predicciones vs. valores verdaderos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test,"^", color='red')
plt.xlabel('Valor verdadero')
plt.ylabel('Prediccion')
plt.title('Prediccion vs. Valor verdadero')
plt.savefig('Prediccion.png')
plt.show()
