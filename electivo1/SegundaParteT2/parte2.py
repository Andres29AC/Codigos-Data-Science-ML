import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
#NOTE: Objetivo de Prediccion
##Objetivo de prediccion:Analisis de Competencia de Vendedores
#Analiza la relacion entre la competencia de vendedores (numero de vendedores)
#y las metricas de ventas, ingresos y calificaciones de productos
#NOTE: Columnas relevantes
#numberOfSellers --> Numero de vendedores que ofrecen el producto.
#monthlyRevenueEstimate --> Estimacion de los ingresos mensuales generados por el producto.
#monthlyUnitsSold --> Numero de unidades vendidas mensualmente.
#reviewRating -->Calificacion promedio de las reseñas del producto.

#cargando los datos
prod = pd.read_csv('procesado4.csv')

#conociendo la informacion
prod.head()
print(prod.head())
columnas = prod.head().columns
print(columnas)

data = {'c1': prod['numberOfSellers'], 'c2': prod['monthlyRevenueEstimate'], 'c3': prod['monthlyUnitsSold'], 'Rating': prod['reviewRating']}
df = pd.DataFrame(data)
prod.info()

#utilizando plot para que las caracteristicas ayuden al obetivo 
df.plot(kind='scatter',x='c1', y='Rating')
plt.show()
df.plot(kind='scatter',x='c2', y='Rating')
plt.show()
df.plot(kind='scatter',x='c3', y='Rating')
plt.show()

data2 = {'c1': prod['reviewRating'], 'c2': prod['monthlyRevenueEstimate'], 'c3': prod['monthlyUnitsSold'], 'Numero de Vendedores': prod['numberOfSellers']}
df2 = pd.DataFrame(data2)
prod.info()

#utilizando plot para que las caracteristicas ayuden al obetivo 
df2.plot(kind='scatter',x='c1', y='Numero de Vendedores')
plt.show()
df2.plot(kind='scatter',x='c2', y='Numero de Vendedores')
plt.show()
df2.plot(kind='scatter',x='c3', y='Numero de Vendedores')
plt.show()

sns.heatmap(df2.corr(), annot=True)
# Guarda el grafico como un archivo de imagen
plt.savefig('correlation_plot.png')

# Cierra el grafico
plt.close()
#objetivo:predecir Analisis de Competencia de Vendedores
# Normalizacion de caracteristicas numericas
x_numerical = prod[['reviewRating', 'monthlyRevenueEstimate', 'monthlyUnitsSold']]
scaler = MinMaxScaler()
x_numerical = scaler.fit_transform(x_numerical)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x_numerical, prod['numberOfSellers'], test_size=0.2, random_state=42)

# Definir el modelo de regresion
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=100, activation='relu', input_shape=(x_numerical.shape[1],)),
    tf.keras.layers.Dropout(0.2),  # Agregar capa de Dropout
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1, activation='linear')
])

# Compilar el modelo con una tasa de aprendizaje personalizada
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=150, batch_size=50, validation_split=0.2)

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Perdida en el conjunto de prueba: {mse}")
print(f"Coeficiente de determinacion (R^2): {r2}")

# Hacer predicciones con 2 datos de ejemplo
sample_data = np.array([[0.0, 0.0, 0.0], [2667.0, 0.0, 890586.0]])
sample_data = scaler.transform(sample_data)
predictions = model.predict(sample_data)
print("Predicciones:")
for i, pred in enumerate(predictions):
    print(f"Ejemplo {i + 1}: {pred[0]}")

# Graficar la perdida en entrenamiento
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Progreso de entrenamiento del modelo')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend()

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular metricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R^2): {r2}")

# Visualizar las predicciones vs. valores verdaderos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_pred, "^", color="r")
plt.xlabel('Valores Verdaderos')
plt.ylabel('Predicciones del modelo')
plt.title('Predicciones vs. Valores Verdaderos')
# Guardar las graficas
plt.savefig('training_loss.png')

plt.show()

