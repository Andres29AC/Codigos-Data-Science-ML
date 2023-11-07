import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# cargar los datos
prod = pd.read_csv('appearances.csv')
#conociendo la informacion
prod.head()
print(prod.head())

columnas = prod.head().columns
print(columnas)
prod.describe()
print(prod.describe())

# index(['appearance_id', 'game_id', 'player_id', 'player_club_id',
#        'player_current_club_id', 'date', 'player_name', 'competition_id',
#        'yellow_cards', 'red_cards', 'goals', 'assists', 'minutes_played'],
#       dtype='object')


data = {'c1': prod['yellow_cards'], 'c2': prod['red_cards'], 'c3': prod['goals'], 'c4': prod['minutes_played'], 'asistencias': prod['assists']}
df = pd.DataFrame(data)
prod.info()

#todo: utilizando plot para las caracteristicas ayuden al obetivo 
#df.plot(kind='scatter',x='c1', y='asistencias')
#df.plot(kind='scatter',x='c2', y='asistencias')
#df.plot(kind='scatter',x='c3', y='asistencias')
#df.plot(kind='scatter',x='c4', y='asistencias')

#sns.pairplot(prod)

# Seleccionando características numéricas
x_numerical = prod[['yellow_cards', 'red_cards', 'goals', 'minutes_played']]
y = prod['assists']
# Preprocesamiento de la fecha
date = pd.to_datetime(prod['date'])
date_df = pd.DataFrame({'year': date.dt.year, 'month': date.dt.month, 'day': date.dt.day})
date_encoded = pd.get_dummies(date_df, columns=['year', 'month', 'day'], drop_first=True)

# Combinar características numéricas y fecha 
x_numerical = pd.concat([x_numerical, date_encoded], axis=1)

# Convertir características numéricas a float
x_numerical = x_numerical.astype(float)

# Normalización de características
scaler = MinMaxScaler()
x_numerical = scaler.fit_transform(x_numerical)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x_numerical, y, test_size=0.2, random_state=42)

# Definir el modelo de regresión
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=100, activation='relu', input_shape=(x_numerical.shape[1],)),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])
model.summary()


model.compile(optimizer='rmsprop', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=20, batch_size=50, validation_split=0.2)

# Graficar la pérdida en entrenamiento
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Progreso de entrenamiento del modelo')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend()

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular métricas
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
plt.show()

# Crear un DataFrame con la pérdida
loss_df = pd.DataFrame(history.history)

# Visualizar la pérdida a lo largo de las épocas
loss_df.plot()
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.show()
