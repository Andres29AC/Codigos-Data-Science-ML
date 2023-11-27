# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Cargar datos
# prod = pd.read_csv('Water_pond_tanks_2021.csv', encoding='latin-1')

# # Limpiar nombres de columnas
# prod.columns = prod.columns.str.strip()

# # Mostrar las primeras filas y las columnas
# prod.head()
# print(prod.head())
# columnas = prod.head().columns
# print(columnas)

# # Crear el grafico de correlacion
# plt.figure(figsize=(10, 6))
# sns.heatmap(prod.corr().abs(), annot=True)
# corr = prod.corr().abs()

# # Verificar si la columna esta presente en corr
# print(corr.columns)

# # Imprimir las primeras filas de corr
# print(corr.head())

# # Obtener estadisticas descriptivas
# descripcion = prod.describe()
# print(descripcion)

# sns.histplot(prod['Type Water Body'], bins=20, kde=True)
# plt.title('Tipo de cuerpo de agua')
# plt.show()

# # Intentar obtener la columna
# try:
#     corr_SP = corr.loc[:, ['Oxigeno disuelto MAX']]
# except KeyError as e:
#     print(f"Error: {e}")
#     print("La columna 'Oxigeno disuelto MAX' no se encuentra en las columnas de corr.")

# # Corregir el nombre de la columna para el boxplot
# sns.boxplot(x='Temperature\n?C (Max)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
# plt.show()


#Version 2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Cargar datos
# prod = pd.read_csv('Water_pond_tanks_2021.csv', encoding='latin-1')

# # Limpiar nombres de columnas
# prod.columns = prod.columns.str.strip()

# # Mostrar las primeras filas y las columnas
# prod.head()
# print(prod.head())
# columnas = prod.head().columns
# print(columnas)

# # Crear el grafico de correlacion
# plt.figure(figsize=(10, 6))
# sns.heatmap(prod.corr().abs(), annot=True)
# corr = prod.corr().abs()

# # Verificar si la columna esta presente en corr
# print(corr.columns)

# # Imprimir las primeras filas de corr
# print(corr.head())

# # Obtener estadisticas descriptivas
# descripcion = prod.describe()
# print(descripcion)

# sns.histplot(prod['Type Water Body'], bins=20, kde=True)
# plt.title('Tipo de cuerpo de agua')
# plt.show()

# # Imprimir las columnas de corr antes de intentar seleccionar la columna
# print("Columnas de corr:", corr.columns)

# # Intentar obtener la columna
# try:
#     corr_SP = corr.loc[:, ['Oxigeno disuelto MAX']]
# except KeyError as e:
#     print(f"Error: {e}")
#     print("La columna 'Oxigeno disuelto MAX' no se encuentra en las columnas de corr.")

# # Corregir el nombre de la columna para el boxplot
# sns.boxplot(x='Temperature\n?C (Max)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Cargar datos
# prod = pd.read_csv('Water_pond_tanks_2021.csv', encoding='latin-1')
# prod.head()
# # Limpiar nombres de columnas
# prod.columns = prod.columns.str.strip()

# plt.figure(figsize=(10,6))
# sns.heatmap(prod.corr(), annot=True)

# # Crear el DataFrame de correlacion
# columnas_elegidas = ['Temperature\n?C (Max)', 'Temperature\n?C (Min)',
#                      'Conductivity (?mhos/cm) (Min)', 'Conductivity (?mhos/cm) (Max)',
#                      'BOD (mg/L) (Min)', 'BOD (mg/L) (Max)', 'Dissolved Oxygen (mg/L) (Max)']
# corr = prod[columnas_elegidas].corr().abs()
# plt.figure(figsize=(10,6))
# sns.heatmap(corr.corr(), annot=True)
# plt.show()
# # Crear el boxplot
# sns.boxplot(x='Temperature\n?C (Max)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
# plt.show()

# sns.boxplot(x='Temperature\n?C (Min)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Cargar datos
prod = pd.read_csv('Water_pond_tanks_2021.csv', encoding='latin-1')
prod.head()
# Crear la figura
# plt.figure(figsize=(10, 6))
# sns.heatmap(prod.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlacion en el conjunto de datos')
# plt.show()
# Limpiar nombres de columnas
prod.columns = prod.columns.str.strip()

# Manejo de NaN para columnas especificas
columnas_nan = ['Conductivity (?mhos/cm) (Min)', 'Conductivity (?mhos/cm) (Max)',
                'BOD (mg/L) (Min)', 'BOD (mg/L) (Max)','Dissolved Oxygen (mg/L) (Max)']

prod.replace('-', np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
prod[columnas_nan] = imputer.fit_transform(prod[columnas_nan])



columnas_elegidas = ['Temperature\n?C (Max)', 'Temperature\n?C (Min)',
                     'Conductivity (?mhos/cm) (Min)', 'Conductivity (?mhos/cm) (Max)',
                     'BOD (mg/L) (Min)', 'BOD (mg/L) (Max)', 'Dissolved Oxygen (mg/L) (Max)']
corr = prod[columnas_elegidas].corr().abs()
plt.figure(figsize=(10, 6))
sns.heatmap(corr.corr(), annot=True, cmap='coolwarm')
plt.title('Correlacion entre variables seleccionadas')
plt.show()



# Boxplot 1
sns.boxplot(x='Temperature\n?C (Max)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
plt.title('Boxplot de Temperatura (Max) vs. Oxigeno Disuelto (Max)')
plt.show()

# Boxplot 2
sns.boxplot(x='Temperature\n?C (Min)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
plt.title('Boxplot de Temperatura (Min) vs. Oxigeno Disuelto (Max)')
plt.show()
# Boxplot 3
sns.boxplot(x='Conductivity (?mhos/cm) (Min)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
plt.title('Boxplot de Conductividad Min vs. Oxigeno Disuelto (Max)')
plt.show()
# Boxplot 4
sns.boxplot(x='Conductivity (?mhos/cm) (Max)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
plt.title('Boxplot de Conductividad Max vs. Oxigeno Disuelto (Max)')
plt.show()
# Boxplot 5
sns.boxplot(x='BOD (mg/L) (Min)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
plt.title('Boxplot de BOD Min vs. Oxigeno Disuelto (Max)')
plt.show()
# Boxplot 6
sns.boxplot(x='BOD (mg/L) (Max)', y='Dissolved Oxygen (mg/L) (Max)', data=prod)
plt.title('Boxplot de BOD Max vs. Oxigeno Disuelto (Max)')
plt.show()


seleccionadas = corr.loc[:,["Temperature\n?C (Max)","Temperature\n?C (Min)"]]
sns.pairplot(seleccionadas)
plt.show()

X = corr.loc[:,["Temperature\n?C (Max)","Temperature\n?C (Min)"]]  
Y = corr.loc[:,["Dissolved Oxygen (mg/L) (Max)"]]

X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size=0.30,random_state=33)

#La cantidad de elementos implicados para el tamaño del test es 30% de los elementos y aleatorizado
#Lo restante es entrenamiento

lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
lm.coef_
print(str(lm.coef_))

predicciones = lm.predict(X_test)
print(predicciones)


DataFramePredicciones = pd.DataFrame(predicciones)
DataFramePredicciones.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
df_unido = y_test.join(DataFramePredicciones)
print(df_unido)

#Metricas
print('MAE',metrics.mean_absolute_error(y_test,predicciones))
print('MSE',metrics.mean_squared_error(y_test,predicciones))
print('RMSE',np.sqrt(metrics.mean_absolute_error(y_test,predicciones)))
sns.displot(corr.loc[:,['Dissolved Oxygen (mg/L) (Max)']])













