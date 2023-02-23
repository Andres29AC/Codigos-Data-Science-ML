import pandas as pd

fichero = 'precios_carros.csv'
datos = pd.read_csv(fichero)
#Uso de variables dummies 
print(datos.columns)
print(datos['Fuel_Type'])
#transformar variables categoricas a dummies 
columnas_dummies = pd.get_dummies(datos['Fuel_Type'])
print(columnas_dummies)

datos_dummies = pd.get_dummies(datos, columns=['Fuel_Type'])
print(datos.head)
print("*"*25)
print("*"*25)
print(datos_dummies.head())
