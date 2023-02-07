import pandas as pd
import numpy as np

fichero = 'precios_carros.csv'
datos = pd.read_csv(fichero)
#Mostranddo la cabecera 
print(datos.head())

print(datos.dtypes)
#Cambiando un tipo de dato
datos['Unnamed: 0'] = datos['Unnamed: 0'].astype('float64')
print(datos.dtypes)
#Realizar calculos y ponerlos en una nueva columna

# millas = kilometros * 0.621371
datos['millas_driven'] = datos['Kilometers_Driven'] * 0.621371
#Mostrando el dataset
print(datos.head())

#Renombrar la columna
datos.rename(columns={'millas_driven':'Millas'}, inplace=True)
print(datos.head())


#Ahora veremos como normlizr los datos
#Normalizar los datos
#Esto es necesario para algunos algoritmos de machine learning

#Transformaacion min y valor anterior
#Mostrando 2 columnas
print(datos[['Millas', 'Kilometers_Driven']])
datos['Millas_normalizadas'] = datos['Millas']/datos['Millas'].max()
datos['Kilometros_normalizados'] = datos['Kilometers_Driven']/datos['Kilometers_Driven'].max()
print(datos[['Millas_normalizadas','Kilometros_normalizados']])
#Transformacion min y max
print(datos[['Millas', 'Kilometers_Driven']])
datos['Millas_normalizadas'] = (datos['Millas']-datos['Millas'].min())/(datos['Millas'].max()-datos['Millas'].min())
datos['Kilometros_normalizados'] = (datos['Kilometers_Driven']-datos['Kilometers_Driven'].min())/(datos['Kilometers_Driven'].max()-datos['Kilometers_Driven'].min())
print(datos[['Millas_normalizadas','Kilometros_normalizados']])

















