#Importando pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fichero='precios_carros.csv'
datos=pd.read_csv(fichero)
print(datos.columns)

"""
Index(['Unnamed: 0', 'Name', 'Location', 'Year', 'Kilometers_Driven',
       'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power',
       'Seats', 'New_Price'],
      dtype='object')

"""
#pasando español el nombre de las columnas
titulos = ['Unnamed: 0', 'Nombre', 'Ubicacion', 'Año', 'Kilometros', 'Tipo de combustible', 'Transmision', 'Tipo de propietario', 'Consumo', 'Motor', 'Potencia', 'Asientos', 'Precio']
datos.columns = titulos
print(datos.head(5))
print(datos.columns)
print(datos['Kilometros'] )
#Usando numpy para crear intervalos
intervls = np.linspace(min(datos['Kilometros']), max(datos['Kilometros']), 4)
#Explicacion:linspace crea un array de 4 elementos, con valores entre el minimo y el maximo de la columna Kilometros
nombre_grupos = ['pocos','normal','muchos']
datos['kilometros agrupados'] = pd.cut(datos['Kilometros'],intervls,labels=nombre_grupos,include_lowest=True)
#Explicacion:cut crea un array de 4 elementos, con valores entre el minimo y el maximo de la columna Kilometros
#labels=nombre_grupos, incluye los nombres de los grupos en el array de salida (pocos,normal,muchos) 
#include_lowest=True, incluye el valor minimo en el primer grupo
#include_lowest sirve para que el primer grupo incluya el valor minimo
print(datos['kilometros agrupados'])

#Graficando histograma
plt.hist(datos['Kilometros'],bins=intervls,rwidth=0.9,color='green')
plt.title('Histograma kilometros recorridos')
plt.xlabel('Kilometros')
plt.ylabel('Frecuencia')
plt.show()











