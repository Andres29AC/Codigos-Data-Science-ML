#NOTE:Implementaciones:
#1->Cargar BD spyder o Jupyter u otro IDE ->1 punto
#2->Analizar los datos cargados :
#   -Establecer un objetivo de prediccion(el estudiante establece el criterio)
#   -realizar comparativas de CORRELACION con criterios que desea evaluar planificando
#    que realizara al menos 2 modelos de prediccion,(mapas de calor) ->3 puntos
#3->Agregar o reducir o eliminar columnas a los grupos de datos que no correspondan
#  al objetivo de prediccion a las bases de datos.Mencionar lo que hace ->2 puntos
#4->Establecer el grupo de datos:Train y Test a los resultados aplicados al item 3 ->2 puntos
#5->Aplicar regression lineal/Regresion logistica/KNN Vecinos/Arboles de decision
#   /Arboles de decision aleatorios a cada grupo de datos -> 4 puntos si aplica correctamente
#6->Realizar y comparar las predicciones obtenidas por los metodos aplicados a los grupos de datos
#   depurada para determinar cual metodo es el que menos resultados a favor del aprendizaje y viceversa
#   ;aplicado a los grupos de datos DEPURADA.Concluir con graficos,matriz de confusion y 3 tipos de error
#   ->5 puntos ->COMPRACIONES DE PREDICCIONES CON SUS TEST Y CONCLUSIONES
#7->Explicacion en video,subido en aula virtual de forma oportuna con su codigo fuente obtenida y explicada
#   paso a paso ->3 puntos
#   -NOTE:Video Activo
#   -NOTE:Explicado y Disponible con archivo de codigo fuente en el aula virtual
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

prod = pd.read_csv('1ero.csv',encoding='latin-1') #latin-1 ->sirve para caracteres especiales
# prod.head()
# print(prod.head())
# columnas = prod.head().columns
# print(columnas)
mitad_superior = prod.iloc[:len(prod)//2,:]
mitad_inferior = prod.iloc[len(prod)//2:,:]
#Guardar las dos mitades en archivos separados en formato csv
mitad_superior.to_csv('lectora.csv', index=False)
mitad_inferior.to_csv('matematica.csv', index=False)
print(mitad_superior)
print(mitad_inferior)
