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

prod = pd.read_csv('1ero.csv',encoding='latin-1',delimiter=";") #latin-1 ->sirve para caracteres especiales
prod.head()
print(prod.head())
columnas = prod.head().columns
print(columnas)
prod.info()

"""
 N;RED;INSTITUCION_EDUCATIVA;GRADO;SECCION;TURNO;NRO_DE_ESTUDIANTES_MATRICULADOS;NRO_DE_ESTUDIANTES_EVALUADOS
;P 1_C1;P 2_C1;P 3_C2;P 4_C1;P 5_C2;P 6_C2;P 7_C1;P 8_C3;P 9_C3;P 10_C2;P 11_C2;P 12_C2;P 13_C3;P 14_C2;
P 15_C3;P 16;P 17;P 18;P 19;P 20;CAPACIDADES_C1;CAPACIDADES_C2;CAPACIDADES_C3;%_POR_CAPACIDAD_C1;
%_POR_CAPACIDAD_C2;%_POR_CAPACIDAD_C3;PROMEDIO_X_SECCION
"""
#N->Numeracion de la institucion educativas en el registro
#RED->Red a la que pertenece la institucion educativa
#INSTITUCION_EDUCATIVA->Nombre de la institucion educativao
#GRADO->Grado de la institucion educativa
#SECCION->Seccion de la institucion educativa y tambien el numero de secciones evaluadas del colegio
#TURNO->Turno de la institucion educativa
#NRO_DE_ESTUDIANTES_MATRICULADOS->Total de alumnos matriculados en la institucion educativa
#NRO_DE_ESTUDIANTES_EVALUADOS->Total de alumnos evaluados en la institucion educativa
#P 1_C1->Puntaje en la Competencia 1(C1) de la Pregunta 1(P 1)
#.....
#P 15_C3->Puntaje en la Competencia 3(C3) de la Pregunta 15(P 15)
#P 16->No hay puntaje sobre una competencia para esta pregunta
#......
#P 20->No hay puntaje sobre una competencia para esta pregunta
#CAPACIDADES_C1->Puntaje sobre la capacidad de la competencia 1(C1)
#CAPACIDADES_C2->Puntaje sobre la capacidad de la competencia 2(C2)
#CAPACIDADES_C3->Puntaje sobre la capacidad de la competencia 3(C3)
#%_POR_CAPACIDAD_C1->Porcentaje sobre la capacidad de la competencia 1(C1)
#%_POR_CAPACIDAD_C2->Porcentaje sobre la capacidad de la competencia 2(C2)
#%_POR_CAPACIDAD_C3->Porcentaje sobre la capacidad de la competencia 3(C3)
#PROMEDIO_X_SECCION->Promedio de la seccion de la institucion educativa
#Informacion Adicional:Comprension de textos
#C1 = Competencia 1 = Lee diversos tipos de textos en su lengua materna
#C2 = Competencia 2 = Comprende textos escritos en su lengua materna
#C4 = Competencia 3 = Interpreta y valora textos orales en su lengua materna

#2->Analizar los datos cargados :
#   -Establecer un objetivo de prediccion(el estudiante establece el criterio)
#   -realizar comparativas de CORRELACION con criterios que desea evaluar planificando
#    que realizara al menos 2 modelos de prediccion,(mapas de calor) ->3 puntos

#Objetivo de Prediccion:
#Objetivo de Prediccion General para Todas las Instituciones Educativas:
#Rendimiento Promedio por Competencia y Capacidad
#Descripcion:
#Predecir el rendimiento promedio de todas las instituciones educativas en base al promedio de
#puntajes de todas las competencias y capacidades. Esto permitira una evaluacion global del
#desempeño academico en todas las instituciones.
#Columnas Relevantes:
#->P 1_C1 a P 20
#Capacidades_C1,Capacidades_C2,Capacidades_C3
#Promedio_x_Seccion(columna objetivo)
#NRO_DE_ESTUDIANTES_MATRICULADOS
#NRO_DE_ESTUDIANTES_EVALUADOS

#Borrando algunas columnas de P 16 a P 20
eliminar_columnas = ['P16','P17','P18','P19','P20']

df_formateado=prod.drop(columns=eliminar_columnas,axis=1)
print(df_formateado.head())

columnas_objetivo = ['PROMEDIO_X_SECCION', 'NRO_DE_ESTUDIANTES_MATRICULADOS', 'NRO_DE_ESTUDIANTES_EVALUADOS',
                     'P1_C1', 'P2_C1', 'P3_C2', 'P4_C1', 'P5_C2', 'P6_C2', 'P7_C1',
                     'P8_C3', 'P9_C3', 'P10_C2', 'P11_C2', 'P12_C2', 'P13_C3', 'P14_C2',
                     'P15_C3', 'CAPACIDADES_C1', 'CAPACIDADES_C2', 'CAPACIDADES_C3']
df_objetivo =prod[columnas_objetivo]
print(df_objetivo.head())

#Boxplot 1
plt.figure(figsize=(12,6))
sns.boxplot(x='NRO_DE_ESTUDIANTES_MATRICULADOS', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro estudiantes matriculados vs. Promedio x Seccion')
plt.show()
#Boxplot 2
plt.figure(figsize=(12,6))
sns.boxplot(x='NRO_DE_ESTUDIANTES_EVALUADOS', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro estudiantes evaluados vs. Promedio x Seccion')
plt.show()
#Boxplot 3
plt.figure(figsize=(12,6))
sns.boxplot(x='P1_C1', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro alumnos P1_C1 vs. Promedio x Seccion')
plt.text
plt.show()



matriz_corr = df_objetivo.corr()
plt.figure(figsize=(12,6))
sns.heatmap(matriz_corr,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title('Mapa de calor - Matriz de correlacion')
plt.show()


























