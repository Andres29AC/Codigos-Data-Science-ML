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
#Aplicando regression lineal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
prod = pd.read_csv('1ero.csv',encoding='latin-1',delimiter=";") #latin-1 ->sirve para caracteres especiales
prod.head()
print(prod.head())
columnas = prod.head().columns
print(columnas)
prod.info()


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
df_objetivo =df_formateado[columnas_objetivo]
print(df_objetivo.head())

#Boxplot 1
plt.figure(figsize=(12,6))
sns.boxplot(x='NRO_DE_ESTUDIANTES_MATRICULADOS', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro estudiantes matriculados vs. Promedio x Seccion')
plt.text(35,27,'Tabla de Comprension de Textos',color='red')
plt.show()
#Boxplot 2
plt.figure(figsize=(12,6))
sns.boxplot(x='NRO_DE_ESTUDIANTES_EVALUADOS', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro estudiantes evaluados vs. Promedio x Seccion')
plt.text(35,27,'Tabla de Comprension de Textos',color='red')
plt.show()
#Boxplot 3
plt.figure(figsize=(12,6))
sns.boxplot(x='P1_C1', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro alumnos P1_C1 vs. Promedio x Seccion')
plt.text(1,27,'P1_C1:Nro de estudiantes que hicieron correctamente la competencia 1 de la pregunta 1')
plt.text(35,27,'Tabla de Comprension de Textos',color='red')
plt.show()
#Boxplot 4
plt.figure(figsize=(12,6))
sns.boxplot(x='P2_C1', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro alumnos P2_C1 vs. Promedio x Seccion')
plt.text(1,27,'P2_C1:Nro de estudiantes que hicieron correctamente la competencia 1 de la pregunta 2')
plt.text(35,27,'Tabla de Comprension de Textos',color='red')
plt.show()
#Boxplot 5
plt.figure(figsize=(12,6))
sns.boxplot(x='P3_C2', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro alumnos P3_C2 vs. Promedio x Seccion')
plt.text(1,27,'P2_C1:Nro de estudiantes que hicieron correctamente la competencia 2 de la pregunta 3')
plt.text(35,27,'Tabla de Comprension de Textos',color='red')
plt.show()

#Boxplot 6
plt.figure(figsize=(12,6))
sns.boxplot(x='P4_C1', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot Nro alumnos P4_C1 vs. Promedio x Seccion')
plt.text(1,27,'P2_C1:Nro de estudiantes que hicieron correctamente la competencia 1 de la pregunta 4')
plt.text(35,27,'Tabla de Comprension de Textos',color='red')
plt.show()
#Boxplot 7
plt.figure(figsize=(12,6))
sns.boxplot(x='CAPACIDADES_C1', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot CAPACIDADES_C1 vs. Promedio x Seccion')
plt.text(-1,27,'Nivel 1 o 3:Indica un nivel de desempeño basico o insuficiente en la capacidad evaluada',size=7)
plt.text(-1,26,'Nivel 4 o 5:Puede representar un nivel de desempeño intermedio',size=7)
plt.text(-1,25,'Nivel 6 o 10:Puede indicar un nivel excelente de desenpeño',size=7)
plt.show()
#Boxplot 8
plt.figure(figsize=(12,6))
sns.boxplot(x='CAPACIDADES_C2', y='PROMEDIO_X_SECCION', data=df_objetivo)
plt.title('Boxplot CAPACIDADES_C2 vs. Promedio x Seccion')
plt.text(-1,27,'Nivel 1 o 3:Indica un nivel de desempeño basico o insuficiente en la capacidad evaluada',size=7)
plt.text(-1,26,'Nivel 4 o 5:Puede representar un nivel de desempeño intermedio',size=7)
plt.text(-1,25,'Nivel 6 o 10:Puede indicar un nivel excelente de desenpeño',size=7)
plt.show()
matriz_corr = df_objetivo.corr()
plt.figure(figsize=(12,6))
sns.heatmap(matriz_corr,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title('Mapa de calor - Matriz de correlacion')
plt.show()


seleccionadas=matriz_corr.loc[:,["CAPACIDADES_C1","CAPACIDADES_C2","CAPACIDADES_C3"]]
sns.pairplot(seleccionadas)

plt.show()

#4->Establecer el grupo de datos:Train y Test a los resultados aplicados al item 3 ->2 puntos
X=matriz_corr.loc[:,["CAPACIDADES_C1","CAPACIDADES_C2","CAPACIDADES_C3"]]
Y=matriz_corr.loc[:,["PROMEDIO_X_SECCION"]]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

#5->Aplicar regression lineal/Regresion logistica/KNN Vecinos/Arboles de decision
#   /Arboles de decision aleatorios a cada grupo de datos -> 4 puntos si aplica correctamente
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
print(str(lm.coef_))

predicciones = lm.predict(X_test)
print(predicciones)

DatFrame_predicciones = pd.DataFrame(predicciones)
DatFrame_predicciones.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
df_unido = pd.concat([y_test, DatFrame_predicciones], axis=1)
df_unido.columns = ['y_test', 'predicciones']
print(df_unido)
#Metricas
print('MAE:', metrics.mean_absolute_error(y_test, predicciones))
print('MSE:', metrics.mean_squared_error(y_test, predicciones))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicciones)))
#Grafica de metricas
sns.distplot((y_test-predicciones),bins=50)
plt.show()
            

#2 objetivo de prediccion de tipo clasificacion
#Identificar si un colegio tiene un rendimiento satisfactorio en la Competencia 2 (C2) de la
#Pregunta 3 (P3_C2).
#Se considerara que el rendimiento es satisfactorio si un porcentaje especifico
#(definido por el umbral) de los estudiantes evaluados en esta competencia obtiene un puntaje alto.
umbral_absoluto_C2_P3= 0.8
umbral_calculado_C2_P3=df_formateado['NRO_DE_ESTUDIANTES_EVALUADOS']*umbral_absoluto_C2_P3

#1 si el rendimiento es satisfactorio, 0 si no lo es

df_formateado['Rendimiento_Satisfactorio_C2_P3'] = np.where(df_formateado['P3_C2']>=umbral_calculado_C2_P3,1,0)

columnas_caracteristicas = ['NRO_DE_ESTUDIANTES_MATRICULADOS',
                            'NRO_DE_ESTUDIANTES_EVALUADOS','P3_C2']
columna_objeClass = ['Rendimiento_Satisfactorio_C2_P3']
#Crear un nuevo dataframe con las columnas seleccionadas
df_objeClass = df_formateado[columnas_caracteristicas + columna_objeClass]
print(df_objeClass.head())

#Establecer el grupo de datos:Train y Test a los resultados aplicados
X = df_objeClass.drop('Rendimiento_Satisfactorio_C2_P3',axis=1)
Y = df_objeClass['Rendimiento_Satisfactorio_C2_P3']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=100)

#Aplicando regression logistica
logModel = LogisticRegression()
logModel.fit(X_train,y_train)
predictions = logModel.predict(X_test)
print(predictions)
#Calculando la exactitud
print(logModel.score(X_test,y_test))
print(logModel.score(X_train,y_train))
print(classification_report(y_test, predictions))
confusion_matrix(y_test, predictions)
#Curva ROC
roc_curve(y_test, predictions, pos_label=1)
fpr,tpr,threshold = roc_curve(y_test, predictions, pos_label=1)
plt.plot(fpr,tpr)
plt.show()
print(fpr)
print(tpr)
print(threshold)
plt.plot(fpr,tpr,color='orange',label='Curva ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='Curva ROC nula')
plt.xlabel('Tasa de Falsos Positivos-FPR')
plt.ylabel('Tasa de Verdaderos Positivos-TPR')
plt.title('Curva ROC')
plt.legend()
plt.show()

#Aplicando KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(pred)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
#->Real->Negativo->Negativo(N)->a:(TN)
#->Real->Negativo->Positivo(P)->b:(FP)
#->Real->Positivo->Negativo(N)->c:(FN)
#->Real->Positivo->Positivo(P)->d:(TP)
#Precision ('precision')
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
#Calculando la exactitud
"""
[[11  1]
 [ 0  7]]
"""
print(11+7)
print(11+1+0+7)
print((11+7)/(11+1+0+7))
#o tambien se puede calcular asi
#print((tabla[0][0]+tabla[1][1])/(tabla[0][0]+tabla[0][1]+tabla[1][0]+tabla[1][1]))
#Ccalculamos la puntuacion F1
knn.score(X_test,y_test)
print(knn.score(X_test,y_test))
#Calculamos la puntuacion F1 a nivel de train
knn.score(X_train,y_train)
print(knn.score(X_train,y_train))
#Establecemos un numero de vecinos
vecinos = np.arange(1,25)
print(vecinos)
train_2=np.empty(len(vecinos))
test_2=np.empty(len(vecinos))
print(train_2)
print(test_2)
#Ahora tambien generamos el bucle con los vecinos para entrenamiento como test
#Generamos un bucle para registrar los datos en las matrices
for i,k in enumerate(vecinos):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_2[i]=knn.score(X_train,y_train)
    test_2[i]=knn.score(X_test,y_test)
print(train_2)
print(test_2)
#Grafico de vecinos vs Test
plt.title('k-NN: VECINOS VS TEST')
plt.plot(vecinos,test_2,label='Exactirud de Test')
plt.plot(vecinos,train_2,label='Exactitud de Train')
plt.legend()
plt.xlabel('Numero de Vecinos')
plt.ylabel('Exactitud')
plt.show()
#Aplicando Arboles de decision
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
# predict = decision_tree.predict(X_test)
# print(predict)
#Mostrando el arbol
tree.plot_tree(decision_tree)
plt.show()
#Extremos parte del arbol
X_nombre = list(X.columns)
print(X_nombre)
classes = ['No','Si']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,3), dpi=300)
tree.plot_tree(decision_tree,feature_names=X_nombre,class_names=classes,filled=True)
fig.savefig('arbol.png')
#Creamos las predicciones
predTree = decision_tree.predict(X_test)
print(predTree)
#Calculamos la exactitud
print(decision_tree.score(X_test,y_test))
print(decision_tree.score(X_train,y_train))

print(classification_report(y_test,predTree))
print(confusion_matrix(y_test,predTree))

#Aplicando Arboles de decision aleatorios
random_forest = RandomForestClassifier(n_estimators=40,random_state=33)
random_forest.fit(X_train,y_train)
predict_random = random_forest.predict(X_test)
#Calculamos la exactitud
print(random_forest.score(X_test,y_test))
print(random_forest.score(X_train,y_train))

print(classification_report(y_test,predict_random))
print(confusion_matrix(y_test,predict_random))
