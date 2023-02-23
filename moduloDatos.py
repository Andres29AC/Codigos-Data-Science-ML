import pandas as pd
import numpy as np
# def cargaDatos(fichero):
#     datos = pd.read_csv(fichero)
#     return datos

# if __name__ == '__main__':
#     fichero = 'precios_carros.csv'
#     datos = cargaDatos(fichero)
#     print(datos.head(5))

#def cargaDatos(ruta,fichero):
#   datos = pd.read_csv(ruta+fichero)
#    return datos
def cargarDatos(ruta,fichero,separador=','):
    datos = pd.read_csv(ruta+fichero,sep=separador)
    return datos

#Obtencion de datos de las columnas 
def obtenerDatosColumnas(datos):
    datosColumna = datos.columns
    return datosColumna

def cambiarTitulosColumnas(datos,columnas):
    datos.columns = columnas
    return datos

def renombrarColumna(datos,cambio):
    datos.rename(columns=cambio,inplace=True)
    return datos

   

#Funcion para que nos devuelva las columnas
def guardarDatos(datos,ruta,fichero):
    datos.to_csv(ruta+fichero)
    return True
#Metodos Basicos
#Funcion que reescribe los datos de un fichero
def dameEstadisticos(datos,tipo ='numerico'):
    """tipo numerico o todos"""
    if tipo == 'numerico':
        return datos.describe()
    else:
        return datos.describe(include='all')
def reemplazarValoresNulos(datos,columna):
    media = datos[columna].mean()
    datos[columna].replace(np.nan,media,inplace=True)
    return datos

def cambiarTipos(columna,tipo='float64'):
    columna = columna.astype(tipo)
    return columna
def normalizacionDatos1(datos,columna):
    datos[columna] = datos[columna]/datos[columna].max()
    return datos
def normalizacionDatos2(datos,columna):
    datos[columna] = (datos[columna]-datos[columna].min())/(datos[columna].max()-datos[columna].min())
    return datos
def obtenerValoresDummies(datos,columna):
    datos = pd.get_dummies(datos[columna])
    return datos





if __name__ == '__main__':
    #Tests
    ruta = 'G:\\2023-Proyectos\\Proyectos\Carpeta-Github\\DataScience-MachineLearming\\efectivo1DS\\Codigos\\'
    fichero = 'precios_carros.csv'
    datos = cargarDatos(ruta,fichero)
    print(datos.head(5))
    print(obtenerDatosColumnas(datos))
    print(guardarDatos(datos,ruta,'copia.csv'))
    titulos_cabecera = ['indice','nombre','localizacion','año','kilometros recorridos','tipo de combustible',
                        'tipo de transmision','tipo de propietario','rendimiento','motor','potencia','asientos','precio']
    cambiarTitulosCabecera(datos,titulos_cabecera)
    print(datos.columns)
    print(dameEstadisticos(datos))
    print(dameEstadisticos(datos,'todos'))
    columna = datos['kilometros recorridos']
    columna = cambiarTipos(columna,'int64')
    print(columna.dtypes)
    print(columna)
    datos = renombrarColumna(datos,{'kilometros recorridos':'kilometros'})
    print(datos.columns)
    datos = normalizacionDatos1(datos,'kilometros')
    print(datos['kilometros'])
    datos = normalizacionDatos2(datos,'kilometros')
    print(datos['kilometros'])






