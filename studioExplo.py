import moduloDatos as md

ruta = 'G:\\2023-Proyectos\\Proyectos\Carpeta-Github\\DataScience-MachineLearming\\efectivo1DS\\Codigos\\'
fichero = 'precios_carros.csv'

df = md.cargarDatos(ruta, fichero,',')

print(df.head())
#print(df)
df.set_index('Unnamed: 0', inplace=True)
print(df.head())

print(md.obtenerDatosColumnas(df))
"""
Index(['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type',
       'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats',
       'New_Price'],
      dtype='object')
"""
#pasar a español
titulos_columnas = ['Nombre', 'Ubicacion', 'Año', 'Kilometros', 'Tipo de Combustible',
         'Tipo de Transmision', 'Tipo de Propietario', 'Consumo', 'Motor', 'Potencia', 'Asientos',
            'Precio Nuevo']
md.cambiarTitulosColumnas(df, titulos_columnas)

print(md.obtenerDatosColumnas(df)) 












