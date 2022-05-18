
###################################################################################
#      PROYECTO: MODELO PARA IDENTIFICAR ILÍCITOS EN MEDIDORES DE LUZ
#                      CON MACHINE LEARNING
###################################################################################

#######################################
# Autor: Nahuel Canelo
# Correo: nahuelcaneloaraya@gmail.com
#######################################


########################################
# IMPORTAMOS LAS LIBRERIAS DE INTERÉS
########################################

import warnings
warnings.filterwarnings('once')


#######################################
# CREACIÓN DE VARIABLES ARTIFICIALES
#######################################

# Construimos una base de datos que entrará en el modelo
data_artificial=data[["ilicito","local","sello_roto","impedimento_visual"]]


# Truquito para crear las variables de forma aútomatica: A continuación se crean de forma automatica la programación
# de múltiples variables, están quedan guardads en el objeto "variables" y pueden ser ejecutadas con exec o impresas
# para dejarlas explicitas

list_var=["mes_"]
meses=[12,6,3]
funciones=["sum","mean","max","min"]
variables=[("data_artificial=data_artificial.assign("+var+funcion+str(n_meses)+"=data[['"+var+
            "' +str(i+1) for i in range("+str(n_meses)+")]]."+funcion+"(axis=1))")
           for var in list_var for n_meses in meses for funcion in funciones]

# ejecutamos las funciones
for var in variables:
    exec(var)

# dejamos explicitas las funciones creadas
#print(variables)


# Dimensiones de data_artificial
data_artificial.shape

# Nombre de las columnas
data_artificial.columns


# Se observa que se han creado 51 variables artificiales