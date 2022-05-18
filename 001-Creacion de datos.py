
###################################################################################
#      PROYECTO: MODELO PARA IDENTIFICAR ILÍCITOS EN MEDIDORES DE LUZ
#                      CON MACHINE LEARNING
###################################################################################

#######################################
# Autor: Nahuel Canelo
# Correo: nahuelcaneloaraya@gmail.com
#######################################

########################################
# IMPORTAMOS LAS LIBRERÍAS DE INTERÉS
########################################

import numpy as np
import pandas as pd
import uuid
import random
from numpy.random import rand
import warnings
import seaborn as sns
warnings.filterwarnings('once')

seed=123
np.random.seed(seed) # fijamos la semilla
random.seed(seed)

#############################################
# CREAMOS LAS FUNCIONES QUE VAMOS A UTILIZAR
#############################################

# Construiremos una función que defina el comportamiento histórico del consumo energético del cliente
# según si ha adulterado o no el medidor (0 si no hay ilícito, 1 si lo hay)

# Rezagos: Número de meses anteriores al mes de referencia con información, 1 es el mes más cercano y 12
# es el más lejano
# Valor_max: Es el consumo máximo alcanzado por el cliente durante una ventana de 12 meses

registro_real=[]
def tendencia(x, rezagos, valor_max):
    ds=valor_max/10# ds
    noise = np.random.uniform(-0.3, 0.3, rezagos)  # Se utilizará para agregar ruido
    azar=rand(1)
    x_valores = np.random.normal(valor_max*azar, ds * azar, rezagos)
    x_valores = x_valores + (x_valores * noise)
    x_valores = (x_valores/np.max(x_valores)) * valor_max * azar # consumo en los últimos n meses
    registro_real.append(x_valores)
    salida=tipo_ilicito(x,x_valores) # Según el ilícito que cometan (o no) la serie se verá modificada
    return pd.Series(salida)


# Si el cliente ha cometido ilícito, se ajustará el consumo según el tipo de ilícito cometido
# Si el cliente no ha cometido ilícito, se construyen variables históricas a partir de una función normal

# Tipos de ilícitos y su descripción
# Abrupto: caída abrupta en el consumo y luego mantenido
# Porcentual: caída porcentual del consumo
# Puntual: disminución ciertos meses
# Decreciente: caída porcentual decreciente en el consumo


def tipo_ilicito(x, x_valores):
    print(x_valores)
    salida=x_valores.copy()
    azar = rand(1)
    pos= int(np.random.uniform(2, rezagos-1,1))
    inicio= list(np.ones(pos))
    fin = list(np.ones(rezagos - pos))
    if x==1: # luego desde cierto punto el valor se mantiene fijo
        salida[pos:]=np.min(salida) * azar
    elif x==2: # Se realiza una reducción porcentual fija desde cierta posición
        salida = salida*(inicio + list(fin*azar) )
    elif ((x==3) and (len(fin)>1)): #puntos con caída en ciertos periodos
        puntos = list(np.random.randint(2, size=len(fin)))
        indices = [i for i, x in enumerate(puntos) if x == 0]
        for index in indices: fin[index] = 0.3
        salida = salida * list(inicio + fin)
    elif ((x==4) and (len(fin)>1)):  #reducción porcentual decreciente
        fin=list(np.linspace(0.9,0.1,len(fin)))
        salida = salida * list(inicio + fin)
    else:
        salida=salida
    return salida

# Creamos una función cuya salida sean los ilícitos y los tipos de ilícitos

def patron_ilicito(porcentaje,n):
    n_ones=round((porcentaje/100)*n)
    ilicito1 = list(np.ones(round(10*n_ones/100))*1)
    ilicito2 = list(np.ones(round(50*n_ones/100))*2)
    ilicito3 = list(np.ones(round(30*n_ones /100))*3)
    ilicito4 = list(np.ones(round(10*n_ones/100))*4)
    ilicito= ilicito1 + ilicito2 + ilicito3+ ilicito4
    no_ilicito=list(np.zeros(n-len(ilicito)))
    vector=list(ilicito + no_ilicito)
    return vector


def local(x):
    if x>0:
        valores=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        salida = random.choices(valores, k=1)[0]
    else:
        valores = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        salida = random.choices(valores, k=1)[0]
    return salida


def sello_roto(x):
    if x>0:
        valores = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        salida = random.choices(valores, k=1)[0]
    else:
        valores = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        salida = random.choices(valores, k=1)[0]
    return salida


def impedimento_visual(x):
    if x>0:
        valores = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        salida = random.choices(valores, k=1)[0]
    else:
        valores = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        salida = random.choices(valores, k=1)[0]
    return salida

###########################################################################
# CREAMOS LA BASE DE DATOS SEGÚN TENDENCIAS PLAUSIBLES ASOCIADAS A LA FUGA
###########################################################################

n=1000 # Número de clientes/registros que se van a crear
rezagos=12 # Número de meses anteriores al mes de referencia para los cuales se crearán variables

#Inicializamos un dataframe con los ID de cada cliente
#data=pd.DataFrame({"ID": [uuid.uuid4() for _ in range(n)]})
data=pd.DataFrame({"ID": list(range(n))})

# Agregamos una tasa de ilícitos del 15% (esta se divide en 4 tipos de ilícitos)
tipo_ilicitos=patron_ilicito(15,n)
data=data.assign(tipo_ilicito=random.sample(tipo_ilicitos,n))
data=data.assign(ilicito= lambda x: (x.tipo_ilicito>=1) * 1)

# Consumo mensual de luz (el mes 12 es el rezago más antigüo y el mes 1 el más reciente)
data_consumo=data["tipo_ilicito"].apply(tendencia,args=(rezagos, 500000))
data_consumo.columns=np.array(["mes_"+str(i) for i in range(rezagos,0,-1)])
data=pd.concat([data,data_consumo],axis=1)

# Local
data["local"]=data.apply(lambda x: local(x.tipo_ilicito),axis=1)

# Marca de sello adulterado
data["sello_roto"]=data.apply(lambda x: sello_roto(x.tipo_ilicito),axis=1)

# Marca de impedimento visual
data["impedimento_visual"]=data.apply(lambda x: impedimento_visual(x.tipo_ilicito),axis=1)



##############################
# VISUALIZAMOS ALGUNOS CASOS
##############################

def plot_data(data,tipo_ilicito,rezagos,n):
    variables=np.array(["mes_"+str(i) for i in range(rezagos,0,-1)])
    data_consumo=data.query("tipo_ilicito=={}".format(tipo_ilicito))[['ID'] + list(variables)].sample(n).copy()
    data_consumo_melt=pd.melt(data_consumo,id_vars='ID')
    sns.set(font_scale=1.3)
    fig=sns.lineplot(data=data_consumo_melt, x="variable", y="value", hue="ID", palette="tab10", linewidth=2.5)
    fig.set(xlabel="Registro mensual", ylabel="Consumo de luz ($)",
            title="Variación histórica del consumo")
    return fig

plot_data(data,1,12,3)

