# Ilícitos en medidores de luz (I parte: Desarrollo del modelo)

La intervención de los medidores eléctricos son una práctica ilícita realizada con el fin de disminuir el consumo energético registrado, estas intervenciones son comunes tanto en casas como en locales. Sin embargo, hay que tener presente que generan problemas de suministro al sector o posibles incendios, poniendo en riesgo a las personas que viven o asisten al lugar o que son del sector.\
\
En este repositorio encontraran el desarrollo metodológico para identificar, a través de Machine Learning, aquellos clientes que han intervenido su medidor eléctrico, en particular:

* Se utilizó el modelo RandomForest, una metodología que se basa en árboles de decisiones.
* Se construyó una base de datos con 1000 registros y una tasa de ilícitos del 15%.
* Los datos se construyeron a partir de antecedentes y supuestos plausibles sobre el cambio en el consumo eléctrico condicionado a si existe o no intervención en el medidor y al tipo de intervención.
* Para este proyecto, se asumió la existencia de 4 tipos de intervenciones, con sus respectivos efectos en la tendencia de consumo una vez que han sido implementados.
\
**Intervención tipo 1**: Se congela el consumo energético en un valor determinado, manteniéndolo constante a través del tiempo.
\
**Intervención tipo 2**: Se reduce el consumo energético en cierto valor porcentual, ej. lo reduce en un 20% cada mes, permitiendo la variación mensual y estacional.
\
**Intervención tipo 3**: Se reduce el consumo energético en ciertos periodos, siendo estos decididos por el infractor (tanto en frecuencia como en cantidad).
\
**Intervención tipo 4**: Se reduce de forma paulatina el consumo energético, llegando a reducirse hasta el 10% del valor real.

* El consumo de los clientes fue obtenido desde una distribución normal, a la cual se agregó ruido por medio de una distribución uniforme. En los casos de que hubiese ilícitos se modificaron las tendencias según lo mencionado en el punto anterior.
* Se crearon marcas para caracterizar a cada cliente, “local” para indicar si el registro corresponde a un local o una casa, “sello_roto” para indicar si el medidor mostrabas signos de aperturas y “impedimento_visual” para indicar si existían objetos que no permitían realizar una inspección sobre el medidor. Cada una de las variables mencionadas están asociadas a si se comete o uno un ilícito (revisar syntax).
* El modelo fue construido a través de una muestra de construcción, validado mediante cross-validation y testeado en una muestra out of sample. Finalmente, el desempeño del modelo fue estimado a través de indicadores como el **racall** y **gini**.

# Resultados
A continuación, se presentan aquellos resultados más relevantes relacionados al desarrollo metodológico, la eficiencia del modelo y su implementación.

#  Consumo energético sin intervención
Se muestra la serie de tiempo del consumo energético en los 12 meses anteriores al mes de referencia, el mes 12 es el mes más antiguo (mas lejos en el pasado) y el mes 1 es el mes más cercano al de referencia, esto aplica para todas las imágenes. Los valores son obtenidos desde una distribución normal a la cual se le agregó ruido.

[![Consumo-energia.png](https://i.postimg.cc/3rGLPnhY/Consumo-energia.png)](https://postimg.cc/t7Xt6d7M)

# Intervención del medidor tipo1
A continuación, se muestra el efecto que tiene la intervención del tipo 1 sobre el consumo energético. Se puede apreciar que a partir de cierto punto el valor del consumo se “congela”.

[![Tipo-ilicito1.png](https://i.postimg.cc/W1x8Wkyh/Tipo-ilicito1.png)](https://postimg.cc/3kC2WdYT)

# Intervención del medidor tipo2
A continuación, se muestra el efecto que tiene la intervención del tipo 2 sobre el consumo energético. Se puede apreciar que hay una reducción porcentual en el consumo energético.

[![Tipo-ilicito2.png](https://i.postimg.cc/C1cftyKm/Tipo-ilicito2.png)](https://postimg.cc/zbRBbcKg)

# Intervención del medidor tipo3
A continuación, se muestra el efecto que tiene la intervención del tipo 3 sobre el consumo energético. Se puede apreciar que existen ciertos meses en los cuales el consumo cae de forma considerable.

[![Tipo-ilicito3.png](https://i.postimg.cc/c1KQjHQ3/Tipo-ilicito3.png)](https://postimg.cc/G9C8Tbg3)

# Intervención del medidor tipo4
A continuación, se muestra el efecto que tiene la intervención del tipo 4 sobre el consumo energético. Se puede apreciar que existen una reducción decreciente en el consumo energético a través del tiempo.

[![Tipo-ilicito4.png](https://i.postimg.cc/2SjyscPt/Tipo-ilicito4.png)](https://postimg.cc/xc79HRjG)
 
# Cross validation
Para asegurar la robustez del modelo y su correcta parametrización, se optó por utilizar la metodología de croos-validation, que consiste en utilizar cierto porcentaje de la muestra de desarrollo para entrenar el modelo y el resto para probar el efecto que tiene los parámetros sobre el desempeño del modelo ante datos nuevos. Para este proyecto se dejaron fijos valores como la profundidad del árbol, el porcentaje de variables que se van a utilizar, etc. y se hizo variar el número de árboles del modelo (n_estimators), con el fin de obtener el conjunto de parámetros que asegurasen la robustez del modelo ante un conjunto de datos nuevos.

[![cross-validation.png](https://i.postimg.cc/4yrXpS3y/cross-validation.png)](https://postimg.cc/QKJL3S1Z)

# Accuracy Cross-validation vs Train
Del proceso anterior, se graficó el desempeño del modelo durante su construcción y durante cross validation, a lo largo de distintos números de árboles (n_estimators). En rojo se encuentra marcado el número de árboles óptimos para lograr el mejor desempeño durante cross-validation. Para el caso, el número de árbol óptimos es 81.

[![acc-cross-validation.png](https://i.postimg.cc/B6wnHHNn/acc-cross-validation.png)](https://postimg.cc/LYLSMqrc)

# Métricas de eficiencia
A continuación, se muestran distintos indicadores de la eficiencia del modelo.
/
De los indicadores el que más nos interesa para este proyecto es el **recall** que mide el porcentaje de los registros que han cometido ilícito y que somos capaces de identificar. Para nuestro caso corresponde al **54%**. Por lo cual, si vamos a visitar a todos los registros que el modelo etiqueta como malos, solo el 54% de ellos efectivamente estaría cometiendo ilícitos. Hay que tener presente que esta es una medida global y que puede ser mejorada a través de tablas de eficiencia.

[![metricas-eficiencia.png](https://i.postimg.cc/MTDdD0FV/metricas-eficiencia.png)](https://postimg.cc/dkh2QCvV)

# Tablas de eficiencia
Para el modelo se obtiene un gini igual a 0.897 indicador de que el modelo tiene una alta capacidad para discretizar entre clientes que cometen ilícitos y los que no.
A continuación, se muestra la tabla de eficiencia y el número de registros que comenten ilícitos según tramos de probabilidad. Se puede observar que el **“recall”** del grupo más riesgoso es de un **88%** siendo este grupo al que se debería ir a visitar para capturar clientes que han intervenido sus medidores.

[![Tabla-eficiencia.png](https://i.postimg.cc/MGbTKtTC/Tabla-eficiencia.png)](https://postimg.cc/FfzmTy0p)

# SIGUIENTES ETAPAS
* Desarrollo de un análisis al modelo y definición de gestiones
* Realizar análisis según tipo de intervención para guiar la gestión de aquellos que generen mayor retorno.
* Desarrollo de un Dashbord para su ejecución mensual
