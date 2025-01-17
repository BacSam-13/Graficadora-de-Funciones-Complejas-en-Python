from Graficadora_de_Funciones_Complejas import *
"""
¿Como puedo graficar algo?

1. Genera tu funcion de parametrizacion G, puedes declararla tu o usar alguna de las funciones disponibles
2. Determina el intervalo [a,b] sobre el cual quieres evaluar tu funcion g para generar la curva de parametrización
3. Genera tu funcion para evaluar f, puedes declararla tu o usar alguna de las funciones disponibles
4. Llamamos a una funcion graficadora de la siguiente forma:

  graficadora(G, a, b, f, continua = status)
  graficadora_slider(G, a, b, f, continua = status, animacion = status_2)

_slider se usa en caso de querergenerar una barra para ver con detenimiento 
como se proyecta un punto en la parametrizacion y la proyeccion.

En caso de querer mandar varias parametrizaciones con sus respectivos colores, funciones e intervalos. Mande tales
datos en forma de arreglos, y utilice las contrapartes _multiple() de las funciones anteriores

Asegurese de reemplazar los valores como se menciona a continuacion
  G: Parametrizacion
  a: Extremo inicial del intervalo de la parametrizacion
  b: Extremo final del intervalo de la parametrizacion
  f: Funcion con la cual graficar
  status: Indica si la Parametrizacion es continua(False si no lo es, True en otro caso)
  status_2: Indica si se quiere generar una animacion o no


Con el paso del tiempo se pueden agregar mas funciones a este proyecto, el funcionamiento de estas futuras funciones seguira
esta misma idea.
"""
