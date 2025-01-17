# Graficadora de Funciones Complejas
# Autor: Baruc Samuel Cabrera Garcia

"""
El siguiente codigo es una implementacion de una graficadora de funciones complejas.
Se declararon varias funciones complejas, así como funciones constructoras, las cuales
permiten al usuario generar una gran variedad de parametrizaciones para ser evaluadas en funciones.
Por esto mismo, se desarrollaron varios métodos de graficación para ciertos casos.
"""

############### Librerias necesarias para el codigo
import cmath
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.widgets import Slider
import math
import matplotlib.animation as animation
import sympy as sp

# Se usaran para la funcion get_K_funcion, en conjunto de graficadora_3d_K
y0 = sp.Symbol('y', real=True)
x0 = sp.Symbol('x', real=True)
I = sp.I # Numero i para uso de sympy

# Margen para las parametrizaciones discontinuas
epsilon = 0.000001

############### Declaracion de funciones base
pi = cmath.pi

# Funcion que dado un numero z = x + jy, regresa (x,y)
def split(z):#Regresa la Parte Real e imaginaria de un numero z
  #return sp.re(z), sp.im(z)
  return z.real, z.imag

# Funcion que regresa la Parte Real de un numero z
def Re(z): # Funcion
  #return sp.re(z)
  return z.real

# Funcion que regresa la parte imaginaria de un numero z
def Im(z):
  #return sp.im(z)
  return z.imag

# Funcion que dado un numero z = x + jy, regresa x - jy (el conjugado de z)
def conjug(z):
  #return sp.conjugate(z)
  return z.conjugate()

# Funcion que regresa la norma de un numero z
def norma_c(z):
  return (z*conjug(z))**(1/2)

# Funcion que regresa la exponencial compleja de z
def exp_c(z):
  x, y = split(z)
  return np.exp(x)*( np.cos(y) + 1j*np.sin(y) )

# Funcion que regresa el coseno complejo de z
def cos_c(z):
  return (exp_c(1j*z) + exp_c(-1j*z))/2

# Funcion que regresa el seno complejo de z
def sen_c(z):
  return (exp_c(1j*z) - exp_c(-1j*z))/(1j*2)

# Funcion tangente compleja
def tan_c(z):
  return sen_c(z)/cos_c(z)

# Funcion que regresa el argumento de un numero z (angulo de z respecto al eje real)
def arg_c(z):
  x, y = split(z)
  # El angulo estara en el intervalo [0,2pi
  return np.where(y < 0, 2*pi + np.arctan2(y, x), np.arctan2(y, x)) # Quitamos el signo al angulo

# Variante de arg_c que escala el resultado de  [0,2pi) a [0,360)
def arg_c_grados(z): 
  return arg_c(z) * (360/(2*pi))

# Funcion logaritmo compleja de un numero z
def log_c(z):
  return np.log(norma_c(z)) + 1j*arg_c(z)



############### Funciones constructuras de funciones (crean funciones segun ciertos parametros)

# Funcion que regresa la funcion f(z) = (az+b)/(cz+d), si los parametros son validos
def build_Moebius(a,b,c,d):
  if(a*d == b*c):# Verificamos que se cumpla la condicion de parametros
    print("Cambie los parametros, porfavor")
    return 'error'
  else:
    def Moebius(z):
      return (a*z + b)/(c*z + d)
    return Moebius

# Funcion generadora de polinomios, con coeficientes A en formato de lista (de mayor a menor orden)
# Incluimos bandera para indicar si queremos que el polinomio sea de la forma 1/p(z)
def build_pol_coef(A, flag = False):
  n = len(A)# Numero de coeficientes
  def f(z):
    sum = 0
    for i in range(n):
      sum += A[-1-i]*z**(i)
    if flag:
      return 1/sum
    return sum
  return f


# Funcion generadora de polinomios, con sus ceros en formato de lista
# Incluimos bandera para indicar si queremos que el polinomio sea de la forma 1/p(z)
def build_pol_cero(P, flag = False):
  def f(z):
    prod = 1
    for p in P:
      prod *= (z-p)
    if flag:
      return 1/prod
    return prod
  return f


# Funcion que regresa una recta que une a los puntos start y end, usando el intervalo [a,b]
def build_Recta(start,end, a, b):
  def recta(t):
    t0 = (t-a)/(b-a)#Para escalar [a,b] en [0,1]
    return t0*end + (1-t0)*start
  return recta

# Funcion que regresa una parametrizacion para una circunferencia centrada en c, de radio r
def build_Circle(c,r):
  def circ(t):
    return c + exp_c(1j*t + np.log(r))
  return circ

# Regresa una parametrizacion para n circunferencias centradas en c,
# donde los radios tienen tamaño de paso Dr.
# Se asuminara que el intervalo para la funcion resultante es [0,n),
# este intervalo determina cuantas circunferencias se generaran.
def build_circunferencias_concentricas(c,Dr):
  def circunferencias(t):
    n_circ = math.floor(t)
    circ = build_Circle(c,Dr*(n_circ+1))
    t_new = (t-n_circ)*2*pi
    return circ(t_new)
  return np.vectorize(circunferencias)


# Regresa un total de 2*n rectas paralelas a la recta ya dada (n a su izquierda y n a su derecha)
# Cada recta esta separada por una distancia r
def build_rectas_paralelas(Recta, a, b, n, r): # Recibimos ademas la recta y su intervalo
  # Usaremos un intervalo total de [0, (2*n+1)) (Para facilitar el manejo)

  p = Recta(a) # Punto inicial de la recta
  q = Recta(b) # Punto final de la recta
  # [k, k+1) es el intervalo de la k-th recta paralela a Recta (desde izquierda a derecha, indexado en cero)
  def paralelas(t):
    # Primero, generaremos el numero complejo que funcionara como vector de desplazamiento (C <=> R^2)
    PQ = q-p # Vector de direccion
    u = PQ/norma_c(PQ) # Vector normalizado
    v = conjug(u) # Vector perpendicular
    # El vector de desplazamiento sera dado por d = (v*r_0)*r, r_0 depende de la línea paralela

    n_paralela = math.floor(t)
    r_0 = -(n - n_paralela)
    d = (v*r_0)*r
    p_new = p + d
    q_new = q + d
    recta_k = build_Recta(p_new, q_new, n_paralela, n_paralela+1)
    return recta_k(t)


  return np.vectorize(paralelas)

# Funcion que construye un cuadrado centrada en c, con radio r
def build_Square(c,r):
  def square(t):
    #Definimos las 4 esquinas del cuadrado
    p1 = c - r - r*1j # Inferior izquierda
    p2 = c + r - r*1j # Inferior derecha
    p3 = c + r + r*1j # Superior derecha
    p4 = c - r + r*1j #Superior izquierda

    t_floor = math.floor(t)
    if t_floor == 0:
      recta_1 = build_Recta(p1, p2, 0, 1)
      return recta_1(t)
    elif t_floor == 1:
      recta_2 = build_Recta(p2, p3, 1, 2)
      return recta_2(t)
    elif t_floor == 2:
      recta_3 = build_Recta(p3, p4, 2, 3)
      return recta_3(t)
    else:
      recta_4 = build_Recta(p4, p1, 3, 4)
      return recta_4(t)

  return np.vectorize(square)

# Funcion que genera una malla de n rectas horizontales y n verticales, "centradas" en c, de longitud l
# Cada una separada de sus paralelas por una distancia de 2r
# Se asuminara que el intervalo para la funcion resultante es [0,2*n)
def build_malla(c, r, l, n):
  if n % 2 == 0: #Si hay una cantidad par de rectas horizontales
    # Enumeraremos las mallas horizontales desde 0 a n-1, desde abajo hacia arriba
    # Enumeraremos las mallas verticales desde n a 2n-1, desde la izquierda hacia la derecha

    # Ubicamos las esquinas para que sea mas facil generar las rectas
    # Horizontales
    esquina_inf_izq = c - (l/2) - (((n/2)-1)*2*r + r)*1j
    esquina_inf_der = c + (l/2) - (((n/2)-1)*2*r + r)*1j
    # Verticales
    esquina_izq_sup = c - (((n/2)-1)*2*r + r) + (l/2)*1j
    esquina_izq_inf = c - (((n/2)-1)*2*r + r) - (l/2)*1j

    def malla(t):
      id_recta = math.floor(t) # Ubicamos la recta asociada al valor t, ya que su intervalo seria [floor(t),ceil(t)]

      if id_recta < n: # Si es de una recta horizontal
        start = esquina_inf_izq + (id_recta)*(2*r)*(1j)
        end = esquina_inf_der + (id_recta)*(2*r)*(1j)

        recta = build_Recta(start, end, id_recta, id_recta + 1)
        return recta(t)
      else: # Si es de una recta vertical
        start = esquina_izq_sup + ((id_recta-n))*(2*r)
        end = esquina_izq_inf + ((id_recta-n))*(2*r)
        recta = build_Recta(start, end, id_recta, id_recta + 1)
        return recta(t)

  else: # Si hay una cantidad inpar de rectas horizontales
    m = (n-1)/2 # Rectas a cada extremo de la cruz del centro (caso impar)
    # Ubicamos las esquinas para generar las rectas
    # Horizontalles
    esquina_inf_izq = c - (m*(2*r))*1j - (l/2)
    esquina_inf_der = c - (m*(2*r))*1j + (l/2)
    # Verticales
    esquina_izq_sup = c - m*(2*r) + (l/2)*1j
    esquina_izq_inf = c - m*(2*r) - (l/2)*1j

    def malla(t):
      id_recta = math.floor(t) # Ubicamos la recta asociada al valor t, ya que su intervalo seria [floor(t),ceil(t)]

      if id_recta < n: # Si es de una recta horizontal
        start = esquina_inf_izq + (2*r)*(id_recta)*1j
        end = esquina_inf_der + (2*r)*(id_recta)*1j

        recta = build_Recta(start, end, id_recta, id_recta + 1)
        return recta(t)
      else: # Si es de una recta vertical
        start = esquina_izq_sup + (2*r)*((id_recta-n))
        end = esquina_izq_inf + (2*r)*((id_recta-n))

        recta = build_Recta(start, end, id_recta, id_recta + 1)
        return recta(t)

  return np.vectorize(malla)

# Funcion que genera una malla polar, desde el angulo theta_0 a theta_1. Este intervalo de angulos se representara en n_theta rayos.
# Ademas, se crearan n_r semi semicirculos, separados entre si por una distancia d_r
# Todo este, se hara con el centro c
# Se asuminara que el intervalo para la funcion resultante es [0, n_r + n_theta)
# Se pide n_theta > 1
def build_malla_polar(c, n_r, d_r, theta_0, theta_1, n_theta):
  skip_theta = (theta_1 - theta_0)/(n_theta-1) # Diferencias entre cada rayo
  l = n_r*d_r # Largo de los rayos

  def malla_polar(t):
    if t < n_theta: # Primero se haran los casos para los rayos que salen del centro
      # Vemos cual rayo estaremos graficando segun el valor de t
      n_rayo = math.floor(t) # Tomamos su valor piso
      end_point = c + l*(np.cos(theta_0 + skip_theta*n_rayo) + 1j*np.sin(theta_0 + skip_theta*n_rayo)) # En base a n_rayo, vemos donde va a terminar tal rayo
      rayo = build_Recta(c, end_point, n_rayo, n_rayo + 1)
      return rayo(t)
    elif t != n_r + n_theta:
      n_circulo = math.floor(t) - n_theta
      semi_circulo = build_Circle(c, (n_circulo + 1)*d_r)
      new_t = (theta_1 - theta_0)*(t - n_theta - n_circulo) + theta_0  # Escalamos a (t - n_theta) del intervalo [n_circulo, n_circulo + 1 ) a [theta_0, theta_1)
      return semi_circulo(new_t)
    
  return np.vectorize(malla_polar)

############### Funciones de graficación y visualización
  
# Funcion que recibe los ceros de un polinomio e imprime su forma
def print_pol_cero(P, flag = False):
  if flag:
    pol = '1/p(z) = '
  else:
    pol = 'p(z) = '

  n = len(P)
  for i in range(n):
    num = str(P[i])
    if i == n-1:
      pol += '(z - '+num+')'
    else:
      pol += '(z - '+num+')*'
  print(pol)

# Funcion que recibe los coeficientes de un polinomio e imprime su forma
def print_pol_coef(P, flag = False):
  if flag:
    pol = '1/p(z) = '
  else:
    pol = 'p(z) = '

  n = len(P)
  for i in range(n):
    num = str(P[i])
    if i == n-1:
      pol += num
    else:
      pol += num+'z^{'+str(n-1-i)+'} + '
  print(pol)

# Dada una parametrizacion con intervalo [a,b] o [a,b), regresa un vector de 100000 puntos 
# que representen a la curva
def curve_generator(g, a, b, continua = True):
  if continua:
    T = np.linspace(a,b,100000)
    return g(T)
  else: # En caso se que g() no sea una parametrizacion continua, obtenemos sus fragmentos por separado
    Curve = []

    for i in range(b-a):
      T = np.linspace(i,i+1-epsilon,100000) # Se toma [i, i+1-epsilon] para no mezclar fragmentos
      Curve.append(g(T))
    return Curve

# Funcion graficadora.
# Para graficar, se manda una parametrizacion g junto con su intervalo [a,b] o [a,b), 
# así como la funcion f sobre la cual graficaremos.
# Tambien se considera la bandera continua, la cual indica si la parametrizacion es de una curva continua
def graficadora(g, a, b, f, color = 'b', continua = True):

  # Graficamos Gamma
  if continua:
    # Curva a mandar a funcion
    Gamma = curve_generator(g, a, b, continua)
    X0, Y0 = split(Gamma)

    # Cuva resultado de funcion
    Gamma2 = f(Gamma)
    X1, Y1 = split(Gamma2)

    plt.subplot(1, 2, 1)
    plt.plot(X0, Y0, c = 'red')
    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginaria')
    plt.title('Gráfico de g(t)')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.plot(X1, Y1, c =  color)
    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginaria')
    plt.title('Gráfico de f(g(t))')
    plt.axis('equal')

  else:
    Gammas = curve_generator(g, a, b, continua)
    plt.subplot(1, 2, 1)
    for i in range(len(Gammas)):
      X0, Y0 = split(Gammas[i])
      plt.plot(X0, Y0, c = 'red')
      plt.xlabel('Parte Real')
      plt.ylabel('Parte Imaginaria')
      plt.title('Gráfico de g(t)')
      plt.axis('equal')

    plt.subplot(1, 2, 2)
    for i in range(len(Gammas)):
      X1, Y1 = split(f(Gammas[i]))
      plt.plot(X1, Y1, c =  color)
      plt.xlabel('Parte Real')
      plt.ylabel('Parte Imaginaria')
      plt.title('Gráfico de f(g(t))')
      plt.axis('equal')
  

  plt.tight_layout()
  #plt.grid(True)
  plt.show()

# Similar a graficadora(), pero permite generar una barra que nos muestra el comportamiento de f()
def graficadora_slider(g, a, b, f, color = 'b', continua = True, animacion = False):

  # Creamos la figura con sus axis para las curvas
  fig, (ax1, ax2) = plt.subplots(1, 2)

  # Prcodemos a graficar de cierto modo si continua esta activa
  if continua:
    Gamma = curve_generator(g, a, b, continua)
    X0, Y0 = split(Gamma)

    # Cuva resultado de funcion
    Gamma2 = f(Gamma)
    X1, Y1 = split(Gamma2)
    # Graficamos a la parametrizacion segun su intervalo
    ax1.plot(X0, Y0, c = color)
    ax1.set_xlabel('Parte Real')
    ax1.set_ylabel('Parte Imaginaria')
    ax1.set_title('Gráfico de g(t)')
    ax1.set_aspect('equal')  # Ajustar la proporción de los ejes

    # Graficamos a la funcion evaluada en la parametrizacion
    ax2.plot(X1, Y1, c = color)
    ax2.set_xlabel('Parte Real')
    ax2.set_ylabel('Parte Imaginaria')
    ax2.set_title('Gráfico de f(t)')
    ax2.set_aspect('equal')  # Ajustar la proporción de los ejes

  else:
    Gammas = curve_generator(g,a,b,continua)
    # Graficamos a la parametrizacion segun su intervalo
    for i in range(len(Gammas)):
      X0, Y0 = split(Gammas[i])
      ax1.plot(X0, Y0, c = color)
      ax1.set_xlabel('Parte Real')
      ax1.set_ylabel('Parte Imaginaria')
      ax1.set_title('Gráfico de g(t)')
      ax1.set_aspect('equal')  # Ajustar la proporción de los ejes

    # Graficamos a la funcion evaluada en la parametrizacion
    for i in range(len(Gammas)):
      X1, Y1 = split(f(Gammas[i]))
      ax2.plot(X1, Y1, c = color)
      ax2.set_xlabel('Parte Real')
      ax2.set_ylabel('Parte Imaginaria')
      ax2.set_title('Gráfico de f(t)')
      ax2.set_aspect('equal')  # Ajustar la proporción de los ejes


  # Ahora, declararemos a los puntos que recorreran ambas curvas según alteremos el valor t del intervalo
  if not continua:
    X0, Y0 = split(Gammas[i])

  punto_1 = ax1.scatter(X0[0],Y0[0], color = 'red')
  punto_2 = ax2.scatter(X1[0],Y1[0], color = 'red')

  # Creamos el slider para mover los puntos
  axcolor = 'lightgoldenrodyellow'
  ax_slider_point = plt.axes([0.15, 0, 0.65, 0.03], facecolor=axcolor)
  slider_point = Slider(ax_slider_point, 'Intervalo', a, b-epsilon, valinit = a)

  def update(val):
    t = slider_point.val# Obtenemos el valor del invervalo del slider

    val_0 = g(t)# Obtenemos el punto que corresponde a g(t)
    x_0, y_0 = split(val_0)

    val_1 = f(val_0)# Obtenemos el punto que corresponde a f(g(t))
    x_1, y_1 = split(val_1)

    # Actualizamos los puntos
    punto_1.set_offsets([x_0,y_0])
    punto_2.set_offsets([x_1,y_1])

    fig.canvas.draw_idle()

  slider_point.on_changed(update)

  ############################# Para generar animaciones
  if animacion:
    # Función para la animación
    def animate(intervalo):
        slider_point.set_val(intervalo)

    # Configuración de la animación
    ani = animation.FuncAnimation(fig, animate, frames=np.arange(a, b, 0.1), interval=100)

  plt.show()


# Funcion para graficar distintos conjuntos de funciones g, f, con sus intervalos y colores respectivos
# Es analoga a graficadora()
def graficadora_multiple(G,I,F,Colores,continua = True):
  n = len(G)# Numero de funciones a parametrizaciones a evaluar

  for i in range(n):
    g = G[i] # Parametrizacion
    a = I[i][0] # Extremo izquierdo del intervalo
    b = I[i][1] # Extremo derecho del intervalo
    f = F[i] # Funcion en la cual evaluar
    color = Colores[i] # Color para graficar

    # Graficamos Gamma
    if continua:
      # Curva a mandar a funcion
      Gamma = curve_generator(g, a, b, continua)
      X0, Y0 = split(Gamma)

      # Cuva resultado de funcion
      Gamma2 = f(Gamma)
      X1, Y1 = split(Gamma2)

      plt.subplot(1, 2, 1)
      plt.plot(X0, Y0, c = 'red')
      plt.xlabel('Parte Real')
      plt.ylabel('Parte Imaginaria')
      plt.title('Gráfico de g(t)')
      plt.axis('equal')


      plt.subplot(1, 2, 2)
      plt.plot(X1, Y1, c =  color)
      plt.xlabel('Parte Real')
      plt.ylabel('Parte Imaginaria')
      plt.title('Gráfico de f(g(t))')
      plt.axis('equal')

    else:
      Gammas = curve_generator(g, a, b, continua)

      plt.subplot(1, 2, 1)
      for i in range(len(Gammas)):
        X0, Y0 = split(Gammas[i])
        plt.plot(X0, Y0, c = 'red')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginaria')
        plt.title('Gráfico de g(t)')
        plt.axis('equal')

        plt.subplot(1, 2, 2)
      for i in range(len(Gammas)):
        X1, Y1 = split(f(Gammas[i]))
        plt.plot(X1, Y1, c =  color)
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginaria')
        plt.title('Gráfico de f(g(t))')
        plt.axis('equal')
    

  plt.tight_layout()
  #plt.grid(True)
  plt.show()


# Graficadora que permite colocar un punto para desplazarse por los intervalos
# Analoga a gradicadora_slider, pero adaptada para recibir mas funciones
def graficadora_slider_multiple(G,I,F,Colores,continua = True, animacion = False, epsilon = 0.000001): #################
  n = len(G)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10))

  #Creamos almacenamiento para los puntos
  X = np.zeros(n)
  Y = np.zeros(n)
  puntos_1 = [ax1.scatter(x,y) for x,y in zip(X,Y)]
  puntos_2 = [ax2.scatter(x,y) for x,y in zip(X,Y)]

  for i in range(n):
    g = G[i] # Parametrizacion
    a = I[i][0] # Extremo izquierdo del intervalo
    b = I[i][1] # Extremo derecho del intervalo
    f = F[i] # Funcion en la cual evaluar
    color = Colores[i] # Color para graficar

    # Graficamos Gamma
    if continua:
      # Curva a mandar a funcion
      Gamma = curve_generator(g, a, b, continua)
      X0, Y0 = split(Gamma)

      # Cuva resultado de funcion
      Gamma2 = f(Gamma)
      X1, Y1 = split(Gamma2)

      plt.subplot(1, 2, 1)
      plt.plot(X0, Y0, c = 'red')
      plt.xlabel('Parte Real')
      plt.ylabel('Parte Imaginaria')
      plt.title('Gráfico de g(t)')
      plt.axis('equal')


      plt.subplot(1, 2, 2)
      plt.plot(X1, Y1, c =  color)
      plt.xlabel('Parte Real')
      plt.ylabel('Parte Imaginaria')
      plt.title('Gráfico de f(g(t))')
      plt.axis('equal')

      puntos_1[i].set_offsets([X0[i],Y0[i]]) # Coordenadas del i-th punto en ax1
      puntos_2[i].set_offsets([X1[i],Y1[i]]) # Coordenadas del i-th punto en ax2

    else:
      Gammas = curve_generator(g, a, b, continua)
      plt.subplot(1, 2, 1)
      for i in range(len(Gammas)):
        X0, Y0 = split(Gammas[i])
        plt.plot(X0, Y0, c = 'red')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginaria')
        plt.title('Gráfico de g(t)')
        plt.axis('equal')
        puntos_1[i].set_offsets([X0[i],Y0[i]]) # Coordenadas del i-th punto en ax1

      plt.subplot(1, 2, 2)
      for i in range(len(Gammas)):
        X1, Y1 = split(f(Gammas[i]))
        plt.plot(X1, Y1, c =  color)
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginaria')
        plt.title('Gráfico de f(g(t))')
        plt.axis('equal')
        puntos_2[i].set_offsets([X1[i],Y1[i]]) # Coordenadas del i-th punto en ax1


  # Creamos el slider para mover los puntos
  # El slider trabajara con porcentajes para los respectivos intervalos
  axcolor = 'lightgoldenrodyellow'
  ax_slider_point = plt.axes([0.15, 0, 0.65, 0.03], facecolor=axcolor)
  slider_point = Slider(ax_slider_point, '% Intervalo', 0, 100-epsilon, valinit = 0)

  def update(val):
    t = slider_point.val# Obtenemos el valor del invervalo del slider

    for i in range(n):
      g = G[i] # Parametrizacion
      a = I[i][0] # Extremo izquierdo del intervalo
      b = I[i][1] # Extremo derecho del intervalo
      f = F[i] # Funcion en la cual evaluar
      t_i = (t/(100-0))*(b-a) + a # El %t del intervalo [a,b] es t_i

      val_0 = g(t_i) # Punto correspondiente a g(t_i)
      x_0, y_0 = split(val_0)

      val_1 = f(val_0)# Obtenemos el punto que corresponde a f(g(t_i))
      x_1, y_1 = split(val_1)

      # Actualizamos los puntos i
      puntos_1[i].set_offsets([x_0,y_0])
      puntos_2[i].set_offsets([x_1,y_1])

    fig.canvas.draw_idle()

  slider_point.on_changed(update)

  ############################# Para generar animaciones
  if animacion:
    # Función para la animación
    def animate(intervalo):
        slider_point.set_val(intervalo)

    # Configuración de la animación
    ani = animation.FuncAnimation(fig, animate, frames=np.arange(a, b, 0.1), interval=100)
    #ani.save('animacion.gif', writer='pillow', fps=10) # Guarda la animacion como un gif en el directorio actual
  #############################
    
  plt.show()


# Generar cuadrado segun los limites lim_x, lim_y (ambos arreglos) de los ejex x,y respectivamente
def get_square_for_3d(lim_x, lim_y):
  x = np.linspace(lim_x[0], lim_x[1], 100)
  y = np.linspace(lim_y[0], lim_y[1], 100)
  X, Y = np.meshgrid(x, y)

  return X,Y

# Generar un disco en un con centro "c"(arreglo) y radio "r"
def get_disc_for_3d(r, c):
  x = np.linspace(c[0]-r, c[0]+r, 100)
  y = np.linspace(c[1]-r, c[1]+r, 100)
  X, Y = np.meshgrid(x, y)
  
  # Lo anterior es similar a get_square_for_3d, pero a continuacion se eliminaran los puntos fuera del disco

  # Creamos una matriz booleana que indica si el punto está dentro del disco
  distancia = np.sqrt((X-c[0])**2 + (Y-c[1])**2)

  en_disco = distancia <= r # Condicion para solo tomar los puntos dentro del disco

  
  # Filtrar los puntos que están dentro del disco
  X_disco = np.where(en_disco, X, np.nan)
  Y_disco = np.where(en_disco, Y, np.nan)

  return X_disco, Y_disco

# Graficadora que de R2 a R1
# Recibe como parametros a X y Y, los cuales formaran la base para graficar, ademas de la funcion compleja f, 
# junto a la funcion g que transforma los valores de f en reales
def graficadora_3d(X, Y, f, g, a = None):
    I = X + 1j*Y
    Z = g(f(I))

    if not (a is None): # Si se incluye curva de nivel
      fig = plt.figure(figsize=(15, 6))
      # Subplot 1 para surface
      ax1 = fig.add_subplot(1, 2, 1, projection='3d')
      surface = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
      ax1.set_title('g(f(z))')

      # Subplot 2 para surface_2
      ax2 = fig.add_subplot(1, 2, 2, projection='3d')
      #surface_2 = ax2.contour(X, Y, Z, levels=[a], offset=a, cmap=cm.coolwarm, alpha=1, linewidths=2.5)
      ax2.contour(X, Y, Z, levels=[a], offset=a, cmap=cm.coolwarm, alpha=1, linewidths=2.5)
      ax2.set_title(f'Curvas de nivel {a}')

      fig.colorbar(surface, ax=[ax1, ax2])

    else: # Si no se incluye curva de nivel
      fig = plt.figure(figsize=(7, 6))
      ax = fig.add_subplot(projection='3d')
      ax.set_title('g(f(z))')
      surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
      fig.colorbar(surface, ax=[ax])

    plt.show()

# Obtenemos la funcion de la curvatura Gaussiana K = -Laplaciano(E)/(2*E^{3}), donde E = 1 + u_x^{2} + u_y{2}
def get_K_funcion(u,v):
  u_x = sp.diff(u,x0,1)
  v_x = sp.diff(v,x0,1)
  E = sp.simplify(1 + u_x**2 + v_x**2)
  #E_y = sp.diff(E,y0,1) # Primera derivada de E respecto a y
  E_2y = sp.diff(E,y0,2) # Segunda derivada de E respecto a y

  #E_x = sp.diff(E,x0,1) # Primera derivada de E respecto a x
  E_2x = sp.diff(E,x0,2) # Segunda derivada de E respecto a x

  Laplaciano_E = sp.simplify(E_2x + E_2y)
  #print(f"Expresion de K sin simplificar: {-Laplaciano_E/(2*(E**3))}")
  K = sp.simplify(-Laplaciano_E/(2*(E**3)))
  #print(f"Expresion de K simplificada: {K}")
  def K_f(x,y):
    return K.subs({x0:x, y0:y}).evalf()
  return K_f, K

# Graficadora que de R2 a R1
# Recibe como parametros a X y Y, los cuales formaran la base para graficar, de las expresiones u, v 
# en terminos de x0, y0 de una funcion f = u + iv
def graficadora_3d_K(X, Y, u, v, a = None):
    K, K_text = get_K_funcion(u,v) # Obtenemos K
    K_vectorized = np.vectorize(K) # Vectorizamos para evaluar en X,Y
    Z = K_vectorized(X,Y)

    if not (a is None): # Si hay curva de nivel
      fig = plt.figure(figsize=(15, 6))

      # Subplot 1 para surface
      ax1 = fig.add_subplot(1, 2, 1, projection='3d')
      surface = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
      #ax1.set_title('K(x,y)')

      # Subplot 2 para surface_2
      ax2 = fig.add_subplot(1, 2, 2, projection='3d')
      ax2.contour(X, Y, Z, levels=[a], offset=a, cmap=cm.coolwarm, alpha=1, linewidths=2.5)
      ax2.set_title(f'Curvas de nivel {a}')

      fig.colorbar(surface, ax=[ax1, ax2])
    else: # Si no hay curva de nivel
      fig = plt.figure(figsize=(7, 6))
      ax = fig.add_subplot(projection='3d')
      #ax.set_title('K(x,y)')
      surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
      fig.colorbar(surface, ax=[ax])
    #plt.title(f'u={u},   v={v}')
    plt.title(f'K = {K_text}')

    plt.show()

############################### EXTRA: PUNTOS CRITICOS DE CURVATURAS DE POLINOMIOS ######################################

# Funcion que busca los puntos criticos de la curvatura de un polinomio de racies que estan en el arreglo
def find_critics(Raices):

  U, V = get_uv(Raices)

  # Primeras derivadas
  Ux = sp.simplify(sp.diff(U, x0))
  Uy = sp.simplify(sp.diff(U, y0))

  Vx = sp.simplify(sp.diff(V, x0))
  Vy = sp.simplify(sp.diff(V, y0))

  # Segundas derivadas
  Uxx = sp.simplify(sp.diff(Ux, x0))
  Uyy = sp.simplify(sp.diff(Uy, y0))

  Vxx = sp.simplify(sp.diff(Vx, x0))
  Vyy = sp.simplify(sp.diff(Vy, y0))

  # Terceras Derivadas
  Uxxx = sp.simplify(sp.diff(Uxx, x0))
  Uyyy = sp.simplify(sp.diff(Uyy, y0))

  Vxxx = sp.simplify(sp.diff(Vxx, x0))
  Vyyy = sp.simplify(sp.diff(Vyy, y0))

  # Calculo de Kx y Ky
  E = 1 + Ux**2 + Vx**2
  Kx = -4*(Uxx*Uxxx + Vxx*Vxxx)*E + 12*(Uxx**2 + Vxx**2)*(Ux*Uxx + Vx*Vxx)
  Ky = -4*(Uyy*Uyyy + Vyy*Vyyy)*E + 12*(Uyy**2 + Vyy**2)*(Uy*Uyy + Vy*Vyy)

  # Calculo de Kxx, Kyy, Kxy
  Kxx = sp.diff(Kx, x0)
  Kyy = sp.diff(Ky, y0)
  Kxy = sp.diff(Kx, y0)

  # Calculo de Hessiano
  Hess = Kxx*Kyy - Kxy**2

  # Ecuaciones
  eq_1 = sp.Eq(Kx, 0)
  eq_2 = sp.Eq(Ky, 0)

  # Calcular soluciones
  solutions = sp.solve([eq_1, eq_2], [x0, y0], dict = True, domain = sp.S.Reals)

  # Filtrar soluciones reales
  Soluciones_Reales = []
  Soluciones_Imaginarias = []
  Etiquetas = []

  cont = 0

  # Revisamos las soluciones, y se almacenan si son reales, y ademas se catalogan
  for sol in solutions:
    X = sol[x0].evalf()
    Y = sol[y0].evalf()

    print(f"Solucion {cont + 1}")
    cont += 1

    if not (X.is_real and Y.is_real):
      X,Y = transform_xy(X, Y)

    print(f"x = {X}, y = {Y}")
    print(f"z = {sp.simplify(X + I*Y)}")

    Hess_val = Hess.subs({x0: X, y0: Y})
    if not Hess_val.is_real:
      print("El Hessiano no es real\n")
      etiqueta = "No Real"
    elif Hess_val == 0:
      print("El Hessiano en el punto es 0\n")
      etiqueta = "Cero"
    elif Hess_val < 0:
      print("Es un punto silla\n")
      etiqueta = "Silla"
    elif Hess_val > 0 and Kxx.subs({x0: X, y0: Y}) > 0:
      print("Es un punto mínimo\n")
      etiqueta = "Mínimo"
    elif Hess_val > 0 and Kxx.subs({x0: X, y0: Y}) < 0:
      print("Es un punto máximo\n")
      etiqueta = "Máximo"


    if X.is_real and Y.is_real:
      Soluciones_Reales.append((X,Y))
      Etiquetas.append(etiqueta)
    else:
      Soluciones_Imaginarias.append((X,Y))


  print(f"Se encontraron {len(Etiquetas)} soluciones reales, de {cont} soluciones totales")
  return Soluciones_Reales, Etiquetas

# Un numero complejo es de la forma z = x + iy, con x,y reales.
# La siguiente funcion recibe x,y imaginarias y regresa x0,y0 reales tales que x + iy = x0 + iy0
def transform_xy(x, y):
  z0 = x + I*y
  x0, y0 = sp.re(z0), sp.im(z0)

  return x0, y0

# Funcion para obtener las partes u,v de un polinomio f = u + iv con un arreglo de Raices
def get_uv(Raices):
  # Generaremo el polinomio, suponiendo que se ingreso al menos una raiz
  f = 1
  for raiz in Raices:
    a, b = sp.re(raiz) , sp.im(raiz)
    f *= ( (x0 - a) + I*(y0 - b) )

  # Optenemos sus partes reales e imaginarias
  U = sp.simplify(sp.re(f))
  V = sp.simplify(sp.im(f))

  return U, V

# Funcion para graficar los puntos criticos, junto con sus etiquetas resultantes de find_critics
def graficar_criticos(datos, etiquetas = None):
    if not all(len(punto) == 2 for punto in datos):
        raise ValueError("Error: Cada elemento del array debe contener exactamente dos valores [x, y].")
    
    if etiquetas and len(etiquetas) != len(datos):
        raise ValueError("Error: El número de etiquetas debe coincidir con el número de puntos.")
    
    # Obtenemos las partes reales e imaginarias
    x = [punto[0] for punto in datos]
    y = [punto[1] for punto in datos]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='blue')

    
    # Agregar etiquetas 
    if etiquetas:
        for i, texto in enumerate(etiquetas):
            plt.text(x[i], y[i], texto, fontsize=9, ha='right', va='bottom', color='red')
            #print(f"{i}) {texto}")
        #print("\n")

        # Generar aristas basadas en las etiquetas
        for i in range(len(datos)):
            for j in range(i + 1, len(datos)):
                # Condiciones para conectar los puntos
                if (etiquetas[i] == "Silla" and etiquetas[j] == "Mínimo") or \
                   (etiquetas[i] == "Mínimo" and etiquetas[j] == "Silla") or \
                   (etiquetas[i] == "Máximo" or etiquetas[j] == "Máximo"):
                   #(etiquetas[i] == "Silla" and etiquetas[j] == "Silla") or \
                   #(etiquetas[i] == "Mínimo" and etiquetas[j] == "Mínimo"):
                    
                    # Dibujamos la arista
                    plt.plot([x[i], x[j]], [y[i], y[j]], 'k--', alpha=0.5)  # Línea punteada negra
                    
                    # Calculamos la distancia entre los vertices
                    #print(f"X: {x[j] - x[i]}\tY: {y[j] - y[i]}")
                    distancia = math.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2)
                    
                    # Punto medio entre vertices
                    x_medio = (x[i] + x[j]) / 2
                    y_medio = (y[i] + y[j]) / 2
                    
                    # Imprimir distancia
                    plt.text(x_medio, y_medio, f"{distancia:.6f}", fontsize=8, color='purple', ha='center', va='center')    

    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginaria')
    plt.title('Gráfico de Puntos Criticos')
    plt.grid(True)
    plt.show()
#############################################################################################
