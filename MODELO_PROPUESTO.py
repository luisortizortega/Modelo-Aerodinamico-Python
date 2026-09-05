##########################################
#Modelo aerodinámico cl y c_d vs Mach
#Interpolación con Makima o Pchip y teoría K-T, Ackeret, polar y Shapiro
##########################################

import os  #permite trabajar con carpetas y rutas
import sys  #permite cerrar el programa con mensaje de error cuando falla
import numpy as np  #importar la calculadora del programa y np es para abreviar
import matplotlib.pyplot as plt #importar lo necesario para las gráficas y también lo abrevio
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator #se importan ambos métodos
from matplotlib.ticker import MultipleLocator # sirve para los ejes de la gráfica


def cargar_datos_entrada(ruta_archivo):  #función para cargar los datos de entrada al modelo y si no se encuentra el archivo datos_entrada.txt salta un error y se para el programa
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encuentra el archivo {ruta_archivo}")
        sys.exit(1)
        
    datos_usuario = {} #creo tres contenedores vacíos, el de datos va a contener los datos que da el usuario y los demás almacenar los puntos opcionales
    puntos_opcionales_cl = [] #la diferencia entre paréntesis y corchetes es que paréntesis es un diccionario donde se guardan carpetas etiquetadas con un nombre dentro de la cual se guarda el valor de esa cosa que va en la carpeta por ejemplo carpeta TC y dentro 0.12
    puntos_opcionales_cd = [] #los corchetes son para crear una lista, se guardan valores uno tras otro en orden para luego pasarlos al interpolador
    
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:  #with asegura que lo que pase dentro de este bloque termine cerrando el archivo que se va a leer. El open(...) busca el archivo utilizando la dirección de la carpeta guardada en la variable ruta_archivo, o sea, busca datos_entrada en la carpeta del main creada llamada archivos_configuracion. r es para leer solamente
        for num_linea, linea in enumerate(archivo, 1): #este bucle va a leer el archivo de forma que toma el texto de la línea y le asigna un número de orden a esa línea que ha leído empezando por el 1, la primera línea será la 1. num_linea guarda el número entero actual mientras se recorre el archivo y linea guarda el texto correspondiente
            linea = linea.strip()#el strip es una función que elimina los espacios en blanco, tabulación, saltos de línea etc al principio y final de la frase, será útil para detectar cuando hay líneas vacías o detectar valores
            
            if not linea or linea.startswith('#'): #si no hay línea o empieza con un # porque es un comentario entonces 'continue' que lo que hace es ignorar el resto del bucle y vuelve a empezar pasando a la siguiente línea
                continue
                
            if '#' in linea: #esto lo que hace es que si hay un comentario al final de una línea elimina eso que hay comentado
                linea = linea[:linea.index('#')].strip() #el index recorre la línea letra a letra hasta encontrar # y el :linea lo quita. Toda la línea ya limpia se guarda en la variable linea
                
            if not linea: #si al quitar el comentario la línea queda vacía pues ignora el resto del bucle y empieza de nuevo en la siguiente línea 
                continue
                
            partes = [parte.strip() for parte in linea.split(',')] #esto permite dividir la línea de datos_entrada por la coma, entonces la línea TC, 0.12 queda guardada en 2 o 3 partes (por eso se llama partes) en el diccionario de datos_usuario o lista en el caso de las líneas opcionales
            if len(partes) < 2: #si no hay al menos 2 partes que sería el texto y el valor pues la línea se descarta
                continue
                
            clave = partes[0].upper() #aqui ya con la línea limpia y dividida en varias partes, toma la primera parte que es la que identifica si es TC, OPCIONAL_CD, MCRIT, etc y lo convierte a mayúsculas para que de igual que se ponga tc o Tc o TC. Le llamo clave porque es la parte clave para identificar el dato
            
            if clave == 'BORDE': #lee el tipo de borde
                datos_usuario['BORDE'] = partes[1].strip().upper()
                continue
                
            if clave == 'METODO': #permite elegir el método de interpolación, lo lee
                datos_usuario['METODO'] = [partes[1].strip().upper()]
                continue
                
            if clave in ['ARCHIVO_REF_CL', 'ARCHIVO_REF_CD']: #lee los archivos de comparación en caso de querer hacerla
                datos_usuario[clave] = [partes[1].strip()]
                continue
                
            try: #aquí lo que se va a hacer es que los textos como '0.12' se conviertan a numeros 0.12 y si no se puede convertir avisa de formato incorrecto
                valores = [float(p) for p in partes[1:] if p.strip()]
            except ValueError:
                print(f"Aviso: Línea {num_linea} ignorada porque el formato es incorrecto.")
                continue
                
            if clave == 'OPCIONAL_CL' and len(valores) >= 2:   #aquí voy a guardar el resto de líneas en la lista o diccionario correspondiente. Si se detecta opcional_cl y tiene al menos 2 valores, hago la siguiente línea
                puntos_opcionales_cl.append((valores[0], valores[1])) #coge los valores (el 0 es el primer valor que sería el mach y 1 es el segundo que sería el cl correspondiente) que haya y los guarda en una lista llamada puntos_opcionales_cl  sin borrar lo que había guardado antes
            elif clave == 'OPCIONAL_CD' and len(valores) >= 2: #igual pero para opcionales cd
                puntos_opcionales_cd.append((valores[0], valores[1])) #se guardan los valores de los puntos opcionales en una lista
            else:
                datos_usuario[clave] = valores #si no detecta ni BORDE ni OPCIONAL_CL ni OPCIONAL_CD guardo en el diccionario el valor con su etiqueta (clave) correspondiente
                
    datos_usuario['OPCIONAL_CL'] = puntos_opcionales_cl #aqui se añaden las listas de puntos opcionales (puntos_opcionales_cl y puntos_opcionales_cd) al diccionario con la etiqueta de si son opcionales_cl o cd
    datos_usuario['OPCIONAL_CD'] = puntos_opcionales_cd
    
    return datos_usuario #la función cargar_datos_entrada devuelve el diccionario completo


def cargar_polar(ruta_archivo): #se define una función para cargar los datos de entrada de la polar al modelo y si no se encuentra el archivo polar.txt salta un error y se para el programa
    if not os.path.exists(ruta_archivo): 
        print(f"Error: No se encuentra el archivo {ruta_archivo}")
        sys.exit(1)
        
    filas_datos = [] #se crea una lista para meter los datos de cada fila
    with open(ruta_archivo, 'r') as archivo: #de la misma manera que en la función anterior, lo que pase en este bucle se cerrará al final
        for linea in archivo:
            linea = linea.strip()
            if not linea or linea.startswith('#'):
                continue
                
            columnas = linea.split()
            if len(columnas) >= 3:
                try:
                    filas_datos.append([float(columnas[0]), float(columnas[1]), float(columnas[2])])
                except ValueError:
                    continue    #Hasta aquí lo que se ha hecho es leer cada línea, si no hay nada o es un comentario, pasa a la siguiente. Si en la línea detecta que hay al menos 3 columnas y son números las guarda como una fila en la lista de filas_datos pero solo las 3 primeras columnas que son alpha, cl y cd
                    
    if not filas_datos: #si no se ha guardado nada entonces pongo un aviso de error de que el archivo de la polar está mal
        print("Error: El archivo de la polar está mal.")
        sys.exit(1)
        
    datos_polar = np.array(filas_datos)
    polar_orden = np.argsort(datos_polar[:, 0])
    datos_polar = datos_polar[polar_orden]    #estas tres líneas convierten la lista de filas_datos en una matriz. Con argsort se ordena la primera columna de alpha de menor a mayor y se aplica esa ordenación a toda la matriz
    
    alpha_unicos = np.concatenate([[True], np.diff(datos_polar[:, 0]) > 1e-6])
    datos_polar = datos_polar[alpha_unicos]   #se eliminan las filas duplicadas donde alpha es prácticamente igual (diferencia menor a 0,000001 grados) ya que airfoiltools tiene dos filas con alpha 0 y luego lo guarda en datos_polar
    
    return datos_polar[:, 0], datos_polar[:, 1], datos_polar[:, 2] #la función devuelve las tres columnas por separado, ángulo de ataque, cl y cd


def cargar_coordenadas(ruta_archivo): #función para cargar las coordenadas del perfil y si no se encuentra el archivo perfil.txt se cierra el programa
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encuentra el archivo {ruta_archivo}")
        sys.exit(1)
        
    puntos = [] #se crea una lista para los puntos de las coordenadas
    with open(ruta_archivo, 'r') as archivo: #igual que en las funciones anteriores lo que haya en el bucle se cerrará
        for linea in archivo:
            linea = linea.strip()
            if not linea or linea.startswith('#'): #se detecta si no hay línea o comentario y se salta
                continue
                
            columnas = linea.split()
            if len(columnas) >= 2:
                try:
                    x = float(columnas[0])
                    y = float(columnas[1])
                    if x > 2.0:
                        x = x / 100.0   #esto se hace por si la coordenada no viene en fracción de cuerda y viene en porcentaje ya que hay algunos formatos así
                        y = y / 100.0
                    puntos.append([x, y])
                except ValueError:
                    continue    #aqui se detectan las columnas y si hay 2 o más columnas, se guarda el valor en las variable x e y y estas, además, en forma de punto en la lista de puntos
                    
    if len(puntos) < 4:    #si no hay al menos 4 puntos saltará un aviso de que no hay las suficientes coordenadas para definir el perfil
        print("Error: No hay suficientes coordenadas para definir el perfil.")
        sys.exit(1)
        
    matriz_coords = np.array(puntos)   #se convierten todos los puntos que hay en la lista en una matriz y luego se separa en 2, el extradós que tienen coordenada 'y' positiva y el intradós con coordenada 'y' negativa
    coord_x = matriz_coords[:, 0] #los dos puntos indica todas las filas, o sea, que se recorra todas las filas de la matriz y empiece por la columna 0, o sea, la primera que es la de X
    coord_y = matriz_coords[:, 1] #con estas líneas de código consigo 2 listas que solo tienen las coordenadas de x y de y
    mascara_extrados = coord_y >= 0
    mascara_intrados = coord_y < 0
    
    if np.sum(mascara_extrados) < 2 or np.sum(mascara_intrados) < 2:    #estas líneas son para verificar que hay al menos 2 puntos en el extradós y 2 en el intradós
        print("Error: Imposible separar el extradós del intradós en las coordenadas.")
        sys.exit(1)
        
    x_extrados = coord_x[mascara_extrados]  #estas 4 líneas revisan los puntos x e y de coord_x y coord_y verificando si pertenecen
    y_extrados = coord_y[mascara_extrados]  #a extradós o intradós con lo que se llama mascara_extrados y mascara_intrados.
    x_intrados = coord_x[mascara_intrados]  #luego lo guarda en una nueva lista para así tener por separados las coordenadas x e y
    y_intrados = coord_y[mascara_intrados]  #tanto de extradós como de intradós. Aunque la condición de mascara es aplicada a coord_y, se pone también a aplicar a x porque mascara saca una lista que pone true o false entonces ya va asociada la x y la y como un pack
    
    coord_sup = np.argsort(x_extrados)  #como pueden estar desordenados los datos de las coordenadas, argsort ordena los valores y 
    coord_inf = np.argsort(x_intrados)  #se guardan en coord_sup y coord_inf
    
    return x_extrados[coord_sup], y_extrados[coord_sup], x_intrados[coord_inf], y_intrados[coord_inf] #devuelve las listas de x e y
    #tanto de extradós como de intradós filtradas y ordenadas mediante [coord_sup] y [coord_inf]. Como tienen estos corchetes en x y en y, quedan en parejas los puntos (X,Y)

def ecuacion_karman_tsien(cl0, Mach): #función para aplicar K-T en subsónico, se le pasa el valor de cl0 y Mach correspondiente
    beta_KT = np.sqrt(max(1.0 - Mach**2, 1e-10)) #el max evita que beta sea 0 si mach llega a 1 y beta es pequeño
    denominador = beta_KT + (Mach**2 / (2.0 * (1.0 + beta_KT))) * cl0
    
    if abs(denominador) > 1e-10:  #esto es una comprobación simplemente para evitar la división por 0 
        return cl0 / denominador
    else:
        return cl0 / beta_KT #si el denominador es muy cercano a 0 se aplica PG


def ecuacion_ackeret_cl(Mach, alpha_ef_rad): #función para ackeret en supersónico para cl y cogerá el mach y alpha efectivo en radianes
    if Mach <= 1.0: #el if es para que cuando mach sea menor a 1, no devuelve nada porque si no daría fallo por la raíz negativa
        return np.nan
    return (4.0 * alpha_ef_rad) / np.sqrt(Mach**2 - 1.0) #ecuación de ackeret para cl supersónico


def ecuacion_cd_inducido(Mach, alpha_ef_rad): #función para cd inducido que usara el mach y el ángulo de ataque efectivo en radianes
    if Mach <= 1.0:
        return np.nan
    return (4.0 * alpha_ef_rad**2) / np.sqrt(Mach**2 - 1.0) #ecuación de cd inducido de shapiro evitando que mach sea menor a uno para evitar errores


def integral_cd_espesor(Mach, integral_cdesp): #función para el cd espesor que usara el mach y la integral que forma parte de la ecuación y que se calcula aparte
    if Mach <= 1.0:
        return np.nan
    return (2.0 / np.sqrt(Mach**2 - 1.0)) * integral_cdesp #ecuación del cd espesor de shapiro evitando error por mach menor a 1


def calcular_integral_cdesp(xs, ys, xi, yi): #función para calcular la integral de cd espesor, usará las coordenadas de x e y extraídas en la función de cargar coordenadas
    gradiente_sup = np.gradient(ys, xs) #en estas dos líneas se estima la pendiente en cada punto y se guardan en gradiente_sup y gradiente_inf
    gradiente_inf = np.gradient(yi, xi)
    
    integral_sup = np.trapz(gradiente_sup**2, xs) #se integran las pendientes al cuadrado
    integral_inf = np.trapz(gradiente_inf**2, xi)
    
    return integral_sup + integral_inf #devuelve la suma de las integrales superior e inferior que es la integral completa


def ecuacion_polar_cd(cl, cd_minimo, factor_k): #función para el cd en subsónico mediante la ecuación de la polar en la que se usa el cl, el cd minimo y el factor k
    return cd_minimo + factor_k * cl**2 #cl proviene del calculado en KT, cd minimo proviene del archivo de la polar y factor k también


TOLERANCIA_MACH_CRITICO = 0.03 #esto sirve para que el código detecte como mach crítico, el valor especificado por el usuario +- una toleracia, es una recomendación para evitar errores
TOLERANCIA_MACH_12 = 0.02

def comprobar_puntos_cercanos(puntos_opcionales, mach_referencia, tolerancia=TOLERANCIA_MACH_CRITICO): #función que recorre todos los puntos opcionales y devuelve True si alguno está suficientemente cerca del Mach de referencia. El guión bajo _ significa que el valor del coeficiente no interesa para esta comprobación, solo el Mach
    return any(abs(M - mach_referencia) < tolerancia for M, _ in puntos_opcionales) #devuelve si hay algun punto dentro de la tolerancia o no


def verificar_zona_supersonica(puntos_opcionales): #función que comprueba que haya un punto cerca de M=1.2 y que haya al menos otro por encima de M=1.25. Solamente si ambas se cumplen devuelve True y se activa la zona supersónica sin ecuación
    tiene_punto_12 = any(abs(M - 1.2) < TOLERANCIA_MACH_12 for M, _ in puntos_opcionales)
    tiene_puntos_mayores = any(M > 1.25 for M, _ in puntos_opcionales)
    return tiene_punto_12 and tiene_puntos_mayores #sirve para ver si hay puntos opcionales en la zona supersónica o no, se usará después por ejemplo en perfiles romos


def extraer_opcionales_supersonicos(puntos_opcionales): #función para extraer los puntos opcionales supersónicos
    puntos_supersonicos = [] #lista para guardar los opcionales supersónicos
    for M, c in puntos_opcionales: #como los puntos van así: (mach, coef), el mach se asigna a M y el coeficiente a c
        if M >= 1.2:
            puntos_supersonicos.append((M, c)) #si el mach es superior a 1.2 entonces se guarda el punto en la lista 
    return puntos_supersonicos #se devuelve la lista filtrada de puntos supersónicos


VARIABLES_OBLIGATORIAS = ['TC', 'ALPHA', 'DESVIACION', 'BORDE', 'M_CLMAX', 'M_CLMIN', 'M_CLREC', 'MCRIT', 'MDD', 'M_CDMAX'] #esto son las variables obligatorias que se piden al susario y que deben ser proporcionadas, si falta alguna se para el programa

def validar_modelo(datos, tc, alpha, cl0, cd_min, factor_k, alpha_ef_rad, integral_cdesp, es_borde_romo): #función que sirve para ver si falta algo, aplicar las condiciones del modelo, verificar que los datos tengan sentido.
    variables_faltantes = [var for var in VARIABLES_OBLIGATORIAS if var not in datos] #se crea una lista en la que se irá guardando del diccionario datos las variables obligatorias que NO estan, si está vacía entonces está correcto, si no, queda dentro lo que falta
    if variables_faltantes:
        print(f"Error, faltan las siguientes variables: {', '.join(variables_faltantes)}") #se muestran las variables que faltan si la lista NO esta vacía y paro el programa
        sys.exit(1) #join sirve para coger los elementos de variables_faltantes y meterlos en una frase separados por , y espacio
        
    if datos['BORDE'] not in ('AFILADO', 'ROMO'): #se obliga a que el dato de la lista denominado borde, sea afilado o romo y si no lo es muestro el error y se para el programa
        print("Error: El tipo de borde debe ser AFILADO o ROMO.")
        sys.exit(1)

    mach_critico = datos['MCRIT'][0] #en estas líneas se extraen del diccionario todos los valores de los puntos predefinidos
    mach_cl_maximo = datos['M_CLMAX'][0] #0 es el numero de mach y 1 es el valor correspondiente, como se ha explicado previamente
    valor_cl_maximo = datos['M_CLMAX'][1] #ya que 0 es la primera posición y 1 es la segunda según se han ido guardando
    mach_cl_minimo = datos['M_CLMIN'][0]
    valor_cl_minimo = datos['M_CLMIN'][1]
    mach_cl_recuperacion = datos['M_CLREC'][0]
    valor_cl_recuperacion = datos['M_CLREC'][1]
    mach_divergencia = datos['MDD'][0]
    valor_cd_divergencia = datos['MDD'][1]
    mach_cd_maximo = datos['M_CDMAX'][0]
    valor_cd_maximo = datos['M_CDMAX'][1]

    lista_errores = [] #aquí se crean 2 listas donde se acumularán los errores que puedan ocurrir y los avisos
    lista_avisos = []
    
    es_perfil_grueso = tc > 0.12  #se evalúa si el perfil es grueso o no
    es_alpha_limite = abs(alpha) > 10.0 #se evalúa si alpha está por encima o debajo del límite fijado

    if es_perfil_grueso or es_alpha_limite: #aquí lo que se hace es crear una lista donde se guardará, en caso de cumplirse que el perfil sea grueso o de alhpa fuera de los limites establecidos, un aviso con el motivo correspondiente
        motivo_anulacion = [] #lista para guardar el motivo de anular la teoría
        if es_perfil_grueso:
            motivo_anulacion.append(f"espesor t/c = {tc:.3f} > 0.12")  #se anula la teoría si el tc del perfil es mayor a 0.12
        if es_alpha_limite:
            motivo_anulacion.append(f"ángulo |α| = {abs(alpha):.1f}° > 10°") #se anula la teoría si alpha está fuera de +-10 grados
            
        texto_motivo = ' y '.join(motivo_anulacion) #en estas dos líneas guardo en texto_motivo el motivo de anular las ecuaciones y luego lo meto en la lista de avisos en orden para que se muestre
        lista_avisos.append(f"Perfil fuera de límites teóricos ({texto_motivo}). Las ecuaciones teóricas se desactivarán.")
        
        opcionales_cl = datos['OPCIONAL_CL'] #se extraen los datos de los puntos opcionales del diccionario
        opcionales_cd = datos['OPCIONAL_CD']
        
        if not comprobar_puntos_cercanos(opcionales_cl, mach_critico): #si el perfil es grueso o alpha fuera del límite, entonces Mcrit es obligatorio y su valor del coeficiente asociado, los compruebo y si no están se para y aviso de ello
            lista_errores.append(f"Error: Se requiere punto de cl para el Mcrit={mach_critico:.2f}.") #digo lo que falta, se mostrará luego cuando llame a esta lista
        if not comprobar_puntos_cercanos(opcionales_cd, mach_critico):
            lista_errores.append(f"Error: Se requiere punto de cd para el Mcrit={mach_critico:.2f}.") #digo lo que falta
            
        if not verificar_zona_supersonica(opcionales_cl): #compruebo si en la zona supersónica hay opcionales de cd y cl, si no hay pues aviso donde se cortarán las curvas
            lista_avisos.append("La curva de cl finalizará en la zona transónica al faltar datos supersónicos.")
        if not verificar_zona_supersonica(opcionales_cd):
            lista_avisos.append("La curva de cd finalizará en la zona transónica al faltar datos supersónicos.")

    if es_borde_romo and not es_perfil_grueso and not es_alpha_limite: #si el perfil es romo y no es grueso ni fuera del límite de alpha, entonces se guarda un aviso de que el perfil es romo y la integral para cdespesor no es aplicable
        lista_avisos.append("Perfil con borde romo, ecuación de cd_esp no aplicable.")
        if not verificar_zona_supersonica(datos['OPCIONAL_CD']): #si no hay los puntos en supersónico necesarios definidos en el modelo, o sea, el punto en mach 1.2 y otro en supersónico, la curva en supersónico no se dibujará y acabará en mach cd_max
            lista_avisos.append("Faltan datos supersónicos para el perfil romo. La curva de cd terminará en el pico de resistencia.")

    if abs(np.degrees(alpha_ef_rad)) < 0.5 and not es_perfil_grueso and not es_alpha_limite: #si el ángulo de ataque es muy pequeño, la ecuación de ackeret generará datos muy pequeños de cl que pueden ser poco representativos
        lista_avisos.append("Aviso: Ángulo de ataque muy pequeño (|αef| < 0.5°) por lo que los resultados supersónicos pueden ser poco representativos.") #si se cumple, se guarda otro aviso

    #acotación de los valores definidos en el modelo y en caso de cumplirse se guarda un error o aviso y se parará el programa o se emite aviso
    #LÍMITES AMPLIADOS 20%
    if not (0.4 <= mach_critico <= 1.08):
        lista_avisos.append(f"Aviso: El Mach crítico ({mach_critico}) está fuera del rango (entre 0.4 y 1.08).")
        
    if not (0.52 <= mach_cl_maximo <= 1.14):
        lista_avisos.append(f"Aviso: El Mach del cl máximo ({mach_cl_maximo}) está fuera del rango (entre 0.52 y 1.14).")
        
    #ORDEN DE LOS PUNTOS
    if not (mach_critico < mach_cl_maximo < mach_cl_minimo < mach_cl_recuperacion < 1.2):
        lista_errores.append("Error: Mcrit < M_clmax < M_clmin < M_clrec < 1.2 no se cumple.")
        
    if not (mach_critico < mach_divergencia < mach_cd_maximo < 1.2):
        lista_errores.append("Error: Mcrit < Mdd < M_cdmax < 1.2 no se cumple.")
        
    #LÍMITES AMPLIADOS 20%
    if valor_cl_maximo > 2.4:
        lista_avisos.append(f"Aviso: El cl máximo ({valor_cl_maximo}) supera el límite esperado de 2.4.")
        
    if valor_cl_minimo < -2.4:
        lista_avisos.append(f"Aviso: El cl mínimo ({valor_cl_minimo}) es inferior al límite esperado de -2.4.")
        
    #ORDEN DE LOS PUNTOS
    if not (valor_cl_minimo < valor_cl_maximo):
        lista_errores.append("Error: Incoherencia en los datos, el cl mínimo debe ser menor que el cl máximo.")
        
    if not (valor_cl_minimo < valor_cl_recuperacion < valor_cl_maximo):
        lista_errores.append("Error: El cl de recuperación debe estar entre el valle y el pico (cl mínimo y máximo).")

    #acotación de los valores definidos en el modelo contra la teoría, si no se cumple se guarda un error o aviso que se mostrará luego y se para el programa o se emite aviso
    if not es_perfil_grueso and not es_alpha_limite:
        teoria_kt = ecuacion_karman_tsien(cl0, mach_critico) #se coge el valor de K-T en mach crítico para ver si se cumple la condición del cl maximo
        if valor_cl_maximo <= teoria_kt:
            lista_errores.append(f"Error: cl max ({valor_cl_maximo:.4f}) debe ser mayor que el valor de Kármán-Tsien ({teoria_kt:.4f}).")
            
        teoria_ackeret = ecuacion_ackeret_cl(1.2, alpha_ef_rad) #se coge el valor de cl en mach 1.2 para ver si se cumple la condición para el cl de recuperacion
        if np.isfinite(teoria_ackeret) and valor_cl_recuperacion < teoria_ackeret: #el isfinite comprueba si el valor de ackeret es finito o es un NaN o infinito que daría errores
            lista_avisos.append(f"Aviso: cl rec ({valor_cl_recuperacion:.4f}) debería estar por encima del valor calculado en mach 1.2 = ({teoria_ackeret:.4f}) según la teoría.")

    #LÍMITE AMPLIADO 20%
    if valor_cd_maximo > 0.36:
        lista_avisos.append(f"Aviso: El cd máximo ({valor_cd_maximo}) es excesivamente alto (el límite ampliado es 0.36).")
        
    #ORDEN DE LOS PUNTOS
    if not (cd_min <= valor_cd_divergencia < valor_cd_maximo):
        lista_errores.append("Error: cd_min <= cd_Mach_divergencia < cd_maximo no se cumple.")

    if not es_perfil_grueso and not es_alpha_limite and not es_borde_romo:
        resistencia_supersonica = cd_min + ecuacion_cd_inducido(1.2, alpha_ef_rad) + integral_cd_espesor(1.2, integral_cdesp)
        if np.isfinite(resistencia_supersonica) and valor_cd_maximo <= resistencia_supersonica:
            lista_errores.append(f"Error: cd max ({valor_cd_maximo:.5f}) debe superar el valor supersónico teórico calculado ({resistencia_supersonica:.5f}).")

    if es_perfil_grueso or es_alpha_limite or es_borde_romo:
        for m_opcional, cd_opcional in datos['OPCIONAL_CD']:
            #se avisa si en la zona supersónica el cd es mayor que el pico transónico
            if m_opcional >= 1.15 and cd_opcional >= valor_cd_maximo: 
                lista_avisos.append(f"Aviso: El dato opcional en M={m_opcional} (cd={cd_opcional}) es mayor o igual que el pico máximo de resistencia (cd_max={valor_cd_maximo}).")
                
    if es_perfil_grueso or es_alpha_limite or es_borde_romo:
        for m_opcional, cl_opcional in datos['OPCIONAL_CL']: 
            #se avisa si en la zona supersónica el cl es mayor que el de recuperación
            if m_opcional >= 1.15 and cl_opcional >= valor_cl_recuperacion:
                lista_avisos.append(f"Aviso: El dato opcional en M={m_opcional} (cl={cl_opcional}) es mayor o igual que el cl de recuperación (cl_rec={valor_cl_recuperacion}).")

    #COMPROBACIÓN DE DESVIACIÓN DE PUNTOS DEL USUARIO RESPECTO A TEORÍA
    if 'DESVIACION' in datos and len(datos['DESVIACION']) > 0: #se ve la desviación puesta por el usuario y luego se guarda en una variable
        desviacion_max = datos['DESVIACION'][0]
        
        if not es_perfil_grueso and not es_alpha_limite:
            #comprobación para cl
            for m_opc, cl_opc in datos['OPCIONAL_CL']: #se coge el punto que ha pasado el usuario para cl y se ve si se encuentra en la zona subsónica o supersónica
                cl_teo = None
                if m_opc <= mach_critico:
                    cl_teo = ecuacion_karman_tsien(cl0, m_opc)
                elif m_opc >= 1.2:
                    cl_teo = ecuacion_ackeret_cl(m_opc, alpha_ef_rad)
                
                if cl_teo is not None and abs(cl_teo) > 1e-6:
                    error_relativo = (abs(cl_opc - cl_teo) / abs(cl_teo)) * 100.0 #se calcula el error relativo entre teoría y opcional
                    if error_relativo > desviacion_max: #si el error relativo es mayor a la desviación impuesta por el usuario se comunica
                        lista_avisos.append(f"Aviso: El punto opcional cl en M={m_opc} ({cl_opc}) difiere un {error_relativo:.1f}% de la teoría ({cl_teo:.3f}), superando el límite del {desviacion_max}%.")
            
            #comprobación para cd, funciona igual que para el coeficiente de sustentación
            for m_opc, cd_opc in datos['OPCIONAL_CD']:
                cd_teo = None
                if m_opc <= mach_critico:
                    cl_teo_k = ecuacion_karman_tsien(cl0, m_opc)
                    cd_teo = ecuacion_polar_cd(cl_teo_k, cd_min, factor_k)
                elif m_opc >= 1.2 and not es_borde_romo:
                    cd_teo = cd_min + ecuacion_cd_inducido(m_opc, alpha_ef_rad) + integral_cd_espesor(m_opc, integral_cdesp)
                    
                if cd_teo is not None and abs(cd_teo) > 1e-6:
                    error_relativo = (abs(cd_opc - cd_teo) / abs(cd_teo)) * 100.0
                    if error_relativo > desviacion_max:
                        lista_avisos.append(f"Aviso: El punto opcional cd en M={m_opc} ({cd_opc}) difiere un {error_relativo:.1f}% de la teoría ({cd_teo:.4f}), superando el límite del {desviacion_max}%.")

    for aviso in lista_avisos: #aquí lo que se hace es mostrar los avisos de la lista de avisos donde se han ido guardando en el terminal
        print(aviso)
        
    if lista_errores:    #aquí lo que se hace es indicar que si hay algo (no vacío) en la lista de errores, se avisa de que el programa se para, se muestra el motivo por el terminal guardado en la lista y se para el programa
        print("\nEl proceso ha sido abortado debido a errores de los datos:")
        for error in lista_errores:
            print(f"  -> {error}")
        sys.exit(1)

    return es_perfil_grueso, es_alpha_limite #devuelve si el perfil es grueso o no y si el alpha está fuera de los límites o no, para usarse posteriormente


def generar_puntos_cl_base(datos, cl0, alpha_ef_rad, es_grueso, es_limite_alpha): #función para crear la lista de puntos de cl de la curva teórica (base)
    mach_critico = datos['MCRIT'][0] #se extraen los datos de los puntos de cl obligatorios del diccionario
    mach_max = datos['M_CLMAX'][0]
    cl_max = datos['M_CLMAX'][1]
    mach_min = datos['M_CLMIN'][0]
    cl_min = datos['M_CLMIN'][1]
    mach_rec = datos['M_CLREC'][0]
    cl_rec = datos['M_CLREC'][1]
    
    puntos_mach = [] #se crean dos listas para guardar el mach y coeficiente cl correspondiente
    puntos_coeficientes = []
    
    if not es_grueso and not es_limite_alpha: #se comprueba que el perfil sea delgado y alpha dentro de los límites
        rango_subsonico = np.linspace(0.01, mach_critico, 45) #se crean entre mach 0.01 y mach crítico 45 puntos y se guardan en rango_subsonico
        for m_actual in rango_subsonico: #para cada mach de los creados, se guardan en puntos_mach y con ese mach se llama a la ecuación
            puntos_mach.append(m_actual) #de KT para calcular el cl correspondiente y se guarda en la lista de coeficientes
            puntos_coeficientes.append(ecuacion_karman_tsien(cl0, m_actual))
        rango_supersonico = np.linspace(1.20, 1.80, 45)
        for m_actual in rango_supersonico: #la forma de proceder para supersónico es igual que para subsónico
            puntos_mach.append(m_actual)
            puntos_coeficientes.append(ecuacion_ackeret_cl(m_actual, alpha_ef_rad))
    else: #si es grueso o fuera del límite del ángulo de ataque definido, el mach será 0.01 (el punto de mach 0) y el valor del coeficiente
        puntos_mach.append(0.01) #será el del usuario sacado de la polar (mach 0)
        puntos_coeficientes.append(cl0)
        
    puntos_mach.extend([mach_max, mach_min, mach_rec]) #esto añade a las listas los puntos de transónico, el cl max, cl min y cl rec
    puntos_coeficientes.extend([cl_max, cl_min, cl_rec])
    
        #valor de cl en Mcrit según la ecuación teórica, para poder marcarlo como punto
    if not es_grueso and not es_limite_alpha:
        cl_en_mcrit = ecuacion_karman_tsien(cl0, mach_critico)
    else:
        cl_en_mcrit = cl0  #si no hay teoría, se usa el valor a Mach0 como aproximación

    predefinidos_mach = np.array([0.01, mach_critico, mach_max, mach_min, mach_rec]) #en estas líneas se guardan el valor del mach, del coeficiente correspondiente y se le pone una etiqueta
    predefinidos_valor = np.array([cl0, cl_en_mcrit, cl_max, cl_min, cl_rec])
    predefinidos_nombre = ['c_l,0', 'M_crit', 'M_clmax', 'M_clmin', 'M_clrec']
    
    return (np.array(puntos_mach), np.array(puntos_coeficientes),
            predefinidos_mach, predefinidos_valor, predefinidos_nombre, mach_critico) #la función devuelve las listas como un array porque son vectores que son con los que trabajan luego matplotlib y el interpolador


def generar_puntos_cd_base(datos, cl0, cd_minimo, factor_k, alpha_ef_rad, integral_cdesp, es_grueso, es_limite_alpha, es_romo, permite_cd_teorico, opcionales_supersonicos_romo):
    mach_critico = datos['MCRIT'][0] #esta función sirve para lo mismo que la anterior pero para el coeficiente de resistencia
    mach_divergencia = datos['MDD'][0] #en este caso se extraen del diccionario los datos correspondientes al cd obligatorios
    cd_divergencia = datos['MDD'][1]
    mach_maximo = datos['M_CDMAX'][0]
    cd_maximo = datos['M_CDMAX'][1]
    
    puntos_mach = [] #se crean dos listas para guardar el mach y coeficiente cd correspondiente
    puntos_coeficientes = []

    if not es_grueso and not es_limite_alpha: #comprueba que no sea grueso ni fuera del límite
        rango_subsonico = np.linspace(0.01, mach_critico, 45) #se crean entre mach 0.01 y mach critico 45 puntos y se guardan en rango_subsonico
        for m_actual in rango_subsonico: #para cada mach se calcula el cl y con eso, el cd mínimo y el factor k se calcula el cd
            valor_cl = ecuacion_karman_tsien(cl0, m_actual) #llamando a la ecuación de la polar. Ambos puntos se guardan en las listas
            puntos_mach.append(m_actual)
            puntos_coeficientes.append(ecuacion_polar_cd(valor_cl, cd_minimo, factor_k))
        if permite_cd_teorico: #permite cd teórico viene del main de una comprobación de las condiciones, o sea, que no sea romo ni grueso ni fuera de los límites de alpha
            rango_supersonico = np.linspace(1.20, 1.80, 45) #si es posible el cd teórico entonces se calcula cd total en supersónico con el mach actual
            for m_actual in rango_supersonico: #se llaman a las funciones para el cd inducido y de espesor y, además, con el cd min de los datos 
                cd_inducido = ecuacion_cd_inducido(m_actual, alpha_ef_rad) #a mach 0, se calcula la resistencia total
                cd_espesor = integral_cd_espesor(m_actual, integral_cdesp)
                resistencia_total = cd_minimo + cd_inducido + cd_espesor
                puntos_mach.append(m_actual) #se guardan los mach y cd correspondientes en las listas
                puntos_coeficientes.append(max(resistencia_total, cd_minimo)) #se elige el mayor entre el cd calculado y el cdmin porque no tiene sentido que saliese un cd inferior al mínimo
    else: #si es grueso o excede el límite de alpha entonces el mach será 0.01 y el cd sera el correspondiente a mach 0, o sea, el mínimo
        puntos_mach.append(0.01)
        puntos_coeficientes.append(cd_minimo)

    puntos_mach.extend([mach_divergencia, mach_maximo]) #se añaden a la lista los puntos transónicos dados por el ususario en Mdd y Mcdmax
    puntos_coeficientes.extend([cd_divergencia, cd_maximo])
    
        #valor de cd en Mcrit
    if not es_grueso and not es_limite_alpha:
        cl_temp_mcrit = ecuacion_karman_tsien(cl0, mach_critico)
        cd_en_mcrit = ecuacion_polar_cd(cl_temp_mcrit, cd_minimo, factor_k)
    else:
        cd_en_mcrit = cd_minimo

    predefinidos_mach = np.array([0.01, mach_critico, mach_divergencia, mach_maximo]) #en estas líneas se guardan el valor del mach, del coeficiente correspondiente y se le pone una etiqueta
    predefinidos_valor = np.array([cd_minimo, cd_en_mcrit, cd_divergencia, cd_maximo])
    predefinidos_nombre = ['c_d,0', 'M_crit', 'M_dd', 'M_cdmax']
    
    return (np.array(puntos_mach), np.array(puntos_coeficientes),
            predefinidos_mach, predefinidos_valor, predefinidos_nombre) #la función devuelve las listas como un array porque son vectores que son con los que trabajan luego matplotlib y el interpolador


def generar_puntos_cl_con_usuario(datos, cl0, puntos_usuario, mach_critico, cl_en_mcrit): #función para crear los puntos de cl a partir de lo que introduzca el usuario
    if not puntos_usuario:
        return None, None #si no hay puntos dados por el usuario devuelve None para indicar que no habrá curva con puntos opcionales
        
    mach_max = datos['M_CLMAX'][0] #se extraen los puntos obligatorios del diccionario
    cl_max = datos['M_CLMAX'][1]
    mach_min = datos['M_CLMIN'][0]
    cl_min = datos['M_CLMIN'][1]
    mach_rec = datos['M_CLREC'][0]
    cl_rec = datos['M_CLREC'][1]
    
    lista_mach = [0.01] #lista que empieza con mach 0 (0.01) y el valor de cl correspondiente dado por el usuario mediante el txt de la polar
    lista_coef = [cl0]
    
    for mach, coef in puntos_usuario:
        lista_mach.append(mach)   #para cada mach y coeficiente asociado dentro de la lista de opcionales de cl, se guardan en 2 listas, una para el mach y otra para el coeficiente cl
        lista_coef.append(coef)
        
    #se añade también Mcrit como punto obligatorio, igual que en la curva teórica, para que la curva de usuario también pase por ese punto
    lista_mach.append(mach_critico)
    lista_coef.append(cl_en_mcrit)

    lista_mach.extend([mach_max, mach_min, mach_rec]) #a estas listas se añaden los puntos obligatorios
    lista_coef.extend([cl_max, cl_min, cl_rec])
    
    return np.array(lista_mach), np.array(lista_coef) #la función devuelve las listas como un array con los puntos opcionales que ha proporcionado el usuario y los obligatorios para la curva de cl 


def generar_puntos_cd_con_usuario(datos, cd_minimo, puntos_usuario, es_romo, mach_critico, cd_en_mcrit): #función para crear los puntos de cd a partir de lo que introduzca el usuario
    if not puntos_usuario: #si no hay puntos dados por el ususario para cd entonces devuelve None y no habrá curva con puntos opcionales
        return None, None

    mach_divergencia = datos['MDD'][0] #se extraen los puntos obligatorios de cd para crear la curva
    cd_divergencia = datos['MDD'][1]
    mach_maximo = datos['M_CDMAX'][0]
    cd_maximo = datos['M_CDMAX'][1]

    lista_mach = [0.01] #dos listas que comienzan con el valor de cd a mach aproximadamente 0
    lista_coef = [cd_minimo]
    for mach, coef in puntos_usuario:#para cada mach y coeficiente asociado dentro de la lista de opcionales de cd, se guardan en 2 listas, una para el mach y otra para el coeficiente cd
        lista_mach.append(mach) #el append añade un solo elemento al final de la lista
        lista_coef.append(coef)
        
    #se añade también Mcrit 
    lista_mach.append(mach_critico)
    lista_coef.append(cd_en_mcrit)

    lista_mach.extend([mach_divergencia, mach_maximo]) #todos los valores de mach y cd asociado se guardan en las listas correspondientes
    lista_coef.extend([cd_divergencia, cd_maximo]) #el extend añade varios elementos de golpe, por ejemplo añade una lista en vez de un punto

    return np.array(lista_mach), np.array(lista_coef) #la función devuelve las listas como un array con los puntos de mach y cd asociado dados por el usuario


def preparar_nodos_interpolacion(vector_mach, vector_coeficientes, tolerancia_distancia=5e-4): #función que prepara los puntos o nodos que va a usar el método de interpolación
    indices_ordenados = np.argsort(vector_mach) #argsort crea una ordenación de los valores de mach (lista) desde el más bajo al más alto y se guardan en una lista
    vector_mach = vector_mach[indices_ordenados] #aquí en estas 2 líneas guardo 2 vectores ya con los valores de mach y su coeficiente asociado ordenados
    vector_coeficientes = vector_coeficientes[indices_ordenados]
    
    condicion_distancia = np.concatenate([[True], np.diff(vector_mach) > tolerancia_distancia]) #estas tres líneas sirven para borrar los posibles puntos superpuestos, usa diff para restar el número de mach con el anterior y concatenate para conservar el primer punto de la lista y a partir de ahí, aplicar la tolerancia para ver si esta más cerca de ese valor y borrarlo
    vector_mach = vector_mach[condicion_distancia] #tras verificar la condición se guardan los puntos de nuevo ya filtrados
    vector_coeficientes = vector_coeficientes[condicion_distancia]
    
    valores_finitos = np.isfinite(vector_mach) & np.isfinite(vector_coeficientes) #esta línea hace una revisión de las parejas de puntos de mach y coeficiente y si alguna de las coordenadas es un infinito o NaN se marca como False, porque daría algún error
    
    if not np.all(valores_finitos): #si no son válidos todos los puntos
        nodos_invalidos = np.sum(~valores_finitos) #se guardan los puntos que son inválidos
        print(f"Se han eliminado {nodos_invalidos} nodo/s inválido/s.") #se muestran en el terminal cuántos nodos se han eliminado porque son inválidos
        
    return vector_mach[valores_finitos], vector_coeficientes[valores_finitos] #se devuelven los vectores de mach y coeficientes asociados ya preparados sin puntos que puedan dar errores por ser infinitos o NaN y ordenados


def configurar_interpolador_principal(vect_mach, vect_coef, metodo): #con esta función se consigue preparar el interpolador con los puntos ya preparados, es para la curva con teoría
    vect_mach, vect_coef = preparar_nodos_interpolacion(vect_mach, vect_coef) #primero se llama a la función anterior con los puntos o nodos que va a usar ya ordenados y preparados sin errores
    
    if len(vect_mach) < 3: #filtro con el mínimo número de puntos que requiere para hacer la interpolación, se calculan las pendientes usando los puntos vecinos y para ello necesita mínimo 3 puntos
        print(f"Fallo en la interpolación: El algoritmo {metodo} requiere al menos 3 puntos válidos.")
        sys.exit(1) #si no se cumple que haya al menos 3 puntos válidos entonces se para el programa y se informa
        
    if metodo == 'PCHIP': #se detecta si se usará pchip o makima. Por defecto si no hay nada será makima
        modelo_interpolado = PchipInterpolator(vect_mach, vect_coef, extrapolate=False) #se llama al método de interpolación de scipy
    else:
        modelo_interpolado = Akima1DInterpolator(vect_mach, vect_coef, method='makima') #se llama al método de interpolación de scipy
    return modelo_interpolado #se devuelve el resultado anterior y cuando se necesite pintar la gráfica por ejemplo en mach 0.8 este modelo_interpolado resolverá el polinomio correspondiente y devolverá la solución del coeficiente


def configurar_interpolador_secundaria(vect_mach, vect_coef, metodo): #esta función hace lo mismo que la anterior pero para la curva de puntos opcionales
    vect_mach, vect_coef = preparar_nodos_interpolacion(vect_mach, vect_coef)
    
    if len(vect_mach) < 3: #el funcionamiento es igual que la anterior función pero en este caso si hay menos de 3 nodos devuelve
        return None #None para permitir que el programa continúe aunque no haya suficientes puntos opcionales
        
    if metodo == 'PCHIP': #mismo funcionamiento que en configurar_interpolador_principal
        modelo_interpolado = PchipInterpolator(vect_mach, vect_coef, extrapolate=False)
    else:
        modelo_interpolado = Akima1DInterpolator(vect_mach, vect_coef, method='makima')
    return modelo_interpolado

def cargar_datos_referencia(ruta): #función para leer el csv con los datos de referencia en dos columnas, una de mach y otra de coeficiente. Lo traduce a formato americano

    if not os.path.exists(ruta): #si no se encuentra el archivo en la ruta avisa y no devuelve nada
        print(f"Aviso: No se encuentra el archivo de referencia {ruta}")
        return None, None
    try: #si se encuentra el archivo en la ruta
        #se abre y lee (modo lectura) todo el texto del archivo
        with open(ruta, 'r', encoding='utf-8') as f:
            texto_bruto = f.read() #se guarda lo que se ha leído todo de seguido en texto_bruto
            
        #se limpia el formato y se pasa todo a formato americano
        if ';' in texto_bruto:
            #Se cambian las comas (decimales) por puntos
            texto_limpio = texto_bruto.replace(',', '.')
            #se cambian los puntos y comas (columnas) por comas
            texto_limpio = texto_limpio.replace(';', ',')
        else:
            texto_limpio = texto_bruto
            
        #se pasa el texto limpio y separado por líneas al lector de NumPy
        lineas = texto_limpio.splitlines() #splitlines hace esto: Busca cada vez que haya un salto de línea (\n), corta por ahí y guarda los pedazos separados en una lista
        #la variable texto limpio era así: 0.8, 0.2\n0.85, 0.25\n0.9, 0.3 y la variable líneas ahora guarda esto: [ "0.8, 0.2", "0.85, 0.25", "0.9, 0.3" ]

        datos = np.loadtxt(lineas, delimiter=',', skiprows=1) #se pasa la lista a numpy para hacerla una matriz, donde las columnas se separan con comas
        
        return datos[:, 0], datos[:, 1] #se devuelve la matriz datos con las dos columnas y se extrae por un lado la columna 0 que serán los mach y por otra la columna 1 que son los coeficientes
        
    except Exception as e: #si ocurre cualquier cosa imprevista se avisa de error y no se devuelve nada
        print(f"Error al leer el archivo de referencia: {e}")
        return None, None

def calcular_parametros_comparacion(M_ref, coef_ref, interpolador_modelo, mach_min=None, mach_max=None): #función para calcular el RMSE y R^2 comparando el modelo con una serie de datos de referencia que se proporcionen
    if interpolador_modelo is None: #si no hay modelo (por ejemplo, no hay puntos opcionales) no se calcula nada
        return None

    #si se indica un tramo de Mach, se filtran los datos de referencia para quedarnos solo con los puntos de ese tramo
    if mach_min is not None or mach_max is not None:
        limite_inferior = mach_min if mach_min is not None else -np.inf
        limite_superior = mach_max if mach_max is not None else np.inf
        dentro_del_tramo = (M_ref >= limite_inferior) & (M_ref < limite_superior)
        M_ref = M_ref[dentro_del_tramo]
        coef_ref = coef_ref[dentro_del_tramo]

    coef_modelo = interpolador_modelo(M_ref) #se evalúa el modelo justo en los mismos Mach que tiene la referencia
    validos = np.isfinite(coef_modelo) & np.isfinite(coef_ref) #se descartan los puntos donde el modelo no tiene valor definido (fuera del rango cubierto por la interpolación) o donde la referencia no es un número válido

    if np.sum(validos) < 2: #si quedan menos de 2 puntos válidos no tiene sentido calcular las métricas
        return None

    coef_modelo = coef_modelo[validos]
    coef_ref_validos = coef_ref[validos]
    n = len(coef_ref_validos)

    error = coef_modelo - coef_ref_validos
    rmse = np.sqrt(np.mean(error**2)) #RMSE, diferencia de error entre el modelo y el real elevado al cuadrado, se calcula la media de esos errores al cuadrado y después la raíz

    media_ref = np.mean(coef_ref_validos)
    ss_total = np.sum((coef_ref_validos - media_ref)**2)
    ss_residual = np.sum(error**2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 1e-12 else 0.0 #coeficiente de determinación R^2

    return {'n': n, 'rmse': rmse, 'r2': r2}


def grafica_comparacion(M_ref_cl, cl_ref, M_ref_cd, cd_ref, modelo_cl_base, modelo_cd_base,
    modelo_cl_usuario, modelo_cd_usuario, met_cl_base, met_cd_base, met_cl_usuario, met_cd_usuario, 
    es_caso_extremo, alpha_grad, tc, carpeta,
    lim_mach_cl_base, lim_mach_cd_base, lim_mach_cl_opc, lim_mach_cd_opc,
    predef_cl_m, predef_cl_v, predef_cl_n,
    predef_cd_m, predef_cd_v, predef_cd_n,
    opc_cl_m, opc_cl_v, opc_cd_m, opc_cd_v):                    #función para generar una gráfica de comparación superponiendo los datos de referencia con los modelos de teoría y/o usuario.
  
    os.makedirs(carpeta, exist_ok=True) #se asegura que exista la carpeta donde se va a guardar
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) #se hace una figura con 1 fila y 2 columnas para las 2 gráficas
    M_plot = np.linspace(0.01, 1.60, 800) #se crean 800 puntos de mach hasta 1.6

    #cl
    ax = axes[0] #se selecciona la columna de la izquierda de la figura
    if M_ref_cl is not None:
        ax.scatter(M_ref_cl, cl_ref, color='black', s=40, zorder=6, label='Datos de [30]', marker='o') #se dibujan los datos sueltos
        
        #dibujar teoría (Si es válida)
        if not es_caso_extremo:
            filtro_base = M_plot <= lim_mach_cl_base #esto hace que se llegue hasta un límite el mach en la gráfica
            cl_modelo = modelo_cl_base(M_plot[filtro_base]) #se evalúa el modelo en los puntos de mach correspondientes
            ax.plot(M_plot[filtro_base], cl_modelo, 'b-', lw=2.5, label='Modelo propio') #se pinta en azul una línea del modelo teórico
            
        #dibujar usuario (Si existe)
        if modelo_cl_usuario is not None:
            filtro_opc = M_plot <= lim_mach_cl_opc
            cl_usuario = modelo_cl_usuario(M_plot[filtro_opc]) #el funcionamiento es como el anterior para la teoría pero con los datos del usuario y en línea discontinua
            estilo = 'g-' if es_caso_extremo else 'g--'
            etiqueta = 'Modelo usuario' if es_caso_extremo else 'Curva con puntos opcionales'
            ax.plot(M_plot[filtro_opc], cl_usuario, estilo, lw=2.5, label=etiqueta)

    #puntos predefinidos de cl y opcionales del usuario se dibujan
    ax.scatter(predef_cl_m, predef_cl_v, color='red', s=60, zorder=8, marker='o',
               edgecolors='black', linewidths=0.5, label='Puntos predefinidos')
    for m, v, nom in zip(predef_cl_m, predef_cl_v, predef_cl_n):
        ax.annotate(nom, (m, v), textcoords="offset points", xytext=(5, 5), fontsize=8)

    if len(opc_cl_m) > 0:
        ax.scatter(opc_cl_m, opc_cl_v, color='green', s=60, zorder=7, marker='^',
                   alpha=0.85, label='Puntos opcionales')


    ax.set_xlabel('Número de Mach ($M_\\infty$)', fontsize=13)
    ax.set_ylabel('Coeficiente $c_l$', fontsize=13)
    ax.set_title('Comparación — Sustentación', fontsize=12)
    ax.set_xlim(0, 1.65)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10, loc='upper right') #líneas para el dibujo de la gráfica de comparación

    #cd, el funcionamienteo es igual que para el cl
    ax = axes[1] #se selecciona la columna derecha de la figura, la segunda gráfica que habrá en la imagen
    if M_ref_cd is not None:
        ax.scatter(M_ref_cd, cd_ref, color='black', s=40, zorder=6, label='Datos de [30]', marker='o')
        
        #dibujar teoría (Si es válida) (ya explicado para el cl)
        if not es_caso_extremo:
            filtro_base = M_plot <= lim_mach_cd_base
            cd_modelo = np.maximum(modelo_cd_base(M_plot[filtro_base]), 1e-6)
            ax.plot(M_plot[filtro_base], cd_modelo, 'r-', lw=2.5, label='Modelo propio')
                
        #dibujar usuario (Si existe) (ya explicado para el cl)
        if modelo_cd_usuario is not None:
            filtro_opc = M_plot <= lim_mach_cd_opc
            cd_usuario = np.maximum(modelo_cd_usuario(M_plot[filtro_opc]), 1e-6)
            estilo = 'm-' if es_caso_extremo else 'm--'
            etiqueta = 'Modelo usuario' if es_caso_extremo else 'Curva con puntos opcionales'
            ax.plot(M_plot[filtro_opc], cd_usuario, estilo, lw=2.5, label=etiqueta)

    #puntos predefinidos de cd y opcionales del usuario se dibujan
    ax.scatter(predef_cd_m, predef_cd_v, color='darkorange', s=60, zorder=8, marker='o',
               edgecolors='black', linewidths=0.5, label='Puntos predefinidos')
    for m, v, nom in zip(predef_cd_m, predef_cd_v, predef_cd_n):
        ax.annotate(nom, (m, v), textcoords="offset points", xytext=(5, 5), fontsize=8)

    if len(opc_cd_m) > 0:
        ax.scatter(opc_cd_m, opc_cd_v, color='purple', s=60, zorder=7, marker='^',
                   alpha=0.85, label='Puntos opcionales')

    ax.set_xlabel('Número de Mach ($M_\\infty$)', fontsize=13)
    ax.set_ylabel('Coeficiente $c_d$', fontsize=13)
    ax.set_title('Comparación — Resistencia', fontsize=12)
    ax.set_xlim(0, 1.65)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10, loc='lower right') #líneas para el dibujo de la gráfica de cd de comparación

    fig.suptitle(f'Comparación del modelo — $t/c={tc:.2f}$, $\\alpha={alpha_grad:.1f}°$', fontsize=12)
    fig.tight_layout()
    ruta = os.path.join(carpeta, f'comparacion_1_alpha{alpha_grad:.1f}_tc{tc:.2f}.png')
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    print(f"Gráfica de comparación generada en: {ruta}")

def generar_exportacion_grafica(mach_simulado, curva_cl_base, curva_cd_base, lim_mach_cl_base,
    lim_mach_cd_base,
    curva_cl_opcional, curva_cd_opcional, lim_mach_cl_opc, lim_mach_cd_opc, nodos_mach_cl_opc,
    nodos_c_cl_opc, nodos_mach_cd_opc, nodos_c_cd_opc, existen_puntos_usuario_cl,
    existen_puntos_usuario_cd, valor_alpha, valor_espesor, es_borde_romo,
    predefinidos_cl_mach, predefinidos_cl_valor, predefinidos_cl_nombre,   
    predefinidos_cd_mach, predefinidos_cd_valor, predefinidos_cd_nombre,   
    ruta_directorio_salida):#esta función lo que hace es recoger las listas definitivas de mach, cd y cl y las prepara para exportar los datos en un csv y también para trazar las curvas
    os.makedirs(ruta_directorio_salida, exist_ok=True) #se crea una carpeta de resultados si no existe
    figura, paneles = plt.subplots(1, 2, figsize=(14, 6)) #se crea una figura de matplotlib con dos paneles vacíos que ahí irán las gráficas
    
    etiqueta_borde = "borde romo" if es_borde_romo else "borde afilado" #se crea una etiqueta según sea el borde del perfil para luego usarla para tomar decisión al dibujar
    
    es_caso_extremo = (valor_espesor > 0.12) or (abs(valor_alpha) > 10.0) #si ocurre que es grueso o fuera del límite de alpha, me indica que es el caso extremo que no dibujará curva teórica

    #gráfica del coeficiente de sustentación
    grafico_sustentacion = paneles[0] #el gráfico de sustentación irá en el panel izquierdo (0)
    
    if not es_caso_extremo: #si no hay caso extremo entonces se dibuja la curva teórica (base) en azul hasta el límite de mach
        filtro_mach_base = mach_simulado <= lim_mach_cl_base #se revisan los números de mach simulados y pone true a los que sean menores o igual al límite, este límite se usa para ver hasta que mach se usa cada ecuación, por ejemplo, el límite para KT será el Mcrit
        grafico_sustentacion.plot(mach_simulado[filtro_mach_base], curva_cl_base[filtro_mach_base], 'b-', lw=2.5, label='Modelo propio') #se dibuja la gráfica

    #se marcan los puntos predefinidos obligatorios de cl
    grafico_sustentacion.scatter(predefinidos_cl_mach, predefinidos_cl_valor, color='red',
        s=60, zorder=7, marker='o', edgecolors='black', linewidths=0.5, label='Puntos predefinidos')
    for m, v, nom in zip(predefinidos_cl_mach, predefinidos_cl_valor, predefinidos_cl_nombre):
        grafico_sustentacion.annotate(nom, (m, v), textcoords="offset points", xytext=(5, 5), fontsize=8)

    if existen_puntos_usuario_cl and curva_cl_opcional is not None: #si además de la teoría, o no hay teoría pero sí hay curva de puntos obligatorios y puntos opcionales de cl proporcionados, se dibuja en verde discontinua
        filtro_mach_opc = mach_simulado <= lim_mach_cl_opc #de nuevo se ponen a true los valores de mach inferiores al límite para usarlos para dibujar
        texto_leyenda_opc = 'Modelo solo datos de usuario' if es_caso_extremo else 'Curva con puntos opcionales' #leyenda de la línea
        estilo_linea = 'g-' if es_caso_extremo else 'g--' #si es grueso o excede límites de alpha es línea continua y si no, discontinua
        grosor_linea = 2.5 if es_caso_extremo else 2.0 #grosor de línea según sea grueso y/o fuera de límites de alpha o no
        
        grafico_sustentacion.plot(mach_simulado[filtro_mach_opc], curva_cl_opcional[filtro_mach_opc], estilo_linea, lw=grosor_linea, label=texto_leyenda_opc) #se dibuja la gráfica 
        if len(nodos_mach_cl_opc) > 0: #si la lista de puntos opcionales es mayor que cero se dibujan los puntos sobre los que pasará la gráfica de puntos opcionales
            filtro_nodos_opc = nodos_mach_cl_opc <= lim_mach_cl_opc
            grafico_sustentacion.scatter(nodos_mach_cl_opc[filtro_nodos_opc], nodos_c_cl_opc[filtro_nodos_opc], color='green', s=40, zorder=6, alpha=0.8, marker='^', label='Puntos opcionales')
    elif es_caso_extremo: #si hay perfil grueso y/o se está fuera de los límites de alpha y NO hay puntos opcionales, se emite un aviso de que no hay suficientes puntos para hacer la curva opcional
        grafico_sustentacion.text(0.5, 0.5, 'Información insuficiente\npara trazar la curva', transform=grafico_sustentacion.transAxes, ha='center', va='center', fontsize=11, color='gray')

    grafico_sustentacion.set_xlabel('Número de Mach ($M_\\infty$)', fontsize=13)
    grafico_sustentacion.set_ylabel('Coeficiente $c_l$', fontsize=13)
    grafico_sustentacion.set_title('Gráfica de Sustentación', fontsize=12)
    grafico_sustentacion.set_xlim(0, 1.65)
    grafico_sustentacion.xaxis.set_major_locator(MultipleLocator(0.2))
    grafico_sustentacion.grid(True, alpha=0.35)
    grafico_sustentacion.legend(fontsize=9, loc='upper left') #todo esto es para los títulos de ejes, de la gráfica, etc

    #gráfica del coeficiente de resistencia
    grafico_resistencia = paneles[1] #se dibujará en el segundo panel
    
    if not es_caso_extremo:
        filtro_mach_base = mach_simulado <= lim_mach_cd_base #mismo filtro de cl que pone a true cuando el mach está por debajo del límite
        etiqueta_cd_base = 'Modelo propio'
        if es_borde_romo:
            etiqueta_cd_base += ' (Borde romo)' #esto solamente se indica en la leyenda porque que sea romo solamente afecta al cd supersónico, por eso se pone solo aquí y no en cl
            
        grafico_resistencia.plot(mach_simulado[filtro_mach_base], curva_cd_base[filtro_mach_base], 'r-', lw=2.5, label=etiqueta_cd_base)
            #se dibuja la gráfica continua de cd teórico

    #puntos predefinidos de cd
    grafico_resistencia.scatter(predefinidos_cd_mach, predefinidos_cd_valor, color='darkorange',
        s=60, zorder=7, marker='o', edgecolors='black', linewidths=0.5, label='Puntos predefinidos')
    for m, v, nom in zip(predefinidos_cd_mach, predefinidos_cd_valor, predefinidos_cd_nombre):
        grafico_resistencia.annotate(nom, (m, v), textcoords="offset points", xytext=(5, 5), fontsize=8)

    if existen_puntos_usuario_cd and curva_cd_opcional is not None: #caso de puntos opcionales y, además, curva cd opcional
        filtro_mach_opc = mach_simulado <= lim_mach_cd_opc #el funcionamiento es igual que el cl
        texto_leyenda_opc = 'Modelo solo datos de usuario' if es_caso_extremo else 'Curva con puntos opcionales'
        estilo_linea = 'm-' if es_caso_extremo else 'm--'
        grosor_linea = 2.5 if es_caso_extremo else 2.0
        
        grafico_resistencia.plot(mach_simulado[filtro_mach_opc], curva_cd_opcional[filtro_mach_opc], estilo_linea, lw=grosor_linea, label=texto_leyenda_opc)
                                 
        if len(nodos_mach_cd_opc) > 0:
            filtro_nodos_opc = nodos_mach_cd_opc <= lim_mach_cd_opc #filtro y puntos de cd si hay la lista tiene puntos opcionales y no está vacía
            grafico_resistencia.scatter(nodos_mach_cd_opc[filtro_nodos_opc], nodos_c_cd_opc[filtro_nodos_opc], color='purple', s=40, zorder=6, alpha=0.8, marker='^', label='Puntos opcionales')
    elif es_caso_extremo: #caso de perfil grueso y/o fuera de los límites de alpha y sin puntos opcionales, funciona igual que cl
        grafico_resistencia.text(0.5, 0.5, 'Información insuficiente\npara trazar la curva', transform=grafico_resistencia.transAxes, ha='center', va='center', fontsize=11, color='gray')

    grafico_resistencia.set_xlabel('Número de Mach ($M_\\infty$)', fontsize=13)
    grafico_resistencia.set_ylabel('Coeficiente $c_d$', fontsize=13)
    grafico_resistencia.set_title('Gráfica de Resistencia', fontsize=12)
    grafico_resistencia.set_xlim(0, 1.65)
    grafico_resistencia.set_ylim(bottom=0)
    grafico_resistencia.xaxis.set_major_locator(MultipleLocator(0.2))
    grafico_resistencia.grid(True, alpha=0.35)
    grafico_resistencia.legend(fontsize=9, loc='upper left') #estas líneas son para los títulos de gráfica, ejes, etc

    titulo_principal = f'Análisis Perfil — Espesor $t/c={valor_espesor:.2f}$, ' #título con el espesor, el alpha simulado y tipo de borde
    titulo_principal += f'$\\alpha={valor_alpha:.1f}^\\circ$ ({etiqueta_borde})'
    figura.suptitle(titulo_principal, fontsize=12)
    figura.tight_layout()
    
    nombre_archivo = f'curvas_alpha{valor_alpha:.1f}_tc{valor_espesor:.2f}.png' #estas líneas son para cómo se guardan y dónde las curvas que se generan
    ruta_guardado = os.path.join(ruta_directorio_salida, nombre_archivo)
    figura.savefig(ruta_guardado, dpi=300)
    plt.close(figura)
    
    print(f"La figura se ha generado correctamente en la ruta: {ruta_guardado}") #se muestra que se han generado las curvas bien y la ruta por si el usuario no lo encuentra


def main(): #la función main es la principal, que es la que da las órdenes para coordinar el resto de funciones, primero lee los datos, los valida y prepara, luego calcula, luego construye y genera la gráfica
    #inicialización de variables de comparación vacías que se rellenarán si hay archivos de referencia para hacer la comparación. Si no los hay se quedan vacías
    M_ref_cl, cl_ref = None, None
    M_ref_cd, cd_ref = None, None

    directorio_actual = os.path.dirname(os.path.abspath(__file__)) #busca el directorio donde está el script de python este
    archivo_configuracion = os.path.join(directorio_actual, "datos_entrada.txt") #estas tres líneas guardan la dirección de los archivos necesarios para luego pasar la dirección a las funciones de carga de datos
    archivo_polar = os.path.join(directorio_actual, "polar.txt")
    archivo_geometria = os.path.join(directorio_actual, "perfil.txt")
    directorio_resultados = os.path.join(directorio_actual, "resultados") #define la ruta de la carpeta de resultados para guardar los csv y gráficas

    print("\nIniciando simulación del perfil...") #se irán poniendo textos de lo que va ocurriendo para ver en que paso va fallando el código en caso de error
    print("---------------------------------------------------------")

    print("Cargando datos de entrada...") #a partir de aquí se empiezan a cargar los datos
    parametros = cargar_datos_entrada(archivo_configuracion) #se llama a la función de carga de datos de entrada y se extraen los datos del diccionario
    
    metodo = parametros.get('METODO', ['MAKIMA'])[0].upper() #se extrae el método elegido (por defecto MAKIMA)
    
    espesor_relativo = parametros['TC'][0] #se extrae del diccionario el dato de espesor relativo y ángulo de ataque para poder usarlos
    angulo_ataque_geom = parametros['ALPHA'][0]
    
    if 'BORDE' not in parametros: #con estas líneas se detecta si existe la etiqueta del borde del perfil en el diccionario y si no está se para el programa y se avisa
        print("Fallo: Se debe especificar el tipo de BORDE.")
        sys.exit(1)
        
    perfil_redondeado = parametros['BORDE'] == 'ROMO' #si el dato del diccionario es ROMO, la variable perfil_redondeado se pone a true y si es afilado a false

    print("Cargando datos de la polar...") #aquí se empiezan a cargar los datos de la polar
    vector_alpha, vector_cl, vector_cd = cargar_polar(archivo_polar) #se llama a la función de carga de datos de la polar que devuelve alpha, cl y cd y guarda esos datos en las listas de vector_alpha, cd, cl 
    
    cl_base_mach0 = float(np.interp(angulo_ataque_geom, vector_alpha, vector_cl)) #estas 2 líneas usan los datos de la polar y calcula interpolando, el cl a mach 0 y el cd para el ángulo de ataque que el usuario quiere simular
    cd_base_mach0 = float(np.interp(angulo_ataque_geom, vector_alpha, vector_cd)) #esto se hace por si el ángulo alpha no se encuentra exactamente en el archivo de la polar
    cd_polar_minimo = float(np.min(vector_cd)) #se extrae el valor de cd mínimo de los datos de la polar
    
    if abs(cl_base_mach0) > 1e-6: #se calcula el factor k a partir de los valores de la polar para el alpha que se usa. Si cl es muy muy pequeño, k será 0 para evitar divisiones por 0
        factor_forma_k = (cd_base_mach0 - cd_polar_minimo) / cl_base_mach0**2
    else:
        factor_forma_k = 0.0

    try: #este bloque sirve para calcular el ángulo de sustentación nula a partir de la polar. Como la polar casi nunca da el punto 0 exactamente, se busca el corte con el eje y se hace una interpolación entre el primer punto positivo y el primero negativo
        zona_lineal = np.where((vector_alpha >= -15) & (vector_alpha <= 10))[0] #filtra la lista completa de datos de la polar y se queda solamente con los valores que están entre -15 y +10 grados, con su posición
        alpha_recortado = vector_alpha[zona_lineal] #esto son dos listas que contienen ya únicamente los valores fitrados que son los de las posiciones detectadas en la línea anterior
        cl_recortado = vector_cl[zona_lineal]
        
        cambios_signo = np.where(np.diff(np.sign(cl_recortado)))[0] #se detecta cuándo ocurre el cambio de signo y el where detecta en que posición de la lista ocurre. El diff hace resta entre una posición y otra, entonces, si hay cambio de signo saldrá una resta mayor y ese es el cruce con el eje. El cambio de signo es en el cl
        if len(cambios_signo) > 0: #si se ha producido el cambio de signo entonces...
            indice_cruce = cambios_signo[0] #se guarda en indice_cruce la posición exacta del punto 1 (el último punto que estaba por debajo de cero), indice + 1 será ya con el signo cambiado
            delta_alpha = alpha_recortado[indice_cruce+1] - alpha_recortado[indice_cruce] #en estas dos líneas se calcula la distancia entre los puntos que cambian de signo
            delta_cl = cl_recortado[indice_cruce+1] - cl_recortado[indice_cruce]
            alpha_sustentacion_nula = alpha_recortado[indice_cruce] - cl_recortado[indice_cruce] * delta_alpha / delta_cl #se calcula el alpha de sustentación nula con la ecuación de una recta, interpolación lineal
        else:
            alpha_sustentacion_nula = 0.0 #si no hay cambio de signo el ángulo de sustentación nula es 0
            print("Aviso: No se ha podido localizar el cruce de c_l=0. Se usará α_L0 = 0°.")
    except Exception: #si los cálculos fallan por cualquier cosa se asume ángulo de sustentación nula para evitar errores
        alpha_sustentacion_nula = 0.0

    angulo_efectivo_rad = np.radians(angulo_ataque_geom - alpha_sustentacion_nula) #calcula el ángulo de ataque efectivo en radianes. np.radians convierte de grados a radianes porque las ecuaciones de Ackeret trabajan en radianes

    print("Cargando datos geométricos del perfil...") #se pasa a cargar las coordenadas del perfil
    coord_x_sup, coord_y_sup, coord_x_inf, coord_y_inf = cargar_coordenadas(archivo_geometria) #se llama a la función de carga de coordenadas y se guardan en listas
    
    if perfil_redondeado: #si el perfil es romo, o sea la variable es true, la integral de cdesp será 0 porque no es aplicable para estos perfiles
        integral_espesor = 0.0
        print("Borde romo en el perfil. Se anula la integral de cd de espesor.")
    else: #si no es romo, se llama la función para calcular la integral de cdesp y se devuelve el valor y se muestra
        integral_espesor = calcular_integral_cdesp(coord_x_sup, coord_y_sup, coord_x_inf, coord_y_inf)
        print(f"Cálculo de cd_esp hecho = {integral_espesor:.6f}")

    print("Realizando comparación de los datos y condiciones...")
    es_caso_grueso, es_caso_limite = validar_modelo(parametros, espesor_relativo, angulo_ataque_geom, cl_base_mach0, cd_polar_minimo, factor_forma_k, angulo_efectivo_rad, integral_espesor, perfil_redondeado)
    #se llama a la función para validar los datos, condiciones, etc. Devuelve si el perfil es grueso o no y si se está fuera del límite de alpha o no y se guarda en 2 variables
    print("Comparación completada sin errores.")

    lista_opcionales_cl = parametros['OPCIONAL_CL'] #se extraen del diccionario de parámetros las listas de datos opcionales y se guardan en listas nuevas
    lista_opcionales_cd = parametros['OPCIONAL_CD']
    
    usuario_ingreso_cl = len(lista_opcionales_cl) > 0 #se mira si hay puntos opcionales en la lista o no

    puntos_sup_cd_romo = extraer_opcionales_supersonicos(lista_opcionales_cd) if perfil_redondeado else [] #si el perfil es romo se llama a la función para extraer de la lista los puntos opcionales supersónicos y si es afilado devuelve una lista vacía
    dispone_sup_romo = len(puntos_sup_cd_romo) > 0 and verificar_zona_supersonica(lista_opcionales_cd) #se verifica que haya valores en la lista extraída y que además esos puntos cumplan con las condiciones de que sea en mach 1.2 y otro mas en supersónico

    es_viable_ecuacion_cd_sup = (not perfil_redondeado) and (not es_caso_grueso) and (not es_caso_limite) #se comprueba que el perfil no sea romo ni grueso ni fuera de los límites de alpha para posteriormente usarlo para dar vía libre al uso de las ecuaciones teóricas de cd
    usuario_cubre_sup_cl = verificar_zona_supersonica(lista_opcionales_cl) #se guardan los puntos opcionales de cl y cd, verificadas las condiciones que deben cumplir estos puntos, en 2 listas para usarlas luego
    usuario_cubre_sup_cd = verificar_zona_supersonica(lista_opcionales_cd)

    if es_caso_grueso or es_caso_limite: #calcula hasta que número de Mach llega la curva de cl. Para grueso o alpha límite llega hasta el último opcional supersónico del usuario (si los hay), o hasta M_CLREC si no nos hay. Para perfil normal llega hasta 1.60
        if usuario_cubre_sup_cl:
            tope_mach_cl_base = max(M for M, _ in lista_opcionales_cl)
        else:
            tope_mach_cl_base = parametros['M_CLREC'][0]
    else:
        tope_mach_cl_base = 1.60

    if es_viable_ecuacion_cd_sup: #si es afilado el perfil, el mach hasta el que llega la curva es 1.6
        tope_mach_cd_base = 1.60
    elif usuario_cubre_sup_cd and (es_caso_grueso or es_caso_limite): #si el perfil es grueso o fuera de los límites de alpha y además el usuario ha proporcionado puntos válidos en supersónico, el último mach será el último proporcionado por el usuario
        tope_mach_cd_base = max(M for M, _ in lista_opcionales_cd)
    else: #si es un caso de perfil grueso y fuera de límites de alpha y, además, no tengo datos en supersónico, se dibuja como mucho hasta Mcdmax
        tope_mach_cd_base = parametros['M_CDMAX'][0]

    print("\nGenerando modelo...") #se genera el modelo que se va a dibujar en la gráfica
    vectores_mach_cl_base, vectores_coef_cl_base, predef_cl_m, predef_cl_v, predef_cl_n, mach_critico_val = generar_puntos_cl_base(
        parametros, cl_base_mach0, angulo_efectivo_rad, es_caso_grueso, es_caso_limite)
    vectores_mach_cd_base, vectores_coef_cd_base, predef_cd_m, predef_cd_v, predef_cd_n = generar_puntos_cd_base(
        parametros, cl_base_mach0, cd_polar_minimo, factor_forma_k, angulo_efectivo_rad, integral_espesor,
        es_caso_grueso, es_caso_limite, perfil_redondeado, es_viable_ecuacion_cd_sup, puntos_sup_cd_romo)
    #se generan las listas con los puntos de cd y cl de la curva teórica
    
    modelo_cl_base = configurar_interpolador_principal(vectores_mach_cl_base, vectores_coef_cl_base, metodo)
    modelo_cd_base = configurar_interpolador_principal(vectores_mach_cd_base, vectores_coef_cd_base, metodo)
    #se aplica el interpolador a estos puntos

    #se extraen el valor de cl y cd en Mcrit
    cl_en_mcrit_val = predef_cl_v[1]
    cd_en_mcrit_val = predef_cd_v[1]

    vectores_mach_cl_opc, vectores_coef_cl_opc = generar_puntos_cl_con_usuario(parametros, cl_base_mach0, lista_opcionales_cl, mach_critico_val, cl_en_mcrit_val)
    vectores_mach_cd_opc, vectores_coef_cd_opc = generar_puntos_cd_con_usuario(parametros, cd_polar_minimo, lista_opcionales_cd, perfil_redondeado, mach_critico_val, cd_en_mcrit_val)
    #se llama a la función que genera los vectores con los puntos de cl y cd del usuario
    modelo_cl_usuario = None
    modelo_cd_usuario = None
    #se crean dos variables vacías que luego se rellenan con el modelo del usuario
    tope_mach_cl_usuario = parametros['M_CLREC'][0]
    tope_mach_cd_usuario = parametros['M_CDMAX'][0]
    #se establece de primeras un valor inicial para el límite de Mach de cd y cl
    if vectores_mach_cl_opc is not None: #si hay valores opcionales de cl, entonces se genera el modelo con los datos del usuario con el interpolador y se establece el nuevo límite de mach en el último proporcionado por el usuario
        modelo_cl_usuario = configurar_interpolador_secundaria(vectores_mach_cl_opc, vectores_coef_cl_opc, metodo)
        if modelo_cl_usuario is not None:
            tope_mach_cl_usuario = float(np.max(vectores_mach_cl_opc))

    if vectores_mach_cd_opc is not None:  #si hay valores de cd opcionales, se genera el modelo con datos del usuario con el interpolador y se establece el nuevo límite de mach como el último proporcionado
        modelo_cd_usuario = configurar_interpolador_secundaria(vectores_mach_cd_opc, vectores_coef_cd_opc, metodo)
        if modelo_cd_usuario is not None:
            tope_mach_cd_usuario = float(np.max(vectores_mach_cd_opc))

    usuario_ingreso_cd_valido = len(lista_opcionales_cd) > 0 and modelo_cd_usuario is not None #se guarda un true si hay puntos opcionales de cd y, además, si el modelo no está vacío

    escala_mach = np.linspace(0.02, 1.60, 800) #se crean 800 puntos de mach entre 0.02 y 1.6

    curva_suavizada_cl_base = modelo_cl_base(escala_mach) #se guardan en estas listas los puntos de la curva de cl y de cd calculados antes con el interpolador pero el modelo_cl_base no era una lista de números si no una función que calculaba los valores de cl
    curva_suavizada_cd_base = np.maximum(modelo_cd_base(escala_mach), 1e-6) #en esta línea además evita que si se ha generado un punto por debajo de cero, no lo use y use un valor muy cercano a 0 pero positivo (no tiene sentido cd negativo)

    if modelo_cl_usuario is not None: #se evalúa si existe la curva opcional de cl y se guardan los puntos de cl calculados para cada mach
        curva_suavizada_cl_usuario = modelo_cl_usuario(escala_mach)
    else: #si no existe, se deja vacío
        curva_suavizada_cl_usuario = None
        
    if modelo_cd_usuario is not None: #igual que para modelo_cl_usuario anterior
        curva_suavizada_cd_usuario = np.maximum(modelo_cd_usuario(escala_mach), 1e-6)
    else:
        curva_suavizada_cd_usuario = None

    if usuario_ingreso_cl: #si hay puntos opcionales, extrae los mach y los coeficientes de cl opcionales en arrays separados para poder dibujar la gráfica
        nodos_usuario_cl_m = np.array([m for m, _ in lista_opcionales_cl]) #una línea rellena un vector (nodos_usuario_cl_m) con todos los valores de mach
        nodos_usuario_cl_c = np.array([c for _, c in lista_opcionales_cl]) #la segunda hace lo mismo pero solamente con los valores de cl
    else: #si no hay, los deja vacíos
        nodos_usuario_cl_m = np.array([])
        nodos_usuario_cl_c = np.array([])

    nodos_usuario_cd_m = np.array([m for m, _ in lista_opcionales_cd]) #hacen la misma función que las de cl
    nodos_usuario_cd_c = np.array([c for _, c in lista_opcionales_cd])
    #esto de separar mach y coeficiente por dos partes se hace para que se puedan dibujar los puntos en la gráfica

    print("Creando gráficas y archivos csv...")
    generar_exportacion_grafica(escala_mach, curva_suavizada_cl_base, curva_suavizada_cd_base,
        tope_mach_cl_base, tope_mach_cd_base,
        curva_suavizada_cl_usuario, curva_suavizada_cd_usuario,
        tope_mach_cl_usuario, tope_mach_cd_usuario, nodos_usuario_cl_m, nodos_usuario_cl_c,
        nodos_usuario_cd_m, nodos_usuario_cd_c, usuario_ingreso_cl, usuario_ingreso_cd_valido,
        angulo_ataque_geom, espesor_relativo, perfil_redondeado,
        predef_cl_m, predef_cl_v, predef_cl_n,   
        predef_cd_m, predef_cd_v, predef_cd_n,   
        directorio_resultados)#se llama a la función de gráficas pasando todos los datos necesarios calculados
    os.makedirs(directorio_resultados, exist_ok=True) #se crea la carpeta de resultados

    tope_base_global = min(tope_mach_cl_base, tope_mach_cd_base) #se determina hasta que mach de las curvas teóricas tiene sentido sacar datos para el csv, que sera el mínimo entre el límite de cl y cd
    filtro_guardado_base = escala_mach <= tope_base_global #se revisan los puntos de mach, los 800, y marca como válidos solamente los que estén por debajo del límite teórico anterior
    ruta_csv_base = os.path.join(directorio_resultados, f'modelo_teoria_alpha{angulo_ataque_geom:.1f}_tc{espesor_relativo:.2f}.csv')
    #se crea la dirección donde se guarda el archivo generado con su nombre
    np.savetxt(ruta_csv_base, np.column_stack([escala_mach[filtro_guardado_base], curva_suavizada_cl_base[filtro_guardado_base], curva_suavizada_cd_base[filtro_guardado_base]]), delimiter=',', header='Mach,c_l,c_d', comments='', fmt='%.6f')
    #se escriben los datos de la gráfica de teoría en el csv, tres columnas, mach, cl y cd
    print(f"Archivo de datos de teoría generado en: {ruta_csv_base}")

    if modelo_cl_usuario is not None or modelo_cd_usuario is not None: #si hay modelo de cl y cd del usuario, o sea, opcional
        tope_opc_global = min(tope_mach_cl_usuario, tope_mach_cd_usuario) #se determina de nuevo el límite de mach entre el mínimo que haya entre cd y cl proporcionados
        filtro_guardado_opc = escala_mach <= tope_opc_global #se revisan los puntos de mach y se ponen en true solamente los que cumplan el límite

        if curva_suavizada_cl_usuario is not None: #se evalúa si existe la curva opcional de cl y en ese caso extrae los datos de la curva de cl
            columna_cl_csv = curva_suavizada_cl_usuario[filtro_guardado_opc]
        else: #si no existe, hace lo mismo pero con los datos de cl de la curva teórica
            columna_cl_csv = curva_suavizada_cl_base[filtro_guardado_opc]
            
        if curva_suavizada_cd_usuario is not None: #si existe la curva opcional de cd, copia los datos de cd de la curva opcional
            columna_cd_csv = curva_suavizada_cd_usuario[filtro_guardado_opc]
        else: #si no existe, usa los datos de la curva teórica
            columna_cd_csv = curva_suavizada_cd_base[filtro_guardado_opc]

        
        ruta_csv_opc = os.path.join(directorio_resultados, f'modelo_usuario_alpha{angulo_ataque_geom:.1f}_tc{espesor_relativo:.2f}.csv')
        #se construye la ruta donde irá el archivo csv
        np.savetxt(ruta_csv_opc, np.column_stack([escala_mach[filtro_guardado_opc], columna_cl_csv, columna_cd_csv]), delimiter=',', header='Mach,c_l,c_d', comments='', fmt='%.6f')
        #se rellena el csv con los datos
        print(f"Archivo de datos opcionales generado en: {ruta_csv_opc}")

    #COMPARACIÓN
    archivo_ref_cl = parametros.get('ARCHIVO_REF_CL', [None])[0] if 'ARCHIVO_REF_CL' in parametros else None
    archivo_ref_cd = parametros.get('ARCHIVO_REF_CD', [None])[0] if 'ARCHIVO_REF_CD' in parametros else None
    #si el usuario ha escrito el nombre del archivo de referencia para la comparación, se guarda el nombre del archivo y si no se guarda un none para usar después este nombre para abrir el archivo
    es_caso_extremo = es_caso_grueso or es_caso_limite #se ve si hay caso limitante, o sea, grueso o fuera de límite de alpha

    #variables donde se guardan RMSE y R^2. Estan vacías de primeras
    met_cl_base, met_cl_usuario = None, None
    met_cd_base, met_cd_usuario = None, None

    if archivo_ref_cl: #se comprueba si se guardó el nombre del archivo de referencia para cl
        ruta_ref = os.path.join(directorio_actual, str(archivo_ref_cl)) #se construye la dirección para encontrar el archivo
        M_ref_cl, cl_ref = cargar_datos_referencia(ruta_ref) #se extraen dos listas de datos del número de mach y coeficiente de referencia
        if M_ref_cl is not None: #se calculan las métricas de cl si hay datos de referencia cargados
            #se calculan las métricas de la curva completa
            print("\n Parámetros cl - CURVA COMPLETA")
            if not es_caso_extremo:
                met_cl_base_completa = calcular_parametros_comparacion(M_ref_cl, cl_ref, modelo_cl_base)
                if met_cl_base_completa: print(f"    TEORÍA  -> RMSE: {met_cl_base_completa['rmse']:.4f} | R²: {met_cl_base_completa['r2']:.4f}")
            if modelo_cl_usuario is not None:
                met_cl_usuario_completa = calcular_parametros_comparacion(M_ref_cl, cl_ref, modelo_cl_usuario)
                if met_cl_usuario_completa: print(f"    USUARIO -> RMSE: {met_cl_usuario_completa['rmse']:.4f} | R²: {met_cl_usuario_completa['r2']:.4f}")

            #se calculan las métricas por separado para cada tramo
            print("\n Parámetros cl - SUBSONICO (M < Mcrit)")
            if not es_caso_extremo:
                met_cl_base_sub = calcular_parametros_comparacion(M_ref_cl, cl_ref, modelo_cl_base, mach_max=mach_critico_val)
                if met_cl_base_sub: print(f"    TEORÍA  -> RMSE: {met_cl_base_sub['rmse']:.4f} | R²: {met_cl_base_sub['r2']:.4f}")
            if modelo_cl_usuario is not None:
                met_cl_usuario_sub = calcular_parametros_comparacion(M_ref_cl, cl_ref, modelo_cl_usuario, mach_max=mach_critico_val)
                if met_cl_usuario_sub: print(f"    USUARIO -> RMSE: {met_cl_usuario_sub['rmse']:.4f} | R²: {met_cl_usuario_sub['r2']:.4f}")

            print("\n Parámetros cl - TRANSONICO (Mcrit <= M < 1.2)")
            if not es_caso_extremo:
                met_cl_base_tra = calcular_parametros_comparacion(M_ref_cl, cl_ref, modelo_cl_base, mach_min=mach_critico_val, mach_max=1.2)
                if met_cl_base_tra: print(f"    TEORÍA  -> RMSE: {met_cl_base_tra['rmse']:.4f} | R²: {met_cl_base_tra['r2']:.4f}")
            if modelo_cl_usuario is not None:
                met_cl_usuario_tra = calcular_parametros_comparacion(M_ref_cl, cl_ref, modelo_cl_usuario, mach_min=mach_critico_val, mach_max=1.2)
                if met_cl_usuario_tra: print(f"    USUARIO -> RMSE: {met_cl_usuario_tra['rmse']:.4f} | R²: {met_cl_usuario_tra['r2']:.4f}")

            print("\n Parámetros cl - SUPERSONICO (M >= 1.2)")
            if not es_caso_extremo:
                met_cl_base_sup = calcular_parametros_comparacion(M_ref_cl, cl_ref, modelo_cl_base, mach_min=1.2)
                if met_cl_base_sup: print(f"    TEORÍA  -> RMSE: {met_cl_base_sup['rmse']:.4f} | R²: {met_cl_base_sup['r2']:.4f}")
            if modelo_cl_usuario is not None:
                met_cl_usuario_sup = calcular_parametros_comparacion(M_ref_cl, cl_ref, modelo_cl_usuario, mach_min=1.2)
                if met_cl_usuario_sup: print(f"    USUARIO -> RMSE: {met_cl_usuario_sup['rmse']:.4f} | R²: {met_cl_usuario_sup['r2']:.4f}")

    if archivo_ref_cd: #el funcionamiento es igual que para el cl en las líneas anteriores
        ruta_ref = os.path.join(directorio_actual, str(archivo_ref_cd))
        M_ref_cd, cd_ref = cargar_datos_referencia(ruta_ref)
        if M_ref_cd is not None: #se calculan las métricas de cd si hay datos de referencia cargados
            #métricas de la curva completa para cd, igual que para cl
            print("\n Parámetros cd - CURVA COMPLETA")
            if not es_caso_extremo:
                met_cd_base_completa = calcular_parametros_comparacion(M_ref_cd, cd_ref, modelo_cd_base)
                if met_cd_base_completa: print(f"    TEORÍA  -> RMSE: {met_cd_base_completa['rmse']:.5f} | R²: {met_cd_base_completa['r2']:.4f}")
            if modelo_cd_usuario is not None:
                met_cd_usuario_completa = calcular_parametros_comparacion(M_ref_cd, cd_ref, modelo_cd_usuario)
                if met_cd_usuario_completa: print(f"    USUARIO -> RMSE: {met_cd_usuario_completa['rmse']:.5f} | R²: {met_cd_usuario_completa['r2']:.4f}")

            #se calcula por tramos las métricas para cd
            print("\n Parámetros cd - SUBSONICO (M < Mcrit)")
            if not es_caso_extremo:
                met_cd_base_sub = calcular_parametros_comparacion(M_ref_cd, cd_ref, modelo_cd_base, mach_max=mach_critico_val)
                if met_cd_base_sub: print(f"    TEORÍA  -> RMSE: {met_cd_base_sub['rmse']:.5f} | R²: {met_cd_base_sub['r2']:.4f}")
            if modelo_cd_usuario is not None:
                met_cd_usuario_sub = calcular_parametros_comparacion(M_ref_cd, cd_ref, modelo_cd_usuario, mach_max=mach_critico_val)
                if met_cd_usuario_sub: print(f"    USUARIO -> RMSE: {met_cd_usuario_sub['rmse']:.5f} | R²: {met_cd_usuario_sub['r2']:.4f}")

            print("\n Parámetros cd - TRANSONICO (Mcrit <= M < 1.2)")
            if not es_caso_extremo:
                met_cd_base_tra = calcular_parametros_comparacion(M_ref_cd, cd_ref, modelo_cd_base, mach_min=mach_critico_val, mach_max=1.2)
                if met_cd_base_tra: print(f"    TEORÍA  -> RMSE: {met_cd_base_tra['rmse']:.5f} | R²: {met_cd_base_tra['r2']:.4f}")
            if modelo_cd_usuario is not None:
                met_cd_usuario_tra = calcular_parametros_comparacion(M_ref_cd, cd_ref, modelo_cd_usuario, mach_min=mach_critico_val, mach_max=1.2)
                if met_cd_usuario_tra: print(f"    USUARIO -> RMSE: {met_cd_usuario_tra['rmse']:.5f} | R²: {met_cd_usuario_tra['r2']:.4f}")

            print("\n Parámetros cd - SUPERSONICO (M >= 1.2)")
            if not es_caso_extremo:
                met_cd_base_sup = calcular_parametros_comparacion(M_ref_cd, cd_ref, modelo_cd_base, mach_min=1.2)
                if met_cd_base_sup: print(f"    TEORÍA  -> RMSE: {met_cd_base_sup['rmse']:.5f} | R²: {met_cd_base_sup['r2']:.4f}")
            if modelo_cd_usuario is not None:
                met_cd_usuario_sup = calcular_parametros_comparacion(M_ref_cd, cd_ref, modelo_cd_usuario, mach_min=1.2)
                if met_cd_usuario_sup: print(f"    USUARIO -> RMSE: {met_cd_usuario_sup['rmse']:.5f} | R²: {met_cd_usuario_sup['r2']:.4f}")

    if M_ref_cl is not None or M_ref_cd is not None:
        #se genera la gráfica de comparación con la referencia
        grafica_comparacion(M_ref_cl, cl_ref, M_ref_cd, cd_ref, modelo_cl_base, modelo_cd_base,
            modelo_cl_usuario, modelo_cd_usuario, None, None, None, None,
            es_caso_extremo, angulo_ataque_geom, espesor_relativo, directorio_resultados,
            tope_mach_cl_base, tope_mach_cd_base, tope_mach_cl_usuario, tope_mach_cd_usuario,
            predef_cl_m, predef_cl_v, predef_cl_n,
            predef_cd_m, predef_cd_v, predef_cd_n,
            nodos_usuario_cl_m, nodos_usuario_cl_c, nodos_usuario_cd_m, nodos_usuario_cd_c)

    print(f"Simulacion del perfil finalizada. Todos los archivos se encuentran en {directorio_resultados}")

if __name__ == "__main__": #al ejecutar, name se le asigna el valor de main y al llegar a esta línea como la condición es true se ejecuta la función main
    main()