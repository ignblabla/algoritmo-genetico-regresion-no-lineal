import numpy as np
import random
import copy
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


#%%
#=======================================================================
# Clase AG
#=======================================================================
#

class AG:

    """
    Esta clase implementa las propiedades y métodos necesarios para la resolución mediante un algoritmo genético de un problema de regresión no lineal
    de la forma y = c_1*x_1^e_1 + c_2*x_2ê_2 + ... + c_k*x_k^e_k + C.
    Los individuos (cromosomas) que componen la población son 2*k+1-uplas de la forma (c_1,e_1,c_2,e_2,...,c_k,e_k,C), donde los c_i y C son valores reales 
    y los e_i son valores enteros. 
    Se complementa con la clase Individuo.
    """

    #==========================================================================================================================
    #
    def __init__(self, datos_train, datos_test, seed=11939, nInd=50, maxIter=100, nTorneo=3, nMutacion=1, tElitismo=0.2, pCruce=0.8, pMutacion=0.1, tol=1e-6,
                 min_c=-2, max_c=2, min_e=-3, max_e=3, min_const=-5, max_const=5):

        '''
        Constructor de la clase. Recibe como argumento los distintos parámetros que regulan el funcionamiento del algoritmo genético (número de
        iteraciones, tamañod de la población, probabilidades de cruce y de mutación, etc.), así como los datos de entrenamiento y validación.
        Inicializa las propiedades del objeto.  
        '''
        # Nombre del fichero de datos de entrenamiento
        self.datos_train = datos_train
        # Nombre del fichero de datos de validación
        self.datos_test = datos_test
        # Valor de la semilla del generador de números aleatorios
        self.seed = seed
        # Número de individuos de la población
        self.nInd = nInd
        # Número máximo de iteraciones del algoritmo
        self.maxIter = maxIter
        # Número de individuos que participan en cada torneo en la selección por torneo
        self.nTorneo = nTorneo
        # Número de genes a mutar
        self.nMutacion = nMutacion
        # Tasa de elistismo (para selección de padres)
        self.tElitismo = tElitismo
        # Probabilidad de cruce 
        self.pCruce = pCruce
        # Probabilidad de mutacion
        self.pMutacion = pMutacion
        # Individuos que componen la población
        self.poblacion = list()
        # Nümero de variables predictoras del modelo de regresión
        self.nVar = None
        # Dimensión de cada individuo (número de genes)
        self.nGen = None
        # Datos de entrenamiento
        self.train = None
        # Número de instancias en conjunto de entrenamiento 
        self.nTrain = None
        # Datos de validación
        self.test = None
        # Número de instancias en conjunto de validación
        self.nTest = None
        # Mejor individuo obtenido
        self.IndOpt = None
        # Fitness del mejor individuo obtenido
        self.FitOpt = float('inf')
        # Tolerancia: el algoritmo se detendrá cuando el valor del fitness sea inferior a este valor
        self.tol = tol
        # Valor mínimo del coeficiente multiplicativo de las variables predictoras x_i
        self.min_c = min_c
        # Valor máximo del coeficiente multiplicativo de las variables predictoras x_i
        self.max_c = max_c
        # Valor mínimo del exponente de las variables predictoras x_i
        self.min_e = min_e
        # Valor máximo del exponente de las variables predictoras x_i
        self.max_e = max_e
        # Valor mínimo constante aditiva C
        self.min_const = min_const
        # Valor máximo constante aditiva C
        self.max_const = max_const
        

        #----------------------------------------------------------
        # Validación 
        #----------------------------------------------------------

        if self.nInd%2 == 1:
            raise Exception(
                "El número de individuos en la población debe ser par"
            )    
        if self.datos_train is None:
            raise Exception(
                "El nombre del fichero de datos de entrenamiento no puede estar en blanco"
            )
        if self.datos_test is None:
            raise Exception(
                "El nombre del fichero de datos de validación no puede estar en blanco"
            )
        if self.tElitismo<0 and self.tElitismo>1:
            raise Exception(
                "El valor de la tasa de elitismo es incorrecto"
            )
        if self.pCruce<0 or self.pCruce>1:
            raise Exception(
                "El valor de la probabilidad de cruce es incorrecto"
            )
        if self.pMutacion<0 or self.pMutacion>1:
            raise Exception(
                "El valor de la probabilidad de mutación es incorrecto"
            )


    #==========================================================================================================================
    #
    def Lectura_datos(self):

        ''' 
        Esta función realiza la lectura de los ficheros csv que contiene los datos de entrenamiento y de validación para el 
        algoritmo genético. Además, determina ciertos valores a partir de estos ficheros, como el número de variables del modelo 
        de regresión (nVar), el número de genes que contendrá cada cromosoma (nGen) o el número de instancias en las muestras de 
        entrenamiento y validación
        '''
        # Leer fichero de datos de entrenamiento
        self.train = pd.read_csv(self.datos_train)

        # Determinar número de filas y columnas (training)
        Filas_train, Col_train = self.train.shape

        self.nTrain = Filas_train
        self.nVar = Col_train - 1            # La última columna es la variable respuesta
        self.nGen = 2*self.nVar+1

        # Leeer fichero de datos de validación
        self.test = pd.read_csv(self.datos_test)

        # Determinar número de filas y columnas (test)
        Filas_test, Col_test = self.test.shape    

        self.nTest = Filas_test
        if Col_test != Col_train:
            raise Exception(
                "Número de columnas no coinciden en ficheros de entrenamiento y validación"
            )

    #==========================================================================================================================
    #
    def Generar_Poblacion(self):

        '''
        Esta función genera los individuos que compondrán la población mediante un bucle
        en el que se crean tantas instancias de la clase Individuo como se requieran (Nind)
        '''

        self.poblacion = [Individuo(self.nGen,self.min_c,self.max_c,self.min_e,self.max_e,self.min_const,self.max_const) for _ in range(self.nInd)] 

        
    #==========================================================================================================================
    #
    def Evaluar_Poblacion(self):

        '''
        Esta función calcula el fitness de los individuos de la población que no hayan sido evaluados con anterioridad. Además,
        actualiza el mejor valor obtenido hasta el momento, así como el individuo en el que se obtiene, en caso de que se
        obtenga un valor fitness que lo mejore (se trata de un problema de minimización)
        '''

        # Extraer valores de las variables predictoras y de la variable respuesta del conjunto de entrenamiento
        x = self.train.iloc[:,0:self.nVar]         # Variables predictoras
        y = self.train.iloc[:,self.nVar]           # Variable respuesta

        # Bucle de procesamiento de los individuos de la población
        for i in range(self.nInd):
            
            ind = self.poblacion[i]

            # Solo se evalúan los individuos no evaluados anteriormente
            if ind.fitness is None:
            
                # Calcular fitness del individuo
                fit = ind.Calcular_Fitness(x, y)
            
                # Actualizar valores óptimos, si procede
                if fit < self.FitOpt:
                    self.FitOpt = fit
                    self.IndOpt = ind

    #==========================================================================================================================
    #
    def Seleccionar_Padres(self):

        '''
        Esta función realiza la selección de padres por torneo para la siguiente generación. El número de padres seleccionados 
        (mediante llemada a la función Realizar_Torneo) es igual al número de individuos de la población. Devuelve una lista
        con los padres seleccionados. 
        '''

        Padres_Sel = list()
        for _ in range(self.nInd):
            sel_ind = self.Realizar_Torneo()
            Padres_Sel.append(sel_ind)

        # Padres_Sel = [self.Realizar_Torneo() for _ in range(self.nInd)]           # De otra forma

        return(Padres_Sel)

    #==========================================================================================================================
    #
    def Realizar_Torneo(self):

        '''
        Esta función realiza la selección de un individuo por torneo. Para ello se seleccionan aleatoriamente un número 
        determinado (nTorneo) de individuos de la población y se escoge el que presente el mejor fitness (mínimo fitness) 
        Devuelve el individuo ganador del torneo.
        '''

        # Inicializar el fitness óptimo a +infinito.
        fitness_opt = float('inf')

        for _ in range(self.nTorneo):

            # Elegir aleatoriamente un individuo de la población
            ind = random.choice(self.poblacion)
        
            # Si se mejora el fitness óptimo obtenido hasta el momento, actualizarlo
            if ind.fitness < fitness_opt:
                fitness_opt = ind.fitness
                ind_opt = ind

        return(ind_opt)        
    
    #==========================================================================================================================
    #      
    def Cruzar_Individuos(self,Padres):

        '''
        Esta función realiza aleatoriamente el cruce de los individuos que recibe e una lista de individuos (Padres). 
        Va seleccionando secuencialmente parejas de individuos de la lista y con probabilidad pCruce procede a realizar el 
        cruce de los mismos mediante llamada a la función Cruce_Puntual. Para cada pareja, los nuevos individuos resultantes
        del cruce o los individuos originales, si no ha habido cruce, se almacenan en la lista Descendencia, que se devuelve
        como salida de la función 
        '''

        # Inicializar lista de salida
        Descendencia = []

        # Recorrer de dos en dos la lista de individuos a cruzar 
        for i in range(0,len(Padres),2):

            # Generar valor aleatorio entre 0 y 1
            p = random.random()

            # Si el valor aleaotorio es inferior a la probabilidad de cruce, este se realiza
            if p <= self.pCruce:
                # Realizar cruce
                Hijos = self.Cruce_Puntual(Padres[i],Padres[i+1])
                # Añadir los nuevos individuos a la descendencia
                Descendencia.extend(Hijos)
            else:       
                # No hay cruce: permanece la pareja original
                Descendencia.extend([Padres[i],Padres[i+1]])

        return Descendencia        

    #==========================================================================================================================
    #
    def Cruce_Puntual(self,Padre1,Padre2):
        
        '''
        Esta función realiza el cruce en un punto, elegido aleatoriamente, de los individuos recibidos como entrada, dando 
        lugar a dos nuevos individuos. Devuelve una lista con dos individuos generados.
        '''

        # Determinar posición (aleatoria) del cruce
        pos = random.randint(1,self.nGen-1)

        # Generar dos instancias de la clase Individuo correspondientes a los hijos
        Hijo1 = Individuo(self.nGen,self.min_c,self.max_c,self.min_e,self.max_e,self.min_const,self.max_const)
        Hijo2 = Individuo(self.nGen,self.min_c,self.max_c,self.min_e,self.max_e,self.min_const,self.max_const)

        # Realizar el cruce de los individuos en la posición pos
        Hijo1.genes = Padre1.genes[:pos] + Padre2.genes[pos:] 
        Hijo2.genes = Padre2.genes[:pos] + Padre1.genes[pos:]

        return [Hijo1,Hijo2] 

    #==========================================================================================================================
    #
    def Mutar_Poblacion(self,Poblacion):

        '''
        Para cada individuo recibido como argumento de llamada, realiza una llamada a la función de mutación individual 
        para decidir si se realiza una mutación en algún gen del cromosoma correspondiente.
        '''    

        return [ind.Mutar_Individuo(self.pMutacion,self.nMutacion) for ind in Poblacion]

    #==========================================================================================================================
    #
    def run(self):

        '''
        Esta función ejecuta las distintas operaciones que componen el algoritmo genético. Tras realizar distintas tareas
        de inicialización (inicialización de la semilla del generador de números aleatorios, lectura de fichero de datos, 
        generación y evaluación de la población inicial), se lleva a cabo el bucle de optimización, en el que se ejecutan 
        en cada iteración los distintos pasos del algoritmo genético: selección, cruce y mutación, para dar lugar a una 
        nueva generación. Devuelve los coeficientes óptimos del modelo de regresión, así como las predicciones realizadas 
        sobre el conjunto de test utilizando estos coeficientes.
        '''

        # Establecer semilla del generador de números aleatorios
        random.seed(self.seed)

        # Lectura de datos de entrenamiento y validación    
        self.Lectura_datos()

        # Generar la población inicial
        self.Generar_Poblacion()

        # Evaluar fitness de los individuos de la población
        self.Evaluar_Poblacion()

        # Bucle de optimización
        for i in range(self.maxIter):

            #print('Iter:',i,'   mse:',self.FitOpt)

            # Seleccionar padres por torneo
            Padres_Sel = self.Seleccionar_Padres() 
            
            # Inicializar lista de individuos de la nueva generación
            Nueva_Generacion = list()
            
            # Si hay selección elitista
            if self.tElitismo > 0:

                # Número de individuos con mejor fitness que pasan directamente a la siguiente generación
                nElite = int(np.ceil(self.nInd*self.tElitismo))

                # Determinar longitud de la lista de padres seleccionados
                L_Padres_Sel = len(Padres_Sel)

                # Obtener un array numpy con los valores de fitness de los padres 
                fitness = np.empty(L_Padres_Sel)

                # Copiar en el array numpy los valores fitness de los padres seleccionados
                for i in range(L_Padres_Sel):
                    fitness[i] = Padres_Sel[i].fitness

                # Ordenar el array de fitness y obtener los índices de la ordenación    
                fitness_indices = np.argsort(fitness)

                # Obtener los nElite padres con mejor fitness a partir de los índices de la ordenación
                Elite = [Padres_Sel[i] for i in fitness_indices[0:nElite]]

                # Eliminar de la lista de padres los elegidos mediante selección elitista
                Padres_Sel = copy.deepcopy([Padres_Sel[i] for i in fitness_indices[nElite:L_Padres_Sel]])
                
                # Almacenar en la lista Nueva_Generacion los individuos de la selección elitista
                Nueva_Generacion = copy.deepcopy(Elite)

            # Cruzar los padres que no intervienen en la selección elitista
            Descendencia = self.Cruzar_Individuos(Padres_Sel)

            # Añadir a la lista Nueva_Generación los individuos resultantes del cruce
            Nueva_Generacion.extend(Descendencia)

            # Mutar los individuos de la siguiente generación
            Nueva_Generacion = self.Mutar_Poblacion(Nueva_Generacion)

            # Reemplazar la generación actual
            self.poblacion = Nueva_Generacion

            # Evaluar individuos de la población
            self.Evaluar_Poblacion()

            if (self.FitOpt < self.tol): 
                break
            
        # Obtener valores de las variables predictoras en el conjunto de test
        x = self.test.iloc[:,0:self.nVar]
        
        # Devolver coeficientes óptimos y predicciones sobre el conjunto test
        return self.IndOpt.genes, self.IndOpt.Calcular_Predicciones(x)
    


#%%
#=======================================================================
# Clase Individuo
#=======================================================================
#

class Individuo:
    
    """
    Esta clase permite la representación de cada individuo de la población utilizada en el algoritmo genético para 
    resolución del problema de regresión no lineal considerado.
       
    """
    #==========================================================================================================================
    #
    def __init__(self, nGen, min_c=-2, max_c=2, min_e=-3, max_e=3, min_const=-5, max_const=5):

        '''
        Constructor de la clase. Recibe el tamaño que tendrán los cromosomas (individuos) y determina a partir de él el número de 
        variables explicativas del modelo de regresión. Además de inicializar otras propiedades del objeto, genera aleatoriamente 
        el valor de los genes del individuo.  
        '''
        # Dimensión del individuo (número de genes del cromosoma)
        self.nGen = nGen
        # Genes
        self.genes = None
        # Fitness del individuo
        self.fitness = None
        # Número de variables predictoras en el modelo de regresión
        self.nVar = None
        # Valor mínimo del coeficiente multiplicativo de las variables predictoras x_i
        self.min_c = min_c
        # Valor máximo del coeficiente multiplicativo de las variables predictoras x_i
        self.max_c = max_c
        # Valor mínimo del exponente de las variables predictoras x_i
        self.min_e = min_e
        # Valor máximo del exponente de las variables predictoras x_i
        self.max_e = max_e
        # Valor mínimo constante aditiva C
        self.min_const = min_const
        # Valor máximo constante aditiva C
        self.max_const = max_const

        #--------------------------------------------------
        # Validación
        #--------------------------------------------------

        if self.nGen is None or (self.nGen is not None and self.nGen <= 0) or self.nGen%2 == 0:
            raise Exception(
                "Valor incorrecto del número de genes"
            )
        
        if min_c > max_c:
            raise Exception(
                "Error en límites de los coeficientes multiplicativos"
            )
        
        if min_e > max_e:
            raise Exception(
                "Error en límites de los exponentes"
            )
        
        if min_const > max_const:
            raise Exception(
                "Error en límites de la constante aditiva"
            )

        #--------------------------------------------------
        # Cálculo del número de variables (regresión)
        #--------------------------------------------------

        self.nVar = int((self.nGen-1)/2)

        #--------------------------------------------------
        # Creación del individuo
        #--------------------------------------------------

        lGen = [self.Generar_Gen(i) for i in range(self.nGen)]

        self.genes = tuple(lGen)

    #==========================================================================================================================
    #
    def Generar_Gen(self,pos):
        
        '''
        Genera un valor aleatorio para un gen del cromosoma (individuo). Dependiendo de la posición del gen, que se recibe
        como argumento, se determina si se debe generar el valor de un coeficiente multiplicativo, de un exponente o de la
        constante aditiva final del modelo de regresión y = c_1*x_1^e_1 + c_2*x_2^e_2 + ... + c_k*x_k^e_k + C
         '''
        
        if pos == self.nGen-1:
            # Generar constante aditiva final C
            valor = random.uniform(self.min_const,self.max_const)
        else:
            if pos%2 == 0:
                # Coeficiente multiplicativo c_i
                valor = random.uniform(self.min_c,self.max_c)
            else:
                # Exponente e_i
                valor = random.randint(self.min_e,self.max_e)

        return valor

    #==========================================================================================================================
    #
    def Calcular_Fitness(self, x, y):
        
        '''
        Evalúa el fitness del individuo actual a partir de los valores de las variables explicativas "x" y del valor de la 
        variable respuesta "y" para un conjunto de instancias. Se han considerado dos medidas de bondad de ajuste (error 
        cuadrático medio y error absoluto medio) para evaluar el fitness del individuo.
        '''

        # Calcular predicciones para los valores recibidos de las variables explicativas
        pred = self.Calcular_Predicciones(x)

        # Calcular el error cuadrático de las predicciones
        mse = mean_squared_error(y,pred)

        # Alternativa: Calcular el error absoluto medio de las predicciones
        # mae = mean_absolute_error(y,pred)

        # Almacenar fitness
        self.fitness = mse

        return(mse)

    #==========================================================================================================================
    #
    def Calcular_Predicciones(self,x):

        '''
        Calcula el valor pronosticado de la variable respuesta "y" en el modelo de regresión considerado para todas las 
        instancias recibidas en el Dataframe "x". Devuelve una lista con las predicciones obtenidas.
        '''
        # Determinar número de filas y columnas del dataframe x
        filas, cols = x.shape
        
        # Calcular la predicción para cada instancia (fila) del Dataframe x
        pred = [self.Calcular_Prediccion_Individual(x.iloc[i,:]) for i in range(filas)]

        return(pred)


    #==========================================================================================================================
    #
    def Calcular_Prediccion_Individual(self,x_ind):

        '''
        Calcula la predicción de la variable respuesta "y" en el modelo de regresión 
        y = c_1*x_1^e_1 + c_2*x_2^e_2 + ... + c_k*x_k^e_k + C
        para una instancia (x_1,x_2,...,x_k) recibida en la llamada.
        '''

        # Inicializar valor acumulativo de la predicción
        suma = 0.0

        # Bucle para procesamiento de cada variable explicativa x_i
        for i in range(self.nVar):
            # Si el valor de la variabel predictora x_i no es cero, se calcula c_i^*x_i^e_i y se añade a la suma
            if x_ind.iloc[i] != 0.0:
                suma += self.genes[i*2]*np.power(x_ind.iloc[i],self.genes[i*2+1])
        
        # Añadir constante aditiva final
        suma += self.genes[self.nGen-1]

        return(suma)
    

    #==========================================================================================================================
    #
    def Mutar_Individuo(self,pMutacion,nMutacion):

        '''
        Efectúa, con probabilidad pMutacion, una mutación en nMutacion genes elegidos aleatoriamente
        '''

        if pMutacion < 0 or pMutacion > 1:
            raise Exception(
                "El valor de la probabilidad de mutación es incorrecto"
            )

        # Generar valor aleatorio entre 0 y 1.
        p = random.random()

        # Si el valor elegido es inferior a pMutacion, realizar la mutación
        if p < pMutacion:

            for i in range(nMutacion):

                # Elegir aleatoriamente el gen a mutar
                pos = random.randint(0,self.nGen-1)
            
                # Borrar el valor de fitness almacenado, pues es un nuevo individuo
                self.fitness = None

                # Generar nuevo gen mutado (tener en cuenta que es una tupla)
                self.genes = self.genes[:pos] + (self.Generar_Gen(pos),) + self.genes[pos+1:]  
            
        return self
        