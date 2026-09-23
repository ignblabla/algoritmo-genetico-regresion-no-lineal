# 🧬 Algoritmo Genético para Regresión No Lineal

Este repositorio contiene la implementación en Python de un algoritmo genético diseñado para resolver problemas de regresión no lineal, optimizando los hiperparámetros de una función predictiva.

Este proyecto fue desarrollado en el marco del Departamento de Ciencias de la Computación e Inteligencia Artificial de la **Universidad de Sevilla**.

## 📖 Descripción del Problema

El objetivo principal es encontrar la función $\hat{y} = f(x_1, ..., x_n)$ que mejor aproxime los valores de una variable dependiente $y$. A diferencia de la regresión lineal tradicional, este proyecto se centra en optimizar la siguiente familia de funciones no lineales:

$$
\hat{y} = c_1 x_1^{e_1} + c_2 x_2^{e_2} + ... + c_k x_k^{e_k} + C
$$

Donde los coeficientes a determinar mediante evolución computacional son:

* $c_i \in \mathbb{R}$: Coeficientes multiplicativos.
* $e_i \in \mathbb{Z}$: Exponentes enteros (necesarios para evitar números complejos con bases negativas).
* $C \in \mathbb{R}$: Constante aditiva.

El algoritmo busca minimizar el **Error Cuadrático Medio (MSE / RMSE)**, utilizado como métrica principal para la función de *fitness*.

## ⚙️ Arquitectura del Software

El proyecto sigue el paradigma de Programación Orientada a Objetos (POO) estructurado en dos clases principales:

1. **`Clase AG` (Algoritmo Genético):** Controla el ciclo de vida de la evolución.
   * Lectura de datos y configuración del modelo.
   * Evaluación de la población poblacional.
   * Selección por torneo (`nTorneo = 3`).
   * Cruzamiento puntual (`pCruce = 0.8`).
   * Mutación iterativa poblacional.
   * Elitismo para preservar las mejores soluciones.

2. **`Clase Individuo` (Cromosoma):** Representa una posible solución matemática.
   * Generación y mutación de genes específicos (coeficientes, exponentes, constante).
   * Cálculo de *fitness* (MSE) utilizando las variables predictoras y respuestas.
   * Realización de predicciones individuales o sobre un conjunto de datos.

## 📊 Espacio de Búsqueda y Parámetros

Los valores por defecto que dirigen la exploración del algoritmo (configurables según el conjunto de datos) son:

* $c_i \in [-2, 2]$
* $e_i \in \{-3, -2, -1, 0, 1, 2, 3\}$
* $C \in [-5, 5]$
* **Tamaño de población:** 50 individuos
* **Iteraciones máximas:** 100
* **Probabilidad de mutación:** 0.1 (mutando 1 gen)

## 📁 Conjuntos de Datos Soportados

El algoritmo ha sido probado y validado con los siguientes conjuntos de datos experimentales, los cuales deben dividirse en entrenamiento (train) y validación (test):

* **`toy1`**: 5 variables (100 train / 30 test)
* **`synt1`**: 100 variables (1000 train / 200 test)
* **`housing`**: 8 variables (11558 train / 1634 test)

## 🚀 Instalación y Uso

### Prerrequisitos

* Python 3.x
* Librerías estándar de manejo de datos (ej. `pandas`, `numpy`, `math`).

### Ejecución Básica

```python
# Ejemplo de uso de la librería (Basado en la estructura descrita)
from ag_regression import AG

# Inicializar el Algoritmo Genético con parámetros y rutas de datos
algoritmo = AG(
    iteraciones=100, 
    tam_poblacion=50, 
    p_cruce=0.8, 
    p_mutacion=0.1,
    train_file="data/toy1_train.csv",
    test_file="data/toy1_test.csv"
)

# Ejecutar la optimización
coeficientes_optimos, predicciones_test = algoritmo.run()

print(f"Mejores coeficientes encontrados: {coeficientes_optimos}")
```

## 👥 Autores

* **Ignacio Blanquero Blanco** (`ignblabla@alum.us.es`)
* **Adrián Cabello Martín** (`adrcabmar@alum.us.es`)

Departamento de Ciencias de la Computación e Inteligencia Artificial, Universidad de Sevilla (España).
