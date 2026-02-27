# 🎵 Predicción de Popularidad en Spotify (1921-2020)
Trabajo Final - Asignatura de Aprendizaje Profundo 
Autores: Javier Beny Rodríguez y Adrián Blat Campos  
Fecha: Febrero 2026  


## 📚 1. Definición del Problema y Estado del Arte

### 1.1. Descripción del Problema
El objetivo principal de este proyecto es determinar la capacidad predictiva de las características de audio intrínsecas de una canción sobre su popularidad comercial. Se trata de un problema de regresión supervisada, donde el modelo debe aprender una función que mapee variables acústicas (como 'tempo', 'energy', 'danceability', etc.) a un valor continuo de popularidad en el rango [0-100].

### 1.2. Análisis del Estado del Arte (SOTA)
En la literatura sobre Music Information Retrieval (MIR) y predicción de éxitos, se ha establecido que la relación entre las características de audio y la popularidad no es lineal. Por ello, es necesario comparar modelos lineales clásicos frente a arquitecturas no lineales más complejas.

Para este proyecto, se ha diseñado una tabla de experimentación que se irá completando con los resultados obtenidos, comparando nuestro enfoque de Deep Learning frente a un modelo base estándar.

### Tabla de Modelos y Resultados
La siguiente tabla recoge los modelos seleccionados para el estudio y sus métricas de rendimiento (se completará tras la fase de entrenamiento):

| Modelo              | Tipo de Algoritmo        | MSE (Test) | RMSE (Test) | R² (Test) | Estado |
|---------------------|--------------------------|------------|-------------|-----------|--------|
| Regresión Lineal    | Modelo clásico           | 114.93     | 10.72       | 0.7505    | Baseline |
| Random Forest       | Machine Learning (Trees) | 89.44      | 9.46        | 0.8058    | ⭐ Mejor modelo |
| Red Neuronal (MLP)  | Deep Learning            | 107.44     | 10.37       | 0.7668    | Estable |

Métricas de Evaluación:  
Para este problema de Regresión, las métricas estándar utilizadas son:  
- MSE (Mean Squared Error): Penaliza los errores grandes.  
- RMSE (Root Mean Squared Error): Error promedio en las mismas unidades que la popularidad (0-100).  
- R² (Coeficiente de Determinación): Indica qué porcentaje de la varianza de la popularidad es explicada por el modelo.  


## 📊 2. El Dataset
Fuente: Spotify Dataset 1921-2020 (https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks)  
Tamaño original: ~170.000 muestras.  
Variables de entrada (Features): acousticness, danceability, duration_ms, energy, explicit, instrumentalness, liveness, loudness, speechiness, tempo, valence, year.  


## 🧹 3. Preprocesamiento y Limpieza de Datos
Uno de los desafíos principales de este dataset es la gran cantidad de valores con popularity = 0. Para garantizar que el modelo aprenda patrones musicales reales, realizamos la siguiente distinción:

### 3.1. Distinción: Impopularidad Genuina vs. Ruido Técnico
✅ SE MANTIENEN (Señal Histórica): Canciones antiguas (1920-1960) o de nicho con popularidad 0. Justificación: Representan datos válidos sobre cómo la antigüedad penaliza el éxito.  

❌ SE ELIMINAN (Ruido Técnico): Archivos que no son canciones musicales.
- Duración < 40s: Intros, efectos de sonido.  
- Speechiness > 0.90: Audiolibros, discursos.  
- Tempo = 0: Errores de datos.

### 3.2. Resultado del Preprocesamiento
Tras aplicar filtros y eliminar outliers extremos de duración (+15 min), el dataset se redujo a ~166.000 muestras. Se ha aplicado normalización (StandardScaler) a todas las variables numéricas para el correcto funcionamiento de la Red Neuronal.


## 📈 4. Modelo Simple 1: Regresión Lineal Simple

### 🔍 Preparación de los datos

Para este primer enfoque se ha utilizado un modelo de **regresión lineal** con el objetivo de predecir la **popularidad de las canciones en Spotify** a partir de sus características acústicas.

Antes del entrenamiento, se realizó una limpieza de datos para eliminar ruido que pudiera distorsionar el modelo:

- Se eliminaron canciones con:
  - Duración inferior a 40 segundos.
  - `speechiness` superior a 0.90 (audiolibros, charlas o pistas casi exclusivamente de voz).
  - Tempo igual a 0.
- En total, se eliminaron **4.665 canciones**, reduciendo el dataset de **170.653 a 165.988 muestras**.
- A pesar de la limpieza, se conservan **19.073 canciones antiguas con popularidad 0**, lo que refleja un fuerte desequilibrio temporal en los datos.

Las variables no predictivas o identificativas (`id`, `name`, `artists`, fechas, etc.) fueron eliminadas, manteniendo únicamente las **features numéricas relevantes**.  
La variable objetivo (`y`) es la **popularidad**.

---

### 📊 División del dataset

El conjunto de datos se dividió de la siguiente forma:

- **70% Entrenamiento**: 116.181 canciones  
- **15% Validación**: 24.896 canciones  
- **15% Test**: 24.896 canciones  

Antes de entrenar el modelo, las variables predictoras fueron **estandarizadas** mediante `StandardScaler`, garantizando que todas las features contribuyeran de forma equilibrada.

---

### 🧠 Modelo

Se entrenó un modelo de **Regresión Lineal** (`LinearRegression`) sobre los datos escalados.  
Este modelo actúa como **baseline**, proporcionando una referencia sencilla e interpretable frente a modelos más complejos que se evaluarán posteriormente.

---

### 📈 Resultados

| Conjunto        | MSE    | RMSE | R²     |
|-----------------|--------|------|--------|
| Entrenamiento   | 120.07 | 10.96 | 0.7420 |
| Validación      | 119.48 | 10.93 | 0.7442 |
| Test            | 114.93 | 10.72 | 0.7505 |

---

### 📝 Interpretación

- El modelo explica aproximadamente **el 75% de la varianza de la popularidad**, lo cual es un resultado notable para un modelo lineal simple.
- El **RMSE (~11 puntos de popularidad)** indica que, de media, las predicciones se desvían en torno a 11 puntos sobre una escala de 0 a 100.
- La similitud entre los resultados de entrenamiento, validación y test sugiere que **no existe sobreajuste**.
- Sin embargo, la naturaleza lineal del modelo limita su capacidad para capturar relaciones complejas entre las variables acústicas y la popularidad.

En conclusión, este modelo proporciona una **base sólida y estable**, útil como referencia para evaluar las mejoras introducidas por modelos más avanzados (regularización, árboles de decisión, boosting, etc.).


## 🌲 5. Modelo Simple 2: Random Forest Regressor

### 🔍 Preparación de los datos

Para este segundo modelo se utilizó un **Random Forest Regressor**, un algoritmo de **aprendizaje automático basado en árboles de decisión** que permite capturar relaciones no lineales entre las características acústicas y la popularidad de las canciones en Spotify.

La limpieza y preparación de los datos fue idéntica al modelo lineal:

- Eliminación de canciones con:
  - Duración inferior a 40 segundos.
  - `speechiness` superior a 0.90.
  - Tempo igual a 0.
- Se mantuvieron únicamente las **features numéricas relevantes**, eliminando identificadores y nombres.
- La variable objetivo (`y`) es la **popularidad**.

Los datos fueron escalados con `StandardScaler` antes de entrenar el modelo, garantizando que todas las features contribuyeran de manera equilibrada al aprendizaje.

---

### 📊 División del dataset

La división de los datos fue la misma que en el modelo lineal:

- **70% Entrenamiento**: 116.181 canciones  
- **15% Validación**: 24.896 canciones  
- **15% Test**: 24.896 canciones  

---

### 🧠 Modelo

Se entrenó un **Random Forest Regressor** con los siguientes parámetros:

- `n_estimators=50` (número de árboles)
- `max_depth=15` (profundidad máxima de cada árbol)
- `n_jobs=-1` (uso de todos los núcleos disponibles)
- `random_state=42` (reproducibilidad)

Este modelo es capaz de **capturar relaciones complejas y no lineales** que el modelo lineal no puede representar.

---

### 📈 Resultados

| Conjunto        | MSE    | RMSE | R²     |
|-----------------|--------|------|--------|
| Entrenamiento   | 57.20  | 7.56 | 0.8771 |
| Validación      | 94.23  | 9.71 | 0.7983 |
| Test            | 89.44  | 9.46 | 0.8058 |

- **Número total de parámetros (nodos): 734.216**, mostrando la complejidad del modelo y su capacidad para modelar patrones detallados.

---

### 📝 Interpretación

- El modelo logra un **R² de ~0.81 en test**, mejorando notablemente sobre el modelo lineal (~0.75). Esto indica que el Random Forest captura mejor la varianza de la popularidad.
- El **RMSE (~9.5 puntos)** es menor que en el modelo lineal (~10.7 puntos), lo que significa predicciones más precisas.
- Existe cierta diferencia entre entrenamiento y validación, lo que refleja que el modelo es más flexible y puede **ajustarse a patrones complejos**, aunque sin sobreajustarse excesivamente gracias a la limitación de profundidad (`max_depth=15`).
- La gran cantidad de nodos evidencia la complejidad y capacidad de memoria del modelo, necesaria para capturar las relaciones no lineales.

En conclusión, el **Random Forest** ofrece un **balance entre precisión y interpretabilidad**, y sirve como un segundo baseline robusto frente a modelos lineales, especialmente útil para detectar interacciones complejas entre las características de las canciones.


## 🧠 6. Modelo Simple 3: Red Neuronal Artificial (Neural Network)

### 🔍 Preparación de los datos

El tercer y último enfoque consiste en una **Red Neuronal Artificial (NN)**, diseñada para modelar relaciones no lineales entre las características acústicas y la **popularidad de las canciones en Spotify**.

La preparación de los datos es idéntica a la utilizada en los modelos anteriores, con el objetivo de garantizar una comparación justa:

- Eliminación de canciones con:
  - Duración inferior a 40 segundos.
  - `speechiness` superior a 0.90.
  - Tempo igual a 0.
- Eliminación de variables identificativas o no predictivas.
- La variable objetivo (`y`) es la **popularidad**.

Los datos se dividieron en entrenamiento, validación y test (70/15/15) y se **escalaron mediante `StandardScaler`**, paso especialmente importante para el correcto funcionamiento de redes neuronales.

---

### 🧠 Arquitectura del modelo

Se implementó una **red neuronal completamente conectada (feedforward)** sencilla, con el objetivo de evaluar si una arquitectura mínima es capaz de mejorar los modelos clásicos:

- **Capa oculta**:
  - 4 neuronas
- **Capa de salida**:
  - 1 neurona (regresión)
- **Parámetros totales**: **61**
- Función de pérdida: **MSE**
- Entrenamiento:
  - **20 épocas**
  - `batch_size = 256`

Esta arquitectura ligera permite analizar el poder de generalización del modelo sin introducir una complejidad excesiva.

---

### 📉 Curvas de aprendizaje

Durante el entrenamiento, tanto el error de entrenamiento como el de validación **disminuyen de forma progresiva y estable** a lo largo de las 20 épocas, sin divergencias entre ambas curvas.

Esto indica que:
- El modelo **aprende correctamente**.
- No se observa sobreajuste.
- El entrenamiento converge de forma estable.

---

### 📈 Resultados

| Conjunto        | MSE    | RMSE | R²     |
|-----------------|--------|------|--------|
| Entrenamiento   | 112.11 | 10.59 | 0.7591 |
| Validación      | 111.60 | 10.56 | 0.7611 |
| Test            | 107.44 | 10.37 | 0.7668 |

---

### 📝 Interpretación

- El modelo neuronal alcanza un **R² ≈ 0.77 en test**, mejorando ligeramente al modelo lineal, pero quedando por debajo del Random Forest.
- El **RMSE (~10.4 puntos)** es similar al modelo lineal, lo que indica que la red, en esta configuración, no logra explotar plenamente las relaciones no lineales del problema.
- La estabilidad entre entrenamiento, validación y test sugiere una **buena capacidad de generalización**, sin señales de sobreajuste.
- La arquitectura es extremadamente compacta (solo 61 parámetros), lo que limita su capacidad expresiva frente a modelos más complejos.

En conclusión, esta red neuronal actúa como un **modelo intermedio**: mejora ligeramente la regresión lineal manteniendo una alta estabilidad, pero no alcanza el rendimiento del Random Forest. Resultados potencialmente mejores podrían lograrse con arquitecturas más profundas, regularización o técnicas de ajuste de hiperparámetros.