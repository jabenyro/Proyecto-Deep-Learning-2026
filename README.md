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

| Modelo                        | Tipo de Algoritmo        | MSE (Test) | RMSE (Test) | R² (Test) | Estado |
|-------------------------------|--------------------------|------------|-------------|-----------|--------|
| Regresión Lineal              | Modelo clásico           | 114.93     | 10.72       | 0.7505    | Baseline|
| Random Forest                 | Machine Learning (Trees) | 89.44      | 9.46        | 0.8058    | ⭐ Mejor modelo global|
| Red Neuronal (MLP)            | Deep Learning            | 107.44     | 10.37       | 0.7668    | Estable |
| Red Más Neuronas (C1)         | Deep Learning            | 99.27      | 9.96        | 0.7845    | Estable |
| Red Varias Capas (C2)         | Deep Learning            | 93.26      | 9.66        | 0.7975    | Estable |
| Red Con Dropout (C3)          | Deep Learning            | 94.11      | 9.70        | 0.7957    | Estable |
| Red Batch Normalization (C4)  | Deep Learning            | 91.30      | 9.55        | 0.8018    | Estable |
| Red Ancha (C5)                | Deep Learning            | 93.36      | 9.66        | 0.7973    | Estable |
| Red Wide & Deep (C6)          | Deep Learning            | 91.91      | 9.59        | 0.8005    | Estable |
| Red Residual (C7)             | Deep Learning            | 91.33      | 9.56        | 0.8017    | Estable |
| Red Profunda Optimizada (C8)  | Deep Learning            | 90.59      | 9.52        | 0.8033    | Estable |
| Red Alta Convergencia (C9)    | Deep Learning            | 90.03      | 9.49        | 0.8046    | Estable |
| Red Swish (C10)               | Deep Learning            | 89.97      | 9.49        | 0.8047    | ⭐ Mejor modelo de Red Neuronal|

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


## 🧠 Modelo Complejo 1: Red Neuronal con Más Neuronas

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal Artificial (MLP - Multilayer Perceptron)** para predecir la popularidad de canciones.

El objetivo de este modelo complejo es **aumentar la capacidad de aprendizaje de la red neuronal incrementando el número de neuronas en la capa oculta**. De esta forma, el modelo puede capturar relaciones más complejas entre las variables del dataset musical.

La arquitectura utilizada consta de:

- 🔹 **1 capa oculta con 16 neuronas**
- 🔹 **1 capa de salida con 1 neurona** para la predicción de la popularidad

El modelo se implementa utilizando **TensorFlow / Keras**.

---

### 🧹 2. Preparación de los datos

Antes de entrenar la red neuronal se realiza un proceso de **limpieza y preparación de los datos**.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Este filtrado ayuda a mejorar la calidad de los datos utilizados para el entrenamiento.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que simplemente actúan como identificadores.

Variables eliminadas:

id
name
artists
id_artists
release_date
mode

La variable objetivo del modelo es:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esto permite evaluar el modelo correctamente y evitar sobreajuste.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Esto es especialmente importante en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite que el modelo **converja más rápido**
- Evita que variables con escalas grandes dominen el aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal Feedforward** utilizando **TensorFlow/Keras**.

#### 🧩 Arquitectura del modelo

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Capa oculta | Dense | 16 | 224 |
| Capa salida | Dense | 1 | 17 |

Total de parámetros entrenables:

241

---

#### ⚡ Configuración del entrenamiento

Parámetros principales utilizados:

- **Epochs máximas:** `500`
- **Batch size:** `64`
- **Función de pérdida:** `Mean Squared Error (MSE)`
- **Optimizador:** `Adam`

Para evitar sobreentrenamiento se utiliza **Early Stopping**, que monitoriza el error de validación y restaura automáticamente los mejores pesos del modelo.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar en problemas de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 103.53 | 10.17 | 0.7775 |
| 🟡 Validación | 103.23 | 10.16 | 0.7790 |
| 🔵 Test | 99.27 | 9.96 | 0.7845 |

---

### 🔎 5. Interpretación de resultados

Los resultados muestran que la red neuronal consigue explicar aproximadamente **un 78% de la variabilidad de la popularidad de las canciones**, según el valor de **R² en el conjunto de test (0.7845)**.

Observaciones principales:

✅ Los valores de **MSE y RMSE son muy similares en entrenamiento, validación y test**, lo que indica que el modelo **generaliza correctamente**.

✅ No se observa **sobreajuste significativo**, ya que el rendimiento es consistente entre los distintos conjuntos de datos.

✅ El aumento del número de neuronas permite que el modelo **capture relaciones más complejas entre las características musicales**.

El **RMSE cercano a 10** indica que el modelo comete, de media, un error de aproximadamente **10 puntos en la escala de popularidad (0–100)**.

---

### 🧾 6. Conclusiones

El modelo de red neuronal con **mayor número de neuronas en la capa oculta** muestra un rendimiento sólido en la predicción de la popularidad de canciones.

Principales conclusiones:

🎯 El modelo presenta **buena capacidad de generalización**.

📊 El rendimiento es **estable entre entrenamiento, validación y test**.

🧠 Incrementar el número de neuronas permite **capturar patrones más complejos en los datos musicales**.

Este modelo será comparado posteriormente con otras arquitecturas de redes neuronales para analizar **cómo afectan los cambios en la arquitectura al rendimiento predictivo**.

---


## 🧠 Modelo Complejo 2: Red Neuronal con Varias Capas

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal Artificial (MLP - Multilayer Perceptron)** con **varias capas ocultas**, para capturar relaciones más complejas entre las variables de entrada y mejorar la predicción de popularidad.

La arquitectura utilizada consta de:

- 🔹 **Capa oculta 1:** 32 neuronas  
- 🔹 **Capa oculta 2:** 16 neuronas  
- 🔹 **Capa oculta 3:** 8 neuronas  
- 🔹 **Capa de salida:** 1 neurona

El modelo se implementa utilizando **TensorFlow / Keras**.

---

### 🧹 2. Preparación de los datos

Se realiza la misma preparación de datos que en el Modelo Complejo 1:

### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**  
- Canciones con **speechiness mayor a 0.90**  
- Canciones con **tempo menor o igual que 0**

#### 🗑 Eliminación de variables no relevantes

Variables eliminadas:

id
name
artists
id_artists
release_date
mode

Variable objetivo:

popularity

#### 📊 División del dataset

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada y asegurar estabilidad en el entrenamiento.

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal Feedforward** utilizando **TensorFlow/Keras** con **Early Stopping** para evitar sobreentrenamiento.

#### 🧩 Arquitectura del modelo

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Capa oculta 1 | Dense | 32 | 448 |
| Capa oculta 2 | Dense | 16 | 528 |
| Capa oculta 3 | Dense | 8 | 136 |
| Capa salida | Dense | 1 | 9 |

Total de parámetros entrenables:

1121

---

#### ⚡ Configuración del entrenamiento

- **Epochs máximas:** 500  
- **Batch size:** 64  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `Adam`  
- **Early Stopping:** monitoriza la pérdida en validación, restaura los mejores pesos y paciencia de 30 epochs

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa con:

- 📉 **MSE (Mean Squared Error)**  
- 📊 **RMSE (Root Mean Squared Error)**  
- 📏 **R² (Coeficiente de determinación)**

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 96.96 | 9.85 | 0.7917 |
| 🟡 Validación | 97.83 | 9.89 | 0.7906 |
| 🔵 Test | 93.26 | 9.66 | 0.7975 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **un 79% de la variabilidad de la popularidad** según R² en test (0.7975).  
✅ Los valores de **MSE y RMSE son consistentes** entre entrenamiento, validación y test, indicando buena **generalización**.  
✅ Las múltiples capas permiten que la red capture **relaciones más complejas y jerárquicas** entre las características musicales.  
✅ El **RMSE cercano a 9.7** indica que, de media, el error de predicción es de unos 10 puntos en la escala de popularidad (0–100).

---

### 🧾 6. Conclusiones

El modelo de red neuronal con **varias capas ocultas** muestra:

🎯 Excelente capacidad de **generalización**.  
📊 Rendimiento **estable entre conjuntos de datos**.  
🧠 Captura patrones complejos gracias a su **arquitectura jerárquica**.

Se compara favorablemente con el **Modelo Complejo 1** de más neuronas, mostrando que **incrementar profundidad puede mejorar la predicción sin sobreajustar**.

---


## 🧠 Modelo Complejo 3: Regularización con Dropout

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** a partir de diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal Artificial (MLP)** que incorpora **técnicas de regularización mediante Dropout**.

El objetivo de este modelo es **reducir el riesgo de sobreajuste (overfitting)** que puede aparecer cuando se utilizan arquitecturas más complejas con un mayor número de parámetros.

La técnica **Dropout** consiste en desactivar aleatoriamente un porcentaje de neuronas durante el entrenamiento, lo que obliga a la red a aprender representaciones más robustas y generalizables.

La arquitectura utilizada consta de:

- 🔹 **Capa oculta 1:** 64 neuronas  
- 🔹 **Dropout**
- 🔹 **Capa oculta 2:** 32 neuronas  
- 🔹 **Dropout**
- 🔹 **Capa oculta 3:** 16 neuronas  
- 🔹 **Capa de salida:** 1 neurona

El modelo se implementa utilizando **TensorFlow / Keras**.

---

### 🧹 2. Preparación de los datos

La preparación de los datos es la misma que en los modelos anteriores.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Esto permite trabajar con datos más consistentes y mejorar el rendimiento del modelo.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que actúan únicamente como identificadores.

Variables eliminadas:

id
name
artists
id_artists
release_date
mode

Variable objetivo:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esta división permite evaluar correctamente la capacidad de generalización del modelo.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Esto es fundamental en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite una **convergencia más rápida**
- Evita que variables con escalas grandes dominen el aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal Feedforward** utilizando **TensorFlow/Keras**.

#### 🧩 Arquitectura del modelo

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Capa oculta 1 | Dense | 64 | 896 |
| Regularización | Dropout | - | 0 |
| Capa oculta 2 | Dense | 32 | 2080 |
| Regularización | Dropout | - | 0 |
| Capa oculta 3 | Dense | 16 | 528 |
| Capa salida | Dense | 1 | 17 |

Total de parámetros entrenables:

3521

---

#### ⚡ Configuración del entrenamiento

Parámetros principales:

- **Epochs máximas:** 500  
- **Batch size:** 64  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `Adam`

También se utiliza **Early Stopping** para detener el entrenamiento cuando el error en validación deja de mejorar.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 98.57 | 9.93 | 0.7882 |
| 🟡 Validación | 99.47 | 9.97 | 0.7871 |
| 🔵 Test | 94.11 | 9.70 | 0.7957 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **un 79.6% de la variabilidad de la popularidad** según el valor de **R² en test (0.7957)**.

✅ Los valores de **MSE y RMSE son muy similares entre entrenamiento, validación y test**, lo que indica una **buena capacidad de generalización**.

✅ La incorporación de **Dropout ayuda a reducir el riesgo de sobreajuste**, incluso en arquitecturas con más parámetros.

✅ El **RMSE cercano a 9.7** indica que el error medio del modelo es de aproximadamente **10 puntos en la escala de popularidad (0–100)**.

---

### 🧾 6. Conclusiones

El modelo con **regularización mediante Dropout** presenta un rendimiento sólido y estable.

Principales conclusiones:

🎯 La técnica **Dropout ayuda a mejorar la capacidad de generalización del modelo**.

📊 El rendimiento es **muy consistente entre entrenamiento, validación y test**.

🧠 La arquitectura más profunda combinada con regularización permite capturar **patrones complejos sin generar sobreajuste significativo**.

Este modelo será comparado con los otros modelos complejos para analizar **qué arquitectura ofrece el mejor equilibrio entre complejidad y rendimiento predictivo**.

---


## 🧠 Modelo Complejo 4: Batch Normalization

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal Artificial (MLP)** que incorpora **Batch Normalization** para mejorar la estabilidad del entrenamiento.

La técnica **Batch Normalization** normaliza las activaciones de cada capa durante el entrenamiento, lo que permite:

- 🔹 Acelerar la **convergencia del modelo**
- 🔹 Reducir el problema de **internal covariate shift**
- 🔹 Mejorar la **estabilidad del entrenamiento**
- 🔹 Facilitar el uso de arquitecturas más profundas

La arquitectura utilizada consta de:

- 🔹 **Capa oculta 1:** 64 neuronas  
- 🔹 **Batch Normalization**
- 🔹 **Capa oculta 2:** 32 neuronas  
- 🔹 **Batch Normalization**
- 🔹 **Capa oculta 3:** 16 neuronas  
- 🔹 **Capa de salida:** 1 neurona

El modelo se implementa utilizando **TensorFlow / Keras**.

---

### 🧹 2. Preparación de los datos

La preparación de los datos es la misma que en los modelos anteriores.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Esto permite trabajar con un conjunto de datos más consistente y fiable.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que actúan únicamente como identificadores.

Variables eliminadas:

id
name
artists
id_artists
release_date
mode

Variable objetivo:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esta división permite evaluar correctamente la capacidad de generalización del modelo.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Este paso es especialmente importante en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite una **convergencia más rápida**
- Evita que variables con escalas muy diferentes afecten al aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal Feedforward** utilizando **TensorFlow/Keras**.

#### 🧩 Arquitectura del modelo

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Capa oculta 1 | Dense | 64 | 896 |
| Normalización | BatchNormalization | - | 256 |
| Capa oculta 2 | Dense | 32 | 2080 |
| Normalización | BatchNormalization | - | 128 |
| Capa oculta 3 | Dense | 16 | 528 |
| Capa salida | Dense | 1 | 17 |

Total de parámetros:

3905

Parámetros entrenables:

3713

---

#### ⚡ Configuración del entrenamiento

Parámetros principales utilizados:

- **Epochs máximas:** 500  
- **Batch size:** 64  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `Adam`

También se utiliza **Early Stopping** para detener el entrenamiento cuando el error en validación deja de mejorar.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 93.59 | 9.67 | 0.7989 |
| 🟡 Validación | 96.09 | 9.80 | 0.7943 |
| 🔵 Test | 91.30 | 9.55 | 0.8018 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **un 80% de la variabilidad de la popularidad**, según el valor de **R² en test (0.8018)**.

✅ La incorporación de **Batch Normalization mejora la estabilidad del entrenamiento**, permitiendo alcanzar mejores resultados.

✅ Los valores de **MSE y RMSE son consistentes entre entrenamiento, validación y test**, indicando **buena capacidad de generalización**.

✅ El **RMSE cercano a 9.5** indica que el modelo comete un error medio de aproximadamente **10 puntos en la escala de popularidad (0–100)**.

---

### 🧾 6. Conclusiones

El modelo con **Batch Normalization** muestra uno de los mejores rendimientos entre las arquitecturas evaluadas.

Principales conclusiones:

🎯 La normalización de las activaciones ayuda a **mejorar la estabilidad y eficiencia del entrenamiento**.

📊 El modelo presenta **excelente capacidad de generalización**.

🧠 La combinación de varias capas con **Batch Normalization permite capturar relaciones complejas entre las características musicales**.

Este modelo será comparado con los demás modelos complejos para identificar **qué arquitectura ofrece el mejor rendimiento global en la predicción de popularidad**.

---


## 🧠 Modelo Complejo 5: Red Ancha

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal Ancha (Wide Neural Network)**, que prioriza **un gran número de neuronas por capa** en lugar de aumentar la profundidad de la red.

La idea detrás de este enfoque es que una red más ancha puede:

- 🧠 Capturar **más combinaciones de características en cada capa**
- 📊 Aprender **relaciones complejas entre variables**
- ⚡ Mejorar la capacidad de representación del modelo

Para mejorar la estabilidad del entrenamiento y evitar el sobreajuste, se incorporan además:

- 🔹 **Batch Normalization**
- 🔹 **Dropout**

La arquitectura utilizada consta de:

- 🔹 **Capa oculta 1:** 512 neuronas  
- 🔹 **Batch Normalization**
- 🔹 **Dropout**
- 🔹 **Capa oculta 2:** 256 neuronas  
- 🔹 **Batch Normalization**
- 🔹 **Dropout**
- 🔹 **Capa de salida:** 1 neurona

El modelo se implementa utilizando **TensorFlow / Keras**.

---

### 🧹 2. Preparación de los datos

La preparación de los datos es la misma que en los modelos anteriores.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Esto permite trabajar con un conjunto de datos más consistente y fiable.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que actúan únicamente como identificadores.

Variables eliminadas:

id
name
artists
id_artists
release_date
mode

Variable objetivo:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esta división permite evaluar correctamente la capacidad de generalización del modelo.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Este paso es especialmente importante en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite una **convergencia más rápida**
- Evita que variables con escalas muy diferentes afecten al aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal Ancha** utilizando **TensorFlow/Keras**.

#### 🧩 Arquitectura del modelo

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Capa oculta 1 | Dense | 512 | 7168 |
| Normalización | BatchNormalization | - | 2048 |
| Regularización | Dropout | - | 0 |
| Capa oculta 2 | Dense | 256 | 131328 |
| Normalización | BatchNormalization | - | 1024 |
| Regularización | Dropout | - | 0 |
| Capa salida | Dense | 1 | 257 |

Total de parámetros:

141825

Parámetros entrenables:

14289

---

#### ⚡ Configuración del entrenamiento

Parámetros principales utilizados:

- **Epochs máximas:** 500  
- **Batch size:** 64  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `Adam`

También se utiliza **Early Stopping** para detener el entrenamiento cuando el error en validación deja de mejorar.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 97.36 | 9.87 | 0.7908 |
| 🟡 Validación | 97.59 | 9.88 | 0.7911 |
| 🔵 Test | 93.36 | 9.66 | 0.7973 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **un 79-80% de la variabilidad de la popularidad**, según el valor de **R² en test (0.7973)**.

✅ La arquitectura ancha permite al modelo **aprender muchas combinaciones de variables en cada capa**.

⚠️ Sin embargo, el incremento significativo en el número de parámetros (**más de 140.000**) **no se traduce en una mejora clara del rendimiento** frente a arquitecturas más compactas.

📊 El **RMSE de 9.66** indica un error medio de aproximadamente **10 puntos en la escala de popularidad (0–100)**.

---

### 🧾 6. Conclusiones

El modelo de **Red Ancha** explora el impacto de aumentar significativamente el número de neuronas en cada capa.

Principales conclusiones:

🧠 El modelo tiene una **gran capacidad de representación** gracias al alto número de parámetros.

⚖️ Sin embargo, **una arquitectura más grande no siempre implica mejor rendimiento**.

📊 Los resultados obtenidos son **similares a modelos más simples**, lo que sugiere que el problema puede resolverse eficazmente con redes más compactas.

---


## 🧠 Modelo Complejo 6: Red Wide & Deep

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una arquitectura **Wide & Deep**, una técnica utilizada en sistemas de recomendación y predicción que combina:

- 🔹 Un **modelo profundo (Deep)** capaz de aprender **relaciones no lineales complejas**
- 🔹 Un **modelo ancho (Wide)** que permite mantener **relaciones directas entre variables originales**

Este enfoque permite que el modelo:

- 🧠 Aprenda **representaciones complejas de los datos**
- 📊 Mantenga información directa de las **variables originales**
- ⚡ Mejore la capacidad de generalización del modelo

La arquitectura utiliza la **API funcional de Keras**, lo que permite crear estructuras más flexibles que el modelo secuencial.

El modelo combina:

- 🔹 Capas densas profundas
- 🔹 Batch Normalization
- 🔹 Dropout
- 🔹 Concatenación de las variables originales con la representación profunda

---

### 🧹 2. Preparación de los datos

La preparación de los datos es la misma que en los modelos anteriores.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Esto permite trabajar con un conjunto de datos más consistente y fiable.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que actúan únicamente como identificadores.

Variables eliminadas:

id
name
artists
id_artists
release_date
mode

Variable objetivo:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esta división permite evaluar correctamente la capacidad de generalización del modelo.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Este paso es especialmente importante en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite una **convergencia más rápida**
- Evita que variables con escalas muy diferentes afecten al aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal Wide & Deep** utilizando **TensorFlow/Keras** con la **API funcional**.

#### 🧩 Arquitectura del modelo

El modelo combina una **rama profunda** con las **variables originales**, que posteriormente se concatenan antes de la capa final.

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Entrada | InputLayer | 13 | 0 |
| Capa profunda 1 | Dense | 128 | 1792 |
| Normalización | BatchNormalization | - | 512 |
| Regularización | Dropout | - | 0 |
| Capa profunda 2 | Dense | 64 | 8256 |
| Normalización | BatchNormalization | - | 256 |
| Regularización | Dropout | - | 0 |
| Capa profunda 3 | Dense | 32 | 2080 |
| Concatenación | Concatenate | - | 0 |
| Capa salida | Dense | 1 | 46 |

Total de parámetros:

12942

Parámetros entrenables:

12558

---

#### ⚡ Configuración del entrenamiento

Parámetros principales utilizados:

- **Epochs máximas:** 500  
- **Batch size:** 64  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `Adam`

También se utiliza **Early Stopping** para detener el entrenamiento cuando el error en validación deja de mejorar.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 96.07 | 9.80 | 0.7936 |
| 🟡 Validación | 96.63 | 9.83 | 0.7931 |
| 🔵 Test | 91.91 | 9.59 | 0.8005 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **el 80% de la variabilidad de la popularidad**, según el valor de **R² en test (0.8005)**.

✅ La arquitectura **Wide & Deep permite combinar el aprendizaje profundo con información directa de las variables originales**.

📊 El **RMSE de 9.59** indica un error medio aproximado de **10 puntos en la escala de popularidad (0–100)**.

⚖️ Los resultados obtenidos son **muy similares a otras arquitecturas profundas**, lo que indica que la combinación wide-deep **no aporta una mejora significativa en este dataset concreto**.

---

### 🧾 6. Conclusiones

El modelo **Wide & Deep** explora una arquitectura más avanzada que combina dos tipos de aprendizaje.

Principales conclusiones:

🧠 Permite integrar **representaciones profundas y relaciones directas entre variables**.

⚡ Ofrece **buen rendimiento y buena estabilidad durante el entrenamiento**.

📊 Sin embargo, en este problema concreto **no supera claramente a las arquitecturas neuronales profundas más simples**.

---


## 🧠 Modelo Complejo 7: Red Neuronal Residual

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal Residual**, una arquitectura inspirada en las **ResNet**, que introduce **conexiones residuales (skip connections)** entre capas de la red.

Las conexiones residuales permiten que la información de una capa anterior se **sume directamente a una capa posterior**, facilitando el flujo de información durante el entrenamiento.

Este enfoque ayuda a resolver problemas típicos en redes profundas como:

- 🔹 **Desvanecimiento del gradiente**
- 🔹 **Dificultad para entrenar redes profundas**
- 🔹 **Pérdida de información entre capas**

Gracias a estas conexiones, el modelo puede **aprender funciones más complejas sin perder estabilidad durante el entrenamiento**.

La arquitectura utiliza la **API funcional de Keras**, lo que permite crear estructuras más flexibles que el modelo secuencial.

El modelo incluye:

- 🔹 Capas densas profundas
- 🔹 Batch Normalization
- 🔹 Dropout
- 🔹 Una **conexión residual** que combina información de distintas capas

---

### 🧹 2. Preparación de los datos

La preparación de los datos es la misma que en los modelos anteriores.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Esto permite trabajar con un conjunto de datos más consistente y fiable.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que actúan únicamente como identificadores.

Variables eliminadas:

id  
name  
artists  
id_artists  
release_date  
mode  

Variable objetivo:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esta división permite evaluar correctamente la capacidad de generalización del modelo.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Este paso es especialmente importante en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite una **convergencia más rápida**
- Evita que variables con escalas muy diferentes afecten al aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal Residual** utilizando **TensorFlow/Keras** con la **API funcional**.

#### 🧩 Arquitectura del modelo

La arquitectura incorpora una **conexión residual** que suma la salida de una capa con la salida de una capa posterior.

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Entrada | InputLayer | 13 | 0 |
| Capa profunda 1 | Dense | 128 | 1792 |
| Normalización | BatchNormalization | - | 512 |
| Capa profunda 2 | Dense | 128 | 16512 |
| Normalización | BatchNormalization | - | 512 |
| Regularización | Dropout | - | 0 |
| Conexión residual | Add | - | 0 |
| Capa profunda 3 | Dense | 64 | 8256 |
| Normalización | BatchNormalization | - | 256 |
| Regularización | Dropout | - | 0 |
| Capa salida | Dense | 1 | 65 |

Total de parámetros:

27905

Parámetros entrenables:

27265

---

#### ⚡ Configuración del entrenamiento

Parámetros principales utilizados:

- **Epochs máximas:** 500  
- **Batch size:** 64  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `Adam`

También se utiliza **Early Stopping** para detener el entrenamiento cuando el error en validación deja de mejorar.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 91.82 | 9.58 | 0.8027 |
| 🟡 Validación | 95.66 | 9.78 | 0.7952 |
| 🔵 Test | 91.33 | 9.56 | 0.8017 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **el 80% de la variabilidad de la popularidad**, según el valor de **R² en test (0.8017)**.

✅ Las **conexiones residuales ayudan a mejorar el flujo de información dentro de la red**, facilitando el aprendizaje de relaciones más complejas entre las variables.

📊 El **RMSE de 9.56** indica un error medio aproximado de **10 puntos en la escala de popularidad (0–100)**.

⚖️ Los resultados son **muy similares entre entrenamiento, validación y test**, lo que sugiere que el modelo **generaliza correctamente y no presenta sobreajuste significativo**.

---

### 🧾 6. Conclusiones

La **Red Neuronal Residual** introduce una arquitectura más avanzada que mejora el flujo de información dentro de la red mediante conexiones residuales.

Principales conclusiones:

🧠 Permite entrenar redes más profundas de forma **estable y eficiente**.

⚡ Ofrece **buen rendimiento y buena capacidad de generalización**.

📊 En este problema concreto, los resultados obtenidos son **ligeramente mejores que algunos modelos neuronales anteriores**, aunque las mejoras no son muy grandes debido a la naturaleza del dataset.

---


## 🧠 Modelo Complejo 8: Red Profunda Optimizada

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal Profunda (Deep Neural Network)** que combina varias técnicas para mejorar el rendimiento y la estabilidad del entrenamiento.

La arquitectura integra:

- 🔹 **Capas densas profundas**
- 🔹 **Batch Normalization**
- 🔹 **Dropout**

Estas técnicas permiten:

- 📈 Mejorar la **capacidad de aprendizaje del modelo**
- ⚡ Acelerar la **convergencia**
- 🛡 Reducir el **overfitting**
- 🧠 Capturar relaciones **más complejas entre variables**

La arquitectura utilizada consta de:

- 🔹 **Capa oculta 1:** 128 neuronas  
- 🔹 **Batch Normalization**
- 🔹 **Dropout**
- 🔹 **Capa oculta 2:** 64 neuronas  
- 🔹 **Batch Normalization**
- 🔹 **Dropout**
- 🔹 **Capa oculta 3:** 32 neuronas  
- 🔹 **Batch Normalization**
- 🔹 **Capa oculta 4:** 16 neuronas  
- 🔹 **Capa de salida:** 1 neurona

El modelo se implementa utilizando **TensorFlow / Keras**.

---

### 🧹 2. Preparación de los datos

La preparación de los datos es la misma que en los modelos anteriores.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Esto permite trabajar con un conjunto de datos más consistente y fiable.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que actúan únicamente como identificadores.

Variables eliminadas:

id
name
artists
id_artists
release_date
mode

Variable objetivo:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esta división permite evaluar correctamente la capacidad de generalización del modelo.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Este paso es especialmente importante en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite una **convergencia más rápida**
- Evita que variables con escalas muy diferentes afecten al aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal Profunda** utilizando **TensorFlow/Keras**.

#### 🧩 Arquitectura del modelo

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Capa oculta 1 | Dense | 128 | 1792 |
| Normalización | BatchNormalization | - | 512 |
| Regularización | Dropout | - | 0 |
| Capa oculta 2 | Dense | 64 | 8256 |
| Normalización | BatchNormalization | - | 256 |
| Regularización | Dropout | - | 0 |
| Capa oculta 3 | Dense | 32 | 2080 |
| Normalización | BatchNormalization | - | 128 |
| Capa oculta 4 | Dense | 16 | 528 |
| Capa salida | Dense | 1 | 17 |

Total de parámetros:

13569

Parámetros entrenables:

13121

---

#### ⚡ Configuración del entrenamiento

Parámetros principales utilizados:

- **Epochs máximas:** 500  
- **Batch size:** 64  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `Adam`

También se utiliza **Early Stopping** para detener el entrenamiento cuando el error en validación deja de mejorar.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 94.08 | 9.70 | 0.7979 |
| 🟡 Validación | 95.55 | 9.78 | 0.7954 |
| 🔵 Test | 90.59 | 9.52 | 0.8033 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **un 80% de la variabilidad de la popularidad**, con un **R² en test de 0.8033**, el valor más alto entre los modelos evaluados.

✅ La combinación de **profundidad de la red + Batch Normalization + Dropout** permite mejorar la capacidad de aprendizaje del modelo.

✅ Los valores de **MSE y RMSE son consistentes entre entrenamiento, validación y test**, lo que indica **buena capacidad de generalización**.

✅ El **RMSE cercano a 9.5** indica que el modelo comete un error medio aproximado de **10 puntos en la escala de popularidad (0–100)**.

---

### 🧾 6. Conclusiones

El modelo de **Red Profunda** representa la arquitectura más compleja evaluada en el proyecto.

Principales conclusiones:

🧠 La mayor profundidad permite capturar **relaciones no lineales más complejas entre las variables musicales**.

⚡ La combinación de **Batch Normalization y Dropout mejora la estabilidad del entrenamiento y reduce el overfitting**.

📊 Este modelo obtiene **el mejor rendimiento global hasta ahora en el conjunto de test**, alcanzando el mayor valor de **R² (0.8033)** entre los modelos de Deep Learning evaluados.

---


## 🧠 Modelo Complejo 9: Red de Alta Convergencia

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal optimizada para una convergencia rápida y estable**, incorporando diferentes técnicas de regularización y control del entrenamiento.

La arquitectura busca mejorar la estabilidad del aprendizaje mediante:

- 🔹 **Batch Normalization**, que estabiliza la distribución de los datos entre capas.
- 🔹 **Dropout**, que ayuda a reducir el sobreajuste.
- 🔹 **ReduceLROnPlateau**, que reduce dinámicamente la tasa de aprendizaje cuando el modelo deja de mejorar.
- 🔹 **Early Stopping**, que detiene el entrenamiento cuando el modelo alcanza su mejor rendimiento.

Estas técnicas permiten que el modelo:

- 🧠 Aprenda patrones complejos de forma más estable
- ⚡ Converja más rápidamente durante el entrenamiento
- 📊 Mejore la capacidad de generalización

La arquitectura utiliza **TensorFlow/Keras con modelo secuencial** para construir una red neuronal profunda optimizada.

---

### 🧹 2. Preparación de los datos

La preparación de los datos es la misma que en los modelos anteriores.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Esto permite trabajar con un conjunto de datos más consistente y fiable.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que actúan únicamente como identificadores.

Variables eliminadas:

id  
name  
artists  
id_artists  
release_date  
mode  

Variable objetivo:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esta división permite evaluar correctamente la capacidad de generalización del modelo.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Este paso es especialmente importante en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite una **convergencia más rápida**
- Evita que variables con escalas muy diferentes afecten al aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal profunda optimizada para alta convergencia** utilizando **TensorFlow/Keras**.

#### 🧩 Arquitectura del modelo

La arquitectura incluye varias capas densas junto con técnicas de normalización y regularización.

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Capa 1 | Dense | 128 | 1792 |
| Normalización | BatchNormalization | - | 512 |
| Regularización | Dropout | - | 0 |
| Capa 2 | Dense | 64 | 8256 |
| Normalización | BatchNormalization | - | 256 |
| Regularización | Dropout | - | 0 |
| Capa 3 | Dense | 32 | 2080 |
| Normalización | BatchNormalization | - | 128 |
| Capa 4 | Dense | 16 | 528 |
| Capa salida | Dense | 1 | 17 |

Total de parámetros:

13569

Parámetros entrenables:

13121

---

#### ⚡ Configuración del entrenamiento

Parámetros principales utilizados:

- **Epochs máximas:** 500  
- **Batch size:** 256  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `Adam`

Se utilizan además dos técnicas importantes de control del entrenamiento:

**Early Stopping**

- Monitoriza `val_loss`
- Paciencia de **25 épocas**
- Restaura los **mejores pesos del modelo**

**ReduceLROnPlateau**

- Reduce la tasa de aprendizaje cuando la validación deja de mejorar
- Factor de reducción **0.5**
- Paciencia de **8 épocas**
- Tasa mínima de aprendizaje **0.00001**

Esto permite mejorar la estabilidad del aprendizaje y evitar sobreentrenamiento.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 92.05 | 9.59 | 0.8022 |
| 🟡 Validación | 95.31 | 9.76 | 0.7960 |
| 🔵 Test | 90.03 | 9.49 | 0.8046 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **el 80% de la variabilidad de la popularidad**, según el valor de **R² en test (0.8046)**.

⚡ La combinación de **Early Stopping y ReduceLROnPlateau permite una convergencia más eficiente**, evitando entrenamientos innecesariamente largos.

📊 El **RMSE de 9.49** indica un error medio aproximado de **9–10 puntos en la escala de popularidad (0–100)**.

⚖️ Los resultados entre entrenamiento, validación y test son **muy consistentes**, lo que indica una **buena capacidad de generalización del modelo**.

---

### 🧾 6. Conclusiones

La **Red de Alta Convergencia** introduce técnicas adicionales de control del entrenamiento que mejoran la estabilidad del modelo.

Principales conclusiones:

🧠 La combinación de **Batch Normalization, Dropout y control dinámico del learning rate** permite un aprendizaje más estable.

⚡ El uso de **ReduceLROnPlateau y Early Stopping** mejora la eficiencia del entrenamiento.

📊 El modelo consigue **uno de los mejores resultados del proyecto**, con un **RMSE de 9.49 y R² de 0.8046 en test**.

---


## 🧠 Modelo Complejo 10: Red Neuronal con Activación Swish

Este modelo forma parte de la sección de **modelos de Deep Learning** del proyecto, cuyo objetivo es **predecir la popularidad de canciones en Spotify** utilizando diferentes características musicales.

La variable objetivo **popularity** toma valores entre **0 y 100**, representando el nivel de popularidad de una canción en la plataforma.

---

### 📌 1. Introducción del modelo

En este modelo se implementa una **Red Neuronal que incorpora la función de activación Swish**, una función moderna que ha demostrado mejorar el rendimiento en arquitecturas de deep learning.

La función **Swish** permite una propagación más suave del gradiente durante el entrenamiento, lo que facilita la optimización del modelo y puede mejorar la capacidad de aprendizaje.

La arquitectura utiliza además diferentes técnicas de estabilización del entrenamiento:

- 🔹 **Batch Normalization**, que estabiliza la distribución de los datos entre capas.
- 🔹 **Dropout**, que ayuda a reducir el sobreajuste.
- 🔹 **ReduceLROnPlateau**, que reduce dinámicamente la tasa de aprendizaje cuando el modelo deja de mejorar.
- 🔹 **Early Stopping**, que detiene el entrenamiento cuando el modelo alcanza su mejor rendimiento.

Estas técnicas permiten que el modelo:

- 🧠 Aprenda patrones complejos de forma más eficiente
- ⚡ Mantenga un entrenamiento estable
- 📊 Mejore la capacidad de generalización

La arquitectura se implementa utilizando **TensorFlow/Keras con un modelo secuencial optimizado**.

---

### 🧹 2. Preparación de los datos

La preparación de los datos es la misma que en los modelos anteriores.

#### 🔍 Filtrado de valores atípicos

Se eliminan registros con valores extremos o poco representativos:

- Canciones con duración menor a **40 segundos**
- Canciones con **speechiness mayor a 0.90**
- Canciones con **tempo menor o igual que 0**

Esto permite trabajar con un conjunto de datos más consistente y fiable.

---

#### 🗑 Eliminación de variables no relevantes

Se eliminan variables que no aportan valor predictivo o que actúan únicamente como identificadores.

Variables eliminadas:

id  
name  
artists  
id_artists  
release_date  
mode  

Variable objetivo:

popularity

---

#### 📊 División del dataset

El dataset se divide en tres subconjuntos:

| Conjunto | Porcentaje |
|--------|--------|
| 🟢 Entrenamiento | 70% |
| 🟡 Validación | 15% |
| 🔵 Test | 15% |

Esta división permite evaluar correctamente la capacidad de generalización del modelo.

---

#### ⚙️ Escalado de variables

Se aplica **StandardScaler** para normalizar las variables de entrada.

Este paso es especialmente importante en redes neuronales porque:

- Mejora la **estabilidad del entrenamiento**
- Permite una **convergencia más rápida**
- Evita que variables con escalas muy diferentes afecten al aprendizaje

---

### 🏗 3. Entrenamiento del modelo

Se entrena una **Red Neuronal profunda con activación Swish** utilizando **TensorFlow/Keras**.

#### 🧩 Arquitectura del modelo

La arquitectura incluye varias capas densas junto con técnicas de normalización y regularización.

| Capa | Tipo | Neuronas | Parámetros |
|-----|-----|-----|-----|
| Capa 1 | Dense | 128 | 1792 |
| Normalización | BatchNormalization | - | 512 |
| Regularización | Dropout | - | 0 |
| Capa 2 | Dense | 64 | 8256 |
| Normalización | BatchNormalization | - | 256 |
| Regularización | Dropout | - | 0 |
| Capa 3 | Dense | 32 | 2080 |
| Normalización | BatchNormalization | - | 128 |
| Capa 4 | Dense | 16 | 528 |
| Capa salida | Dense | 1 | 17 |

Total de parámetros:

13569

Parámetros entrenables:

13121

---

#### ⚡ Configuración del entrenamiento

Parámetros principales utilizados:

- **Epochs máximas:** 500  
- **Batch size:** 256  
- **Función de pérdida:** `Mean Squared Error (MSE)`  
- **Optimizador:** `AdamW`

Se utilizan además dos técnicas importantes de control del entrenamiento:

**Early Stopping**

- Monitoriza `val_loss`
- Paciencia de **30 épocas**
- Restaura los **mejores pesos del modelo**

**ReduceLROnPlateau**

- Reduce la tasa de aprendizaje cuando la validación deja de mejorar
- Factor de reducción **0.5**
- Paciencia de **8 épocas**
- Tasa mínima de aprendizaje **0.00001**

Esto permite mejorar la estabilidad del aprendizaje y evitar sobreentrenamiento.

---

### 📈 4. Evaluación del modelo

El rendimiento del modelo se evalúa utilizando tres métricas estándar de regresión:

- 📉 **MSE (Mean Squared Error)**
- 📊 **RMSE (Root Mean Squared Error)**
- 📏 **R² (Coeficiente de determinación)**

---

#### 📊 Resultados obtenidos

| Conjunto | MSE | RMSE | R² |
|--------|--------|--------|--------|
| 🟢 Entrenamiento | 91.36 | 9.56 | 0.8037 |
| 🟡 Validación | 94.74 | 9.73 | 0.7972 |
| 🔵 Test | 89.97 | 9.49 | 0.8047 |

---

### 🔎 5. Interpretación de resultados

Observaciones principales:

✅ El modelo consigue explicar aproximadamente **el 80% de la variabilidad de la popularidad**, según el valor de **R² en test (0.8047)**.

⚡ La utilización de la **función de activación Swish** permite mejorar ligeramente el rendimiento respecto a modelos anteriores.

📊 El **RMSE de 9.49** indica un error medio aproximado de **9–10 puntos en la escala de popularidad (0–100)**.

⚖️ Los resultados entre entrenamiento, validación y test son **muy consistentes**, lo que indica una **buena capacidad de generalización del modelo**.

---

### 🧾 6. Conclusiones

La **Red Neuronal con activación Swish** representa una mejora adicional dentro de los modelos de deep learning del proyecto.

Principales conclusiones:

🧠 La función de activación **Swish** permite una optimización más eficiente del entrenamiento.

⚡ La combinación de **Batch Normalization, Dropout y control dinámico del learning rate** mantiene un aprendizaje estable.

📊 El modelo consigue **el mejore resultado del proyecto con las redes neuronales**, con un **RMSE de 9.49 y R² de 0.8047 en test**, posicionándose como uno de los modelos más precisos para predecir la popularidad de canciones.
