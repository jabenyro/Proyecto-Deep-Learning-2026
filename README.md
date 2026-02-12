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

| Modelo | Tipo de Algoritmo | MSE | RMSE | R² | Estado |
| Regresión Lineal | Clásico | - | - | - |
| Red Neuronal (MLP) | Deep Learning | - | - | - |

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