# 🎵 Predicción de Popularidad en Spotify (1921-2020)
Trabajo Final - Asignatura de Aprendizaje Profundo  
Autores: Javier Beny Rodríguez y Adrián Blat Campos  
Fecha: Febrero 2026  

# 📖 Descripción del Proyecto
Este proyecto tiene como objetivo predecir la popularidad de una canción (variable continua 0-100) basándose en sus características de audio (bailabilidad, energía, acústica, etc.) y metadatos. Se comparará el rendimiento de un modelo clásico (Regresión Lineal) frente a una arquitectura de Deep Learning (Perceptrón Multicapa - MLP).


## 📚 1. Definición del Problema
Descripción del Problema
El problema de la predicción de popularidad consiste en entrenar modelos de aprendizaje supervisado para estimar un valor numérico. El desafío radica en la subjetividad de la "popularidad".


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


## 🧠 4. Modelos y Metodología
Métricas de Evaluación
Para este problema de Regresión, las métricas estándar utilizadas son:
- MSE (Mean Squared Error): Penaliza los errores grandes.
- RMSE (Root Mean Squared Error): Error promedio en las mismas unidades que la popularidad (0-100).
- R² (Coeficiente de Determinación): Indica qué porcentaje de la varianza de la popularidad es explicada por el modelo.
