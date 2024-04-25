# Proyecto de Machine Learning para Predicción de Descargas de Aplicaciones de Google Play Store
Este proyecto de machine learning tiene como objetivo predecir el número de descargas de aplicaciones en la tienda Google Play Store utilizando datos disponibles públicamente. A continuación, se detallan los pasos realizados en el desarrollo del proyecto:

### 1. Obtención y Exploración de Datos
Se descargó el conjunto de datos "Google Play Store Applications" de Kaggle, proporcionado por el usuario BHAVIK JIKADARA. Este conjunto de datos contiene 10833 entradas con 13 columnas que incluyen información como el nombre de la aplicación, categoría, calificación, número de revisiones, tamaño, instalaciones, tipo, precio, clasificación de contenido, géneros, última actualización, versión actual y versión de Android.
### 2. Preprocesamiento de Datos
Se identificaron y eliminaron 1074 valores duplicados del conjunto de datos.
La variable objetivo, "INSTALLS", se agrupó en tres categorías: menos de 10000, entre 10000 y 1000000, y más de 1000000.
Se convirtieron todas las variables categóricas en variables numéricas para su posterior procesamiento.
### 3. Feature Engineering
Se realizó Feature Engineering para mejorar la calidad de los datos y prepararlos para el entrenamiento del modelo.
### 4. Entrenamiento de Modelos
Se exploraron varios modelos de aprendizaje automático, incluyendo RandomForest, pipelines con diferentes configuraciones de preprocesamiento (StandardScaler, MinMaxScaler, PCA y RandomUnderSampler).
El modelo que mejor rendimiento mostró fue el XGBoost, utilizando StandardScaler, SelectKBest (k=42), Learning_rate=0.01, Min_child_weight=13 y N_estimators=621.
### 5. Evaluación del Modelo
Se logró una precisión del modelo del 65%, que es el porcentaje de predicciones correctas sobre el total de predicciones realizadas.
Se identificó que la mejora del modelo podría lograrse mediante la expansión del conjunto de datos con nuevas muestras obtenidas a través de web scrapping, y la revisión y ajuste del preprocesamiento de los datos.
## Próximos Pasos
Para mejorar el modelo, se recomienda obtener más datos mediante web scrapping para aumentar la diversidad del conjunto de entrenamiento.
Se puede explorar la aplicación de técnicas avanzadas de preprocesamiento y modelado para obtener un mejor rendimiento del modelo.
Se pueden realizar ajustes adicionales en los hiperparámetros del modelo y evaluar su impacto en la precisión y generalización del modelo.
Este proyecto ofrece una visión general del proceso de desarrollo de un modelo de machine learning para predecir las descargas de aplicaciones en la Google Play Store, así como sugerencias para mejorar y continuar el trabajo en el futuro.






