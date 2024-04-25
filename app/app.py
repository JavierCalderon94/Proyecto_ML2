import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import streamlit.components.v1 as c
from datetime import datetime
from data_processing_app import transformacion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, confusion_matrix, recall_score, precision_score
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import matplotlib.pyplot as plt

with open(r'C:\Users\javie\Documents\Bootcamp\Alumno\PROYECTO MACHINE LEARNING - copia\Proyecto_ML\models\best_model\xgb_+rating_MS072_ACC652_PS686_RS641_.pkl', 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)

img = Image.open("data/GetItOnGooglePlay.png")
df = pd.read_csv(r"C:\Users\javie\Documents\Bootcamp\Alumno\PROYECTO MACHINE LEARNING - copia\Proyecto_ML\data\train.csv")

st.set_page_config(page_title="Descargas Play Store",
                   page_icon="data/icono.png")


seleccion = st.sidebar.selectbox("Seleccione menu", ["Home", "Predictor"])

#----------------------------------------------------------------------------------------------------
# MENÚ HOME
#----------------------------------------------------------------------------------------------------

if seleccion == "Home":
    st.title("Estimador de descargas")
    
    with st.expander("¿Cómo funciona?"):
        st.write("En el lado izquierdo de la pantalla dispone de un menú desplegable, haga click sobre éste para seleccionar \"Predictor\". ")
        st.write("Una vez que lo seleccione, debe elegir dentro de todas las opciones disponibles y el sistema le facilitará una estimación de las descargas que puede esperar bajo los criterios seleccionados.")
    img = Image.open("data/GetItOnGooglePlay.png")

    st.image(img)


#----------------------------------------------------------------------------------------------------

elif seleccion == "Predictor":
    
    st.title("Predicciones:")

    
    
    categoria = st.sidebar.selectbox("Categoría:", sorted(list(df["Category"].unique())))
    rating = st.sidebar.slider("Rating:", min_value=0.0, max_value=5.0, step=0.1)
    size = st.sidebar.slider("Memoria (kB):\t 0 si es desconocido", min_value=0, max_value=100000, step=1)
    precio = st.sidebar.slider("Precio", min_value=0.0, max_value=400.0, step=0.5)
    content_rating = st.sidebar.selectbox("Clasificación Edad:", sorted(list(df["Content Rating"].unique())))
    genero = st.sidebar.selectbox("Género:", sorted(list(df["Genres"].unique())))
    novedad = st.sidebar.selectbox("¿Es una aplicación nueva o ya existente?", ["Nueva","Existente"])
    if novedad=="Existente":
        tiempo_sin_actualizar = st.sidebar.slider("¿Cuanto lleva sin actualizarse? (Días):", min_value=1, max_value=2702, step=1) + 2082
    else:
        tiempo_sin_actualizar = 2082
    version_android = st.sidebar.selectbox("Versión Android Compatible:", sorted(list(df["Android Ver"].unique())))

#----------------------------------------------------------------------------------------------------
# CONVERSIONES
#----------------------------------------------------------------------------------------------------    

    df_test, X = transformacion(rating,precio,categoria,size,content_rating,genero,tiempo_sin_actualizar,version_android,df)

    testeo = modelo_cargado.predict(X)
    probs = (modelo_cargado.predict_proba(X)[0] * 100).tolist()
    clases = ['0 - 10.000', '10.000 - 1.000.000', '1.000.000+']
    clase_max = clase_max = np.argmax(probs)
    colors = ['red' if i != clase_max else 'blue' for i in range(len(clases))]

    plt.figure(figsize=(6, 4))
    bars_red = []
    bars_blue = []
    for i in range(len(clases)):
        if colors[i] == 'red':
            bars_red.append(plt.bar(clases[i], probs[i], color=colors[i]))
        else:
            bars_blue.append(plt.bar(clases[i], probs[i], color=colors[i]))
        plt.text(clases[i], probs[i], f'{probs[i]:.2f}'+"%", ha='center', va='bottom')

    plt.title('Probabilidad de pertenencia')
    plt.xlabel('Descargas')
    plt.ylabel('% de Probabilidad')
    plt.xticks(rotation=0)
    plt.legend([bars_blue[0], bars_red[0]], ['Resultado', 'Resto'])
    




    
    st.pyplot(plt)


    if testeo==0:
    
        st.write("Se estima una cantidad de descargas inferior a 10.000.")
    
    elif testeo==1:

        st.write("Se estima una cantidad de descargas entre 10.000 y 1.000.000.")

    elif testeo==2:

        st.write("Se estima una cantidad de descargas superior a 1.000.000.")



st.sidebar.image(img)