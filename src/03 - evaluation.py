import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, confusion_matrix, recall_score, precision_score
import pickle

os.chdir(os.path.dirname(os.getcwd()))


df = pd.read_csv("data/train.csv")

X = df[["Rating",'Price',
       'Categoria',
       'tamaño', 'tamaño2', 'Gratuito', 'tipo_contenido', 'Genero1',
       'dias_sin_actualizar', 'meses_sin_actualizar', 'grupomeses',
       'version_android',
       'Genero_grp_Descargas_mean', 'Genero_grp_Descargas_median',
       'Genero_grp_Descargas_mode', 'Genero_grp_Rating_mean',
       'Genero_grp_Rating_median', 'Genero_grp_Rating_mode',
       'Genero_grp_Reviews_mean', 'Genero_grp_Reviews_median',
       'Genero_grp_Reviews_mode', 'Genero_grp_Descargas2_mean',
       'Genero_grp_Descargas2_median', 'Genero_grp_Descargas2_mode',
       'Genero_grp_Descargas3_mean', 'Genero_grp_Descargas3_median',
       'Genero_grp_Descargas3_mode', 'Tamaño2_grp_Descargas_mean',
       'Tamaño2_grp_Descargas_median', 'Tamaño2_grp_Descargas_mode',
       'Tamaño2_grp_Descargas2_mean', 'Tamaño2_grp_Descargas2_median',
       'Tamaño2_grp_Descargas2_mode', 'Tamaño2_grp_Descargas3_mean',
       'Tamaño2_grp_Descargas3_median', 'Tamaño2_grp_Descargas3_mode',
       'Tamaño2_grp_Reviews_mean', 'Tamaño2_grp_Reviews_median',
       'tipo_contenido_grp_Descargas_mean',
       'tipo_contenido_grp_Descargas_median',
       'tipo_contenido_grp_Descargas2_mean',
       'tipo_contenido_grp_Descargas2_median',
       'tipo_contenido_grp_Descargas3_mean',
       'tipo_contenido_grp_Descargas3_median',
       'tipo_contenido_grp_Reviews_mean', 'tipo_contenido_grp_Reviews_median',
       'grupomeses_grp_Reviews_mean', 'grupomeses_grp_Reviews_median',
       'version_android_grp_Reviews_mean',
       'version_android_grp_Reviews_median', 'med_versandroid']]
y = df[['Descargas3']]




final_test = pd.read_csv("data/test.csv")
final_test_x = final_test[["Rating",'Price',
       'Categoria',
       'tamaño', 'tamaño2', 'Gratuito', 'tipo_contenido', 'Genero1',
       'dias_sin_actualizar', 'meses_sin_actualizar', 'grupomeses',
       'version_android',
       'Genero_grp_Descargas_mean', 'Genero_grp_Descargas_median',
       'Genero_grp_Descargas_mode', 'Genero_grp_Rating_mean',
       'Genero_grp_Rating_median', 'Genero_grp_Rating_mode',
       'Genero_grp_Reviews_mean', 'Genero_grp_Reviews_median',
       'Genero_grp_Reviews_mode', 'Genero_grp_Descargas2_mean',
       'Genero_grp_Descargas2_median', 'Genero_grp_Descargas2_mode',
       'Genero_grp_Descargas3_mean', 'Genero_grp_Descargas3_median',
       'Genero_grp_Descargas3_mode', 'Tamaño2_grp_Descargas_mean',
       'Tamaño2_grp_Descargas_median', 'Tamaño2_grp_Descargas_mode',
       'Tamaño2_grp_Descargas2_mean', 'Tamaño2_grp_Descargas2_median',
       'Tamaño2_grp_Descargas2_mode', 'Tamaño2_grp_Descargas3_mean',
       'Tamaño2_grp_Descargas3_median', 'Tamaño2_grp_Descargas3_mode',
       'Tamaño2_grp_Reviews_mean', 'Tamaño2_grp_Reviews_median',
       'tipo_contenido_grp_Descargas_mean',
       'tipo_contenido_grp_Descargas_median',
       'tipo_contenido_grp_Descargas2_mean',
       'tipo_contenido_grp_Descargas2_median',
       'tipo_contenido_grp_Descargas3_mean',
       'tipo_contenido_grp_Descargas3_median',
       'tipo_contenido_grp_Reviews_mean', 'tipo_contenido_grp_Reviews_median',
       'grupomeses_grp_Reviews_mean', 'grupomeses_grp_Reviews_median',
       'version_android_grp_Reviews_mean',
       'version_android_grp_Reviews_median', 'med_versandroid']]

final_test_y = final_test['Descargas3']



with open(r'C:\Users\javie\Documents\Bootcamp\Alumno\PROYECTO MACHINE LEARNING - copia\Proyecto_ML\models\best_model\xgb_+rating_MS072_ACC652_PS686_RS641_.pkl', 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)



pdns = modelo_cargado.predict(final_test_x)



print("Model Score:\t",modelo_cargado.score(X, y))
print("Accuracy Score:\t",accuracy_score(final_test_y, pdns))
print("Precision Score:",precision_score(final_test_y, pdns, average="macro"))
print("Recall Score:\t",recall_score(final_test_y, pdns, average="macro"))