from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
import statistics
import pickle

import os

os.chdir(os.path.dirname(os.getcwd()))
df=pd.read_csv("data/processed.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42)

#-----------------------------------------------------------------------------------------
# CONVERSIONES TRAIN / TEST
#-----------------------------------------------------------------------------------------

for i,j in enumerate(list(train.groupby("Genero1")["Descargas"].mean().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Descargas_mean"]=j
test["Genero_grp_Descargas_mean"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas_mean"]

for i,j in enumerate(list(train.groupby("Genero1")["Descargas"].median().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Descargas_median"]=j
test["Genero_grp_Descargas_median"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas_median"]

moda_por_genero_dict = train.groupby("Genero1")["Descargas"].apply(lambda x: x.mode().iloc[0]).to_dict()
train["Genero_grp_Descargas_mode"] = train["Genero1"].map(moda_por_genero_dict)
test["Genero_grp_Descargas_mode"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas_mode"]

for i,j in enumerate(list(train.groupby("Genero1")["Rating"].mean().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Rating_mean"]=j
test["Genero_grp_Rating_mean"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Rating_mean"]

for i,j in enumerate(list(train.groupby("Genero1")["Rating"].median().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Rating_median"]=j
test["Genero_grp_Rating_median"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Rating_median"]

moda_por_genero_dict = train.groupby("Genero1")["Rating"].apply(lambda x: x.mode().iloc[0]).to_dict()
train["Genero_grp_Rating_mode"] = train["Genero1"].map(moda_por_genero_dict)
test["Genero_grp_Rating_mode"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Rating_mode"]

for i,j in enumerate(list(train.groupby("Genero1")["Reviews"].mean().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Reviews_mean"]=j
test["Genero_grp_Reviews_mean"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Reviews_mean"]

for i,j in enumerate(list(train.groupby("Genero1")["Reviews"].median().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Reviews_median"]=j
test["Genero_grp_Reviews_median"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Reviews_median"]

moda_por_genero_dict = train.groupby("Genero1")["Reviews"].apply(lambda x: x.mode().iloc[0]).to_dict()
train["Genero_grp_Reviews_mode"] = train["Genero1"].map(moda_por_genero_dict)
test["Genero_grp_Reviews_mode"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Reviews_mode"]

for i,j in enumerate(list(train.groupby("Genero1")["Descargas2"].mean().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Descargas2_mean"]=j
test["Genero_grp_Descargas2_mean"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas2_mean"]

for i,j in enumerate(list(train.groupby("Genero1")["Descargas2"].median().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Descargas2_median"]=j
test["Genero_grp_Descargas2_median"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas2_median"]

moda_por_genero_dict = train.groupby("Genero1")["Descargas2"].apply(lambda x: x.mode().iloc[0]).to_dict()
train["Genero_grp_Descargas2_mode"] = train["Genero1"].map(moda_por_genero_dict)
test["Genero_grp_Descargas2_mode"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas2_mode"]

for i,j in enumerate(list(train.groupby("Genero1")["Descargas3"].mean().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Descargas3_mean"]=j
test["Genero_grp_Descargas3_mean"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas3_mean"]

for i,j in enumerate(list(train.groupby("Genero1")["Descargas3"].median().sort_values().values)):
    train.loc[train["Genero1"]==i,"Genero_grp_Descargas3_median"]=j
test["Genero_grp_Descargas3_median"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas3_median"]

moda_por_genero_dict = train.groupby("Genero1")["Descargas3"].apply(lambda x: x.mode().iloc[0]).to_dict()
train["Genero_grp_Descargas3_mode"] = train["Genero1"].map(moda_por_genero_dict)
test["Genero_grp_Descargas3_mode"]=pd.merge(train, test, on="Genero1", how="left")["Genero_grp_Descargas3_mode"]

df["tamaño2"].unique()

for i,j in enumerate(list(train.groupby("tamaño2")["Descargas"].mean().values)):
    train.loc[train["tamaño2"]==i,"Tamaño2_grp_Descargas_mean"]=j
test["Tamaño2_grp_Descargas_mean"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas_mean"]

for i,j in enumerate(list(train.groupby("tamaño2")["Descargas"].median().sort_values().values)):
    train.loc[train["tamaño2"]==i,"Tamaño2_grp_Descargas_median"]=j
test["Tamaño2_grp_Descargas_median"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas_median"]

moda_por_genero_dict = train.groupby("tamaño2")["Descargas"].apply(lambda x: x.mode().iloc[0]).to_dict()
train["Tamaño2_grp_Descargas_mode"] = train["tamaño2"].map(moda_por_genero_dict)
test["Tamaño2_grp_Descargas_mode"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas_mode"]

for i,j in enumerate(list(train.groupby("tamaño2")["Descargas2"].mean().values)):
    train.loc[train["tamaño2"]==i,"Tamaño2_grp_Descargas2_mean"]=j
test["Tamaño2_grp_Descargas2_mean"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas2_mean"]

for i,j in enumerate(list(train.groupby("tamaño2")["Descargas2"].median().sort_values().values)):
    train.loc[train["tamaño2"]==i,"Tamaño2_grp_Descargas2_median"]=j
test["Tamaño2_grp_Descargas2_median"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas2_median"]

moda_por_genero_dict = train.groupby("tamaño2")["Descargas2"].apply(lambda x: x.mode().iloc[0]).to_dict()
train["Tamaño2_grp_Descargas2_mode"] = train["tamaño2"].map(moda_por_genero_dict)
test["Tamaño2_grp_Descargas2_mode"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas2_mode"]

for i,j in enumerate(list(train.groupby("tamaño2")["Descargas3"].mean().values)):
    train.loc[train["tamaño2"]==i,"Tamaño2_grp_Descargas3_mean"]=j
test["Tamaño2_grp_Descargas3_mean"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas3_mean"]

for i,j in enumerate(list(train.groupby("tamaño2")["Descargas3"].median().sort_values().values)):
    train.loc[train["tamaño2"]==i,"Tamaño2_grp_Descargas3_median"]=j
test["Tamaño2_grp_Descargas3_median"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas3_median"]

moda_por_genero_dict = train.groupby("tamaño2")["Descargas3"].apply(lambda x: x.mode().iloc[0]).to_dict()
train["Tamaño2_grp_Descargas3_mode"] = train["tamaño2"].map(moda_por_genero_dict)
test["Tamaño2_grp_Descargas3_mode"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Descargas3_mode"]

for i,j in enumerate(list(train.groupby("tamaño2")["Reviews"].mean().values)):
    train.loc[train["tamaño2"]==i,"Tamaño2_grp_Reviews_mean"]=j
test["Tamaño2_grp_Reviews_mean"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Reviews_mean"]

for i,j in enumerate(list(train.groupby("tamaño2")["Reviews"].median().sort_values().values)):
    train.loc[train["tamaño2"]==i,"Tamaño2_grp_Reviews_median"]=j
test["Tamaño2_grp_Reviews_median"]=pd.merge(train, test, on="tamaño2", how="left")["Tamaño2_grp_Reviews_median"]

df["tipo_contenido"].unique()

for i,j in enumerate(list(train.groupby("tipo_contenido")["Descargas"].mean().values)):
    train.loc[train["tipo_contenido"]==i,"tipo_contenido_grp_Descargas_mean"]=j
test["tipo_contenido_grp_Descargas_mean"]=pd.merge(train, test, on="tipo_contenido", how="left")["tipo_contenido_grp_Descargas_mean"]

for i,j in enumerate(list(train.groupby("tipo_contenido")["Descargas"].median().sort_values().values)):
    train.loc[train["tipo_contenido"]==i,"tipo_contenido_grp_Descargas_median"]=j
test["tipo_contenido_grp_Descargas_median"]=pd.merge(train, test, on="tipo_contenido", how="left")["tipo_contenido_grp_Descargas_median"]

for i,j in enumerate(list(train.groupby("tipo_contenido")["Descargas2"].mean().values)):
    train.loc[train["tipo_contenido"]==i,"tipo_contenido_grp_Descargas2_mean"]=j
test["tipo_contenido_grp_Descargas2_mean"]=pd.merge(train, test, on="tamaño2", how="left")["tipo_contenido_grp_Descargas2_mean"]

for i,j in enumerate(list(train.groupby("tipo_contenido")["Descargas2"].median().sort_values().values)):
    train.loc[train["tipo_contenido"]==i,"tipo_contenido_grp_Descargas2_median"]=j
test["tipo_contenido_grp_Descargas2_median"]=pd.merge(train, test, on="tipo_contenido", how="left")["tipo_contenido_grp_Descargas2_median"]

for i,j in enumerate(list(train.groupby("tipo_contenido")["Descargas3"].mean().values)):
    train.loc[train["tipo_contenido"]==i,"tipo_contenido_grp_Descargas3_mean"]=j
test["tipo_contenido_grp_Descargas3_mean"]=pd.merge(train, test, on="tamaño2", how="left")["tipo_contenido_grp_Descargas3_mean"]

for i,j in enumerate(list(train.groupby("tipo_contenido")["Descargas3"].median().sort_values().values)):
    train.loc[train["tipo_contenido"]==i,"tipo_contenido_grp_Descargas3_median"]=j
test["tipo_contenido_grp_Descargas3_median"]=pd.merge(train, test, on="tipo_contenido", how="left")["tipo_contenido_grp_Descargas3_median"]

for i,j in enumerate(list(train.groupby("tipo_contenido")["Reviews"].mean().values)):
    train.loc[train["tipo_contenido"]==i,"tipo_contenido_grp_Reviews_mean"]=j
test["tipo_contenido_grp_Reviews_mean"]=pd.merge(train, test, on="tipo_contenido", how="left")["tipo_contenido_grp_Reviews_mean"]

for i,j in enumerate(list(train.groupby("tipo_contenido")["Reviews"].median().sort_values().values)):
    train.loc[train["tipo_contenido"]==i,"tipo_contenido_grp_Reviews_median"]=j
test["tipo_contenido_grp_Reviews_median"]=pd.merge(train, test, on="tipo_contenido", how="left")["tipo_contenido_grp_Reviews_median"]


for i,j in enumerate(list(train.groupby("grupomeses")["Reviews"].mean().values)):
    train.loc[train["grupomeses"]==i,"grupomeses_grp_Reviews_mean"]=j
test["grupomeses_grp_Reviews_mean"]=pd.merge(train, test, on="grupomeses", how="left")["grupomeses_grp_Reviews_mean"]

for i,j in enumerate(list(train.groupby("grupomeses")["Reviews"].median().sort_values().values)):
    train.loc[train["grupomeses"]==i,"grupomeses_grp_Reviews_median"]=j
test["grupomeses_grp_Reviews_median"]=pd.merge(train, test, on="grupomeses", how="left")["grupomeses_grp_Reviews_median"]


for i,j in enumerate(list(train.groupby("version_android")["Reviews"].mean().values)):
    train.loc[train["version_android"]==i,"version_android_grp_Reviews_mean"]=j
test["version_android_grp_Reviews_mean"]=pd.merge(train, test, on="version_android", how="left")["version_android_grp_Reviews_mean"]

for i,j in enumerate(list(train.groupby("version_android")["Reviews"].median().sort_values().values)):
    train.loc[train["version_android"]==i,"version_android_grp_Reviews_median"]=j
test["version_android_grp_Reviews_median"]=pd.merge(train, test, on="version_android", how="left")["version_android_grp_Reviews_median"]

train['med_versandroid'] = train.groupby("version_android")["Descargas2"].transform('mean').round()
test["med_versandroid"]=pd.merge(train, test, on="version_android", how="left")["med_versandroid"]

#-----------------------------------------------------------------------------------------
# EXPORTAMOS TRAIN Y TEST
#-----------------------------------------------------------------------------------------

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

#-----------------------------------------------------------------------------------------
# SEPARAMOS TRAIN EN "X" e "Y" Y A SU VEZ EN TRAIN/TEST
#-----------------------------------------------------------------------------------------

X = train[["Rating",'Price',
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
y = train[['Descargas3']]



best_params = {
    'selectkbest__k': 42,
    'scaler': StandardScaler(),
    'classifier': XGBClassifier(learning_rate=0.01, 
                                min_child_weight=13, 
                                n_estimators=621,
                                )
}


best_pipeline = Pipeline(steps=[
    ("scaler", best_params['scaler']),
    ("selectkbest", SelectKBest(k=best_params['selectkbest__k'])),
    ("classifier", best_params['classifier'])
])

best_pipeline.fit(X,y)

filename = 'models/best_model/best_model.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(best_pipeline, archivo_salida)