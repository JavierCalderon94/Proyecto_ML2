import pandas as pd
import numpy as np
from datetime import datetime
import os
os.chdir(os.path.dirname(os.getcwd()))

df = pd.read_csv("data/raw/googleplaystore.csv", index_col=0)

#-----------------------------------------------------------------------------------------
# LIMPIEZA DE DUPLICADOS Y RATING
#-----------------------------------------------------------------------------------------
df.loc[df["Reviews"]==0,"Rating"]=0.0
df.sort_values(by="App", inplace=True)
df.reset_index(inplace=True,drop=True)

for i in range(0,len(df)):
    if i==0:
        continue
    if df.loc[i,"App"]==df.loc[(i-1),"App"] and df.loc[i,"Category"]==df.loc[(i-1),"Category"]:
        df.loc[i,"duplicado"]=1
    else:
        df.loc[i,"duplicado"]=0

df.drop(df[df["duplicado"]==1].index,axis=0, inplace=True)
df.drop("duplicado",axis=1,inplace=True)

df["Rating"] = np.where(df["Rating"].str.contains(","),df["Rating"].str.replace(",","."),df["Rating"])
df.loc[df["Rating"].isna(),"Rating"]=0
df["Rating"]=df["Rating"].astype(float)


#-----------------------------------------------------------------------------------------
#LIMPIEZA DE INSTALLS (VARIABLE OBJETIVO) Y GENERAMOS NUEVAS VARIABLES CATEGÓRICAS
#-----------------------------------------------------------------------------------------

df["Installs"]=df["Installs"].str.replace(",","")
df["Installs"]=df["Installs"].str.replace("+","")
df["Installs"]=df["Installs"].astype(int)

for i,j in enumerate(list(df["Installs"].sort_values().unique())):
    df.loc[df["Installs"]==j, "Descargas"] = i

df.loc[df["Installs"]>=0,"Descargas2"]=0
df.loc[df["Installs"]>=1000,"Descargas2"]=1
df.loc[df["Installs"]>=10000,"Descargas2"]=2
df.loc[df["Installs"]>=100000,"Descargas2"]=3
df.loc[df["Installs"]>=1000000,"Descargas2"]=4
df.loc[df["Installs"]>=10000000,"Descargas2"]=5
df["Descargas2"].value_counts()

df["Descargas3"] = np.where((df["Descargas2"] == 0) | (df["Descargas2"] == 1), 0,
                           np.where((df["Descargas2"] == 2) | (df["Descargas2"] == 3), 1,
                                    np.where((df["Descargas2"] == 4) | (df["Descargas2"] == 5), 2, df["Descargas2"])))


#-----------------------------------------------------------------------------------------
# CONVERTIMOS CATEGÓRICA EN VALORES ORDENADOS EN FUNCIÓN DE LA MEDIA CON DESCARGAS
#-----------------------------------------------------------------------------------------

for i,j in enumerate(list(df.groupby("Category")["Descargas3"].mean().sort_values().index)):
    df.loc[df["Category"]==j,"Categoria"]=i


#-----------------------------------------------------------------------------------------
# CONVERTIMOS EL TAMAÑO DEL ARCHIVO EN INT Y AGRUPAMOS.
# LOS GRUPOS IRÁN ORDENADOS EN FUNCIÓN DE LA MEDIA
#-----------------------------------------------------------------------------------------

df["tamaño"] = np.where(df["Size"].str.contains("k"), df["Size"].str.replace("k",""), np.where(df["Size"].str.contains("M"), df["Size"].str.replace("M",""), np.where(df["Size"].str.contains("Varies with device"), df["Size"].str.replace("Varies with device","0"), df["Size"])) )
df["tamaño"]=df[["tamaño"]].astype(float)
df["tamaño"]=np.where(df["Size"].str.contains("M"),df["tamaño"]*1000,df["tamaño"] )

df.loc[df["tamaño"]==0, "tamaño2"]=0
df.loc[df["tamaño"]>0, "tamaño2"]=1
df.loc[df["tamaño"]>3500, "tamaño2"]=2
df.loc[df["tamaño"]>7000, "tamaño2"]=3
df.loc[df["tamaño"]>13000, "tamaño2"]=4
df.loc[df["tamaño"]>25000, "tamaño2"]=5
df.loc[df["tamaño"]>45000, "tamaño2"]=6

df["tamaño2"]= np.where(df["tamaño2"]==0, 6, 
                        np.where(df["tamaño2"]==1, 0,
                                 np.where(df["tamaño2"]==2, 1,
                                          np.where(df["tamaño2"]==3, 2,
                                                   np.where(df["tamaño2"]==4, 3,
                                                            np.where(df["tamaño2"]==5, 4,
                                                                     np.where(df["tamaño2"]==6, 5,df["tamaño2"])))))))


#-----------------------------------------------------------------------------------------
# LIMPIEZA DE TYPE Y PRICE
#-----------------------------------------------------------------------------------------

df.loc[df["Type"]=="Free","Gratuito"]=1
df.loc[df["Type"]=="Paid","Gratuito"]=0

df["Price"]=df["Price"].str.replace("$","").copy()
df["Price"]=df["Price"].astype(float).copy()

#-----------------------------------------------------------------------------------------
# LIMPIEZA DE CONTENT RATING (PÚBLICO OBJETIVO)
# HAY GRUPOS APENAS SIN VALORES, LOS DROPEAMOS.
# CONVERTIMOS CATEGÓRICA EN VALORES ORDENADOS EN FUNCIÓN DE LA MEDIA CON DESCARGAS
#-----------------------------------------------------------------------------------------

df.drop(df[df["Content Rating"]=="Adults only 18+"].index,axis=0,inplace=True)
df.drop(df[df["Content Rating"]=="Unrated"].index,axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)

for i,j in enumerate(list(df.groupby("Content Rating")["Descargas3"].mean().sort_values().index)):
    df.loc[df["Content Rating"]==j,"tipo_contenido"]=i

#-----------------------------------------------------------------------------------------
# LIMPIEZA DE GÉNERO
# HAY GRUPOS DUPLICADOS, LOS JUNTAMOS.
# CONVERTIMOS CATEGÓRICA EN VALORES ORDENADOS EN FUNCIÓN DE LA MEDIA CON DESCARGAS
#-----------------------------------------------------------------------------------------

df.loc[df["Genres"]=="Music & Audio","Genres"]="Music"
for i,j in enumerate(list(range(0,len(df)))):
    df.loc[i,"Genero1"] = df.loc[i,"Genres"].split(";")[0]

for i,j in enumerate(list(df.groupby("Genero1")["Descargas3"].mean().sort_values().index)):
    df.loc[df["Genero1"]==j,"Genero1"]=i


#-----------------------------------------------------------------------------------------
# LIMPIEZA DE FECHA DE ÚLTIMA ACTUALIZACIÓN, OBTENEMOS LOS DÍAS SIN ACTUALIZAR
# OBTENEMOS MESES SIN ACTUALIZAR
# CONVERTIMOS CATEGÓRICA EN VALORES ORDENADOS EN FUNCIÓN DE LA MEDIA CON DESCARGAS
# AGRUPAMOS MESES CON LA MEDIA DE DESCARGAS PARA OBTENER 6 GRUPOS
#-----------------------------------------------------------------------------------------

df["Last Updated"]=np.where(df["Last Updated"].str.contains("Jan"), df["Last Updated"].str.replace("Jan","01"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Feb"), df["Last Updated"].str.replace("Feb","02"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Mar"), df["Last Updated"].str.replace("Mar","03"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Apr"), df["Last Updated"].str.replace("Apr","04"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("May"), df["Last Updated"].str.replace("May","05"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Jun"), df["Last Updated"].str.replace("Jun","06"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Jul"), df["Last Updated"].str.replace("Jul","07"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Aug"), df["Last Updated"].str.replace("Aug","08"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Sep"), df["Last Updated"].str.replace("Sep","09"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Oct"), df["Last Updated"].str.replace("Oct","10"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Nov"), df["Last Updated"].str.replace("Nov","11"), df["Last Updated"])
df["Last Updated"]=np.where(df["Last Updated"].str.contains("Dec"), df["Last Updated"].str.replace("Dec","12"), df["Last Updated"])

for i,j in enumerate(list(range(0,len(df)))):
    df.loc[i,"Last Updated"] = "20" + df.loc[i,"Last Updated"].split("-")[2] + "-" + df.loc[i,"Last Updated"].split("-")[1] + "-" + df.loc[i,"Last Updated"].split("-")[0]

df['Last Updated'] = pd.to_datetime(df['Last Updated'])
fecha_actual = datetime.now()
df['dias_sin_actualizar'] = (fecha_actual - df['Last Updated']).dt.days

df["meses_sin_actualizar"] = round(df["dias_sin_actualizar"] / 30 ,0)

media_ins_mes = pd.DataFrame(df.groupby("meses_sin_actualizar")["Descargas2"].mean().sort_values()).reset_index().sort_values(by="meses_sin_actualizar").reset_index(drop=True)
for i,j in enumerate(list(media_ins_mes["meses_sin_actualizar"])):
    if media_ins_mes.loc[i,"Descargas2"]>=0:
        df.loc[df["meses_sin_actualizar"]==j,"grupomeses"]=0
    if media_ins_mes.loc[i,"Descargas2"]>=2:
        df.loc[df["meses_sin_actualizar"]==j,"grupomeses"]=1
    if media_ins_mes.loc[i,"Descargas2"]>=2.2:
        df.loc[df["meses_sin_actualizar"]==j,"grupomeses"]=2
    if media_ins_mes.loc[i,"Descargas2"]>=2.4:
        df.loc[df["meses_sin_actualizar"]==j,"grupomeses"]=3
    if media_ins_mes.loc[i,"Descargas2"]>=2.6:
        df.loc[df["meses_sin_actualizar"]==j,"grupomeses"]=4
    if media_ins_mes.loc[i,"Descargas2"]>=3:
        df.loc[df["meses_sin_actualizar"]==j,"grupomeses"]=5


#-----------------------------------------------------------------------------------------
# LIMPIEZA DE VERSIÓN ACTUAL DE LA APLICACIÓN.
# FINALMENTE NO SE USARÁ
#-----------------------------------------------------------------------------------------

df["version_actual"] = df["Current Ver"].str[0:2]
df["version_actual"] = df["version_actual"].str.replace(",","")
df["version_actual"] = df["version_actual"].str.replace(".","")
df["version_actual"] = df["version_actual"].str.replace("v","")
df["version_actual"] = df["version_actual"].str.replace("_","")
df["version_actual"] = df["version_actual"].str.replace("/","")
df["version_actual"] = df["version_actual"].str.replace("r","")

mask = df['version_actual'].str.contains(r'[a-zA-Z]', regex=True)
df.loc[mask, 'version_actual'] = '100'

df["version_actual"]=df["version_actual"].astype(int)
df["version_actual"].sort_values().unique()

media_ins_act = pd.DataFrame(df.groupby("version_actual")["Descargas2"].mean().sort_values()).reset_index().sort_values(by="Descargas2").reset_index(drop=True)
for i,j in enumerate(list(media_ins_act["version_actual"])):
    if media_ins_act.loc[i,"Descargas2"]>=0:
        df.loc[df["version_actual"]==j,"grupoversiones"]=0
    if media_ins_act.loc[i,"Descargas2"]>=2.2:
        df.loc[df["version_actual"]==j,"grupoversiones"]=1
    if media_ins_act.loc[i,"Descargas2"]>=3:
        df.loc[df["version_actual"]==j,"grupoversiones"]=2

#-----------------------------------------------------------------------------------------
# LIMPIEZA DE VERSIÓN ANDROID COMPATIBLE. AGRUPAMOS Y ORDENAMOS EN FUNCION DE LA MEDIA
#-----------------------------------------------------------------------------------------

for i,j in enumerate(list(df["Android Ver"].unique())):
    df.loc[df["Android Ver"]==j,"version_android"] = i

df.loc[df["version_android"]>=14,"version_android"]=14
df.loc[df["version_android"]==2,"version_android"]=3

version = pd.DataFrame(df.groupby("version_android")["Descargas2"].mean().sort_values().reset_index()).reset_index()
df["version_android"] = pd.merge(df,version, on=["version_android"],how="left")["index"]


#-----------------------------------------------------------------------------------------
# EXPORTAMOS EL MODELO FINAL
#-----------------------------------------------------------------------------------------
df.to_csv("data/processed.csv",index=False)