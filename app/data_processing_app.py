import pandas as pd
import numpy as np

def transformacion(rating,precio,categoria,size,content_rating,genero,tiempo_sin_actualizar,version_android,df):

    df_test = pd.DataFrame({"Rating": [rating], "Price": [precio], "Category": [categoria], "tamaño": [size],
                             "Content Rating": [content_rating], "Genres": [genero], "dias_sin_actualizar": [tiempo_sin_actualizar],
                             "Android Ver": [version_android] })
    

    mapping_categoria = df.set_index('Category')['Categoria'].to_dict()
    df_test['Categoria'] = df_test['Category'].map(mapping_categoria)
    

    df_test.loc[df_test["tamaño"]==0, "tamaño2"]=6
    df_test.loc[df_test["tamaño"]>0, "tamaño2"]=0
    df_test.loc[df_test["tamaño"]>3500, "tamaño2"]=1
    df_test.loc[df_test["tamaño"]>7000, "tamaño2"]=2
    df_test.loc[df_test["tamaño"]>13000, "tamaño2"]=3
    df_test.loc[df_test["tamaño"]>25000, "tamaño2"]=4
    df_test.loc[df_test["tamaño"]>45000, "tamaño2"]=5
    

    df_test['Gratuito'] = np.where(df_test["Price"]==0, 1, 0)


    mapping_tipo_contenido = df.set_index('Content Rating')['tipo_contenido'].to_dict()
    df_test['tipo_contenido'] = df_test['Content Rating'].map(mapping_tipo_contenido)


    mapping_Genero = df.set_index('Genres')['Genero1'].to_dict()
    df_test['Genero1'] = df_test['Genres'].map(mapping_Genero)



    df_test["meses_sin_actualizar"] = round(df_test["dias_sin_actualizar"] / 30 ,0)
    df_test['grupomeses'] = pd.merge_asof(df_test, df.sort_values('meses_sin_actualizar'), on='meses_sin_actualizar', direction='nearest', )["grupomeses"]

    mapping_version_android = df.set_index('Android Ver')['version_android'].to_dict()
    df_test['version_android'] = df_test['Android Ver'].map(mapping_version_android)

    mapping_Genero_grp_Descargas_mean = df.set_index('Genero1')['Genero_grp_Descargas_mean'].to_dict()
    df_test['Genero_grp_Descargas_mean'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas_mean)

    mapping_Genero_grp_Descargas_median = df.set_index('Genero1')['Genero_grp_Descargas_median'].to_dict()
    df_test['Genero_grp_Descargas_median'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas_median)

    mapping_Genero_grp_Descargas_mode = df.set_index('Genero1')['Genero_grp_Descargas_mode'].to_dict()
    df_test['Genero_grp_Descargas_mode'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas_mode)

    mapping_Genero_grp_Rating_mean = df.set_index('Genero1')['Genero_grp_Rating_mean'].to_dict()
    df_test['Genero_grp_Rating_mean'] = df_test['Genero1'].map(mapping_Genero_grp_Rating_mean)

    mapping_Genero_grp_Rating_median = df.set_index('Genero1')['Genero_grp_Rating_median'].to_dict()
    df_test['Genero_grp_Rating_median'] = df_test['Genero1'].map(mapping_Genero_grp_Rating_median)

    mapping_Genero_grp_Rating_mode = df.set_index('Genero1')['Genero_grp_Rating_mode'].to_dict()
    df_test['Genero_grp_Rating_mode'] = df_test['Genero1'].map(mapping_Genero_grp_Rating_mode)

    mapping_Genero_grp_Reviews_mean = df.set_index('Genero1')['Genero_grp_Reviews_mean'].to_dict()
    df_test['Genero_grp_Reviews_mean'] = df_test['Genero1'].map(mapping_Genero_grp_Reviews_mean)

    mapping_Genero_grp_Reviews_median = df.set_index('Genero1')['Genero_grp_Reviews_median'].to_dict()
    df_test['Genero_grp_Reviews_median'] = df_test['Genero1'].map(mapping_Genero_grp_Reviews_median)

    mapping_Genero_grp_Reviews_mode = df.set_index('Genero1')['Genero_grp_Reviews_mode'].to_dict()
    df_test['Genero_grp_Reviews_mode'] = df_test['Genero1'].map(mapping_Genero_grp_Reviews_mode)

    mapping_Genero_grp_Descargas2_mean = df.set_index('Genero1')['Genero_grp_Descargas2_mean'].to_dict()
    df_test['Genero_grp_Descargas2_mean'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas2_mean)

    mapping_Genero_grp_Descargas2_median = df.set_index('Genero1')['Genero_grp_Descargas2_median'].to_dict()
    df_test['Genero_grp_Descargas2_median'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas2_median)

    mapping_Genero_grp_Descargas2_mode = df.set_index('Genero1')['Genero_grp_Descargas2_mode'].to_dict()
    df_test['Genero_grp_Descargas2_mode'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas2_mode)

    mapping_Genero_grp_Descargas3_mean = df.set_index('Genero1')['Genero_grp_Descargas3_mean'].to_dict()
    df_test['Genero_grp_Descargas3_mean'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas3_mean)

    mapping_Genero_grp_Descargas3_median = df.set_index('Genero1')['Genero_grp_Descargas3_median'].to_dict()
    df_test['Genero_grp_Descargas3_median'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas3_median)

    mapping_Genero_grp_Descargas3_mode = df.set_index('Genero1')['Genero_grp_Descargas3_mode'].to_dict()
    df_test['Genero_grp_Descargas3_mode'] = df_test['Genero1'].map(mapping_Genero_grp_Descargas3_mode)

    mapping_Tamaño2_grp_Descargas_mean = df.set_index('tamaño2')['Tamaño2_grp_Descargas_mean'].to_dict()
    df_test['Tamaño2_grp_Descargas_mean'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas_mean)

    mapping_Tamaño2_grp_Descargas_median = df.set_index('tamaño2')['Tamaño2_grp_Descargas_median'].to_dict()
    df_test['Tamaño2_grp_Descargas_median'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas_median)

    mapping_Tamaño2_grp_Descargas_mode = df.set_index('tamaño2')['Tamaño2_grp_Descargas_mode'].to_dict()
    df_test['Tamaño2_grp_Descargas_mode'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas_mode)

    mapping_Tamaño2_grp_Descargas2_mean = df.set_index('tamaño2')['Tamaño2_grp_Descargas2_mean'].to_dict()
    df_test['Tamaño2_grp_Descargas2_mean'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas2_mean)

    mapping_Tamaño2_grp_Descargas2_median = df.set_index('tamaño2')['Tamaño2_grp_Descargas2_median'].to_dict()
    df_test['Tamaño2_grp_Descargas2_median'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas2_median)

    mapping_Tamaño2_grp_Descargas2_mode = df.set_index('tamaño2')['Tamaño2_grp_Descargas2_mode'].to_dict()
    df_test['Tamaño2_grp_Descargas2_mode'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas2_mode)

    mapping_Tamaño2_grp_Descargas3_mean = df.set_index('tamaño2')['Tamaño2_grp_Descargas3_mean'].to_dict()
    df_test['Tamaño2_grp_Descargas3_mean'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas3_mean)

    mapping_Tamaño2_grp_Descargas3_median = df.set_index('tamaño2')['Tamaño2_grp_Descargas3_median'].to_dict()
    df_test['Tamaño2_grp_Descargas3_median'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas3_median)

    mapping_Tamaño2_grp_Descargas3_mode = df.set_index('tamaño2')['Tamaño2_grp_Descargas3_mode'].to_dict()
    df_test['Tamaño2_grp_Descargas3_mode'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Descargas3_mode)

    mapping_Tamaño2_grp_Reviews_mean = df.set_index('tamaño2')['Tamaño2_grp_Reviews_mean'].to_dict()
    df_test['Tamaño2_grp_Reviews_mean'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Reviews_mean)

    mapping_Tamaño2_grp_Reviews_median = df.set_index('tamaño2')['Tamaño2_grp_Reviews_median'].to_dict()
    df_test['Tamaño2_grp_Reviews_median'] = df_test['tamaño2'].map(mapping_Tamaño2_grp_Reviews_median)

    mapping_tipo_contenido_grp_Descargas_mean = df.set_index('tipo_contenido')['tipo_contenido_grp_Descargas_mean'].to_dict()
    df_test['tipo_contenido_grp_Descargas_mean'] = df_test['tipo_contenido'].map(mapping_tipo_contenido_grp_Descargas_mean)

    mapping_tipo_contenido_grp_Descargas_median = df.set_index('tipo_contenido')['tipo_contenido_grp_Descargas_median'].to_dict()
    df_test['tipo_contenido_grp_Descargas_median'] = df_test['tipo_contenido'].map(mapping_tipo_contenido_grp_Descargas_median)

    mapping_tipo_contenido_grp_Descargas2_mean = df.set_index('tipo_contenido')['tipo_contenido_grp_Descargas2_mean'].to_dict()
    df_test['tipo_contenido_grp_Descargas2_mean'] = df_test['tipo_contenido'].map(mapping_tipo_contenido_grp_Descargas2_mean)

    mapping_tipo_contenido_grp_Descargas2_median = df.set_index('tipo_contenido')['tipo_contenido_grp_Descargas2_median'].to_dict()
    df_test['tipo_contenido_grp_Descargas2_median'] = df_test['tipo_contenido'].map(mapping_tipo_contenido_grp_Descargas2_median)

    mapping_tipo_contenido_grp_Descargas3_mean = df.set_index('tipo_contenido')['tipo_contenido_grp_Descargas3_mean'].to_dict()
    df_test['tipo_contenido_grp_Descargas3_mean'] = df_test['tipo_contenido'].map(mapping_tipo_contenido_grp_Descargas3_mean)

    mapping_tipo_contenido_grp_Descargas3_median = df.set_index('tipo_contenido')['tipo_contenido_grp_Descargas3_median'].to_dict()
    df_test['tipo_contenido_grp_Descargas3_median'] = df_test['tipo_contenido'].map(mapping_tipo_contenido_grp_Descargas3_median)

    mapping_tipo_contenido_grp_Reviews_mean = df.set_index('tipo_contenido')['tipo_contenido_grp_Reviews_mean'].to_dict()
    df_test['tipo_contenido_grp_Reviews_mean'] = df_test['tipo_contenido'].map(mapping_tipo_contenido_grp_Reviews_mean)

    mapping_tipo_contenido_grp_Reviews_median = df.set_index('tipo_contenido')['tipo_contenido_grp_Reviews_median'].to_dict()
    df_test['tipo_contenido_grp_Reviews_median'] = df_test['tipo_contenido'].map(mapping_tipo_contenido_grp_Reviews_median)

    mapping_grupomeses_grp_Reviews_mean = df.set_index('grupomeses')['grupomeses_grp_Reviews_mean'].to_dict()
    df_test['grupomeses_grp_Reviews_mean'] = df_test['grupomeses'].map(mapping_grupomeses_grp_Reviews_mean)

    mapping_grupomeses_grp_Reviews_median = df.set_index('grupomeses')['grupomeses_grp_Reviews_median'].to_dict()
    df_test['grupomeses_grp_Reviews_median'] = df_test['grupomeses'].map(mapping_grupomeses_grp_Reviews_median)

    mapping_version_android_grp_Reviews_mean = df.set_index('version_android')['version_android_grp_Reviews_mean'].to_dict()
    df_test['version_android_grp_Reviews_mean'] = df_test['version_android'].map(mapping_version_android_grp_Reviews_mean)

    mapping_version_android_grp_Reviews_median = df.set_index('version_android')['version_android_grp_Reviews_median'].to_dict()
    df_test['version_android_grp_Reviews_median'] = df_test['version_android'].map(mapping_version_android_grp_Reviews_median)

    mapping_med_versandroid = df.set_index('version_android')['med_versandroid'].to_dict()
    df_test['med_versandroid'] = df_test['version_android'].map(mapping_med_versandroid)

    X = df_test[["Rating",'Price',
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
    
    return df_test, X