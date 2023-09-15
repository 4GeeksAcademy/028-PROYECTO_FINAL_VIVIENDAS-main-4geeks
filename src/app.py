import streamlit as st
import pandas as pd
import os
import joblib

# Cargar los modelos manualmente para cada provincia
model_dir = "/workspaces/028-PROYECTO_FINAL_VIVIENDAS-main/src/PROVINCIAS/modelos_finales"

# Diccionario de modelos por provincia
models = {
    "alava": "alava_randomforest_gridsearch_gradientboosting_default_42.sav",
    "alicante": "alicante_randomforest_gridsearch_gradientboosting_default_42.sav",
    "albacete": "albacete_randomforest_gridsearch_gradientboosting_default_42.sav",
    "almeria": "almeria_randomforest_gridsearch_gradientboosting_default_42.sav",
    "asturias": "asturias_randomforest_gridsearch_gradientboosting_default_42.sav",
    "avila": "avila_randomforest_gridsearch_gradientboosting_default_42.sav",
    "barcelona": "barcelona_randomforest_gridsearch_gradientboosting_default_42.sav",
    "burgos": "burgos_randomforest_gridsearch_gradientboosting_default_42.sav",
    "caceres": "caceres_randomforest_gridsearch_gradientboosting_default_42.sav",
    "cadiz": "cadiz_randomforest_gridsearch_gradientboosting_default_42.sav",
    "cantabria": "cantabria_randomforest_gridsearch_gradientboosting_default_42.sav",
    "castellon": "castellon_randomforest_gridsearch_gradientboosting_default_42.sav",
    "ciudad_real": "ciudad_real_randomforest_gridsearch_gradientboosting_default_42.sav",
    "cordoba": "cordoba_randomforest_gridsearch_gradientboosting_default_42.sav",
    "cuenca": "cuenca_randomforest_gridsearch_gradientboosting_default_42.sav",
    "girona": "girona_randomforest_gridsearch_gradientboosting_default_42.sav",
    "granada": "granada_randomforest_gridsearch_gradientboosting_default_42.sav",
    "guadalajara": "guadalajara_randomforest_gridsearch_gradientboosting_default_42.sav",
    "guipuzcoa": "guipuzcoa_randomforest_gridsearch_gradientboosting_default_42.sav",
    "huelva": "huelva_randomforest_gridsearch_gradientboosting_default_42.sav",
    "huesca": "huesca_randomforest_gridsearch_gradientboosting_default_42.sav",
    "jaen": "jaen_randomforest_gridsearch_gradientboosting_default_42.sav",
    "a_coruña": "a_coruña_randomforest_gridsearch_gradientboosting_default_42.sav",
    "la_rioja": "la_rioja_randomforest_gridsearch_gradientboosting_default_42.sav",
    "las_palmas": "las_palmas_randomforest_gridsearch_gradientboosting_default_42.sav",
    "leon": "leon_randomforest_gridsearch_gradientboosting_default_42.sav",
    "lleida": "lleida_randomforest_gridsearch_gradientboosting_default_42.sav",
    "lugo": "lugo_randomforest_gridsearch_gradientboosting_default_42.sav",
    "madrid": "madrid_randomforest_gridsearch_gradientboosting_default_42.sav",
    "malaga": "malaga_randomforest_gridsearch_gradientboosting_default_42.sav",
    "murcia": "murcia_randomforest_gridsearch_gradientboosting_default_42.sav",
    "navarra": "navarra_randomforest_gridsearch_gradientboosting_default_42.sav",
    "ourense": "ourense_randomforest_gridsearch_gradientboosting_default_42.sav",
    "palencia": "palencia_randomforest_gridsearch_gradientboosting_default_42.sav",
    "pontevedra": "pontevedra_randomforest_gridsearch_gradientboosting_default_42.sav",
    "salamanca": "salamanca_randomforest_gridsearch_gradientboosting_default_42.sav",
    "santa_cruz_de_tenerife": "santa_cruz_de_tenerife_randomforest_gridsearch_gradientboosting_default_42.sav",
    "segovia": "segovia_randomforest_gridsearch_gradientboosting_default_42.sav",
    "sevilla": "sevilla_randomforest_gridsearch_gradientboosting_default_42.sav",
    "soria": "soria_randomforest_gridsearch_gradientboosting_default_42.sav",
    "tarragona": "tarragona_randomforest_gridsearch_gradientboosting_default_42.sav",
    "teruel": "teruel_randomforest_gridsearch_gradientboosting_default_42.sav",
    "toledo": "toledo_randomforest_gridsearch_gradientboosting_default_42.sav",
    "valencia": "valencia_randomforest_gridsearch_gradientboosting_default_42.sav",
    "valladolid": "valladolid_randomforest_gridsearch_gradientboosting_default_42.sav",
    "vizcaya": "vizcaya_randomforest_gridsearch_gradientboosting_default_42.sav",
    "zamora": "zamora_randomforest_gridsearch_gradientboosting_default_42.sav",
    "zaragoza": "zaragoza_randomforest_gridsearch_gradientboosting_default_42.sav",
}

# Diccionario de rangos por provincia para Metros Cuadrados
rango_m2_por_provincia = {
    "alava": {"min": 25, "max": 900, "value": 100},
    "albacete": {"min": 25, "max": 1000, "value": 100},
    "alicante": {"min": 30, "max": 5000, "value": 100},
    "almeria": {"min": 35, "max": 4500, "value": 100},
    "cadiz": {"min": 35, "max": 4500, "value": 100},
    "cordoba": {"min": 35, "max": 4500, "value": 100},
    "granada": {"min": 35, "max": 4500, "value": 100},
    "huelva": {"min": 35, "max": 4500, "value": 100},
    "jaen": {"min": 35, "max": 4500, "value": 100},
    "malaga": {"min": 35, "max": 4500, "value": 100},
    "sevilla": {"min": 35, "max": 4500, "value": 100},
    "avila": {"min": 0, "max": 0, "value": 0},
    "badajoz": {"min": 0, "max": 0, "value": 0},
    "barcelona": {"min": 0, "max": 0, "value": 0},
    "burgos": {"min": 0, "max": 0, "value": 0},
    "caceres": {"min": 0, "max": 0, "value": 0},
    "castellon": {"min": 0, "max": 0, "value": 0},
    "ciudad_real": {"min": 0, "max": 0, "value": 0},
    "cuenca": {"min": 0, "max": 0, "value": 0},
    "girona": {"min": 0, "max": 0, "value": 0},
    "guadalajara": {"min": 0, "max": 0, "value": 0},
    "guipuzcoa": {"min": 0, "max": 0, "value": 0},
    "huesca": {"min": 0, "max": 0, "value": 0},
    "illes_balears": {"min": 0, "max": 0, "value": 0},
    "jaen": {"min": 25, "max": 150, "value": 0},
    "la_coruna": {"min": 0, "max": 0, "value": 0},
    "la_rioja": {"min": 0, "max": 0, "value": 0},
    "las_palmas": {"min": 0, "max": 0, "value": 0},
    "leon": {"min": 0, "max": 0, "value": 0},
    "lleida": {"min": 0, "max": 0, "value": 0},
    "lugo": {"min": 0, "max": 0, "value": 0},
    "madrid": {"min": 0, "max": 0, "value": 0},
    "murcia": {"min": 0, "max": 0, "value": 0},
    "navarra": {"min": 0, "max": 0, "value": 0},
    "ourense": {"min": 0, "max": 0, "value": 0},
    "palencia": {"min": 0, "max": 0, "value": 0},
    "pontevedra": {"min": 0, "max": 0, "value": 0},
    "salamanca": {"min": 0, "max": 0, "value": 0},
    "santa_cruz_de_tenerife": {"min": 0, "max": 0, "value": 0},
    "segovia": {"min": 0, "max": 0, "value": 0},
    "soria": {"min": 0, "max": 0, "value": 0},
    "tarragona": {"min": 0, "max": 0, "value": 0},
    "teruel": {"min": 0, "max": 0, "value": 0},
    "toledo": {"min": 0, "max": 0, "value": 0},
    "valencia": {"min": 0, "max": 0, "value": 0},
    "valladolid": {"min": 0, "max": 0, "value": 0},
    "vizcaya": {"min": 0, "max": 0, "value": 0},
    "zamora": {"min": 0, "max": 0, "value": 0},
    "zaragoza": {"min": 0, "max": 0, "value": 0}
}

# Diccionario de rangos por provincia para Número de Habitaciones (inventa valores aquí)
rango_habitaciones_por_provincia = {
    "alava": {"min": 0, "max": 14, "value": 3},
    "alicante": {"min": 0, "max": 12, "value": 3},
    "albacete": {"min": 0, "max": 16, "value": 3},
    "almeria": {"min": 0, "max": 14, "value": 3},
    "badajoz": {"min": 0, "max": 13, "value": 0},
    "barcelona": {"min": 0, "max": 18, "value": 0},
    "burgos": {"min": 0, "max": 15, "value": 0},
    "caceres": {"min": 0, "max": 12, "value": 0},
    "cadiz": {"min": 0, "max": 8, "value": 3},
    "castellon": {"min": 0, "max": 10, "value": 0},
    "ciudad_real": {"min": 0, "max": 14, "value": 0},
    "cordoba": {"min": 0, "max": 18, "value": 3},
    "cuenca": {"min": 0, "max": 10, "value": 0},
    "girona": {"min": 0, "max": 12, "value": 0},
    "granada": {"min": 0, "max": 18, "value": 3},
    "guadalajara": {"min": 0, "max": 20, "value": 0},
    "guipuzcoa": {"min": 0, "max": 20, "value": 0},
    "huelva": {"min": 0, "max": 18, "value": 3},
    "huesca": {"min": 0, "max": 12, "value": 0},
    "islas_baleares": {"min": 0, "max": 0, "value": 0},
    "jaen": {"min": 0, "max": 10, "value": 3},
    "la_coruna": {"min": 0, "max": 15, "value": 0},
    "la_rioja": {"min": 0, "max": 13, "value": 0},
    "las_palmas": {"min": 0, "max": 18, "value": 0},
    "leon": {"min": 0, "max": 30, "value": 0},
    "lleida": {"min": 0, "max": 20, "value": 0},            
    "lugo": {"min": 0, "max": 15, "value": 0},
    "madrid": {"min": 0, "max": 12, "value": 0},
    "malaga": {"min": 0, "max": 14, "value": 3},
    "murcia": {"min": 0, "max": 15, "value": 0},
    "navarra": {"min": 0, "max": 25, "value": 0},
    "ourense": {"min": 0, "max": 18, "value": 0},
    "palencia": {"min": 0, "max": 20, "value": 0},
    "pontevedra": {"min": 0, "max": 10, "value": 0},
    "salamanca": {"min": 0, "max": 15, "value": 0},
    "santa_cruz_de_tenerife": {"min": 0, "max": 12, "value": 0},
    "segovia": {"min": 0, "max": 19, "value": 0},
    "sevilla": {"min": 0, "max": 30, "value": 3},
    "soria": {"min": 0, "max": 17, "value": 0},
    "tarragona": {"min": 0, "max": 15, "value": 0},
    "teruel": {"min": 0, "max": 25, "value": 0},
    "toledo": {"min": 0, "max": 18, "value": 0},
    "valencia": {"min": 0, "max": 20, "value": 0},
    "valladolid": {"min": 0, "max": 15, "value": 0},
    "vizcaya": {"min": 0, "max": 20, "value": 0},
    "zamora": {"min": 0, "max": 20, "value": 0},
    "zaragoza": {"min": 0, "max": 25, "value": 0}
}





# Comunidades y provincias
comunidades = {
    "andalucia": ["almeria", "cadiz", "cordoba", "granada", "huelva", "jaen", "malaga", "sevilla"],
    "aragon": ["huesca", "teruel", "zaragoza"],
    "asturias": ["asturias"],
    "islas_baleares": ["islas_baleares"],
    "canarias": ["las_palmas", "santa_cruz_de_tenerife"],
    "cantabria": ["cantabria"],
    "castilla_y_leon": ["avila", "burgos", "leon", "palencia", "salamanca", "segovia", "soria", "valladolid", "zamora"],
    "castilla-la_mancha": ["albacete", "ciudad_real", "cuenca", "guadalajara", "toledo"],
    "cataluna": ["barcelona", "girona", "lleida", "tarragona"],
    "extremadura": ["badajoz", "caceres"],
    "galicia": ["a_coruña", "lugo", "ourense", "pontevedra"],
    "madrid": ["madrid"],
    "murcia": ["region_de_murcia"],
    "navarra": ["navarra"],
    "la_rioja": ["la_rioja"],
    "comunidad_valenciana": ["alicante", "castellon", "valencia"],
    "pais_vasco": ["alava", "guipuzcoa", "vizcaya"]
}

# Características
comunidad = st.selectbox("Comunidad Autónoma", list(comunidades.keys()))
provincia = st.selectbox("Provincia", comunidades[comunidad])

# Establecer los valores de Metros Cuadrados según la provincia seleccionada
if provincia in rango_m2_por_provincia:
    m2_min, m2_max, m2_value = (
        rango_m2_por_provincia[provincia]["min"],
        rango_m2_por_provincia[provincia]["max"],
        rango_m2_por_provincia[provincia]["value"]
    )
else:
    # Valores predeterminados globales para Metros Cuadrados
    m2_min, m2_max, m2_value = (27, 600, 100)

# Usar los valores para el slider de Metros Cuadrados
m2 = st.slider("Metros cuadrados", m2_min, m2_max, m2_value)

# Establecer los valores de Número de Habitaciones según la provincia seleccionada
if provincia in rango_habitaciones_por_provincia:
    habitaciones_min, habitaciones_max, habitaciones_value = (
        rango_habitaciones_por_provincia[provincia]["min"],
        rango_habitaciones_por_provincia[provincia]["max"],
        rango_habitaciones_por_provincia[provincia]["value"]
    )
else:
    # Valores predeterminados globales para Número de Habitaciones
    habitaciones_min, habitaciones_max, habitaciones_value = (2, 8, 3)

# Usar los valores para el slider de Número de Habitaciones
habitaciones = st.slider("Número de habitaciones", habitaciones_min, habitaciones_max, habitaciones_value)

# Función para cargar el modelo
def load_model(provincia):
    model_name = f"{provincia}_randomforest_gridsearch_gradientboosting_default_42.sav"
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

# Suponiendo que min_price y max_price son los valores mínimos y máximos originales del precio
min_price = 34000  # Valor mínimo del precio después de excluir ceros
max_price = 2900000  # Valor máximo del precio después de excluir ceros

# Función para normalizar las características
def normalize_features(m2, habitaciones):
    min_m2 = 27  # Valor mínimo de 'm2'
    max_m2 = 6000  # Valor máximo de 'm2'
    min_habitaciones = 0  # Reemplaza con el valor mínimo real
    max_habitaciones = 14  # Reemplaza con el valor máximo real

    normalized_m2 = (m2 - min_m2) / (max_m2 - min_m2)
    normalized_habitaciones = (habitaciones - min_habitaciones) / (max_habitaciones - min_habitaciones)

    return normalized_m2, normalized_habitaciones

# Botón para predecir el precio
if st.button("Ver precio de mi vivienda"):
    if provincia in models:
        model = load_model(provincia)
        if model is not None:
            # Normalizar las características
            normalized_m2, normalized_habitaciones = normalize_features(m2, habitaciones)
            
            features = pd.DataFrame({
                "m2": [normalized_m2],  # Usar el valor normalizado
                "Habitaciones": [normalized_habitaciones]  # Usar el valor normalizado
            })
            prediction_scaled = model.predict(features)
            
            # Aplicar transformación inversa para obtener el precio en la escala original
            prediction = prediction_scaled * (max_price - min_price) + min_price
            
            # Convertir el precio a número entero
            prediction_int = int(prediction[0])

            # Formatear el precio con millares
            formatted_price = (prediction_int)
            
            st.write(f"El precio estimado de tu vivienda en {provincia.capitalize()} es: {formatted_price} euros")
        else:
            st.error(f"No se encontró un modelo para la provincia {provincia.capitalize()}.")
    else:
        st.error("Provincia no válida. Por favor, selecciona una provincia válida.")
