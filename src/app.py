import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np


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
    "islas_baleares": "islas_baleares_randomforest_gridsearch_gradientboosting_default_42.sav",
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
    "alava": {"min": 30, "max": 3800, "value": 100},
    "albacete": {"min": 30, "max": 8000, "value": 100},
    "alicante": {"min": 30, "max": 7000, "value": 100},
    "almeria": {"min": 30, "max": 500, "value": 100},
    "asturias": {"min": 30, "max": 1500, "value": 100},
    "cadiz": {"min": 30, "max": 3870, "value": 100},
    "cordoba": {"min": 30, "max": 4000, "value": 100},
    "granada": {"min": 30, "max": 4200, "value": 100},
    "huelva": {"min": 30, "max": 7400, "value": 100},
    "jaen": {"min": 30, "max": 6000, "value": 100},
    "malaga": {"min": 30, "max": 4800, "value": 100},
    "sevilla": {"min": 30, "max": 6500, "value": 100},
    "avila": {"min": 30, "max": 7000, "value": 100},
    "badajoz": {"min": 30, "max": 8000, "value": 100},
    "barcelona": {"min": 30, "max": 7000, "value": 100},
    "burgos": {"min": 30, "max": 4000, "value": 100},
    "caceres": {"min": 30, "max": 8000, "value": 100},
    "castellon": {"min": 30, "max": 8200, "value": 100},
    "ciudad_real": {"min": 30, "max": 1200, "value": 100},
    "cuenca": {"min": 30, "max": 4200, "value": 100},
    "girona": {"min": 30, "max": 4300, "value": 100},
    "guadalajara": {"min": 30, "max": 5300, "value": 100},
    "guipuzcoa": {"min": 30, "max": 6000, "value": 100},
    "huesca": {"min": 30, "max": 7500, "value": 100},
    "islas_baleares": {"min": 30, "max": 5000, "value": 100},
    "jaen": {"min": 30, "max": 5000, "value": 100},
    "la_coruna": {"min": 30, "max": 5000, "value": 100},
    "la_rioja": {"min": 30, "max": 8500, "value": 100},
    "las_palmas": {"min": 30, "max": 8000, "value": 100},
    "leon": {"min": 30, "max": 8300, "value": 100},
    "lleida": {"min": 30, "max": 6000, "value": 100},
    "lugo": {"min": 30, "max": 4500, "value": 100},
    "madrid": {"min": 30, "max": 2700, "value": 100},
    "murcia": {"min": 30, "max": 6000, "value": 100},
    "navarra": {"min": 30, "max": 7000, "value": 100},
    "ourense": {"min": 30, "max": 6000, "value": 100},
    "palencia": {"min": 30, "max": 3000, "value": 100},
    "pontevedra": {"min": 30, "max": 4300, "value": 100},
    "salamanca": {"min": 30, "max": 8500, "value": 0},
    "santa_cruz_de_tenerife": {"min": 30, "max": 7000, "value": 100},
    "segovia": {"min": 30, "max": 8000, "value": 100},
    "soria": {"min": 30, "max": 5000, "value": 100},
    "tarragona": {"min": 30, "max": 8000, "value": 100},
    "teruel": {"min": 30, "max": 8500, "value": 100},
    "toledo": {"min": 30, "max": 5000, "value": 100},
    "valencia": {"min": 30, "max": 8000, "value": 100},
    "valladolid": {"min": 30, "max": 7000, "value": 100},
    "vizcaya": {"min": 30, "max": 5000, "value": 100},
    "zamora": {"min": 30, "max": 6700, "value": 100},
    "zaragoza": {"min": 30, "max": 3600, "value": 100}
}

# Diccionario de rangos por provincia para Número de Habitaciones (inventa valores aquí)
rango_habitaciones_por_provincia = {
    "alava": {"min": 0, "max": 14, "value": 3},
    "alicante": {"min": 0, "max": 12, "value": 3},
    "albacete": {"min": 0, "max": 16, "value": 3},
    "almeria": {"min": 0, "max": 14, "value": 3},
    "asturias": {"min": 0, "max": 14, "value": 3},
    "badajoz": {"min": 0, "max": 13, "value": 3},
    "barcelona": {"min": 0, "max": 18, "value": 3},
    "burgos": {"min": 0, "max": 15, "value": 3},
    "caceres": {"min": 0, "max": 12, "value": 3},
    "cadiz": {"min": 0, "max": 8, "value": 3},
    "castellon": {"min": 0, "max": 10, "value": 3},
    "ciudad_real": {"min": 0, "max": 14, "value": 3},
    "cordoba": {"min": 0, "max": 18, "value": 3},
    "cuenca": {"min": 0, "max": 10, "value": 3},
    "girona": {"min": 0, "max": 12, "value": 3},
    "granada": {"min": 0, "max": 18, "value": 3},
    "guadalajara": {"min": 0, "max": 20, "value": 3},
    "guipuzcoa": {"min": 0, "max": 20, "value": 3},
    "huelva": {"min": 0, "max": 18, "value": 3},
    "huesca": {"min": 0, "max": 12, "value": 3},
    "islas_baleares": {"min": 0, "max": 15, "value": 3},
    "jaen": {"min": 0, "max": 10, "value": 3},
    "la_coruna": {"min": 0, "max": 15, "value": 3},
    "la_rioja": {"min": 0, "max": 13, "value": 3},
    "las_palmas": {"min": 0, "max": 18, "value": 3},
    "leon": {"min": 0, "max": 30, "value": 3},
    "lleida": {"min": 0, "max": 20, "value": 3},            
    "lugo": {"min": 0, "max": 15, "value": 3},
    "madrid": {"min": 0, "max": 12, "value": 3},
    "malaga": {"min": 0, "max": 14, "value": 3},
    "murcia": {"min": 0, "max": 15, "value": 3},
    "navarra": {"min": 0, "max": 25, "value": 3},
    "ourense": {"min": 0, "max": 18, "value": 3},
    "palencia": {"min": 0, "max": 20, "value": 3},
    "pontevedra": {"min": 0, "max": 10, "value": 3},
    "salamanca": {"min": 0, "max": 15, "value": 3},
    "santa_cruz_de_tenerife": {"min": 0, "max": 12, "value": 3},
    "segovia": {"min": 0, "max": 19, "value": 3},
    "sevilla": {"min": 0, "max": 20, "value": 3},
    "soria": {"min": 0, "max": 17, "value": 3},
    "tarragona": {"min": 0, "max": 15, "value": 3},
    "teruel": {"min": 0, "max": 25, "value": 3},
    "toledo": {"min": 0, "max": 18, "value": 3},
    "valencia": {"min": 0, "max": 20, "value": 3},
    "valladolid": {"min": 0, "max": 15, "value": 3},
    "vizcaya": {"min": 0, "max": 20, "value": 3},
    "zamora": {"min": 0, "max": 20, "value": 3},
    "zaragoza": {"min": 0, "max": 25, "value": 3}
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
    "pais_vasco": ["Alava", "guipuzcoa", "vizcaya"]
}

# Características
comunidad = st.selectbox("Seleccionar Comunidad Autónoma", ["Seleccionar"] + list(comunidades.keys()))
if comunidad != "Seleccionar":
    provincia = st.selectbox("Seleccionar Provincia", ["Seleccionar"] + comunidades[comunidad])
else:
    provincia = "Seleccionar"

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

# Botón para encontrar el precio más bajo dentro del rango
if st.button("VER PRECIO"):
    if provincia in models:
        model_name = models[provincia]
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            model = joblib.load(model_path)

            # Obtener los rangos específicos de la provincia seleccionada
            if provincia in rango_m2_por_provincia and provincia in rango_habitaciones_por_provincia:
                m2_min, m2_max, m2_value = (
                    rango_m2_por_provincia[provincia]["min"],
                    rango_m2_por_provincia[provincia]["max"],
                    rango_m2_por_provincia[provincia]["value"]
                )
                habitaciones_min, habitaciones_max, habitaciones_value = (
                    rango_habitaciones_por_provincia[provincia]["min"],
                    rango_habitaciones_por_provincia[provincia]["max"],
                    rango_habitaciones_por_provincia[provincia]["value"]
                )
            else:
                # Valores predeterminados globales si no se encuentran rangos específicos
                m2_min, m2_max, m2_value = (27, 6000, 100)
                habitaciones_min, habitaciones_max, habitaciones_value = (2, 8, 3)

            # Normalizar las características según los rangos específicos de la provincia
            normalized_m2 = (m2 - m2_min) / (m2_max - m2_min)
            normalized_habitaciones = (habitaciones - habitaciones_min) / (habitaciones_max - habitaciones_min)

            # Calcular el precio más bajo y alto dentro del rango
            min_price = 34000  # Valor mínimo del precio después de excluir ceros
            max_price = 2900000  # Valor máximo del precio después de excluir ceros
            min_price_scaled = min_price / (max_price - min_price)
            max_price_scaled = max_price / (max_price - min_price)

            # Crear una lista de posibles valores de precio dentro del rango
            price_values = [min_price_scaled + i * 100 for i in range(int((max_price_scaled - min_price_scaled) / 100) + 1)]

            # Crear una lista de resultados de predicción para cada valor de precio
            predictions = []
            for price_scaled in price_values:
                features = pd.DataFrame({
                    "m2": [normalized_m2],  # Usar el valor normalizado
                    "Habitaciones": [normalized_habitaciones]  # Usar el valor normalizado
                })
                prediction_scaled = model.predict(features)
                prediction = prediction_scaled * (max_price - min_price) + min_price
                predictions.append(prediction[0])

            # Encontrar el precio más bajo dentro del rango
            lowest_price = min(predictions)

            # Formatear el precio para mostrar solo las 6 primeras cifras
            formatted_lowest_price = str(int(lowest_price))[:6]

            st.write(f"El precio medio según los datos es de: {formatted_lowest_price} euros")
        else:
            st.error(f"No se encontró un modelo para la provincia {provincia}.")
    else:
        st.error("Selección no válida. Por favor, selecciona las opciones.")



