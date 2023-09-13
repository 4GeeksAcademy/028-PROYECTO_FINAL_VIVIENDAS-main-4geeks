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
    "a_coruÑa": "la_coruÑa_randomforest_gridsearch_gradientboosting_default_42.sav",
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
    "galicia": ["a_coruÑa", "lugo", "ourense", "pontevedra"],
    "madrid": ["comunidad_de_madrid"],
    "murcia": ["region_de_murcia"],
    "navarra": ["navarra"],
    "la_rioja": ["la_rioja"],
    "comunidad_valenciana": ["alicante", "castellon", "valencia"],
    "pais_vasco": ["alava", "guipuzcoa", "vizcaya"]
}

# Características
comunidad = st.selectbox("Comunidad Autónoma", list(comunidades.keys()))
provincia = st.selectbox("Provincia", comunidades[comunidad])
tipo_inmueble = st.selectbox("Tipo de Inmueble", ["estudio", "piso", "ático", "chalet", "duplex", "casa_rural"])
m2 = st.slider("Metros cuadrados", 40, 500, 200)
habitaciones = st.slider("Número de habitaciones", 0, 12, 3)

# Función para cargar el modelo
def load_model(provincia):
    model_name = f"{provincia}_randomforest_gridsearch_gradientboosting_default_42.sav"
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

# Suponiendo que min_price y max_price son los valores mínimos y máximos originales del precio
min_price = 10000  # Reemplaza con el valor mínimo real
max_price = 1000000  # Reemplaza con el valor máximo real

# Función para normalizar las características
def normalize_features(m2, habitaciones):
    min_m2 = 40  # Reemplaza con el valor mínimo real
    max_m2 = 500  # Reemplaza con el valor máximo real
    min_habitaciones = 0  # Reemplaza con el valor mínimo real
    max_habitaciones = 12  # Reemplaza con el valor máximo real

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
            
            st.write(f"El precio estimado de tu vivienda en {provincia.capitalize()} es: {prediction_int} euros")
        else:
            st.error(f"No se encontró un modelo para la provincia {provincia.capitalize()}.")
    else:
        st.error("Provincia no válida. Por favor, selecciona una provincia válida.")
