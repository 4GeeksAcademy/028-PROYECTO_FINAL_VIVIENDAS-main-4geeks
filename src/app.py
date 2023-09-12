import streamlit as st
import pandas as pd
import os
import joblib
import random

# Cargar los modelos
model_dir = "/workspaces/028-PROYECTO_FINAL_VIVIENDAS-main/src/PROVINCIAS/modelos_finales"
model_files = os.listdir(model_dir)
models = {}

for model_file in model_files:
    if model_file.endswith(".sav"):
        province = model_file.split("_")[0].capitalize()
        model = joblib.load(os.path.join(model_dir, model_file))
        models[province] = model

# Comunidades y provincias
comunidades = {
    "Andalucía": ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Sevilla"],
    "Aragón": ["Huesca", "Teruel", "Zaragoza"],
    "Asturias": ["Asturias"],
    "Islas Baleares": ["Islas Baleares"],
    "Canarias": ["Las Palmas", "Santa Cruz de Tenerife"],
    "Cantabria": ["Cantabria"],
    "Castilla y León": ["Ávila", "Burgos", "León", "Palencia", "Salamanca", "Segovia", "Soria", "Valladolid", "Zamora"],
    "Castilla-La Mancha": ["Albacete", "Ciudad Real", "Cuenca", "Guadalajara", "Toledo"],
    "Cataluña": ["Barcelona", "Girona", "Lleida", "Tarragona"],
    "Extremadura": ["Badajoz", "Cáceres"],
    "Galicia": ["A Coruña", "Lugo", "Ourense", "Pontevedra"],
    "Madrid": ["Comunidad de Madrid"],
    "Murcia": ["Región de Murcia"],
    "Navarra": ["Navarra"],
    "La Rioja": ["La Rioja"],
    "Comunidad Valenciana": ["Alicante", "Castellón", "Valencia"],
    "País Vasco": ["Álava", "Guipúzcoa", "Vizcaya"]
}

# Características
comunidad = st.selectbox("Comunidad Autónoma", list(comunidades.keys()))
provincia = st.selectbox("Provincia", comunidades[comunidad])
tipo_inmueble = st.selectbox("Tipo de Inmueble", ["estudio", "piso", "ático", "chalet", "duplex", "casa rural"])
m2 = st.slider("Metros cuadrados", 40, 500, 200)
habitaciones = st.slider("Número de habitaciones", 0, 12, 3)

# Botón para predecir el precio
if st.button("Ver precio de mi vivienda"):
    if comunidad and provincia and tipo_inmueble and m2 and habitaciones:
        model_name = f"{provincia.lower()}_randomforest_gridsearch_gradientboosting_default_42"
        if model_name in models:
            model = models[model_name]
            features = pd.DataFrame({
                "Tipo de inmueble": [tipo_inmueble],
                "m2": [m2],
                "Habitaciones": [habitaciones]
            })
            prediction = model.predict(features)
            st.write(f"El precio estimado de tu vivienda en {provincia} es: {prediction[0]:.2f} euros")
        else:
            st.error("No se encontró un modelo para la provincia seleccionada.")
