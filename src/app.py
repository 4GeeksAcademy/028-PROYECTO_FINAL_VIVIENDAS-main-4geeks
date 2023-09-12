import streamlit as st
import pandas as pd
import os
import joblib

# Cargar el modelo de Almería
model_dir = "/workspaces/028-PROYECTO_FINAL_VIVIENDAS-main/src/PROVINCIAS/modelos_finales"
model_file = "almeria_randomforest_gridsearch_gradientboosting_default_42.sav"
model_path = os.path.join(model_dir, model_file)

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("No se encontró el modelo de Almería. Verifica la ruta del modelo.")

# Comunidades y provincias
comunidades = {
    "andalucia": ["almeria", "cadiz", "cordoba", "granada", "huelva", "jaen", "malaga", "sevilla"],
    # ... Otras comunidades ...
}

# Características
comunidad = st.selectbox("Comunidad Autónoma", list(comunidades.keys()))
provincia = st.selectbox("Provincia", comunidades[comunidad])
tipo_inmueble = st.selectbox("Tipo de Inmueble", ["estudio", "piso", "ático", "chalet", "duplex", "casa_rural"])
m2 = st.slider("Metros cuadrados", 40, 500, 200)
habitaciones = st.slider("Número de habitaciones", 0, 12, 3)

# Suponiendo que min_price y max_price son los valores mínimos y máximos originales del precio
min_price = 10000  # Reemplaza con el valor mínimo real
max_price = 1000000  # Reemplaza con el valor máximo real

# Botón para predecir el precio
if st.button("Ver precio de mi vivienda"):
    st.write(f"m2: {m2}")
    st.write(f"Habitaciones: {habitaciones}")
    
    if provincia == "almeria" and os.path.exists(model_path):
        features = pd.DataFrame({
            "m2": [m2],
            "Habitaciones": [habitaciones]
        })
        prediction_scaled = model.predict(features)
        
        # Aplicar transformación inversa para obtener el precio en la escala original
        prediction = prediction_scaled * (max_price - min_price) + min_price
        
        st.write(f"El precio estimado de tu vivienda en Almería es: {prediction[0]:.2f} euros")
    elif provincia != "almeria":
        st.error("Este modelo solo está disponible para Almería. Por favor, selecciona Almería como provincia.")
