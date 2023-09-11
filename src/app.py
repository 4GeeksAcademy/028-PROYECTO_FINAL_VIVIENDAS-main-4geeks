import streamlit as st
import pandas as pd
import joblib

# Cargar los modelos de las 50 provincias (asegúrate de tener los archivos .sav en el directorio correcto)
modelos_provincias = {   
    "Álava": joblib.load("/workspaces/028-PROYECTO_FINAL_VIVIENDAS-main/src/PROVINCIAS/modelos_finales/alava_randomforest_gridsearch_gradientboosting_default_42.sav"),
    "Vizcaya": joblib.load("/workspaces/028-PROYECTO_FINAL_VIVIENDAS-main/src/PROVINCIAS/modelos_finales/vizcaya_randomforest_gridsearch_gradientboosting_default_42.sav"),
    "Guipuzcoa": joblib.load("/workspaces/028-PROYECTO_FINAL_VIVIENDAS-main/src/PROVINCIAS/modelos_finales/guipuzcoa_randomforest_gridsearch_gradientboosting_default_42.sav"),
    # Agrega el resto de provincias aquí
}

# Lista de comunidades autónomas y provincias
comunidades_autonomas = ["País Vasco"]
provincias = list(modelos_provincias.keys())

# Crear la aplicación con Streamlit
st.title("Predicción de Precio de Vivienda")

# Agregar botón "Calcular Provincias" en la barra lateral que lleve a la app
if st.sidebar.button("Ver Provincias"):
    st.title("Lista de Provincias")
    st.write(provincias)

# Mostrar todas las columnas en el cuerpo principal
st.header("Características de la Vivienda")
comunidad_autonoma_seleccionada = st.selectbox("Seleccione la Comunidad Autónoma", comunidades_autonomas)
provincia_seleccionada = st.selectbox("Seleccione la Provincia", provincias)
tipo_inmueble = st.selectbox("Tipo de Inmueble", ["piso", "chalet", "ático", "dúplex", "estudio", "casa rural"])
m2 = st.slider("Metros Cuadrados", 20, 1000, 80)
habitaciones = st.slider("Número de Habitaciones", 0, 15, 3)

# Agregar botón "Calcular mi Precio" dentro de las columnas
if st.button("Calcular mi Precio"):
    # Crear un DataFrame con las características de entrada para la predicción
    data = pd.DataFrame({
        'comunidad_autonoma': [comunidad_autonoma_seleccionada],
        'tipo_inmueble': [tipo_inmueble],
        'm2': [m2],
        'habitaciones': [habitaciones]
    })

    # Realizar one-hot encoding para la comunidad autónoma y la provincia
    data_encoded = pd.get_dummies(data, columns=['comunidad_autonoma'])
    data_encoded = pd.get_dummies(data_encoded, columns=['tipo_inmueble'])

    # Asegurarse de que todas las columnas necesarias estén presentes
    missing_cols = set(['comunidad_autonoma_' + comunidad for comunidad in comunidades_autonomas]) - set(data_encoded.columns)
    for col in missing_cols:
        data_encoded[col] = 0

    # Realizar la predicción con el modelo de la provincia seleccionada
    modelo_provincia = modelos_provincias[provincia_seleccionada]
    caracteristicas = data_encoded.values[0].reshape(1, -1)
    precio_predicho = modelo_provincia.predict(caracteristicas)

    # Mostrar el precio predicho
    st.success(f"Precio predicho para {tipo_inmueble} en {provincia_seleccionada}: {precio_predicho[0]} €")
