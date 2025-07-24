
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import os
import io
import tempfile
from datetime import datetime

# URLs de los recursos en GitHub
MODEL_URL = "https://raw.githubusercontent.com/magnades/DurhamSlope/main/model_81_30_31_best.keras"
PREPROCESSOR_URL = "https://raw.githubusercontent.com/magnades/DurhamSlope/main/preprocessor_81_30_31_best.pkl"
ENCODER_URL = "https://raw.githubusercontent.com/magnades/DurhamSlope/main/stability_encoder.pkl"

@st.cache_resource(show_spinner=True)
def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    model = tf.keras.models.load_model(tmp_path)
    os.remove(tmp_path)
    return model

@st.cache_data(show_spinner=True)
def load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

@st.cache_resource(show_spinner=True)
def load_resources():
    try:
        st.info("Loading resources from GitHub...")
        model = load_model_from_url(MODEL_URL)
        preprocessor = load_pickle_from_url(PREPROCESSOR_URL)
        encoder = load_pickle_from_url(ENCODER_URL)
        st.success("Resources loaded from GitHub!")
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        raise
    return model, preprocessor, encoder

# Carga los modelos y objetos
model, preprocessor, stability_encoder = load_resources()

# Resto de tu lógica de widgets, inputs y predicción aquí
st.write("App loaded. Ready for inputs.")
