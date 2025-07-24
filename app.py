
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

# ------------------------------
# Define features y opciones
# ------------------------------
features_ordered = [
    'PhysiographicRegion', 'RainfallCategory', 'Longitude', 'Latitude', 'ElevationDEM', 'ElevationSite',
    'AverageCutHeight', 'AverageCutSlope', 'FinalOverallSlope', 'Dominant',
    'CracksOnSlope', 'SeepageOfSlopeSurface', 'RecentFailureDebrisOnSlope', 'Erosion', 'CracksAtSlopeSides',
    'BlockedRoadsideDrain', 'BlockedSlopeDrainage', 'DrainageCat2',
    'PercentageOfSoil', 'CutSlopeBioCat1', 'Aspect', 'CutWidth',
    'StructureCat2', 'BelowSlope', 'AverageTopsoilThickness',
    'RockStrength', 'LithologyRockTypeCat2',
    'WeatheringGradeCat2', 'BelowHeight', 'DrainageCondition',
    'MaxTopsoilThickness'
]

numeric_defaults = {
    'Longitude': 82.3597222222, 'Latitude': 28.5341666667, 'ElevationDEM': 1868., 'ElevationSite': 1868.,
    'AverageCutHeight': 10., 'AverageCutSlope': 60, 'FinalOverallSlope': 0., 'PercentageOfSoil': 50.,
    'CutWidth': 25., 'BelowSlope': 70., 'AverageTopsoilThickness': 5, 'BelowHeight': 50, 'MaxTopsoilThickness': 5,
}

categorical_options = {
    'PhysiographicRegion': ['MidHill', 'Chure', 'UpperHill'],
    'RainfallCategory': ['L', 'H', 'M'],
    'Dominant': ['Both', 'Soil', 'Rock'],
    'CracksOnSlope': ['No', 'Yes'],
    'SeepageOfSlopeSurface': ['No', 'Yes'],
    'RecentFailureDebrisOnSlope': ['Yes', 'No'],
    'Erosion': ['Yes', 'No'],
    'CracksAtSlopeSides': ['No', 'Yes'],
    'BlockedRoadsideDrain': ['No', 'Yes'],
    'BlockedSlopeDrainage': ['No', 'Yes'],
    'DrainageCat2': ['SurfaceDrain', 'NoEffectiveness', 'Combination', 'SlopeDrain'],
    'CutSlopeBioCat1': ['BareOrAltered', 'Herbaceous', 'Forest', 'Shrubby', 'Mixed'],
    'Aspect': ['W', 'N', 'S', 'E', 'NE', 'NW', 'SW', 'SE'],
    'StructureCat2': ['NoEffectiveness', 'RigidStructure', 'FlexibleStructure'],
    'RockStrength': ['Hard', 'VeryWeak', 'Moderate', 'Weak', 'ExtremelyHard', 'VeryHard'],
    'LithologyRockTypeCat2': ['CrystallineBanded', 'CrystallineCarbonate', 'CrystallineHard', 'Cemented', 'FineGrained',
                              'Conglomerate', 'AlluvialDeposit'],
    'WeatheringGradeCat2': ['II', 'I', 'III', 'IV', 'V'],
    'DrainageCondition': ['Functional', 'NeedsRepairOrCleaning', 'UnderConstruction'],
}

st.title("Nepal Slope Stability Classifier")
st.markdown("This app predicts slope stability using a machine learning model trained on Nepal slope data.")

user_input = {}

# Entradas numéricas
st.subheader("Numeric Features")
for feature in numeric_defaults:
    user_input[feature] = st.number_input(feature, value=numeric_defaults[feature])

# Entradas categóricas
st.subheader("Categorical Features")
for feature, options in categorical_options.items():
    user_input[feature] = st.selectbox(feature, options)

# Ejecutar predicción
if st.button("Predict Stability"):
    input_ordered = [user_input[feat] for feat in features_ordered]
    df = pd.DataFrame([input_ordered], columns=features_ordered)
    transformed = preprocessor.transform(df)
    prediction = model.predict(transformed)
    predicted_class = stability_encoder.inverse_transform([np.argmax(prediction)])

    st.success(f"Predicted Class: {predicted_class[0]}")
    st.write(f"Raw probabilities: {prediction.tolist()}")
    st.caption(f"Prediction time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
