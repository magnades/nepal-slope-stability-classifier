import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import io
from datetime import datetime

# ------------------------------
# CONFIGURA TUS URLS DE GITHUB
# ------------------------------
MODEL_URL = "https://raw.githubusercontent.com/USUARIO/REPO/main/modelo.h5"
PREPROCESSOR_URL = "https://raw.githubusercontent.com/USUARIO/REPO/main/preprocessor.pkl"
ENCODER_URL = "https://raw.githubusercontent.com/USUARIO/REPO/main/stability_encoder.pkl"

# ------------------------------
# Cargar modelo y preprocesadores
# ------------------------------
@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    with open("modelo.h5", "wb") as f:
        f.write(response.content)
    return tf.keras.models.load_model("modelo.h5")

@st.cache_resource
def load_pickle_from_url(url):
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

model = load_model()
preprocessor = load_pickle_from_url(PREPROCESSOR_URL)
encoder = load_pickle_from_url(ENCODER_URL)

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
    'LithologyRockTypeCat2': ['CrystallineBanded', 'CrystallineCarbonate', 'CrystallineHard', 'Cemented', 'FineGrained', 'Conglomerate', 'AlluvialDeposit'],
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
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])

    st.success(f"Predicted Class: {predicted_class[0]}")
    st.write(f"Raw probabilities: {prediction.tolist()}")
    st.caption(f"Prediction time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
