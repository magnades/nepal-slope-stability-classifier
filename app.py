import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
from urllib.error import URLError
import requests
import io

# URLs para los archivos en GitHub
MODEL_URL = "https://raw.githubusercontent.com/magnades/DurhamSlope/main/model_81_30_31_best.keras"
PREPROCESSOR_URL = "https://raw.githubusercontent.com/magnades/DurhamSlope/main/preprocessor_81_30_31_best.pkl"
ENCODER_URL = "https://raw.githubusercontent.com/magnades/DurhamSlope/main/stability_encoder.pkl"

# Rutas locales para usar cuando no haya internet
MODEL_PATH = "model_81_30_31_best.keras"
PREPROCESSOR_PATH = "preprocessor_81_30_31_best.pkl"
ENCODER_PATH = "stability_encoder.pkl"

@st.cache_data(show_spinner=True)
def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return tf.keras.models.load_model(io.BytesIO(response.content))

@st.cache_data(show_spinner=True)
def load_pickle_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

def load_resources():
    try:
        st.info("Loading resources from GitHub...")
        model = load_model_from_url(MODEL_URL)
        preprocessor = load_pickle_from_url(PREPROCESSOR_URL)
        encoder = load_pickle_from_url(ENCODER_URL)
        st.success("Resources loaded from GitHub!")
    except Exception as e:
        st.warning(f"Failed to load from GitHub, loading local files: {e}")
        model = tf.keras.models.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        encoder = joblib.load(ENCODER_PATH)
        st.success("Resources loaded from local files.")
    return model, preprocessor, encoder

# Carga los modelos y objetos
model, preprocessor, stability_encoder = load_resources()

# Aquí tus variables, widgets y lógica:
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

numeric_features_base_A = [
    'Longitude',
    'Latitude',
    'ElevationDEM',
    'ElevationSite',
    'AverageCutHeight',
    'AverageCutSlope',
    'FinalOverallSlope',
    'PercentageOfSoil',
    'CutWidth',
    'BelowSlope',
    'AverageTopsoilThickness',
    'BelowHeight',
    'MaxTopsoilThickness'
]

numeric_defaults = {
    'Longitude': 82.3597222222,
    'Latitude': 28.5341666667,
    'ElevationDEM': 1868.,
    'ElevationSite': 1868.,
    'AverageCutHeight': 10.,
    'AverageCutSlope': 60,
    'FinalOverallSlope': 0.,
    'PercentageOfSoil': 50.,
    'CutWidth': 25.,
    'BelowSlope': 70.,
    'AverageTopsoilThickness': 5,
    'BelowHeight': 50,
    'MaxTopsoilThickness': 5,
}

options_dict = {
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

st.title("Slope Stability Prediction - Nepal Case Study")

st.header("Input features")

numeric_inputs = {}
for feature in numeric_features_base_A:
    numeric_inputs[feature] = st.number_input(f"{feature}", value=numeric_defaults.get(feature, 0.0))

categorical_inputs = {}
for feature in options_dict.keys():
    categorical_inputs[feature] = st.selectbox(f"{feature}", options=options_dict[feature])

def predict(model, preprocessor, encoder, numeric_inputs, categorical_inputs):
    # Organizar datos en orden
    inputs = {}
    inputs.update(numeric_inputs)
    inputs.update(categorical_inputs)

    ordered_values = [inputs[feat] for feat in features_ordered]
    input_df = pd.DataFrame([ordered_values], columns=features_ordered)

    preprocessed = preprocessor.transform(input_df)
    prediction = model.predict(preprocessed)
    pred_class_idx = np.argmax(prediction, axis=1)
    pred_class = encoder.inverse_transform(pred_class_idx)
    return prediction, pred_class[0]

if st.button("Predict"):
    prediction, pred_class = predict(model, preprocessor, stability_encoder, numeric_inputs, categorical_inputs)
    st.write(f"Prediction probabilities: {prediction}")
    st.success(f"Predicted class: {pred_class} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

