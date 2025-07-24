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
import pydeck as pdk

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

model, preprocessor, stability_encoder = load_resources()

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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric Features")
    for feature in numeric_defaults:
        if feature not in ['Latitude', 'Longitude']:
            user_input[feature] = st.number_input(feature, value=numeric_defaults[feature], key=f"num_{feature}")

    st.subheader("🌐 Location Input")
    latitude = st.number_input("Latitude", value=28.5341666667, format="%.6f", key="latitude")
    longitude = st.number_input("Longitude", value=82.3597222222, format="%.6f", key="longitude")
    user_input['Latitude'] = latitude
    user_input['Longitude'] = longitude

    st.subheader("🗺️ Slope Location Map")
    map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=10,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=map_data,
                get_position='[lon, lat]',
                get_color='[255, 0, 0, 160]',
                get_radius=500,
            ),
        ],
    ))

with col2:
    st.subheader("Categorical Features")

    # Grupo 1: Binarios Yes/No
    st.markdown("**Slope Conditions**")
    for feature in ['CracksOnSlope', 'SeepageOfSlopeSurface', 'RecentFailureDebrisOnSlope',
                    'Erosion', 'CracksAtSlopeSides', 'BlockedRoadsideDrain', 'BlockedSlopeDrainage']:
        user_input[feature] = st.selectbox(feature, categorical_options[feature], key=f"cat_{feature}")

    # Grupo 2: Regiones y clima
    st.markdown("**Geographic/Climatic**")
    for feature in ['PhysiographicRegion', 'RainfallCategory', 'Aspect']:
        user_input[feature] = st.selectbox(feature, categorical_options[feature], key=f"cat_{feature}")

    # Grupo 3: Geología
    st.markdown("**Geology**")
    for feature in ['Dominant', 'RockStrength', 'LithologyRockTypeCat2', 'WeatheringGradeCat2']:
        user_input[feature] = st.selectbox(feature, categorical_options[feature], key=f"cat_{feature}")

    # Grupo 4: Infraestructura y drenaje
    st.markdown("**Drainage & Structures**")
    for feature in ['DrainageCat2', 'StructureCat2', 'DrainageCondition']:
        user_input[feature] = st.selectbox(feature, categorical_options[feature], key=f"cat_{feature}")

    # Grupo 5: Vegetación
    st.markdown("**Vegetation**")
    user_input['CutSlopeBioCat1'] = st.selectbox('CutSlopeBioCat1', categorical_options['CutSlopeBioCat1'], key="cat_CutSlopeBioCat1")

if st.button("Predict Stability"):
    input_ordered = [user_input[feat] for feat in features_ordered]
    df = pd.DataFrame([input_ordered], columns=features_ordered)
    transformed = preprocessor.transform(df)
    prediction = model.predict(transformed)
    predicted_class = stability_encoder.inverse_transform([np.argmax(prediction)])

    st.success(f"Predicted Class: {predicted_class[0]}")
    st.write(f"Raw probabilities: {prediction.tolist()}")
    st.caption(f"Prediction time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
