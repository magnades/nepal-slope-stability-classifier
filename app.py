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


numeric_ranges = {
    'Longitude': (80.0, 85.0),
    'Latitude': (26.0, 31.0),
    'ElevationDEM': (100, 4000),
    'ElevationSite': (100, 4000),
    'AverageCutHeight': (0, 30),
    'AverageCutSlope': (0, 90),
    'FinalOverallSlope': (0, 90),
    'PercentageOfSoil': (0, 100),
    'CutWidth': (0, 50),
    'BelowSlope': (0, 100),
    'AverageTopsoilThickness': (0, 10),
    'BelowHeight': (0, 100),
    'MaxTopsoilThickness': (0, 10),
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

stability_labels = {
    'A': 'Stable (A)',
    'B': 'Slightly Unstable (B)',
    'C': 'Moderately Unstable (C)',
    'D': 'Highly Unstable (D)'
}

stability_colors = {
    'A': '#4CAF50',  # Green
    'B': '#FFEB3B',  # Yellow
    'C': '#FF9800',  # Orange
    'D': '#F44336',  # Red
}

slopes_categories_dict = {
    'A': 'Stable, A',
    'B': 'Slightly Unstable, B',
    'C': 'Moderately Unstable, C',
    'D': 'Highly Unstable, D',
}
st.title("🧭 Nepal Slope Stability Classifier")
st.markdown("This app predicts slope stability using a machine learning model trained on Nepal slope data.")

user_input = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Numeric Features")
    # for feature in numeric_defaults:
    #     if feature not in ['Latitude', 'Longitude']:
    #         user_input[feature] = st.number_input(feature, value=numeric_defaults[feature], key=f"num_{feature}")

    for feature in numeric_defaults:
        min_val, max_val = numeric_ranges[feature]
        default = numeric_defaults[feature]
        user_input[feature] = st.slider(feature, min_value=min_val, max_value=max_val, value=default,
                                        key=f"slider_{feature}")

    st.subheader("🌐 Location Input")
    latitude = st.number_input("Latitude", value=28.5341666667, format="%.6f", key="latitude")
    longitude = st.number_input("Longitude", value=82.3597222222, format="%.6f", key="longitude")
    user_input['Latitude'] = latitude
    user_input['Longitude'] = longitude

    st.subheader("🗺️ Slope Location Map")
    map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

    st.pydeck_chart(pdk.Deck(
        map_style= None,#'mapbox://styles/mapbox/light-v9',
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
    st.subheader("🧩 Categorical Features")

    # Grupo 1: Binarios Yes/No
    st.markdown("**🌍 Slope Conditions**")
    for feature in ['CracksOnSlope', 'SeepageOfSlopeSurface', 'RecentFailureDebrisOnSlope',
                    'Erosion', 'CracksAtSlopeSides', 'BlockedRoadsideDrain', 'BlockedSlopeDrainage']:
        user_input[feature] = st.selectbox(feature, categorical_options[feature], key=f"cat_{feature}")

    # Grupo 2: Regiones y clima
    st.markdown("**🌧️ Geographic/Climatic**")
    for feature in ['PhysiographicRegion', 'RainfallCategory', 'Aspect']:
        user_input[feature] = st.selectbox(feature, categorical_options[feature], key=f"cat_{feature}")

    # Grupo 3: Geología
    st.markdown("**🧱 Geology**")
    for feature in ['Dominant', 'RockStrength', 'LithologyRockTypeCat2', 'WeatheringGradeCat2']:
        user_input[feature] = st.selectbox(feature, categorical_options[feature], key=f"cat_{feature}")

    # Grupo 4: Infraestructura y drenaje
    st.markdown("**🚧 Drainage & Structures**")
    for feature in ['DrainageCat2', 'StructureCat2', 'DrainageCondition']:
        user_input[feature] = st.selectbox(feature, categorical_options[feature], key=f"cat_{feature}")

    # Grupo 5: Vegetación
    st.markdown("**🌿 Vegetation**")
    user_input['CutSlopeBioCat1'] = st.selectbox('CutSlopeBioCat1', categorical_options['CutSlopeBioCat1'], key="cat_CutSlopeBioCat1")

# if st.button("Predict Stability"):
#     input_ordered = [user_input[feat] for feat in features_ordered]
#     df = pd.DataFrame([input_ordered], columns=features_ordered)
#     transformed = preprocessor.transform(df)
#     prediction = model.predict(transformed)
#     predicted_class = stability_encoder.inverse_transform([np.argmax(prediction)])
#
#     st.success(f"Predicted Class: {slopes_categories_dict[predicted_class[0]]}")
#     st.write(f"Raw probabilities: {prediction.tolist()}")
#     st.caption(f"Prediction time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Prediction logic ---
def make_prediction(input_data):
    ordered_values = [input_data[feat] for feat in features_ordered]
    df = pd.DataFrame([ordered_values], columns=features_ordered)
    transformed = preprocessor.transform(df)
    probabilities = model.predict(transformed)[0]
    pred_index = np.argmax(probabilities)
    pred_class = stability_encoder.inverse_transform([pred_index])[0]
    return pred_class, probabilities

# Trigger prediction
predict_now = True  # Always predict automatically
if st.button("🔍 Predict Stability"):
    predict_now = True

if predict_now:
    pred_class, probabilities = make_prediction(user_input)
    label = stability_labels[pred_class]
    color = stability_colors.get(pred_class, "#D3D3D3")

    st.markdown(
        f"<div style='background-color: {color}; padding: 20px; border-radius: 10px;'>"
        f"<h3 style='color: white;'>Predicted Stability: {label}</h3>"
        f"<p><strong>Confidence:</strong> {np.max(probabilities):.2f}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Bar chart
    prob_df = pd.DataFrame({
        'Stability Level': [stability_labels[c] for c in stability_encoder.classes_],
        'Probability': probabilities
    })
    st.subheader("📈 Stability Class Probabilities")
    st.bar_chart(prob_df.set_index("Stability Level"))

    st.caption(f"Prediction generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
