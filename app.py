import streamlit as st

from src.predict import predict_crop
from src.yield_predict import predict_yield

st.set_page_config(page_title="Crop & Yield Recommendation System")

st.title("🌾 Crop & Yield Recommendation System")
st.write(
    "This system recommends suitable crops based on soil and climate "
    "and estimates expected yield for each recommendation."
)

# ----------- USER INPUTS -----------

N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)

temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall (mm)")

# ----------- PREDICTION -----------

if st.button("Get Recommendations"):
    soil_climate_features = {
        "Temperature": temperature,
        "Rainfall": rainfall,
        "Humidity": humidity,
        "Soil_pH": ph,
        "Soil_Nitrogen": N,
        "Soil_Phosphorus": P,
        "Soil_Potassium": K
    }

    # 1️⃣ Crop recommendations (Top 3)
    crop_results = predict_crop(
        [N, P, K, temperature, humidity, ph, rainfall],
        top_k=3
    )

    st.subheader("🌱 Recommended Crops & Expected Yield")

    for crop, confidence in crop_results:
        yield_value = predict_yield(crop, soil_climate_features)

        st.markdown(
            f"""
            **Crop:** {crop}  
            **Confidence:** {confidence:.2%}  
            **Estimated Yield:** {yield_value:.2f}
            """
        )
