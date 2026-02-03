import streamlit as st
from src.predict import predict_crop

st.set_page_config(page_title="Crop Recommendation System")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and environmental parameters to get crop recommendations.")

N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    features = [N, P, K, temperature, humidity, ph, rainfall]
    results = predict_crop(features)

    st.subheader("🌱 Top Crop Recommendations")
    for crop, prob in results:
        st.write(f"**{crop}** — {prob:.2%} confidence")

    st.success(f"🌱 Recommended Crop: **{crop}**")
