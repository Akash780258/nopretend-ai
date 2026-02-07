import streamlit as st
import pickle
import pandas as pd

MODEL_PATH = "model/productivity_model.pkl"

# -------- LOAD MODEL -------- #
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["label_encoder"]


model, label_encoder = load_model()

# -------- UI -------- #
st.set_page_config(page_title="NoPretend AI", layout="centered")

st.title("🧠 NoPretend AI – Productivity Detector")
st.write(
    "This dashboard predicts **Productive**, **Fake_Productive**, or **Idle** "
    "based on user activity patterns."
)

st.divider()

st.subheader("🔧 Input Activity Features")

mouse_distance = st.slider("Mouse Distance", 0.0, 10000.0, 500.0)
mouse_clicks = st.slider("Mouse Clicks", 0, 20, 1)
key_presses = st.slider("Key Presses", 0, 50, 5)
idle_time = st.slider("Idle Time (seconds)", 0, 200, 0)
activity_intensity = st.slider("Activity Intensity", 0.0, 200.0, 10.0)
typing_ratio = st.slider("Typing Ratio", 0.0, 1.0, 0.3)
switch_rate = st.slider("Window Switches", 0, 5, 0)

# -------- PREDICTION -------- #
if st.button("🔮 Predict Productivity"):
    input_df = pd.DataFrame([[
        mouse_distance,
        mouse_clicks,
        key_presses,
        idle_time,
        activity_intensity,
        typing_ratio,
        switch_rate
    ]], columns=[
        "mouse_distance",
        "mouse_clicks",
        "key_presses",
        "idle_time",
        "activity_intensity",
        "typing_ratio",
        "switch_rate"
    ])

    prediction = model.predict(input_df)
    label = label_encoder.inverse_transform(prediction)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    if label == "Productive":
        st.success("✅ Productive")
    elif label == "Fake_Productive":
        st.warning("⚠️ Fake Productive")
    else:
        st.error("🛑 Idle")
