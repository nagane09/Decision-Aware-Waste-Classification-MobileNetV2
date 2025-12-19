import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(
    page_title="Smart Waste Decision System",
    layout="centered"
)

st.title("♻️ Smart Waste Decision System")
st.subheader("Decision-Aware AI for Real-World Recycling")

st.markdown("---")

@st.cache_resource
def load_trained_model():
    return load_model("waste_classifier_model.h5")

model = load_trained_model()

class_labels = ['cardboard', 'glass', 'metal', 'plastic']
confidence_threshold = 0.70

st.markdown("### Upload Waste Image")
uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

st.markdown("### Select Bin Type")
bin_type = st.selectbox(
    "Which bin is this waste thrown into?",
    ["cardboard", "glass", "metal", "plastic"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class = class_labels[np.argmax(prediction)]

    st.markdown("---")
    st.markdown("## 🔍 Model Output")

    st.write(f"**Predicted Waste Type:** `{predicted_class.upper()}`")
    st.write(f"**Confidence:** `{confidence*100:.2f}%`")

    if confidence < confidence_threshold:
        st.error("⚠️ Low Confidence Prediction")
        st.warning("Manual verification required before recycling.")
    else:
        st.success("✅ High Confidence Prediction")

        st.markdown("## 🧠 Recyclability Decision")

        if predicted_class == "plastic":
            decision = "Recyclable (if clean)"
        elif predicted_class == "glass":
            decision = "Recyclable"
        elif predicted_class == "metal":
            decision = "Recyclable"
        elif predicted_class == "cardboard":
            decision = "Needs Cleaning (check for grease)"
        else:
            decision = "Unknown"

        st.info(f"**Recommended Action:** {decision}")

        st.markdown("## 🗑️ Bin Validation")

        if predicted_class == bin_type:
            st.success("✅ Correct Bin Used")
        else:
            st.error("❌ Wrong Bin Detected")
            st.warning(f"Suggested Bin: **{predicted_class.upper()} BIN**")

st.markdown("---")
st.caption(
    "This system integrates deep learning with rule-based decision reasoning "
    "for smart recycling and waste management applications."
)
