import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set Streamlit page config for a modern look
st.set_page_config(
    page_title="Fractured Bone X-ray Detection AI",
    page_icon="🦴",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for colors and styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #e0e7ff 0%, #f9fafb 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #e0e7ff 0%, #f9fafb 100%);
        }
        .stButton>button {
            color: #fff;
            background-color: #6366f1;
            border-radius: 8px;
            padding: 0.5em 1.5em;
            border: none;
        }
        .stButton>button:hover {
            color: #fff;
            background-color: #4338ca;
        }
        .stFileUploader label {
            color: #4338ca;
            font-weight: bold;
        }
        .header {
            font-size: 2.2em;
            font-weight: 700;
            color: #6366f1;
        }
        .subheader {
            font-size: 1.3em;
            color: #10b981;
        }
        .fracture {
            color: #ef4444;
            font-weight: bold;
        }
        .normal {
            color: #10b981;
            font-weight: bold;
        }
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
        }
        .prediction-card {
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 2px 12px 0 rgba(99,102,241,0.14);
            padding: 2rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource()
def load_model():
    try:
        model = tf.keras.models.load_model('fracture_xray_model.h5')  # Path to your trained model
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Nice title and intro
st.markdown('<div class="header">🦴 Fractured Bone Detection from X-rays</div>', unsafe_allow_html=True)

st.markdown("""
<div style="background: #6366f1; color: #fff; border-radius: 12px; padding: 1.3em 1.5em; margin: 1em 0 2em 0;">
    <b>What is a Bone Fracture?</b><br>
    A fracture is a break in bone continuity due to stress or trauma. Early detection is critical for proper healing and recovery.
    <hr style="border: 0; border-top: 1px solid #fff;">
    <span style="font-size:1.1em;"><b>AI for Detection:</b> This tool uses a trained deep learning model to detect fractures in X-rays.</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="subheader">Upload a Bone X-ray Image for Prediction</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📷 Uploaded X-ray', use_column_width=True)

        with st.spinner("Analyzing X-ray..."):
            # Resize and preprocess the image for your model's expected input
            img = image.resize((64, 64))
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize if your model was trained this way
            img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)

            if model is not None:
                prediction = model.predict(img_array)
                confidence = float(prediction[0][0])
                label = 'Normal' if confidence > 0.5 else 'Fractured'
                # For display
                conf_label = confidence if label == 'Normal' else 1-confidence
                color_class = "normal" if label == "Normal" else "fracture"

                st.markdown(f"""
                    <div class="prediction-card">
                    <span style="font-size:1.6em;">Prediction: <span class="{color_class}">{label}</span></span><br><br>
                    <b>Confidence:</b><br>
                    <div style="background:#e0e7ff; border-radius:10px; padding:0.2em;">
                        <div class="confidence-bar" style="width:{conf_label*100}%; background: {'#10b981' if label=='Normal' else '#ef4444'}; text-align:center; color:#fff;">
                            {conf_label:.2%}
                        </div>
                    </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Model not loaded properly.")
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Add a custom footer
st.markdown("""
<hr>
<div style="text-align:center; color:#6366f1; font-size:1em; margin-top:1em;">
    Made using Streamlit & TensorFlow &bull; <a style="color:#4338ca;" href="https://github.com/AkulaArchana04" target="_blank">AkulaArchana04</a>
</div>
""", unsafe_allow_html=True)