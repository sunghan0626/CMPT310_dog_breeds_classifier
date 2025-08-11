import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import os
import pandas as pd
import altair as alt

# ====== CONFIG ======
MODEL_PATH = "best_model.pth"
DATA_DIR = "data/Images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

def prettify(name):
    breed_only = name.split('-')[-1]
    return breed_only.replace('_', ' ').title()

BREED_NAMES = [prettify(name) for name in CLASS_NAMES]

# ====== TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====== MODEL LOADING ======
def get_resnet_model(num_classes):
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
    return model

model = get_resnet_model(len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ====== PREDICTION ======
def predict_with_confidence(image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    top5_idx = probs.argsort()[-5:][::-1]
    return [(BREED_NAMES[i], float(probs[i])) for i in top5_idx]

# ====== CUSTOM RESTART BUTTON ======
custom_button = """
<div style='text-align:center; margin-top: 30px;'>
    <form action="">
        <button type="submit" style="
            background-color: #ff9933;
            border: none;
            color: white;
            padding: 12px 24px;
            text-align: center;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        " onmouseover="this.style.backgroundColor='#e6862c'" onmouseout="this.style.backgroundColor='#ff9933'">
              Classify Another Dog
        </button>
    </form>
</div>
"""

# ====== STREAMLIT UI ======
st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#ff9933;'>🐶 Dog Breed Classifier 🐶</h1>",
    unsafe_allow_html=True
)

# Session state for uploaded image
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if st.session_state.uploaded_image is None:
    uploaded = st.file_uploader("Upload a dog image to predict its breed", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.session_state.uploaded_image = uploaded
        st.rerun()

if st.session_state.uploaded_image is not None:
    try:
        img = Image.open(st.session_state.uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Classifying..."):
            results = predict_with_confidence(img)

        breed, confidence = results[0]
        st.success(f"**Predicted Breed:** {breed} ({confidence*100:.2f}%)")

        df = pd.DataFrame(results, columns=["Breed", "Confidence"])
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Breed", sort="-y", axis=alt.Axis(labelAngle=0)),
                y="Confidence"
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown(custom_button, unsafe_allow_html=True)

    except UnidentifiedImageError:
        st.error("Cannot read the image file. Please upload a valid JPG, JPEG, or PNG image.")
        st.markdown(custom_button, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.markdown(custom_button, unsafe_allow_html=True)

# Reset state when button clicked (via query param)
if st.query_params:
    st.session_state.uploaded_image = None
    st.experimental_rerun()

# ====== FOOTER ======
st.markdown("""
<hr>
<div style='text-align: center; font-size: 16px; color: #CCCCCC;'>
    <p>
        Developed by Yein Hwang, Sung Han, and Cody Huang (CMPT 310 Group 34).<br>
        This application is an AI-powered dog breed classifier that analyzes a photo of a dog and identifies its breed based on visual features.<br><br>
        Simply upload an image, and our model will predict the most likely breed along with a confidence score.<br>
        Give it a try and see how accurately it performs!
    </p>
</div>
""", unsafe_allow_html=True)
