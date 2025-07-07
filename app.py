# app.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import shap
import pandas as pd
from model import ChestXrayCNN, TabularNN
from interpret import GradCAM, get_shap_values
from data_loader import load_tabular_data

# Set up page
st.set_page_config(page_title="Explainable AI for Healthcare", page_icon="🧠", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧠 Explainable AI for Healthcare Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
cnn_model = ChestXrayCNN().to(device)
cnn_model.load_state_dict(torch.load("models/cnn.pth", map_location=device))
cnn_model.eval()

tabular_model = TabularNN(input_dim=13).to(device)
tabular_model.load_state_dict(torch.load("models/tabular.pth", map_location=device))
tabular_model.eval()

# ----------------------------- 🫁 X-ray Diagnosis -----------------------------

st.subheader("📸 Chest X-ray Diagnosis")
st.caption("Upload a chest X-ray to detect signs of pneumonia with a visual heatmap.")

xray_file = st.file_uploader("Upload a Chest X-ray (JPG or PNG)", type=["jpg", "jpeg", "png"])

if xray_file:
    image = Image.open(xray_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=300)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing X-ray..."):
        with torch.no_grad():
            output = cnn_model(input_tensor)
            pred_class = output.argmax(1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_class].item()

        st.success(f"🧾 Prediction: **{'PNEUMONIA' if pred_class else 'NORMAL'}** with **{confidence:.2%}** confidence.")

        # Grad-CAM
        gradcam = GradCAM(model=cnn_model, target_layer=cnn_model.base_model.layer4)
        cam = gradcam.generate(input_tensor)

        # Generate heatmap
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Prepare original image in the same format
        original = np.array(image.resize((224, 224)).convert("RGB"))
        if original.shape != heatmap.shape:
            heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        if original.dtype != np.uint8:
            original = original.astype(np.uint8)
        if heatmap.dtype != np.uint8:
            heatmap = heatmap.astype(np.uint8)

        # Blend and display
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=False)

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------- ❤️ Heart Risk Prediction -----------------------------

st.subheader("📊 Heart Disease Prediction")
st.caption("Fill in patient data to get a prediction with SHAP-based explanation.")

features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

input_cols = st.columns(4)
input_values = []

for i, feat in enumerate(features):
    val = input_cols[i % 4].number_input(f"{feat}", value=0.0, format="%.2f", step=1.0)
    input_values.append(val)

if st.button("🩺 Predict Heart Risk"):
    if all(v == 0.0 for v in input_values):
        st.warning("Please enter valid patient data.")
    else:
        with st.spinner("Evaluating risk..."):
            input_tensor = torch.tensor([input_values], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = tabular_model(input_tensor)
                pred = output.argmax(1).item()
                prob = torch.softmax(output, dim=1)[0, pred].item()

            st.success(f"🧾 Prediction: **{'Heart Disease' if pred else 'Healthy'}** with **{prob:.2%}** confidence.")

            # SHAP
            _, val_loader, _, _, _ = load_tabular_data("data/tabular/heart.csv")
            background = next(iter(val_loader))[0][:100]
            _, shap_vals = get_shap_values(tabular_model, background, input_tensor.cpu())

            st.markdown("#### 🔍 Feature Contributions (SHAP Values)")
            shap_index = pred if len(shap_vals) > 1 else 0
            shap_expl = zip(features, shap_vals[shap_index][0])
            shap_sorted = sorted(shap_expl, key=lambda x: abs(x[1]), reverse=True)

            for feat, val in shap_sorted:
                color = "red" if val > 0 else "blue"
                st.markdown(f"<span style='color:{color}'>**{feat}**</span>: {val:.4f}", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with ❤️ using PyTorch, SHAP, Grad-CAM, and Streamlit.")
st.markdown("© 2023 Your Name. All rights reserved.")