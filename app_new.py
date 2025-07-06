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
import google.generativeai as genai
import os

# ----------- GOOGLE GEMINI SETUP ------------
genai.configure(api_key="AIzaSyBTJhGt4CTXXwbO0Rv7pIAysOfX2Rf7PvE")

def explain_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ----------- STREAMLIT UI STARTS ------------
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
explanation_requested_xray = False

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

        pred_label = "PNEUMONIA" if pred_class else "NORMAL"
        st.success(f"🧾 Prediction: **{pred_label}** with **{confidence:.2%}** confidence.")

        # Grad-CAM
        gradcam = GradCAM(model=cnn_model, target_layer=cnn_model.base_model.layer4)
        cam = gradcam.generate(input_tensor)
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        original = np.array(image.resize((224, 224)).convert("RGB"))
        if original.shape != heatmap.shape:
            heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        if original.dtype != np.uint8:
            original = original.astype(np.uint8)
        if heatmap.dtype != np.uint8:
            heatmap = heatmap.astype(np.uint8)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=False)

        explanation_requested_xray = st.button("💬 Explain this X-ray diagnosis")
        if explanation_requested_xray:
            xray_prompt = f"""You are an expert AI assistant integrated into a medical diagnostic application that analyzes chest X-ray images.

The user has uploaded a chest X-ray, and a deep learning model has predicted the presence or absence of pneumonia.

The system also generates a Grad-CAM heatmap that visually highlights the regions of the lungs that influenced the model’s prediction.

Here is the prediction output from the system:
- Predicted Class: {pred_label}
- Confidence: {confidence}%
- Grad-CAM: Highlights the most influential lung areas for this prediction

Please explain the result to a clinician:
1. What does the prediction mean clinically?
2. How confident is the model, and what does that confidence imply?
3. What is a Grad-CAM heatmap and how can it help the doctor understand or verify the prediction?
4. What might the highlighted regions suggest pathologically?
5. What should a doctor consider when interpreting this prediction + heatmap together?

Avoid any machine learning jargon. Keep the explanation medically relevant and understandable for clinicians. Your tone should be professional, informative, and supportive. Do not exceed 300 words. End the explanation with a short clinical suggestion (e.g., “Recommend correlation with symptoms and clinical history”)."""
            with st.spinner("Generating explanation using Gemini..."):
                explanation = explain_with_gemini(xray_prompt)
                st.markdown("#### 🧠 Gemini Explanation")
                st.info(explanation)

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
            # Load scaler used during training
            _, _, _, feature_order, scaler = load_tabular_data("data/tabular/heart.csv")
            
            # Reorder user input to match training feature order (just in case)
            input_dict = dict(zip(features, input_values))
            input_ordered = [input_dict[feat] for feat in feature_order]
            
            # Apply scaling to input
            input_scaled = scaler.transform([input_ordered])
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
            
            # Model prediction
            with torch.no_grad():
                output = tabular_model(input_tensor)
                pred = output.argmax(1).item()
                prob = torch.softmax(output, dim=1)[0, pred].item()

        pred_label = "Heart Disease" if pred == 1 else "Healthy"

        # SHAP explanation
        _, val_loader, _, _, _ = load_tabular_data("data/tabular/heart.csv")
        background = next(iter(val_loader))[0][:100]  # use first 100 validation samples
        shap_vals = get_shap_values(tabular_model, background, input_tensor.cpu(), pred)

        # Save result for explanation button
        st.session_state.pred_result = {
            "label": pred_label,
            "prob": prob,
            "shap_vals": shap_vals.tolist(),
            "input_values": input_ordered
        }

# Show results if available
if "pred_result" in st.session_state:
    pred_label = st.session_state.pred_result["label"]
    prob = st.session_state.pred_result["prob"]
    shap_vals = st.session_state.pred_result["shap_vals"]

    st.success(f"🧾 Prediction: **{pred_label}** with **{prob:.2%}** confidence.")

    st.markdown("#### 🔍 Feature Contributions (SHAP Values)")
    shap_expl = zip(features, shap_vals)
    for feat, val in shap_expl:
        if isinstance(val, list):
            val = val[0]  # Flatten nested list
        color = "red" if val > 0 else "blue"
        st.markdown(f"<span style='color:{color}'>**{feat}**</span>: {val:.4f}", unsafe_allow_html=True)

    if st.button("💬 Explain this heart risk prediction"):
        feat_string = ", ".join([f"{k}: {v[0]:+.3f}" if isinstance(v, list) else f"{k}: {v:+.3f}" for k, v in zip(features, shap_vals)])
        heart_prompt = f"""
You are a clinical AI explanation assistant helping explain heart disease risk predictions to medical professionals.

In this case, the user submitted structured patient data including age, sex, cholesterol, oldpeak, and other vitals.

A neural network model predicted whether the patient is at risk of heart disease.

In addition, SHAP values have been calculated to show how each feature contributed to the prediction.

Here is the prediction result:
- Prediction: {pred_label}
- Confidence: {prob}%
- SHAP Feature Contributions (All factors): {feat_string}

Please explain to a clinician:
1. What does this prediction imply for the patient?
2. What does the confidence score represent in a clinical sense?
3. How did the all features influence the model's decision (e.g., which ones increased or reduced risk)?
4. Explain why each of the all features might be medically significant.

Avoid machine learning terminology. Focus on medical interpretation, risk factors, and clinical reasoning. Write in a clear, professional tone suitable for doctors or healthcare practitioners. End with a short suggestion on next steps (e.g., “Recommend lifestyle changes or cardiac stress testing if symptomatic”).

Keep your explanation medically sound, non-alarming, and under 400 words.
"""
        with st.spinner("Generating explanation using Gemini..."):
            explanation = explain_with_gemini(heart_prompt)
            st.markdown("#### 🧠 Gemini Explanation")
            st.info(explanation)


st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with ❤️ using PyTorch, SHAP, Grad-CAM, Gemini, and Streamlit.")
st.markdown("© 2025 Sai Sambhu Prasad Kalaga. All rights reserved.")
